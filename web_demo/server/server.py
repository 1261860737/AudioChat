# Copyright (c) 2025, Alibaba Cloud and its affiliates;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FunAudioChat 语音对话 Web Demo 后端服务（S2S）。

- 通过 WebSocket 接收前端 Opus 音频流与控制信号
- 将用户语音拼接/保存后送入 S2S 模型进行流式生成（文本 + 语音 token）
- 语音 token 通过 TTS detokenizer（CosyVoice）流式还原为波形
- 波形编码为 Opus 帧并实时回传，同时按节奏回传文本
"""

import argparse
import asyncio
import os
import aiohttp
from aiohttp import web
import numpy as np
import sphn
import soundfile as sf
from datetime import datetime
import torch
import torchaudio
import json
import math
import uuid
import queue
import sys
import time
import threading
from pathlib import Path
from typing import Optional
from threading import Thread
import multiprocessing as mp
from multiprocessing import Process, Queue as MPQueue

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party"))
sys.path.insert(0, str(ROOT / "third_party" / "CosyVoice"))

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass
from transformers import AutoConfig, AutoProcessor, AutoModelForSeq2SeqLM

from web_demo.server.protocal import encode_handshake, decode_message
from web_demo.server.funaudiochat_infer import (
    FunaudioChatStreamer,
    remove_generate_text_special_token,
)

from utils.constant import *
from utils.cosyvoice_detokenizer import get_audio_detokenizer, tts_infer_streaming


def log(level, message):
    """简单日志输出：统一打印格式（便于在多线程/多进程下排查）。"""
    print(f"[{level.upper()}] {message}")


def tts_worker_process(
    input_queue: MPQueue,
    output_queue: MPQueue,
    control_queue: MPQueue,
    tts_gpu: int = 1,
):
    """TTS 子进程：将模型生成的语音 token 还原为音频波形。

    使用独立进程运行，主要目的是：
    - 避免 Python GIL 对 TTS 推理的影响
    - 将 TTS 模型与主进程解耦（也便于单独指定 GPU）

    Args:
        input_queue: 输入任务队列，元素为 (uuid, tokens_list, offset, finalize)
        output_queue: 输出队列，元素为 (uuid, audio_array)
        control_queue: 控制队列：('init_cache', uuid) / ('clear_cache', uuid) / ('stop', None)
        tts_gpu: TTS 使用的 GPU 设备 id（默认：1）
    """

    log("info", f"[TTS Process] Starting TTS worker process on cuda:{tts_gpu}...")
    torch.cuda.set_device(tts_gpu)
    tts_device = torch.device(f"cuda:{tts_gpu}")

    log("info", "[TTS Process] Loading TTS model...")
    tts_model = get_audio_detokenizer()

    tts_spk_emb_path = tts_model_config["spk_emb_path"]
    tts_spk_embedding = torch.load(tts_spk_emb_path)["中文女"]["embedding"]
    tts_spk_embedding = tts_spk_embedding.to(tts_device)

    log("info", "[TTS Process] TTS model loaded successfully")

    running = True
    while running:
        try:
            # 检查控制队列
            try:
                while not control_queue.empty():
                    cmd, data = control_queue.get_nowait()
                    if cmd == "init_cache":
                        uuid_str = data
                        tts_model.model.hift_cache_dict[uuid_str] = None
                        log("info", f"[TTS Process] Initialized cache for {uuid_str}")
                    elif cmd == "clear_cache":
                        uuid_str = data
                        if uuid_str in tts_model.model.hift_cache_dict:
                            del tts_model.model.hift_cache_dict[uuid_str]
                            log("info", f"[TTS Process] Cleared cache for {uuid_str}")
                    elif cmd == "stop":
                        running = False
                        log("info", "[TTS Process] Received stop command")
                        break
            except:
                pass

            if not running:
                break

            try:
                task = input_queue.get(timeout=0.1)
            except:
                continue

            uuid_str, tokens_list, offset, finalize = task

            queue_size = input_queue.qsize()
            if queue_size > 10:
                log("warning", f"[TTS Process] Input queue backlog: {queue_size} tasks pending")

            if uuid_str not in tts_model.model.hift_cache_dict:
                log("info", f"[TTS Process] Skipping task for cleared session {uuid_str[:8]}...")
                continue

            this_tokens = torch.tensor(tokens_list, dtype=torch.long, device=tts_device).view(1, -1)

            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            log("info", f"[{timestamp}] [TTS Process] TTS input: {len(tokens_list)} tokens, offset: {offset}")

            speech = tts_infer_streaming(
                tts_model,
                tts_spk_embedding,
                this_tokens,
                offset,
                uuid_str,
                finalize=finalize,
                token_hop_len=TOKEN_HOP_LEN,
                pre_lookahead_len=PRE_LOOKAHEAD_LEN,
                device=f"cuda:{tts_gpu}",
            )

            if speech is not None and speech.shape[-1] > 0:
                speech_array = speech[0].cpu().numpy()
                output_queue.put((uuid_str, speech_array))

                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                log("info", f"[{timestamp}] [TTS Process] TTS generated: {speech_array.shape[-1]} samples")

        except Exception as e:
            log("error", f"[TTS Process] Error: {e}")
            import traceback

            traceback.print_exc()

    log("info", "[TTS Process] TTS worker process stopped")


class GlobalModelManager:
    """全局模型管理器（单例）。

    负责：
    - 加载 HuggingFace Transformers 的 S2S 模型与 Processor
    - 保存生成参数（gen_kwargs / sp_gen_kwargs）
    - 提供目标采样率等全局信息
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self, model_path: str, target_sample_rate: int = 16000):
        """初始化并加载 S2S 主模型。

        Args:
            model_path: 模型权重/配置目录（HF 格式）
            target_sample_rate: 主模型期望的输入音频采样率（默认 16k）
        """

        if self._initialized:
            log("info", "initialized, skipping ...")
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_sample_rate = target_sample_rate

        log("info", f"loading s2s model to {self.device}...")

        config = AutoConfig.from_pretrained(model_path)
        text_config = getattr(config, "text_config", None)
        if text_config and getattr(text_config, "model_type", None) in ["qwen3_moe"]:
            setattr(text_config, "output_router_logits", False)

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, config=config, torch_dtype=torch.bfloat16
        ).to(self.device)

        # 设置生成参数（文本+语音 token 同步生成）
        self.gen_kwargs = DEFAULT_S2M_GEN_KWARGS.copy()
        if ("bad_words_ids" not in self.gen_kwargs or self.gen_kwargs["bad_words_ids"] is None):
            self.gen_kwargs["bad_words_ids"] = [
                [
                    self.processor.tokenizer.convert_tokens_to_ids("<|audio_bos|>"),
                    self.processor.tokenizer.convert_tokens_to_ids("<|sil|>"),
                ]
            ]

        self.model.sp_gen_kwargs = DEFAULT_SP_GEN_KWARGS.copy()

        log("info", f"s2s model loaded (: {self.device})")

        self._initialized = True
        log("info", f"waiting for tts model loading ... ")


class ServerState:
    """WebSocket 服务状态。

    - 维护主模型（S2S）引用与推理配置
    - 启动并管理全局 TTS 子进程及其队列
    - 每个连接在 handle_chat() 内创建多协程/多线程流水线
    """

    def __init__(
        self,
        model_manager: GlobalModelManager,
        sample_rate: int = 24000,
        output_dir: str = "./output",
        tts_gpu: int = 1,
        *,
        enable_diar_asr: bool = True,
        diar_device=None,
        speaker_num=None,
        model_cache_dir=None,
        speaker_embedding_model_path=None,
        vad_model_path=None,
        funasr_model: str = "/data/models/Voice/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        funasr_device: str = "cuda:0",
        funasr_vad_model_path: str = "/data/models/Voice/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        funasr_punc_model_path: str = "/data/models/Voice/iic/punc_ct-transformer_cn-en-common-vocab471067-large",
        instruction: str = DEFAULT_MEETING_INSTRUCTION,
        max_lines: int = 400,
    ):
        """初始化服务端状态并启动全局 TTS 子进程。

        Args:
            model_manager: 已初始化的全局主模型管理器
            sample_rate: WebSocket 侧 Opus/PCM 处理采样率（默认 24k）
            output_dir: 输入/输出音频保存目录
            tts_gpu: TTS 使用的 GPU id
        """

        self.model_manager = model_manager
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        self.tts_gpu = tts_gpu
        self.lock = asyncio.Lock()

        self.template = AUDIO_TEMPLATE
        self.APAD_TOKEN = AUDIO_PAD_TOKEN
        self.token_fps = TOKEN_FPS
        self.system_prompt = SPOKEN_S2M_PROMPT

        self.enable_diar_asr = enable_diar_asr
        self.diar_device = diar_device
        self.speaker_num = speaker_num
        self.model_cache_dir = model_cache_dir
        self.speaker_embedding_model_path = speaker_embedding_model_path
        self.vad_model_path = vad_model_path

        self.funasr_model = funasr_model
        self.funasr_device = funasr_device
        self.funasr_vad_model_path = funasr_vad_model_path
        self.funasr_punc_model_path = funasr_punc_model_path

        self.instruction = instruction
        self.max_lines = max_lines

        self._diarizer = None
        self._asr_transcriber = None
        self._diar_asr_init_lock = threading.Lock()

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "input"), exist_ok=True)
        log("info", f"Output directory: {self.output_dir}")

        # 全局 TTS 队列（主进程 <-> TTS 子进程）
        self.tts_input_queue = MPQueue()
        self.tts_output_queue = MPQueue()
        self.tts_control_queue = MPQueue()

        self.tts_process = Process(
            target=tts_worker_process,
            args=(
                self.tts_input_queue,
                self.tts_output_queue,
                self.tts_control_queue,
                self.tts_gpu,
            ),
            daemon=True,
        )
        self.tts_process.start()
        log("info", f"Global TTS process started (pid: {self.tts_process.pid})")

    def stop_tts_process(self):
        """停止全局 TTS 子进程（用于服务关闭或清理资源）。"""
        if self.tts_process and self.tts_process.is_alive():
            self.tts_control_queue.put(("stop", None))
            self.tts_process.join(timeout=5.0)
            if self.tts_process.is_alive():
                log("warning", "TTS process did not stop in time, terminating...")
                self.tts_process.terminate()
            log("info", "Global TTS process stopped")

    def _ensure_diar_asr_initialized(self) -> None:
        if not self.enable_diar_asr:
            return

        with self._diar_asr_init_lock:
            if self._diarizer is None:
                from audiochat.diarization.diarizer_3dspeaker import (ThreeDSpeakerDiarizer,)

                self._diarizer = ThreeDSpeakerDiarizer(
                    device=self.diar_device,
                    model_cache_dir=self.model_cache_dir,
                    speaker_num=self.speaker_num,
                    speaker_embedding_model_path=self.speaker_embedding_model_path,
                    vad_model_path=self.vad_model_path,
                )

            if self._asr_transcriber is None:
                from audiochat.asr.funasr_asr import FunASRTranscriber

                self._asr_transcriber = FunASRTranscriber(
                    model=self.funasr_model,
                    device=self.funasr_device,
                    vad_model=self.funasr_vad_model_path,
                    punc_model=self.funasr_punc_model_path,
                )

    def diarize_and_transcribe(
        self,
        *,
        waveform_16k_mono: torch.Tensor,
        sample_rate: int,
        speaker_num: Optional[int],
    ):
        """Run 3D-Speaker diarization + FunASR ASR for one turn.

        Returns:
            utterances: list[Utterance]
            diar_segments: list[DiarizationSegment]
        """

        if not self.enable_diar_asr:
            return [], []

        self._ensure_diar_asr_initialized()
        assert self._diarizer is not None
        assert self._asr_transcriber is not None

        from audiochat.audio_io import slice_waveform

        diar_segments = self._diarizer.diarize(
            waveform_16k_mono,
            wav_fs=sample_rate,
            speaker_num=speaker_num,
        )
        diar_segments = sorted(diar_segments, key=lambda s: (s.start_s, s.end_s, s.speaker))

        utterances = []
        raw_asr_items = []
        for seg in diar_segments:
            seg_audio = slice_waveform(
                waveform_16k_mono,
                seg.start_s,
                seg.end_s,
                sample_rate=sample_rate,
            )
            seg_start_ms = int(round(seg.start_s * 1000.0))
            spk = f"spk{seg.speaker}"
            uts, raw = self._asr_transcriber.transcribe_segment(
                seg_audio,
                speaker=spk,
                segment_start_ms=seg_start_ms,
            )
            utterances.extend(uts)
            raw_asr_items.append(raw)

        utterances.sort(key=lambda u: (u.start_ms, u.end_ms, u.speaker))
        return utterances, diar_segments

    def build_meeting_instruction(self, *, utterances, user_instruction: str | None = None) -> str:
        if not self.enable_diar_asr:
            return user_instruction or self.instruction

        from audiochat.prompting import build_llm_instruction

        return build_llm_instruction(
            utterances=utterances,
            user_instruction=user_instruction or self.instruction,
            max_lines=self.max_lines,
        )

    async def handle_chat(self, request):
        """处理单个 WebSocket 连接（/api/chat）。

        该方法内部会启动：
        - 一个接收协程：解析前端 Opus 音频/控制消息
        - 一个 PCM 累积协程：将解码后的 PCM 拼接成完整输入语音
        - 一个推理协程：在 pause/endTurn 时触发模型生成，并把 text/audio token 分流到队列
        - 一个编码线程：TTS 波形 -> Opus 帧
        - 两个发送协程：按节奏发送音频帧与文本
        """

        # 心跳间隔设为 30s（receive_timeout=None 避免中间代理/防火墙误断开）
        ws = web.WebSocketResponse(heartbeat=30.0, receive_timeout=None)
        await ws.prepare(request)

        client_id = f"Client-{id(ws)}"
        turn_counter = 0
        is_recording = True

        # 当前连接/会话的 system prompt（支持前端动态覆盖）
        session_system_prompt = self.system_prompt

        # 当前连接/会话的 user instruction（支持前端动态覆盖）
        session_instruction = self.instruction

        # 对话历史（messages 对应文本模板，audio_list 对应音频输入条目）
        messages = []
        audio_list = []
        current_generation = {
            "streamer": None,
            "accumulated_text": "",
            "generation_thread": None,
            "is_generating": False,
            "interrupt": False,
        }

        async def recv_loop():
            nonlocal close, is_recording, turn_counter, opus_reader
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        log("error", f"{ws.exception()}")
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        log("error", f"unexpected message type {message.type}")
                        continue
                    message = message.data
                    if not isinstance(message, bytes):
                        log("error", f"unsupported message type {type(message)}")
                        continue
                    if len(message) == 0:
                        log("warning", "empty message")
                        continue

                    try:
                        decoded = decode_message(message)
                        msg_type = decoded["type"]

                        if msg_type == "audio":
                            if is_recording:
                                payload = decoded["data"]
                                log("info", f"Received audio data: {len(payload)} bytes")
                                pcm = opus_reader.append_bytes(payload)
                                if pcm is not None and len(pcm) > 0:
                                    await pcm_queue.put(pcm)

                        elif msg_type == "control":
                            action = decoded["action"]
                            if action == "pause":
                                log("info", f"Received PAUSE signal")
                                is_recording = False
                                await save_audio_queue.put(("pause", None))
                            elif action == "start":
                                log("info", f"Received START signal")
                                is_recording = True
                                turn_counter += 1
                                await save_audio_queue.put(("start", None))
                            elif action == "endTurn":
                                log("info", f"Received END_TURN signal")
                                is_recording = False
                                await save_audio_queue.put(("pause", None))
                            else:
                                log("info", f"Received control: {action}")

                        elif msg_type == "ping":
                            log("info", "Received PING")

                        elif msg_type == "text":
                            log("info", f"Received text: {decoded['data']}")

                        elif msg_type == "metadata":
                            metadata = decoded["data"]
                            if isinstance(metadata, dict):
                                if "system_prompt" in metadata:
                                    nonlocal session_system_prompt
                                    session_system_prompt = metadata["system_prompt"]
                                    log("info", f"Received custom system prompt: {session_system_prompt[:100]}...")
                                if "instruction" in metadata:
                                    nonlocal session_instruction
                                    session_instruction = metadata["instruction"]
                                    log("info", f"Received custom instruction: {session_instruction[:100]}...")
                                if ("system_prompt" not in metadata) and ("instruction" not in metadata):
                                    log("info", f"Received metadata: {metadata}")
                            else:
                                log("info", f"Received metadata: {metadata}")

                        else:
                            log("warning", f"Unknown message type: {msg_type}")

                    except Exception as e:
                        log("error", f"Failed to decode message: {e}")
                        kind = message[0]
                        log("info", f"Trying old protocol, kind={kind}")
                        if kind == 1:  # 音频帧
                            if is_recording:
                                payload = message[1:]
                                log("info", f"Received audio data (old): {len(payload)} bytes")
                                
                                pcm = opus_reader.append_bytes(payload)
                                if pcm is not None and len(pcm) > 0:
                                    await pcm_queue.put(pcm)
                        elif kind == 2:  # 暂停信号
                            log("info", f"Received PAUSE signal (old)")
                            is_recording = False
                            await save_audio_queue.put(("pause", None))
                        elif kind == 3:  # 开始录音信号
                            log("info", f"Received START signal (old)")
                            is_recording = True
                            turn_counter += 1
                            await save_audio_queue.put(("start", None))

            finally:
                close = True
                log("info", "connection closed")

        async def save_audio_loop():
            """异步任务：处理录音保存与模型推理（含流式生成监控）。"""
            nonlocal \
                all_recorded_pcm, \
                turn_counter, \
                messages, \
                audio_list, \
                audio_buffer_list, \
                audio_buffer_lock, \
                all_generated_audio, \
                reset_first_frame, \
                reset_send_state, \
                this_uuid, \
                tts_offset, \
                cur_audio_tokens, \
                opus_reader, \
                accumulate_tts_tokens, \
                current_generation


            while True:
                if close:
                    return
                try:
                    signal_type, _ = await asyncio.wait_for(save_audio_queue.get(), timeout=0.1)

                    if signal_type == "pause":
                        # 保存音频并触发模型推理
                        if all_recorded_pcm is not None and len(all_recorded_pcm) > 0:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"{client_id}_turn{turn_counter}_input.wav"
                            filepath = os.path.join(self.output_dir, "input", filename)

                            audio_duration = len(all_recorded_pcm) / self.sample_rate

                            # 如采样率不一致，则重采样到模型期望采样率
                            if (self.sample_rate != self.model_manager.target_sample_rate):
                                audio_tensor = torch.from_numpy(all_recorded_pcm).unsqueeze(0)
                                resampler = torchaudio.transforms.Resample(
                                    self.sample_rate,
                                    self.model_manager.target_sample_rate,
                                )
                                audio_tensor = resampler(audio_tensor)
                                audio_for_model = audio_tensor.squeeze(0).numpy()
                            else:
                                audio_for_model = all_recorded_pcm

                            audio_tensor = torch.from_numpy(audio_for_model).unsqueeze(0)
                            try:
                                x = audio_tensor.detach().cpu().float().numpy()
                                if x.ndim == 2:
                                    x = x.T
                                x = np.clip(x, -1.0, 1.0).astype(np.float32)

                                sf.write(filepath, x, self.model_manager.target_sample_rate, subtype="PCM_16")
                            except Exception as e:
                                logger.exception(f"[WARN] soundfile save failed: {e}")
                            log("info", f"Saved audio to {filepath}, length: {audio_duration:.2f}s")


                            # 构造对话上下文（system/user/assistant）
                            if len(messages) == 0:
                                messages = [{"role": "system", "content": session_system_prompt}]

                            max_messages = (1 + MAX_HISTORY_TURNS * 2)  # 1 for system prompt
                            if len(messages) >= max_messages:
                                messages_to_remove = len(messages) - max_messages + 2
                                if messages_to_remove > 0:
                                    user_messages_removed = sum(1 for m in messages[1 : 1 + messages_to_remove] if m["role"] == "user")
                                    messages = [messages[0]] + messages[1 + messages_to_remove :]
                                    audio_list = audio_list[user_messages_removed:]
                                    log("info", f"Trimmed history: removed {messages_to_remove} messages, {user_messages_removed} audio items")

                            meeting_instruction = session_instruction
                            if self.enable_diar_asr:
                                try:
                                    msg = b"\x02" + bytes("[Diarization+ASR...]", encoding="utf8")
                                    await ws.send_bytes(msg)

                                    (utterances, _diar_segments,) = await asyncio.to_thread(
                                        self.diarize_and_transcribe,
                                        waveform_16k_mono=audio_tensor,
                                        sample_rate=self.model_manager.target_sample_rate,
                                        speaker_num=self.speaker_num,
                                    )
                                    meeting_instruction = self.build_meeting_instruction(
                                        utterances=utterances,
                                        user_instruction=session_instruction,
                                    )
                                except Exception as exc:
                                    log("error", f"Diarization+ASR failed: {exc}")

                            message_item = {"role": "user", "content": self.template + "\n" + meeting_instruction}
                            audio_tokens = self.APAD_TOKEN * int(math.ceil(audio_duration * self.token_fps))
                            audio_item = {
                                "path": filepath,
                                "token": audio_tokens,
                                "text": "",
                            }

                            messages.append(message_item)
                            audio_list.append(json.dumps(audio_item))

                            log("info", f"[Turn {turn_counter}] Preparing model input: {len(messages)} messages, {len(audio_list)} audio items")
                            log("info", f"[Turn {turn_counter}] Message history: {[m['role'] for m in messages]}")

                            log("info", f"[Turn {turn_counter}] Queue status: audio_tokens={audio_tokens_queue.qsize()}, opus_bytes={opus_bytes_queue.qsize()}")

                            text = self.model_manager.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                            inputs = self.model_manager.processor(
                                text=text,
                                audio=audio_list,
                                return_tensors="pt",
                                return_token_type_ids=False,
                            ).to(self.model_manager.device)

                            log("info", f"start inference of ({audio_duration:.2f}s audio)...")

                            msg = b"\x02" + bytes("[Processing...]", encoding="utf8")
                            await ws.send_bytes(msg)

                            group_size = getattr(self.model_manager.model.config.audio_config, "group_size", 5)
                            streamer = FunaudioChatStreamer(
                                self.model_manager.processor,
                                skip_prompt=True,
                                group_size=group_size,
                            )

                            gen_kwargs_with_streamer = (self.model_manager.gen_kwargs.copy())
                            gen_kwargs_with_streamer["streamer"] = streamer

                            generation_error = {"error": None}
                            generation_start_time = time.time()

                            def run_generation():
                                try:
                                    log("info", f"[Turn {turn_counter}] Generation thread started, input_ids shape: {inputs['input_ids'].shape}")
                                    self.model_manager.model.generate(**inputs, **gen_kwargs_with_streamer)
                                    elapsed = time.time() - generation_start_time
                                    log("info", f"[Turn {turn_counter}] Generation thread completed in {elapsed:.2f}s")
                                except Exception as e:
                                    generation_error["error"] = e
                                    elapsed = time.time() - generation_start_time
                                    log("error", f"[Turn {turn_counter}] Generation failed after {elapsed:.2f}s: {e}")
                                    import traceback

                                    traceback.print_exc()

                            generation_thread = Thread(target=run_generation)
                            generation_thread.start()

                            current_generation["streamer"] = streamer
                            current_generation["accumulated_text"] = ""
                            current_generation["generation_thread"] = generation_thread
                            current_generation["is_generating"] = True

                            last_step = 0
                            accumulated_text = ""
                            accumulated_audio_ids = []
                            first_audio_batch = True
                            loop_count = 0
                            last_step_change_time = time.time()
                            stuck_warning_shown = False

                            log("info", f"Start monitoring the generation results loop (turn {turn_counter})")
                            while generation_thread.is_alive() or last_step < len(streamer.get_step_results()):
                                loop_count += 1
                                if current_generation["interrupt"]:
                                    log("info", "Generation interrupted by new turn")
                                    break

                                if (generation_error["error"] is not None and not generation_thread.is_alive()):
                                    raise generation_error["error"]
                                step_results = streamer.get_step_results()

                                current_steps = len(step_results)
                                if current_steps != last_step:
                                    last_step_change_time = time.time()
                                    stuck_warning_shown = False
                                else:
                                    if (not stuck_warning_shown and (time.time() - last_step_change_time) > 30):
                                        log("warning", f"[Turn {turn_counter}] Generation appears stuck: no new steps for 10s (steps={current_steps}, loop={loop_count})")
                                        stuck_warning_shown = True

                                if loop_count % 100 == 0:
                                    elapsed_since_last_step = (time.time() - last_step_change_time)
                                    log("info", f"Monitor loop #{loop_count}: thread_alive={generation_thread.is_alive()}, steps={len(step_results)}, last_step={last_step}, stuck_time={elapsed_since_last_step:.1f}s")

                                for i in range(last_step, len(step_results)):
                                    step = step_results[i]

                                    # 取新增的文本片段（流式）
                                    if step["new_text_str"]:
                                        new_text = step["new_text_str"]
                                        if (new_text and not new_text.startswith("<|")):
                                            accumulated_text += new_text
                                            current_generation["accumulated_text"] = (accumulated_text)

                                    # 取新增的语音 token 并放入队列（供 TTS 流式还原）
                                    if "new_audio_ids" in step:
                                        audio_ids = step["new_audio_ids"]
                                        if audio_ids is not None and len(audio_ids) > 0:
                                            if first_audio_batch:
                                                log("info", f"Skipping first audio batch (prompt): shape={audio_ids.shape}, tokens={audio_ids.shape[1]}")
                                                first_audio_batch = False
                                            else:
                                                try:
                                                    valid_codes = []
                                                    for code in audio_ids[0]:
                                                        code_val = (code.item() if torch.is_tensor(code) else code)
                                                        if 0 <= code_val < 6561:
                                                            valid_codes.append(code_val)
                                                    accumulated_audio_ids.extend(valid_codes)

                                                    for code in valid_codes:
                                                        audio_tokens_queue.put_nowait(code)

                                                    if len(valid_codes) > 0:
                                                        timestamp = (datetime.now().strftime("%H:%M:%S.%f")[:-3])
                                                        log("info", f"[{timestamp}] Added {len(valid_codes)} valid audio codes to queue (skipped {len(audio_ids[0]) - len(valid_codes)} invalid)")
                                                    else:
                                                        log("warning",f"No valid audio codes in this batch (all {len(audio_ids[0])} tokens were invalid)")
                                                except Exception as e:
                                                    log("error", f"Failed to process audio tokens: {e}")

                                last_step = len(step_results)
                                await asyncio.sleep(0.05)

                            log("info", f"Monitor loop ended: total_loops={loop_count}, final_steps={len(streamer.get_step_results())}, interrupted={current_generation['interrupt']}")
                            generation_thread.join()

                            # 若被新一轮录音打断，则跳过后续收尾处理
                            if current_generation["interrupt"]:
                                log("info", "Skipping post-generation processing due to interrupt")

                                current_generation["is_generating"] = False
                                current_generation["streamer"] = None
                                current_generation["accumulated_text"] = ""
                                current_generation["generation_thread"] = None
                                continue

                            # 标记 TTS 侧的 token 生成已结束（用于触发 flush）
                            tts_generation_complete["flag"] = True
                            log("info", "TTS generation marked as complete")

                            # 等待编码线程处理完所有剩余音频（防止最后一小段丢失）
                            max_wait_time = 15  # 最多等待 15 秒
                            wait_interval = 0.1
                            waited = 0
                            while waited < max_wait_time:
                                with audio_buffer_lock:
                                    buffer_empty = len(audio_buffer_list) == 0

                                if buffer_empty and opus_bytes_queue.qsize() == 0:
                                    log("info", f"All audio encoded and sent after {waited:.1f}s")
                                    break

                                await asyncio.sleep(wait_interval)
                                waited += wait_interval

                            if waited >= max_wait_time:
                                log("warning", f"Encoding timeout after {max_wait_time}s")

                            # 获取最终结果（累积的完整文本/语音 token）
                            final_results = streamer.get_accumulated_results()
                            generate_text = final_results["text_str"]

                            log("info", f"Generation completed: {generate_text}")

                            # 保存完整的生成音频（把流式片段拼起来落盘）
                            assistant_audio_path = None
                            assistant_audio_duration = 0.0
                            if len(all_generated_audio) > 0:
                                try:
                                    # # 拼接所有生成的音频片段
                                    # full_generated_audio = np.concatenate(all_generated_audio)
                                    # assistant_audio_duration = (len(full_generated_audio) / self.sample_rate)
                                    # log("info", f"Concatenated generated audio: {len(all_generated_audio)} segments, total length {assistant_audio_duration:.2f}s ({len(full_generated_audio)} samples)")

                                    def save_full_audio(segments, sample_rate, output_path):
                                        full_audio = np.concatenate(segments)
                                        duration = len(full_audio) / sample_rate
                                        audio_to_save = full_audio.astype(np.float32)
                                        if np.abs(audio_to_save).max() > 1.0:
                                            audio_to_save = (audio_to_save / np.abs(audio_to_save).max())
                                        sf.write(output_path, audio_to_save, sample_rate)
                                        return duration, len(segments), len(full_audio)

                                    # 保存为 WAV 文件（在线程中执行，避免阻塞事件循环）
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    output_filename = f"{client_id}_turn{turn_counter}_output_{timestamp}.wav"
                                    assistant_audio_path = os.path.join(self.output_dir, output_filename)


                                    # # 确保音频为 float32 且幅度落在 [-1, 1]
                                    # audio_to_save = full_generated_audio.astype(np.float32)
                                    # if np.abs(audio_to_save).max() > 1.0:
                                    #     audio_to_save = (audio_to_save / np.abs(audio_to_save).max())

                                    segments = list(all_generated_audio)
                                    (assistant_audio_duration, segment_count, sample_count) = await asyncio.to_thread(
                                        save_full_audio,
                                        segments,
                                        self.sample_rate,
                                        assistant_audio_path,
                                    )
                                    log("info", f"Concatenated generated audio: {segment_count} segments, total length {assistant_audio_duration:.2f}s ({sample_count} samples)")

                                    # sf.write(assistant_audio_path, audio_to_save, self.sample_rate)

                                    log("info", f"Saved generated audio: {assistant_audio_path}")

                                    # 清空累计的生成音频，准备下一轮
                                    all_generated_audio.clear()
                                except Exception as e:
                                    log("error", f"Failed to save generated audio: {e}")
                                    import traceback

                                    traceback.print_exc()
                                    assistant_audio_path = None
                                    assistant_audio_path = None

                            # 清理特殊 token，并校准最终文本（如果需要）
                            clean_text = remove_generate_text_special_token(
                                generate_text
                            )
                            if clean_text != accumulated_text:
                                accumulated_text = clean_text
                                current_generation["accumulated_text"] = clean_text

                            messages.append(
                                {"role": "assistant", "content": clean_text}
                            )

                            # 重置本轮生成状态
                            current_generation["is_generating"] = False
                            current_generation["streamer"] = None
                            current_generation["accumulated_text"] = ""
                            current_generation["generation_thread"] = None

                            log("info", f"Processing completed")
                        else:
                            log("warning", "No audio data to save")

                    elif signal_type == "start":
                        # 若上一轮仍在生成中，则先设置 interrupt 并等待其停止
                        if current_generation["is_generating"]:
                            log("info", "Interrupting current generation...")
                            current_generation["interrupt"] = True
                            if current_generation["generation_thread"] is not None:
                                current_generation["generation_thread"].join(
                                    timeout=2.0
                                )
                                if current_generation["generation_thread"].is_alive():
                                    log("warning", "Generation thread did not stop in time")
                            current_generation["is_generating"] = False
                            current_generation["interrupt"] = False
                            log("info", "Previous generation stopped")

                        all_recorded_pcm = None
                        with audio_buffer_lock:
                            all_generated_audio.clear()
                            audio_buffer_list.clear()

                        cleared_count = 0
                        while not opus_bytes_queue.empty():
                            try:
                                opus_bytes_queue.get_nowait()
                                cleared_count += 1
                            except queue.Empty:
                                break
                        if cleared_count > 0:
                            log("info", f"Cleared {cleared_count} opus frames from queue")

                        opus_reader = sphn.OpusStreamReader(self.sample_rate)
                        log("info", "Reset opus_reader for new turn")

                        cleared_pcm = 0
                        try:
                            while True:
                                pcm_queue.get_nowait()
                                cleared_pcm += 1
                        except asyncio.QueueEmpty:
                            pass
                        if cleared_pcm > 0:
                            log("info", f"Cleared {cleared_pcm} PCM chunks from queue")

                        reset_first_frame["flag"] = True
                        reset_send_state["flag"] = True
                        tts_generation_complete["flag"] = False
                        frame_generation_complete["flag"] = False
                        audio_send_started["flag"] = False

                        # 重置 TTS 相关状态（加锁保护：cur_audio_tokens / tts_offset / this_uuid）
                        with tts_state_lock:
                            cur_audio_tokens.clear()
                            tts_offset = 0
                            accumulate_tts_tokens = 0
                            old_uuid = this_uuid
                            this_uuid = str(uuid.uuid4())

                            tts_control_queue.put(("clear_cache", old_uuid))
                            tts_control_queue.put(("init_cache", this_uuid))

                        cleared_tokens = 0
                        try:
                            while True:
                                audio_tokens_queue.get_nowait()
                                cleared_tokens += 1
                        except queue.Empty:
                            pass
                        if cleared_tokens > 0:
                            log("info", f"Cleared {cleared_tokens} audio tokens from queue")


                        log("info", "Cleared audio buffer and TTS state for new recording")

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    log("error", f"Inference failed: {e}")
                    import traceback

                    traceback.print_exc()
                    try:
                        error_msg = b"\x05" + bytes(f"Processing failed: {str(e)}", encoding="utf8")
                        await ws.send_bytes(error_msg)
                    except:
                        pass

        async def accumulate_pcm_loop():
            nonlocal all_recorded_pcm
            total_samples = 0
            last_all_recorded_pcm_id = (id(all_recorded_pcm) if all_recorded_pcm is not None else None)

            while True:
                if close:
                    return

                current_id = (id(all_recorded_pcm) if all_recorded_pcm is not None else None)
                if all_recorded_pcm is None and last_all_recorded_pcm_id is not None:
                    total_samples = 0
                    log("info", "Reset PCM accumulator for new turn")
                last_all_recorded_pcm_id = current_id

                # 从队列读取解码后的 PCM 数据块
                try:
                    pcm = await asyncio.wait_for(pcm_queue.get(), timeout=0.01)
                except asyncio.TimeoutError:
                    continue

                if pcm is None or len(pcm) == 0:
                    continue

                if is_recording:
                    pcm_samples = len(pcm)
                    total_samples += pcm_samples
                    log("info", f"Accumulating PCM: {pcm_samples} samples (total: {total_samples}, {total_samples / self.sample_rate:.2f}s)")

                    if all_recorded_pcm is None:
                        all_recorded_pcm = pcm
                    else:
                        all_recorded_pcm = np.concatenate((all_recorded_pcm, pcm))

        def tts_sender_thread_func():
            """[多进程] 将语音 token 按窗口打包发送给 TTS 子进程。"""
            nonlocal \
                tts_offset, \
                cur_audio_tokens, \
                this_uuid, \
                max_tts_tokens, \
                tts_generation_complete, \
                accumulate_tts_tokens

            token_hop_len = 15
            pre_lookahead_len = 3

            log("info", "TTS sender thread started")

            while not close:
                try:
                    finalize = False
                    try:
                        audio_token = audio_tokens_queue.get(timeout=0.1)

                        with tts_state_lock:
                            cur_audio_tokens.append(audio_token)
                            if (len(cur_audio_tokens) < tts_offset + token_hop_len + pre_lookahead_len):
                                continue
                    except queue.Empty:
                        if tts_generation_complete["flag"]:
                            finalize = True
                            frame_generation_complete["flag"] = True
                            with tts_state_lock:
                                if (len(cur_audio_tokens) <= tts_offset + 1 + pre_lookahead_len):
                                    continue
                        else:
                            continue

                    with tts_state_lock:
                        tokens_to_send = cur_audio_tokens.copy()
                        local_tts_offset = tts_offset
                        local_this_uuid = this_uuid

                    tts_input_queue.put((local_this_uuid, tokens_to_send, local_tts_offset, finalize))

                    # 更新流式 TTS 的 offset（滑窗推进/环形缓冲）
                    with tts_state_lock:
                        tts_offset += token_hop_len
                        if tts_offset >= max_tts_tokens:
                            tts_offset -= max_tts_tokens
                            cur_audio_tokens = cur_audio_tokens[max_tts_tokens:]
                            accumulate_tts_tokens += max_tts_tokens
                            if accumulate_tts_tokens >= MAX_TTS_HISTORY:
                                accumulate_tts_tokens = 0
                                tts_control_queue.put(("clear_cache", this_uuid))
                                tts_control_queue.put(("init_cache", this_uuid))

                except Exception as e:
                    log("error", f"TTS sender thread error: {e}")
                    import traceback

                    traceback.print_exc()

            log("info", "TTS sender thread stopped")

        def tts_receiver_thread_func():
            """[多进程] 接收 TTS 子进程输出（仅保留当前会话 uuid 对应的数据）。"""
            log("info", f"TTS receiver thread started for session {this_uuid}")

            while not close:
                try:
                    try:
                        uuid_str, speech_array = tts_output_queue.get(timeout=0.1)
                    except:
                        continue

                    with tts_state_lock:
                        current_uuid = this_uuid

                    if uuid_str != current_uuid:
                        log("info", f"Discarding TTS output from old session {uuid_str[:8]}... (current: {current_uuid[:8]}...)")
                        continue

                    with audio_buffer_lock:
                        all_generated_audio.append(speech_array.copy())
                        audio_buffer_list.append(speech_array.copy())

                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    log("info", f"[{timestamp}] Received TTS audio: {speech_array.shape[-1]} samples")

                except Exception as e:
                    log("error", f"TTS receiver thread error: {e}")
                    import traceback

                    traceback.print_exc()

            log("info", "TTS receiver thread stopped")

        def encode_thread_func():
            """编码线程：从音频缓冲取出完整帧，编码为 Opus 并放入 opus_bytes_queue。"""
            nonlocal \
                reset_first_frame, \
                tts_generation_complete, \
                frame_generation_complete

            local_opus_writer = sphn.OpusStreamWriter(self.sample_rate)
            frame_size = int(
                self.sample_rate * 0.04
            )  # 40ms 帧（24kHz 下约 960 samples）
            local_buffer = np.array([], dtype=np.float32)
            is_first_frame = True
            first_frame_start_time = None

            log("info", "Encode thread started")

            while not close:
                try:
                    if reset_first_frame["flag"]:
                        is_first_frame = True
                        first_frame_start_time = None
                        reset_first_frame["flag"] = False
                        local_buffer = np.array([], dtype=np.float32)
                        log("info", "Reset first frame flag for new conversation turn")

                    with audio_buffer_lock:
                        if len(audio_buffer_list) > 0:
                            chunk = audio_buffer_list.pop(0)
                        else:
                            chunk = None

                    if chunk is None:
                        if frame_generation_complete["flag"] and len(local_buffer) > 0:
                            log("info", f"TTS completed, flushing remaining {len(local_buffer)} samples")

                            if is_first_frame:
                                is_first_frame = False
                                log("info", "Skipping first frame delay due to TTS completion")

                            while len(local_buffer) >= frame_size:
                                frame = local_buffer[:frame_size]
                                local_buffer = local_buffer[frame_size:]

                                opus_bytes = local_opus_writer.append_pcm(frame)
                                if opus_bytes is not None and len(opus_bytes) > 0:
                                    opus_bytes_queue.put(opus_bytes)

                            # 处理最后不足一帧的尾段（用 0 补齐）
                            if len(local_buffer) > 0:
                                padding = np.zeros(frame_size - len(local_buffer), dtype=np.float32)
                                frame = np.concatenate([local_buffer, padding])
                                local_buffer = np.array([], dtype=np.float32)

                                opus_bytes = local_opus_writer.append_pcm(frame)
                                if opus_bytes is not None and len(opus_bytes) > 0:
                                    opus_bytes_queue.put(opus_bytes)
                                    log("info", "Encoded final partial frame")

                            log("info", "All audio flushed to queue")

                        time.sleep(0.01)
                        continue

                    # 将新到的波形片段追加到本地缓冲
                    local_buffer = np.concatenate([local_buffer, chunk])

                    # 若是本轮第一帧，记录开始时间（用于首帧缓冲/对齐）
                    if (is_first_frame and first_frame_start_time is None and len(local_buffer) > 0):
                        first_frame_start_time = time.time()
                        log("info", f"First audio data received, waiting before encoding...")

                    # 检查缓冲区中是否已经累计出完整帧
                    while len(local_buffer) >= frame_size:
                        # 首帧：可适当等待积累一些数据再开始编码（避免过短导致抖动）
                        if is_first_frame:
                            if first_frame_start_time is not None:
                                elapsed = time.time() - first_frame_start_time

                            is_first_frame = False
                            log("info", f"Starting to encode (buffer: {len(local_buffer)} samples)")

                        frame = local_buffer[:frame_size]
                        local_buffer = local_buffer[frame_size:]

                        # 编码为 Opus 帧
                        opus_bytes = local_opus_writer.append_pcm(frame)
                        if opus_bytes is not None and len(opus_bytes) > 0:
                            opus_bytes_queue.put(opus_bytes)

                except Exception as e:
                    log("error", f"Encode thread error: {e}")
                    import traceback

                    traceback.print_exc()

            log("info", "Encode thread stopped")

        async def send_audio_loop():
            """异步发送：从队列读取 Opus 数据，并按固定间隔发送（控制播放节奏）。"""
            nonlocal reset_send_state, audio_send_started
            frame_interval = 0.04
            next_send_time = None
            frames_sent = 0
            empty_streak = 0
            empty_streak_start = None
            last_empty_log = 0.0
            lag_spike_start = None
            # startup_buffer_frames = 5
            # startup_buffer_ready = False

            while not close:
                try:
                    if reset_send_state["flag"]:
                        next_send_time = None
                        frames_sent = 0
                        reset_send_state["flag"] = False
                        empty_streak = 0
                        empty_streak_start = None
                        last_empty_log = 0.0
                        lag_spike_start = None
                        # startup_buffer_ready = False
                        log("info", "Reset send state for new turn")

                    # if not startup_buffer_ready:
                    #     queued = opus_bytes_queue.qsize()
                    #     if queued < startup_buffer_frames:
                    #         await asyncio.sleep(0.005)
                    #         continue
                    #     startup_buffer_ready = True
                    #     log("info", f"Audio startup buffer ready: {queued} frames queued")

                    try:
                        opus_bytes = opus_bytes_queue.get_nowait()
                    except queue.Empty:
                        now = time.time()
                        if empty_streak == 0:
                            empty_streak_start = now
                        empty_streak += 1
                        if empty_streak_start is not None and (now - empty_streak_start) > 1.0 and (now - last_empty_log) > 1.0:
                            last_empty_log = now
                            log("warning", f"Opus queue empty for {now - empty_streak_start:.1f}s (streak={empty_streak})")
                        await asyncio.sleep(0.005)
                        continue

                    if empty_streak > 0 and empty_streak_start is not None:
                        empty_duration = time.time() - empty_streak_start
                        log("info", f"Opus queue resumed after {empty_duration:.1f}s (streak={empty_streak})")
                        empty_streak = 0
                        empty_streak_start = None

                    current_time = time.time()

                    if next_send_time is None:
                        next_send_time = current_time

                    wait_time = next_send_time - current_time

                    if wait_time > 0:
                        await asyncio.sleep(wait_time)

                    await ws.send_bytes(b"\x01" + opus_bytes)
                    frames_sent += 1

                    # 发送首帧音频后，通知文本发送协程可以开始发送
                    if frames_sent == 1:
                        audio_send_started["flag"] = True
                        log("info", "Audio sending started, text buffer can start sending")

                    next_send_time += frame_interval

                    lag = time.time() - next_send_time
                    if lag > 0.5:
                        if lag_spike_start is None:
                            lag_spike_start = time.time()
                        log("warning", f"Send lag detected: {lag * 1000:.0f}ms, resetting time base")
                        next_send_time = time.time() + frame_interval
                    elif lag_spike_start is not None and lag <= 0.05:
                        recovered = time.time() - lag_spike_start
                        log("info", f"Send lag recovered in {recovered:.1f}s")
                        lag_spike_start = None

                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    if frames_sent % 50 == 0:
                        log("info", f"[{timestamp}] Sent frame #{frames_sent} (queue: {opus_bytes_queue.qsize()}, lag: {lag * 1000:.1f}ms)")
                except Exception as e:
                    log("error", f"Send coroutine error: {e}")
                    import traceback

                    traceback.print_exc()

            log("info", f"Send coroutine stopped, total frames sent: {frames_sent}")

        async def send_text_loop():
            """异步发送：按固定节奏发送文本快照（尾部窗口）。"""
            nonlocal audio_send_started, current_generation
            text_interval = 0.5  # 500ms per snapshot
            max_snapshot_lines = 300
            suffix_tokens = 4096
            guard_tokens = 512
            snapshots_sent = 0
            current_turn_started = False
            last_sent_snapshot = ""
            next_send_time = None
            last_token_count = 0

            log("info", "Text send coroutine started")

            while not close:
                try:
                    if not current_turn_started:
                        while not close and not audio_send_started["flag"]:
                            await asyncio.sleep(0.05)
                        if close:
                            log("info", "Text send coroutine stopped while waiting for audio")
                            break
                        current_turn_started = True
                        last_sent_snapshot = ""
                        last_token_count = 0
                        next_send_time = None
                        log("info", "Audio started, beginning text snapshot transmission")

                    if current_turn_started and not audio_send_started["flag"]:
                        current_turn_started = False
                        last_sent_snapshot = ""
                        last_token_count = 0
                        next_send_time = None
                        log("info", "New turn detected, resetting text send state")
                        await asyncio.sleep(0.05)
                        continue

                    current_time = time.time()
                    if next_send_time is None:
                        next_send_time = current_time
                    wait_time = next_send_time - current_time
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                    current_time = time.time()
                    if next_send_time is None:
                        next_send_time = current_time
                    wait_time = next_send_time - current_time
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)

                    streamer = current_generation.get("streamer")
                    if streamer is None:
                        await asyncio.sleep(0.05)
                        continue

                    token_cache = list(streamer.text_token_cache)
                    token_count = 0
                    for chunk in token_cache:
                        try:
                            token_count += int(chunk.shape[1])
                        except Exception:
                            pass

                    if token_count == 0 or token_count == last_token_count:
                        await asyncio.sleep(0.05)
                        continue

                    last_token_count = token_count

                    def decode_suffix_text():
                        if not token_cache:
                            return ""
                        take = suffix_tokens + guard_tokens
                        total = 0
                        for chunk in token_cache:
                            total += int(chunk.shape[1])
                        if total <= 0:
                            return ""
                        take = min(total, take)
                        remaining = take
                        slices = []
                        for chunk in reversed(token_cache):
                            if remaining <= 0:
                                break
                            try:
                                chunk_cpu = chunk.detach().to("cpu")
                            except Exception:
                                chunk_cpu = chunk
                            chunk_len = int(chunk_cpu.shape[1])
                            if chunk_len <= remaining:
                                slices.append(chunk_cpu)
                                remaining -= chunk_len
                            else:
                                slices.append(chunk_cpu[:, -remaining:])
                                remaining = 0
                        if not slices:
                            return ""
                        slices.reverse()
                        text_ids = torch.cat(slices, dim=1)
                        try:
                            text = streamer.processor.decode(text_ids[0])
                        except Exception:
                            text = ""
                        text = remove_generate_text_special_token(text)
                        if total > (suffix_tokens + guard_tokens):
                            lines = text.splitlines()
                            if len(lines) > 1:
                                text = "\n".join(lines[1:])
                        return text

                    snapshot_source = await asyncio.to_thread(decode_suffix_text)
                    if not snapshot_source:
                        await asyncio.sleep(0.05)
                        continue

                    snapshot_lines = snapshot_source.splitlines()
                    if len(snapshot_lines) > max_snapshot_lines:
                        snapshot_lines = snapshot_lines[-max_snapshot_lines:]
                    snapshot = "\n".join(snapshot_lines)

                    if snapshot and snapshot != last_sent_snapshot:
                        try:
                            msg = b"\x08" + bytes(snapshot, encoding="utf8")
                            await ws.send_bytes(msg)
                            snapshots_sent += 1
                            last_sent_snapshot = snapshot
                            log("info", f"Sent text snapshot #{snapshots_sent} (lines={len(snapshot_lines)})")
                        except Exception as send_err:
                            log("error", f"Failed to send text snapshot: {send_err}")
                            break

                    next_send_time = time.time() + text_interval
                except Exception as e:
                    log("error", f"Text send coroutine error: {e}")
                    import traceback

                    traceback.print_exc()

            log("info", f"Text send coroutine stopped, total snapshots sent: {snapshots_sent}")

        log("info", "accepted connection")
        close = False
        all_recorded_pcm = None  # 累积保存用户录音的 PCM 数据
        all_generated_audio = []  # 累积保存生成的音频片段（用于最终拼接落盘）
        audio_buffer_list = []  # TTS 子进程与主进程之间的音频缓冲（PCM chunk）
        audio_buffer_lock = (threading.Lock())  # 保护 audio_buffer_list 与 all_generated_audio（线程安全）
        reset_first_frame = {"flag": True}  # 控制编码线程重置“首帧”状态（用 dict 便于线程共享）
        reset_send_state = {"flag": False}  # 控制发送协程重置帧计数与时间基准
        tts_generation_complete = {"flag": False}  # 标记语音 token 是否已全部生成完毕
        frame_generation_complete = {"flag": False}  # 标记 TTS 生成（波形还原）是否已结束
        audio_send_started = {"flag": False}  # 标记音频是否已开始发送（文本发送要等首帧音频后才能开始）
        save_audio_queue = asyncio.Queue()  # 录音控制信号队列（start/pause 等）
        audio_tokens_queue = queue.Queue()  # 线程安全队列：避免阻塞 asyncio 事件循环
        pcm_queue = asyncio.Queue()  # 解码后的 PCM 数据队列（由 Opus reader 产出）
        opus_bytes_queue = (queue.Queue())  # Opus 编码后的字节流队列（线程安全，用于发送）
        # text_buffer_queue = queue.Queue()  # 线程安全文本缓冲队列（按节奏发送）
        cur_audio_tokens = []
        tts_offset = 0
        accumulate_tts_tokens = 0
        max_tts_tokens = MAX_TTS_TOKENS
        this_uuid = str(uuid.uuid4())
        tts_state_lock = (threading.Lock())  # 保护 cur_audio_tokens / tts_offset / this_uuid（线程安全）
        # TTS（使用全局子进程）
        tts_input_queue = self.tts_input_queue
        tts_output_queue = self.tts_output_queue
        tts_control_queue = self.tts_control_queue

        tts_control_queue.put(("init_cache", this_uuid))
        log("info", f"Initialized TTS cache for session {this_uuid}")

        opus_reader = sphn.OpusStreamReader(self.sample_rate)

        loop = asyncio.get_event_loop()

        tts_sender_thread = Thread(target=tts_sender_thread_func, daemon=True)
        tts_receiver_thread = Thread(target=tts_receiver_thread_func, daemon=True)
        encode_thread = Thread(target=encode_thread_func, daemon=True)

        tts_sender_thread.start()
        tts_receiver_thread.start()
        encode_thread.start()

        log("info", "All workers started (TTS in separate process, encode thread active)")
        # 发送握手包：告知协议版本与模型类型
        handshake = encode_handshake(version=0, model=0)
        await ws.send_bytes(handshake)
        log("info", "Sent handshake")

        try:
            await asyncio.gather(
                recv_loop(),
                save_audio_loop(),
                accumulate_pcm_loop(),
                send_audio_loop(),
                send_text_loop(),
            )
        finally:
            close = True
            log("info", "Waiting for workers to stop...")

            tts_control_queue.put(("clear_cache", this_uuid))
            log("info", f"Cleared TTS cache for session {this_uuid}")

            tts_sender_thread.join(timeout=2.0)
            tts_receiver_thread.join(timeout=2.0)
            encode_thread.join(timeout=2.0)

            log("info", "All worker threads stopped")

        log("info", "done with connection")
        return ws


def main():
    """服务端入口：解析参数、加载模型、启动 aiohttp Web 服务。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=11236, type=int)
    parser.add_argument(
        "--sample-rate", default=24000, type=int, help="Audio sample rate (Opus)"
    )
    parser.add_argument(
        "--model-sample-rate", default=16000, type=int, help="Model sample rate"
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        type=str,
        help="Directory to save input audio files",
    )
    parser.add_argument(
        "--model-path", type=str, default="model/s2s", help="Path to S2S model"
    )
    parser.add_argument(
        "--tts-gpu",
        default=1,
        type=int,
        help="GPU device id for TTS model (default: 1)",
    )

    parser.add_argument(
        "--disable-diar-asr",
        action="store_true",
        help="Disable 3D-Speaker diarization + FunASR ASR before LLM",
    )

    # 3D-Speaker diarization options
    parser.add_argument("--diar-device", default=None, help="cuda / cuda:0 / cpu")
    parser.add_argument("--speaker-num", type=int, default=None)
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument(
        "--speaker-embedding-model-path",
        default="/data/models/Voice/iic/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt",
        help="Local path to speaker embedding model (.pt file)",
    )
    parser.add_argument(
        "--vad-model-path",
        default="/data/models/Voice/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        help="Local path to VAD model directory",
    )

    # FunASR ASR options
    parser.add_argument(
        "--funasr-model",
        default="/data/models/Voice/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    )
    parser.add_argument("--funasr-device", default="cuda:0")
    parser.add_argument(
        "--funasr-vad-model-path",
        default="/data/models/Voice/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        help="Local path to FunASR VAD model directory",
    )
    parser.add_argument(
        "--funasr-punc-model-path",
        default="/data/models/Voice/iic/punc_ct-transformer_cn-en-common-vocab471067-large",
        help="Local path to FunASR punctuation model directory",
    )

    # Meeting minutes instruction
    parser.add_argument(
        "--instruction",
        default=DEFAULT_MEETING_INSTRUCTION,
    )
    parser.add_argument("--max-lines", type=int, default=400)

    args = parser.parse_args()

    log("info", f"Initializing server with Opus sample rate: {args.sample_rate}, Model sample rate: {args.model_sample_rate}")

    model_manager = GlobalModelManager()
    try:
        model_manager.initialize(model_path=args.model_path, target_sample_rate=args.model_sample_rate)
    except Exception as e:
        log("error", f"Model initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return

    state = ServerState(
        model_manager=model_manager,
        sample_rate=args.sample_rate,
        output_dir=args.output_dir,
        tts_gpu=args.tts_gpu,
        enable_diar_asr=not args.disable_diar_asr,
        diar_device=args.diar_device,
        speaker_num=args.speaker_num,
        model_cache_dir=args.model_cache_dir,
        speaker_embedding_model_path=args.speaker_embedding_model_path,
        vad_model_path=args.vad_model_path,
        funasr_model=args.funasr_model,
        funasr_device=args.funasr_device,
        funasr_vad_model_path=args.funasr_vad_model_path,
        funasr_punc_model_path=args.funasr_punc_model_path,
        instruction=args.instruction,
        max_lines=args.max_lines,
    )

    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)

    protocol = "http"
    log("info", f"Access the Web UI directly at {protocol}://{args.host}:{args.port}")
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
