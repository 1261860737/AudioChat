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

"""FunAudioChat 语音对话 Web Demo 后端服务（Diarization + ASR + vLLM + Qwen3-TTS）。

- 通过 WebSocket 接收前端 Opus 音频流与控制信号
- 在 pause/endTurn 时触发说话人分离 + ASR
- 将 ASR 文本送入本地 vLLM 生成回复（支持流式）
- 生成文本分段送入 Qwen3-TTS，流式回传音频
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
import re
import queue
import sys
import time
import threading
from pathlib import Path
from typing import Optional
from threading import Thread

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party"))
os.environ.setdefault("GITEA_DRY_RUN", "0")

from web_demo.server.protocal import encode_handshake, decode_message

from actions.mail_dispatcher.dispatch_to_gitea import dispatch_master_issue
from actions.mail_dispatcher.meeting_result_parser import (
    get_beijing_date_str,
    parse_meeting_result_from_llm,
    render_meeting_result_markdown,
)
from utils.constant import *


def log(level, message):
    """简单日志输出：统一打印格式（便于在多线程/多进程下排查）。"""
    print(f"[{level.upper()}] {message}")


async def vllm_chat_completion(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 512,
    timeout_s: float = 300.0,
) -> str:
    base_url = base_url.rstrip("/")
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"vLLM HTTP {resp.status}: {text}")
            data = json.loads(text)
    try:
        return str(data["choices"][0]["message"]["content"]).strip()
    except Exception as exc:
        raise RuntimeError(f"Unexpected vLLM response schema: keys={list(data.keys())}") from exc


async def vllm_stream_chat_completion(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 512,
    timeout_s: float = 300.0,
    on_delta,
) -> str:
    base_url = base_url.rstrip("/")
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": True,
    }
    full_text = ""
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"vLLM HTTP {resp.status}: {body}")
            async for raw_line in resp.content:
                for line in raw_line.decode("utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        return full_text
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    try:
                        delta = event["choices"][0]["delta"].get("content", "")
                    except Exception:
                        delta = ""
                    if delta:
                        full_text += delta
                        await on_delta(delta)
    return full_text


_TTS_SENT_END_RE = re.compile(r"(?:[。！？!?]|\\.{3,}|…+|\\n)")
_TTS_MARKDOWN_LINE_RE = re.compile(
    r"(?:^|\\n)(?:#{1,6}\\s+|[-*+]\\s+|\\d+\\.\\s+|- \\[[ xX]\\]\\s+)"
)
_TTS_MAX_CHARS = 120
TTS_MAX_BATCH = 1


def _find_last_markdown_break(text: str) -> int:
    last_idx = -1
    for match in _TTS_MARKDOWN_LINE_RE.finditer(text):
        if match.start() == 0:
            continue
        last_idx = match.start()
    return last_idx


def extract_tts_sentence(buffer: str) -> tuple[str | None, str]:
    """提取一个“句子”用于文本转语音（自然优先，兼顾低延时）。"""
    m = _TTS_SENT_END_RE.search(buffer)
    if m is not None:
        end = m.end()
        sentence = buffer[:end].strip()
        rest = buffer[end:].lstrip()
        if not sentence:
            return None, rest
        return sentence, rest

    md_idx = _find_last_markdown_break(buffer)
    if md_idx > 0:
        sentence = buffer[:md_idx].strip()
        rest = buffer[md_idx:].lstrip()
        if sentence:
            log("info", f"TTS forced markdown split ({len(sentence)} chars)")
            return sentence, rest

    if len(buffer) < _TTS_MAX_CHARS:
        return None, buffer

    sentence = buffer[:_TTS_MAX_CHARS].strip()
    rest = buffer[_TTS_MAX_CHARS:].lstrip()
    if not sentence:
        return None, rest
    log("info", f"TTS forced hard split ({len(sentence)} chars)")
    return sentence, rest


class GlobalModelManager:
    """全局运行时配置（单例）。"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self, target_sample_rate: int = 16000):
        """初始化全局运行时配置（仅保存目标采样率）。"""

        if self._initialized:
            log("info", "initialized, skipping ...")
            return

        self.target_sample_rate = target_sample_rate
        self._initialized = True
        log("info", f"Runtime config ready (target sample rate: {target_sample_rate})")


class ServerState:
    """WebSocket 服务状态。

    - 维护 ASR/diarization 与 vLLM / Qwen3-TTS 运行配置
    - 每个连接在 handle_chat() 内创建多协程/多线程流水线
    """

    def __init__(
        self,
        model_manager: GlobalModelManager,
        sample_rate: int = 24000,
        output_dir: str = "./output",
        *,
        vllm_base_url: str,
        vllm_model: str,
        llm_temperature: float = 0.2,
        llm_max_tokens: int = 512,
        llm_timeout_s: float = 300.0,
        llm_stream: bool = True,
        qwen3_tts_model: str = "/data/models/Voice/Qwen/Qwen3-TTS-12Hz-1___7B-CustomVoice",
        tts_device: str = "cuda:0",
        tts_dtype: str = "bfloat16",
        tts_language: str = "Chinese",
        tts_speaker: str = "Vivian",
        tts_instruct: str = "",
        tts_flash_attn: bool = True,
        tts_local_files_only: bool = True,
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
        """初始化服务端状态。

        Args:
            model_manager: 已初始化的全局运行时配置
            sample_rate: WebSocket 侧 Opus/PCM 处理采样率（默认 24k）
            output_dir: 输入/输出音频保存目录
        """

        self.model_manager = model_manager
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        self.lock = asyncio.Lock()
        # LLM system prompt is fixed (not overridable by client metadata).
        self.llm_system_prompt = DEFAULT_LLM_SYSTEM_PROMPT
        self.owner_to_gitea = dict(DEFAULT_OWNER_TO_GITEA)

        self.vllm_base_url = vllm_base_url
        self.vllm_model = vllm_model
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self.llm_timeout_s = llm_timeout_s
        self.llm_stream = llm_stream

        self.qwen3_tts_model = qwen3_tts_model
        self.tts_device = tts_device
        self.tts_dtype = tts_dtype
        # Fixed defaults for now; later we can expose selection to client.
        self.tts_language = DEFAULT_TTS_LANGUAGE
        self.tts_speaker = DEFAULT_TTS_SPEAKER
        self.tts_instruct = tts_instruct
        self.tts_flash_attn = tts_flash_attn
        self.tts_local_files_only = tts_local_files_only

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
        self._qwen3_tts = None
        self._qwen3_tts_lock = threading.Lock()

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "input"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "text"), exist_ok=True)
        log("info", f"Output directory: {self.output_dir}")

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

    def _ensure_qwen3_tts_initialized(self) -> None:
        with self._qwen3_tts_lock:
            if self._qwen3_tts is not None:
                return
            try:
                from qwen_tts import Qwen3TTSModel
            except Exception as exc:
                raise RuntimeError(
                    "Missing Qwen3-TTS dependencies. Ensure `qwen_tts` is importable "
                    "(pip install qwen-tts or vendor it)."
                ) from exc

            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            tts_dtype = dtype_map.get(self.tts_dtype, torch.bfloat16)
            attn_impl = "sdpa" if self.tts_flash_attn else None

            log("info", "Loading Qwen3-TTS model...")
            self._qwen3_tts = Qwen3TTSModel.from_pretrained(
                self.qwen3_tts_model,
                device_map=self.tts_device,
                dtype=tts_dtype,
                attn_implementation=attn_impl,
                local_files_only=bool(self.tts_local_files_only),
            )
            log("info", "Qwen3-TTS model loaded successfully")

    def diarize_and_transcribe(
        self,
        *,
        waveform_16k_mono: torch.Tensor,
        sample_rate: int,
        speaker_num: Optional[int],
    ):
        """运行3D扬声器定位+FunASR自动语音识别（ASR）进行单轮处理。

        返回：
            语句列表：list[Utterance]
            定位分段列表：list[DiarizationSegment]
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

    @staticmethod
    def write_turn_json(path: str, payload: dict[str, object]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")

    async def handle_chat(self, request):
        """处理单个 WebSocket 连接（/api/chat）。

        该方法内部会启动：
        - 一个接收协程：解析前端 Opus 音频/控制消息
        - 一个 PCM 累积协程：将解码后的 PCM 拼接成完整输入语音
        - 一个推理协程：在 pause/endTurn 时触发 diarization+ASR+vLLM，并把文本分流到队列
        - 一个编码线程：TTS 波形 -> Opus 帧
        - 两个发送协程：按节奏发送音频帧与文本
        """

        # 心跳间隔设为 30s（receive_timeout=None 避免中间代理/防火墙误断开）
        ws = web.WebSocketResponse(heartbeat=30.0, receive_timeout=None)
        await ws.prepare(request)

        client_id = f"Client-{id(ws)}"
        turn_counter = 0
        is_recording = True
        is_processing = False

        # 当前连接/会话的 user instruction（支持前端动态覆盖）
        session_instruction = self.instruction

        # 当前连接/会话的 TTS 风格指令（支持前端动态覆盖）
        tts_params_lock = threading.Lock()
        tts_params = {
            "style_instruct": (DEFAULT_TTS_STYLE_INSTRUCT or ""),
        }

        # 对话历史（vLLM ChatCompletion messages）
        messages = []
        current_generation = {
            "accumulated_text": "",
            "task": None,
            "is_generating": False,
            "interrupt": False,
            "text_output_path": "",
            "turn_record": None,
        }

        async def safe_send_bytes(payload: bytes, *, label: str = "") -> bool:
            nonlocal close
            if close or ws.closed:
                return False
            try:
                await ws.send_bytes(payload)
                return True
            except Exception as exc:
                tag = f" ({label})" if label else ""
                log("warning", f"WebSocket send failed{tag}: {exc}")
                close = True
                return False

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
                                if "instruction" in metadata:
                                    nonlocal session_instruction
                                    session_instruction = metadata["instruction"]
                                    log("info", f"Received custom instruction: {session_instruction[:100]}...")
                                # Repurpose client "system_prompt" to control TTS *style* (emotion) (backward compatible).
                                if "system_prompt" in metadata and str(metadata["system_prompt"]).strip():
                                    with tts_params_lock:
                                        tts_params["style_instruct"] = str(metadata["system_prompt"]).strip()
                                    log("info", f"Received TTS style (from system_prompt): {str(metadata['system_prompt'])[:80]}...")
                                if "tts_instruct" in metadata and str(metadata["tts_instruct"]).strip():
                                    with tts_params_lock:
                                        tts_params["style_instruct"] = str(metadata["tts_instruct"]).strip()
                                    log("info", f"Received TTS style: {str(metadata['tts_instruct'])[:80]}...")

                                # Not supported for now (fixed on backend).
                                if "tts_speaker" in metadata and str(metadata["tts_speaker"]).strip():
                                    log("warning", f"Ignoring client tts_speaker={metadata['tts_speaker']} (fixed speaker={self.tts_speaker})")
                                if "tts_language" in metadata and str(metadata["tts_language"]).strip():
                                    log("warning", f"Ignoring client tts_language={metadata['tts_language']} (fixed language={self.tts_language})")

                                known = {"instruction", "system_prompt", "tts_instruct", "tts_speaker", "tts_language"}
                                if not any(k in metadata for k in known):
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
                all_generated_audio, \
                reset_first_frame, \
                reset_send_state, \
                opus_reader, \
                current_generation, \
                is_processing


            while True:
                if close:
                    return
                try:
                    signal_type, _ = await asyncio.wait_for(save_audio_queue.get(), timeout=0.1)

                    if signal_type == "pause":
                        # 保存音频并触发 diarization + ASR + vLLM + Qwen3-TTS
                        if all_recorded_pcm is not None and len(all_recorded_pcm) > 0:
                            is_processing = True
                            processing_started_at = time.time()
                            log("info", f"[Turn {turn_counter}] Processing started")    
                            if close or ws.closed:
                                is_processing = False
                                return
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"{client_id}_turn{turn_counter}_input.wav"
                            filepath = os.path.join(self.output_dir, "input", filename)

                            audio_duration = len(all_recorded_pcm) / self.sample_rate

                            # 如采样率不一致，则重采样到目标采样率
                            if self.sample_rate != self.model_manager.target_sample_rate:
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
                                log("error", f"[WARN] soundfile save failed: {e}")
                            log("info", f"Saved audio to {filepath}, length: {audio_duration:.2f}s")

                            # 构造对话上下文（system + instruction + user + assistant）
                            if len(messages) == 0:
                                messages = [
                                    {"role": "system", "content": self.llm_system_prompt},
                                    {"role": "system", "content": f"用户要求：{session_instruction}"},
                                ]
                            else:
                                # Update dynamic instruction in-place to avoid growing history.
                                if len(messages) >= 2 and messages[0].get("role") == "system" and messages[1].get("role") == "system":
                                    messages[1]["content"] = f"用户要求：{session_instruction}"

                            max_messages = (2 + MAX_HISTORY_TURNS * 2)  # 2 for system + instruction
                            if len(messages) >= max_messages:
                                messages_to_remove = len(messages) - max_messages + 2
                                if messages_to_remove > 0:
                                    messages = messages[:2] + messages[2 + messages_to_remove :]
                                    log("info", f"Trimmed history: removed {messages_to_remove} messages")

                            transcript_text = ""
                            if self.enable_diar_asr:
                                try:
                                    msg = b"\x02" + bytes("[Diarization+ASR...]", encoding="utf8")
                                    if not await safe_send_bytes(msg, label="status"):
                                        return

                                    (utterances, _diar_segments,) = await asyncio.to_thread(
                                        self.diarize_and_transcribe,
                                        waveform_16k_mono=audio_tensor,
                                        sample_rate=self.model_manager.target_sample_rate,
                                        speaker_num=self.speaker_num,
                                    )
                                    from audiochat.prompting import format_utterances

                                    transcript_text = format_utterances(utterances, max_lines=self.max_lines)
                                except Exception as exc:
                                    log("error", f"Diarization+ASR failed: {exc}")
                            else:
                                # This server relies on ASR text; without it there is no user content to send to LLM.
                                transcript_text = ""
                                log("warning", "Diarization+ASR is disabled; no transcript available for LLM.")

                            if not transcript_text.strip():
                                log("warning", "Empty transcript; skipping vLLM+TTS for this turn")
                                try:
                                    error_msg = b"\x05" + bytes("Empty ASR transcript, cannot call LLM.", encoding="utf8")
                                    await safe_send_bytes(error_msg, label="error")
                                except Exception:
                                    pass
                                is_processing = False
                                continue

                            messages.append({"role": "user", "content": transcript_text})

                            log("info", f"[Turn {turn_counter}] Preparing vLLM input: {len(messages)} messages")
                            log("info", f"[Turn {turn_counter}] Message history: {[m['role'] for m in messages]}")
                            log("info", f"[Turn {turn_counter}] Queue status: opus_bytes={opus_bytes_queue.qsize()}")

                            text_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            text_filename = f"{client_id}_turn{turn_counter}_output_{text_timestamp}.json"
                            text_output_path = os.path.join(self.output_dir, "text", text_filename)
                            current_generation["text_output_path"] = text_output_path
                            current_generation["turn_record"] = {
                                "turn": turn_counter,
                                "timestamp": datetime.now().isoformat(timespec="seconds"),
                                "session_instruction": session_instruction,
                                "user_transcript": transcript_text.strip(),
                                "llm_raw_text": "",
                                "llm_text": "",
                                "meeting_result": None,
                                "gitea_issue_url": None,
                                "dispatch_error": None,
                                "audio_input_path": filepath,
                            }
                            log("info", f"[Turn {turn_counter}] JSON output path: {text_output_path}")

                            msg = b"\x02" + bytes("[Processing...]", encoding="utf8")
                            if not await safe_send_bytes(msg, label="status"):
                                current_generation["text_output_path"] = ""
                                current_generation["turn_record"] = None
                                is_processing = False
                                return

                            current_generation["accumulated_text"] = ""
                            current_generation["is_generating"] = True
                            current_generation["interrupt"] = False

                            frame_generation_complete["flag"] = False

                            with active_turn_lock:
                                active_turn_id = active_turn["id"]
                            messages_snapshot = list(messages)

                            async def run_llm_and_tts():
                                nonlocal messages, all_generated_audio, is_processing
                                accumulated_text = ""
                                assistant_audio_path = None
                                meeting_result: dict[str, object] | None = None
                                dispatch_error: str | None = None
                                gitea_issue_url = ""

                                try:
                                    try:
                                        llm_text = await vllm_chat_completion(
                                            base_url=self.vllm_base_url,
                                            model=self.vllm_model,
                                            messages=messages_snapshot,
                                            temperature=self.llm_temperature,
                                            max_tokens=self.llm_max_tokens,
                                            timeout_s=self.llm_timeout_s,
                                        )
                                        raw_llm_text = (llm_text or "").strip()
                                        accumulated_text = raw_llm_text

                                        if accumulated_text:
                                            beijing_date = get_beijing_date_str()
                                            try:
                                                meeting_result = parse_meeting_result_from_llm(
                                                    accumulated_text,
                                                    default_date=beijing_date,
                                                )
                                                # Use server-side Beijing date to avoid model-hallucinated dates.
                                                meeting_info = meeting_result.get("meeting_info")
                                                if isinstance(meeting_info, dict):
                                                    meeting_info["date"] = beijing_date
                                                accumulated_text = render_meeting_result_markdown(meeting_result)
                                            except Exception as exc:
                                                dispatch_error = f"meeting_result parse failed: {exc}"
                                                log("error", dispatch_error)

                                            if meeting_result is not None:
                                                try:
                                                    gitea_issue_url = await asyncio.to_thread(
                                                        dispatch_master_issue,
                                                        meeting_result,
                                                        self.owner_to_gitea,
                                                    )
                                                    log("info", f"[Turn {turn_counter}] Gitea dispatched: {gitea_issue_url}")
                                                except Exception as exc:
                                                    dispatch_error = f"Gitea dispatch failed: {exc}"
                                                    log("error", dispatch_error)

                                        current_generation["accumulated_text"] = accumulated_text

                                        turn_record = current_generation.get("turn_record")
                                        if isinstance(turn_record, dict):
                                            turn_record["llm_raw_text"] = raw_llm_text
                                            turn_record["llm_text"] = accumulated_text
                                            turn_record["meeting_result"] = meeting_result
                                            turn_record["gitea_issue_url"] = gitea_issue_url or None
                                            turn_record["dispatch_error"] = dispatch_error

                                        if gitea_issue_url:
                                            status_msg = b"\x02" + bytes(
                                                f"[Gitea dispatched] {gitea_issue_url}",
                                                encoding="utf8",
                                            )
                                            await safe_send_bytes(status_msg, label="status")

                                        if accumulated_text:
                                            log("info", f"[Turn {turn_counter}] LLM full response received ({len(accumulated_text)} chars)")
                                    except asyncio.CancelledError:
                                        log("info", "LLM task cancelled")
                                        raise
                                    except Exception as exc:
                                        log("error", f"vLLM call failed: {exc}")
                                        try:
                                            error_msg = b"\x05" + bytes(f"LLM failed: {str(exc)}", encoding="utf8")
                                            await safe_send_bytes(error_msg, label="error")
                                        except Exception:
                                            pass
                                        return

                                    if current_generation["interrupt"]:
                                        log("info", "LLM interrupted, skipping TTS flush")
                                        return

                                    if accumulated_text.strip():
                                        tts_text_queue.put((active_turn_id, accumulated_text, True))
                                    else:
                                        tts_text_queue.put((active_turn_id, "", True))

                                    log("info", "TTS generation marked as complete")

                                    # 等待编码线程处理完所有剩余音频
                                    max_wait_time = 15
                                    wait_interval = 0.1
                                    waited = 0
                                    while waited < max_wait_time:
                                        if (
                                            frame_generation_complete["flag"]
                                            and tts_audio_queue.qsize() == 0
                                            and opus_bytes_queue.qsize() == 0
                                        ):
                                            log("info", f"All audio encoded and sent after {waited:.1f}s")
                                            break
                                        await asyncio.sleep(wait_interval)
                                        waited += wait_interval

                                    if waited >= max_wait_time:
                                        log("warning", f"Encoding timeout after {max_wait_time}s")

                                    with all_generated_audio_lock:
                                        has_audio = len(all_generated_audio) > 0
                                        segments = list(all_generated_audio)
                                    if has_audio:
                                        try:
                                            def save_full_audio(segments, sample_rate, output_path):
                                                full_audio = np.concatenate(segments)
                                                duration = len(full_audio) / sample_rate
                                                audio_to_save = full_audio.astype(np.float32)
                                                if np.abs(audio_to_save).max() > 1.0:
                                                    audio_to_save = (audio_to_save / np.abs(audio_to_save).max())
                                                sf.write(output_path, audio_to_save, sample_rate)
                                                return duration, len(segments), len(full_audio)

                                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                            output_filename = f"{client_id}_turn{turn_counter}_output_{timestamp}.wav"
                                            assistant_audio_path = os.path.join(self.output_dir, output_filename)

                                            (assistant_audio_duration, segment_count, sample_count) = await asyncio.to_thread(
                                                save_full_audio,
                                                segments,
                                                self.sample_rate,
                                                assistant_audio_path,
                                            )
                                            log("info", f"Concatenated generated audio: {segment_count} segments, total length {assistant_audio_duration:.2f}s ({sample_count} samples)")
                                            log("info", f"Saved generated audio: {assistant_audio_path}")

                                            with all_generated_audio_lock:
                                                all_generated_audio.clear()
                                        except Exception as e:
                                            log("error", f"Failed to save generated audio: {e}")
                                            import traceback

                                            traceback.print_exc()
                                            assistant_audio_path = None

                                    if accumulated_text:
                                        messages.append({"role": "assistant", "content": accumulated_text})

                                    log("info", "Processing completed")
                                finally:
                                    if current_generation["is_generating"]:
                                        current_generation["is_generating"] = False
                                    current_generation["task"] = None
                                    if current_generation.get("text_output_path"):
                                        turn_record = current_generation.get("turn_record")
                                        if isinstance(turn_record, dict):
                                            turn_record["assistant_audio_path"] = assistant_audio_path
                                            turn_record["completed_at"] = datetime.now().isoformat(timespec="seconds")
                                            try:
                                                await asyncio.to_thread(
                                                    self.write_turn_json,
                                                    current_generation.get("text_output_path"),
                                                    turn_record,
                                                )
                                                log("info", f"[Turn {turn_counter}] JSON saved")
                                            except Exception as exc:
                                                log("warning", f"Failed to save JSON output: {exc}")
                                    current_generation["text_output_path"] = ""
                                    current_generation["turn_record"] = None
                                    if is_processing:
                                        is_processing = False
                                        elapsed = time.time() - processing_started_at
                                        log("info", f"[Turn {turn_counter}] Processing finished in {elapsed:.2f}s")

                            llm_task = asyncio.create_task(run_llm_and_tts())
                            current_generation["task"] = llm_task

                        else:
                            log("warning", "No audio data to save")

                    elif signal_type == "start":
                        # 若上一轮仍在生成中，则先设置 interrupt 并等待其停止
                        if current_generation["is_generating"]:
                            log("info", "Interrupting current generation...")
                            current_generation["interrupt"] = True
                            if current_generation["task"] is not None:
                                current_generation["task"].cancel()
                            current_generation["is_generating"] = False
                            current_generation["interrupt"] = False
                            current_generation["task"] = None
                            current_generation["accumulated_text"] = ""
                            current_generation["text_output_path"] = ""
                            current_generation["turn_record"] = None
                            log("info", "Previous generation stopped")

                        all_recorded_pcm = None
                        with all_generated_audio_lock:
                            all_generated_audio.clear()
                        while not tts_audio_queue.empty():
                            try:
                                tts_audio_queue.get_nowait()
                            except queue.Empty:
                                break
                        while not tts_text_queue.empty():
                            try:
                                tts_text_queue.get_nowait()
                            except queue.Empty:
                                break

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
                        frame_generation_complete["flag"] = False
                        audio_send_started["flag"] = False

                        with active_turn_lock:
                            active_turn["id"] = turn_counter

                        log("info", "Cleared audio buffer and TTS state for new recording")

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    log("error", f"Inference failed: {e}")
                    import traceback

                    traceback.print_exc()
                    try:
                        error_msg = b"\x05" + bytes(f"Processing failed: {str(e)}", encoding="utf8")
                        await safe_send_bytes(error_msg, label="error")
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

        def tts_worker_thread_func():
            """Qwen3-TTS 工作线程：将文本片段转为音频块并入队。"""
            nonlocal frame_generation_complete

            log("info", "Qwen3-TTS worker thread started")
            try:
                self._ensure_qwen3_tts_initialized()
            except Exception as exc:
                log("error", f"Failed to init Qwen3-TTS: {exc}")
                return

            while not close:
                try:
                    try:
                        turn_id, text, finalize = tts_text_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    with active_turn_lock:
                        current_turn = active_turn["id"]
                    if turn_id != current_turn:
                        continue

                    if not text:
                        if finalize:
                            frame_generation_complete["flag"] = True
                        continue

                    # Micro-batch: keep "one sentence per item" semantics, but allow batching multiple sentences
                    # into one model call to reduce overhead.
                    batch_items = [(turn_id, text, finalize)]
                    max_batch = TTS_MAX_BATCH
                    while len(batch_items) < max_batch:
                        try:
                            nxt_turn_id, nxt_text, nxt_finalize = tts_text_queue.get_nowait()
                        except queue.Empty:
                            break
                        # Drop stale items.
                        with active_turn_lock:
                            current_turn = active_turn["id"]
                        if nxt_turn_id != current_turn:
                            continue
                        if nxt_turn_id != turn_id:
                            # Put back for later (best-effort; preserves content but may reorder slightly).
                            tts_text_queue.put((nxt_turn_id, nxt_text, nxt_finalize))
                            break
                        batch_items.append((nxt_turn_id, nxt_text, nxt_finalize))
                        if nxt_finalize:
                            break

                    try:
                        with tts_params_lock:
                            local_style = str(tts_params.get("style_instruct") or "")
                            
                        local_lang = self.tts_language
                        local_spk = self.tts_speaker

                        base_ins = ((self.tts_instruct or "").strip() or (DEFAULT_TTS_INSTRUCT or "").strip())
                        style_ins = local_style.strip()
                        if base_ins and style_ins:
                            local_ins = f"{base_ins}\n{style_ins}"
                        else:
                            local_ins = base_ins or style_ins

                        texts = [it[1] for it in batch_items]
                        languages = [local_lang] * len(texts)
                        speakers = [local_spk] * len(texts)
                        instructs = [local_ins] * len(texts) if local_ins else None

                        with self._qwen3_tts_lock:
                            wavs, sr = self._qwen3_tts.generate_custom_voice(
                                text=texts,
                                language=languages,
                                speaker=speakers,
                                instruct=instructs,
                            )
                    except Exception as e:
                        log("error", f"Qwen3-TTS generation failed: {e}")
                        if batch_items[-1][2]:
                            frame_generation_complete["flag"] = True
                        continue

                    # wavs: List[np.ndarray]
                    for i, wav in enumerate(wavs):
                        speech = wav
                        if isinstance(speech, torch.Tensor):
                            speech = speech.detach().cpu().numpy()

                        if sr != self.sample_rate:
                            audio_tensor = torch.from_numpy(speech).unsqueeze(0)
                            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                            speech = resampler(audio_tensor).squeeze(0).cpu().numpy()

                        if speech is None or len(speech) == 0:
                            continue

                        speech = speech.astype(np.float32)
                        max_abs = float(np.abs(speech).max()) if speech.size > 0 else 0.0
                        if max_abs > 1.0:
                            speech = speech / max_abs

                        tts_audio_queue.put(speech)
                        with all_generated_audio_lock:
                            all_generated_audio.append(speech.copy())

                        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        log("info", f"[{timestamp}] Qwen3-TTS generated: {speech.shape[-1]} samples (batch={len(wavs)}, idx={i})")

                    if batch_items[-1][2]:
                        frame_generation_complete["flag"] = True

                except Exception as e:
                    log("error", f"Qwen3-TTS worker thread error: {e}")
                    import traceback

                    traceback.print_exc()

            log("info", "Qwen3-TTS worker thread stopped")

        def encode_thread_func():
            """编码线程：从音频缓冲取出完整帧，编码为 Opus 并放入 opus_bytes_queue。"""
            nonlocal \
                reset_first_frame, \
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

                    try:
                        chunk = tts_audio_queue.get_nowait()
                    except queue.Empty:
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
            log_opus_empty = False
            next_send_time = None
            frames_sent = 0
            empty_streak = 0
            empty_streak_start = None
            last_empty_log = 0.0
            lag_spike_start = None

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
                        log("info", "Reset send state for new turn")


                    try:
                        opus_bytes = opus_bytes_queue.get_nowait()
                    except queue.Empty:
                        now = time.time()
                        if empty_streak == 0:
                            empty_streak_start = now
                        empty_streak += 1
                        if log_opus_empty and empty_streak_start is not None and (now - empty_streak_start) > 1.0 and (now - last_empty_log) > 1.0:
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

                    if not await safe_send_bytes(b"\x01" + opus_bytes, label="audio"):
                        break
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

        async def send_ping_loop():
            """在录音/处理中定期发送 ping，避免前端因无消息而断开。"""
            nonlocal is_recording, is_processing
            ping_interval = 10.0
            ping_active = False
            ping_sent = 0

            while not close:
                try:
                    should_ping = is_recording or is_processing
                    if not should_ping:
                        if ping_active:
                            ping_active = False
                            log("info", "Ping loop idle (no recording/processing)")
                        await asyncio.sleep(0.2)
                        continue

                    if not ping_active:
                        ping_active = True
                        log("info", "Ping loop active (recording/processing)")

                    if not await safe_send_bytes(b"\x06", label="ping"):
                        break
                    ping_sent += 1
                    if ping_sent % 6 == 0:
                        log("info", f"Sent ping #{ping_sent} (recording={is_recording}, processing={is_processing})")

                    await asyncio.sleep(ping_interval)
                except Exception as e:
                    log("error", f"Ping loop error: {e}")
                    import traceback

                    traceback.print_exc()
                    await asyncio.sleep(0.5)

        async def send_text_loop():
            """异步发送：按固定节奏发送文本快照（尾部窗口）。"""
            nonlocal audio_send_started, current_generation
            text_interval = 0.5  # 500ms per snapshot
            max_snapshot_lines = 300
            snapshots_sent = 0
            current_turn_started = False
            last_sent_snapshot = ""
            next_send_time = None

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
                        next_send_time = None
                        log("info", "Audio started, beginning text snapshot transmission")

                    if current_turn_started and not audio_send_started["flag"]:
                        current_turn_started = False
                        last_sent_snapshot = ""
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

                    snapshot_source = current_generation.get("accumulated_text", "")
                    if not snapshot_source or snapshot_source == last_sent_snapshot:
                        await asyncio.sleep(0.05)
                        continue

                    snapshot_lines = snapshot_source.splitlines()
                    if len(snapshot_lines) > max_snapshot_lines:
                        snapshot_lines = snapshot_lines[-max_snapshot_lines:]
                    snapshot = "\n".join(snapshot_lines)

                    if snapshot and snapshot != last_sent_snapshot:
                        try:
                            msg = b"\x08" + bytes(snapshot, encoding="utf8")
                            ok = await safe_send_bytes(msg, label="text_snapshot")
                            if not ok:
                                break
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
        all_generated_audio_lock = threading.Lock()
        reset_first_frame = {"flag": True}  # 控制编码线程重置“首帧”状态（用 dict 便于线程共享）
        reset_send_state = {"flag": False}  # 控制发送协程重置帧计数与时间基准
        frame_generation_complete = {"flag": False}  # 标记 TTS 生成（波形还原）是否已结束
        audio_send_started = {"flag": False}  # 标记音频是否已开始发送（文本发送要等首帧音频后才能开始）
        save_audio_queue = asyncio.Queue()  # 录音控制信号队列（start/pause 等）
        tts_text_queue = queue.Queue()
        tts_audio_queue = queue.Queue()
        pcm_queue = asyncio.Queue()  # 解码后的 PCM 数据队列（由 Opus reader 产出）
        opus_bytes_queue = (queue.Queue())  # Opus 编码后的字节流队列（线程安全，用于发送）
        active_turn = {"id": 0}
        active_turn_lock = threading.Lock()

        opus_reader = sphn.OpusStreamReader(self.sample_rate)

        tts_worker_thread = Thread(target=tts_worker_thread_func, daemon=True)
        encode_thread = Thread(target=encode_thread_func, daemon=True)

        tts_worker_thread.start()
        encode_thread.start()
        log("info", "All workers started (Qwen3-TTS worker + encode thread active)")
        # 发送握手包：告知协议版本与模型类型
        handshake = encode_handshake(version=0, model=0)
        await safe_send_bytes(handshake, label="handshake")
        log("info", "Sent handshake")

        try:
            await asyncio.gather(
                recv_loop(),
                save_audio_loop(),
                accumulate_pcm_loop(),
                send_audio_loop(),
                send_ping_loop(),
                send_text_loop(),
            )
        finally:
            close = True
            log("info", "Waiting for workers to stop...")

            tts_worker_thread.join(timeout=2.0)
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
    # vLLM (OpenAI-compatible) server
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm-model", required=True, help="vLLM model name")
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--llm-max-tokens", type=int, default=512)
    parser.add_argument("--llm-timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--llm-stream/--no-llm-stream",
        dest="llm_stream",
        default=False,
        action=argparse.BooleanOptionalAction,
    )

    # Qwen3-TTS(CustomVoice)
    parser.add_argument(
        "--qwen3-tts-model",
        default="/data/models/Voice/Qwen/Qwen3-TTS-12Hz-1___7B-CustomVoice",
        help="Local path to Qwen3-TTS CustomVoice model directory",
    )
    parser.add_argument("--tts-device", default="cuda:0")
    parser.add_argument("--tts-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--tts-language", default="Chinese")
    parser.add_argument("--tts-speaker", default="Vivian")
    parser.add_argument("--tts-instruct", default="", help="Optional instruct for CustomVoice (1.7B supports it)")
    parser.add_argument(
        "--tts-flash-attn/--no-tts-flash-attn",
        dest="tts_flash_attn",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--tts-local-files-only/--no-tts-local-files-only",
        dest="tts_local_files_only",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Force local-only loading for Qwen3-TTS weights/tokenizer (recommended for offline).",
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
        model_manager.initialize(target_sample_rate=args.model_sample_rate)
    except Exception as e:
        log("error", f"Runtime initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return

    state = ServerState(
        model_manager=model_manager,
        sample_rate=args.sample_rate,
        output_dir=args.output_dir,
        vllm_base_url=args.vllm_base_url,
        vllm_model=args.vllm_model,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
        llm_timeout_s=args.llm_timeout_s,
        llm_stream=args.llm_stream,
        qwen3_tts_model=args.qwen3_tts_model,
        tts_device=args.tts_device,
        tts_dtype=args.tts_dtype,
        tts_language=args.tts_language,
        tts_speaker=args.tts_speaker,
        tts_instruct=args.tts_instruct,
        tts_flash_attn=args.tts_flash_attn,
        tts_local_files_only=args.tts_local_files_only,
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
