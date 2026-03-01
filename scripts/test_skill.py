"""
离线 Workflow - ASR/diar + LLM(可调用 skill) + Qwen3-TTS 语音输出

完整流程：
  音频 -> 说话人分离（Diarization）-> ASR 识别 -> LLM(本地服务) 决策/调用 skill -> Qwen3-TTS 合成语音

输出：
  - 分离结果 JSON
  - ASR utterances JSON
  - LLM 文本回复 + TTS 音频路径 JSON

Usage:
  python scripts/test_skill.py \
    --audio examples/2speakers_example.wav \
    --vllm-base-url http://127.0.0.1:8000/v1 \
    --vllm-model Qwen3-Coder-30B-A3B-Instruct-FP8 \
    --qwen3-tts-model /data/models/Voice/Qwen/Qwen3-TTS-12Hz-1___7B-CustomVoice \
    --output-dir AudioChat_saves
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "third_party"))

    parser = argparse.ArgumentParser(
        description="Offline: diarization -> ASR -> vLLM text -> Qwen3-TTS(CustomVoice) wav"
    )
    parser.add_argument("--audio", required=True, help="Input audio path")
    parser.add_argument("--output-dir", default="AudioChat_saves")

    # 3D-Speaker diarization
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

    # FunASR ASR
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

    # vLLM(OpenAI-compatible) server
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm-model", required=True, help="vLLM model name")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--instruction", default="请生成结构化会议纪要，并上传到gitea。")
    parser.add_argument("--max-lines", type=int, default=400)
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--llm-max-tokens", type=int, default=512)
    parser.add_argument("--llm-timeout-s", type=float, default=300.0)
    parser.add_argument("--skills-dir", default="./skills", help="Skill directory")
    parser.add_argument("--agent-max-turns", type=int, default=8, help="Max LLM turns for tool workflow")

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

    args = parser.parse_args()

    from agent_engine import AgentEngine
    from audiochat.audio_io import ensure_mono_16k, slice_waveform
    from audiochat.asr.funasr_asr import FunASRTranscriber
    from audiochat.diarization.diarizer_3dspeaker import ThreeDSpeakerDiarizer
    from audiochat.prompting import format_utterances

    audio = ensure_mono_16k(args.audio)

    diarizer = ThreeDSpeakerDiarizer(
        device=args.diar_device,
        model_cache_dir=args.model_cache_dir,
        speaker_embedding_model_path=args.speaker_embedding_model_path,
        vad_model_path=args.vad_model_path,
    )
    diar_segments = diarizer.diarize(
        audio.waveform,
        wav_fs=audio.sample_rate,
        speaker_num=args.speaker_num,
    )

    transcriber = FunASRTranscriber(
        model=args.funasr_model,
        device=args.funasr_device,
        vad_model=args.funasr_vad_model_path,
        punc_model=args.funasr_punc_model_path,
    )

    utterances = []
    raw_asr_items = []
    for seg in diar_segments:
        seg_audio = slice_waveform(
            audio.waveform, seg.start_s, seg.end_s, sample_rate=audio.sample_rate
        )
        seg_start_ms = int(round(seg.start_s * 1000.0))
        spk = f"spk{seg.speaker}"
        uts, raw = transcriber.transcribe_segment(
            seg_audio,
            speaker=spk,
            segment_start_ms=seg_start_ms,
        )
        utterances.extend(uts)
        raw_asr_items.append(raw)

    utterances.sort(key=lambda u: (u.start_ms, u.end_ms, u.speaker))

    os.makedirs(args.output_dir, exist_ok=True)

    audio_basename = Path(args.audio).stem
    diarization_path = os.path.join(args.output_dir, f"{audio_basename}_diarization.json")
    asr_utterances_path = os.path.join(args.output_dir, f"{audio_basename}_asr_utterances.json")
    llm_reply_path = os.path.join(args.output_dir, f"{audio_basename}_llm_reply_agent_workflow.json")
    tts_audio_path = os.path.join(args.output_dir, f"{audio_basename}_qwen3tts_{uuid.uuid4().hex[:8]}.wav")

    with open(diarization_path, "w", encoding="utf-8") as f:
        json.dump(
            {"audio": args.audio, "segments": [seg.__dict__ for seg in diar_segments]},
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(asr_utterances_path, "w", encoding="utf-8") as f:
        json.dump(
            {"audio": args.audio, "utterances": [u.__dict__ for u in utterances], "raw": raw_asr_items},
            f,
            ensure_ascii=False,
            indent=2,
        )

    workflow_prompt = (
        "你是一个可调用本地技能的智能体。\n"
        "严格规则：\n"
        "1) 如果需要调用工具，只输出 JSON，不得包含任何自然语言。\n"
        "2) JSON 结构必须是：{\"tool\": \"<tool_name>\", \"args\": { ... }}。\n"
        "3) args 必须是 JSON 对象，不能是字符串。\n"
        "5) 如果不需要调用工具，直接输出自然语言（不要输出 JSON）。\n"
        "示例（调用）：{\"tool\":\"send-email\",\"args\":{\"llm_text\":\"...\"}}\n"
        "示例（不调用）：这是普通回复。\n\n"
        f"用户要求：{args.instruction}\n\n"
        "转写内容（供参考）：\n"
        f"{format_utterances(utterances, max_lines=args.max_lines)}\n\n"
        "请开始处理。"
    )

    agent = AgentEngine(
        skills_dir=args.skills_dir,
        vllm_url=args.vllm_base_url,
        model_name=args.vllm_model,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
        llm_timeout_s=args.llm_timeout_s,
        system_prompt=args.system_prompt,
    )
    llm_text = agent.run(workflow_prompt, max_turns=args.agent_max_turns)

    # Qwen3-TTS
    try:
        import torch
        import soundfile as sf
        from qwen_tts import Qwen3TTSModel
    except Exception as exc:
        raise RuntimeError(
            "Missing Qwen3-TTS dependencies. Ensure `qwen_tts` is importable (pip install qwen-tts or vendor it) "
            "and soundfile/torch are installed."
        ) from exc

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    tts_dtype = dtype_map[args.tts_dtype]
    attn_impl = "sdpa" if args.tts_flash_attn else None

    tts = Qwen3TTSModel.from_pretrained(
        args.qwen3_tts_model,
        device_map=args.tts_device,
        dtype=tts_dtype,
        attn_implementation=attn_impl,
        local_files_only=bool(args.tts_local_files_only),
    )

    wavs, sr = tts.generate_custom_voice(
        text=llm_text,
        language=args.tts_language,
        speaker=args.tts_speaker,
        instruct=(args.tts_instruct or ""),
    )
    sf.write(tts_audio_path, wavs[0], sr)

    result = {
        "mode": "asr+agent+skills+qwen3tts",
        "text": llm_text,
        "audio_path": tts_audio_path,
        "diarization_path": diarization_path,
        "asr_utterances_path": asr_utterances_path,
    }

    with open(llm_reply_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
