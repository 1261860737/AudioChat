"""
离线 Pipeline - 文本 + Qwen3-TTS 语音输出（ASR/diarization 同 offline_pipeline_s2s）

完整流程：
  音频 -> 说话人分离（Diarization）-> ASR 识别 -> vLLM(本地服务) 生成文本 -> Qwen3-TTS(CustomVoice) 合成语音

输出：
  - 分离结果 JSON
  - ASR utterances JSON
  - LLM 文本回复 + TTS 音频路径 JSON

Usage:
  python scripts/offline_pipeline_vllm_qwen3tts.py \
    --audio examples/2speakers_example.wav \
    --vllm-base-url http://127.0.0.1:8000/v1 \
    --vllm-model your-llm-name \
    --qwen3-tts-model /data/models/Voice/Qwen/Qwen3-TTS-12Hz-1___7B-CustomVoice \
    --output-dir AudioChat_saves
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float = 300.0) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"HTTP {exc.code} calling {url}: {body}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed calling {url}: {exc}") from exc

    try:
        return json.loads(data)
    except Exception as exc:
        raise RuntimeError(f"Non-JSON response from {url}: {data[:500]}") from exc


def vllm_chat_completion(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 512,
    timeout_s: float = 300.0,
) -> str:
    base_url = base_url.rstrip("/")
    url = f"{base_url}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    resp = _post_json(url, payload, timeout_s=timeout_s)

    try:
        return str(resp["choices"][0]["message"]["content"]).strip()
    except Exception as exc:
        raise RuntimeError(f"Unexpected vLLM response schema: keys={list(resp.keys())}") from exc


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
    parser.add_argument("--instruction", default="请生成结构化会议纪要（要点/结论/行动项）。")
    parser.add_argument("--max-lines", type=int, default=400)
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--llm-max-tokens", type=int, default=512)
    parser.add_argument("--llm-timeout-s", type=float, default=300.0)

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

    from audiochat.audio_io import ensure_mono_16k, slice_waveform
    from audiochat.asr.funasr_asr import FunASRTranscriber
    from audiochat.diarization.diarizer_3dspeaker import ThreeDSpeakerDiarizer
    from audiochat.prompting import build_llm_instruction

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
    llm_reply_path = os.path.join(args.output_dir, f"{audio_basename}_llm_reply_vllm_qwen3tts.json")
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

    user_prompt = build_llm_instruction(
        utterances=utterances,
        user_instruction=args.instruction,
        max_lines=args.max_lines,
    )
    llm_text = vllm_chat_completion(
        base_url=args.vllm_base_url,
        model=args.vllm_model,
        system_prompt=args.system_prompt,
        user_prompt=user_prompt,
        temperature=args.llm_temperature,
        max_tokens=args.llm_max_tokens,
        timeout_s=args.llm_timeout_s,
    )

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
        "mode": "asr+vllm+s2t+qwen3tts",
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
