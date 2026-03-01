"""
离线 Pipeline - 语音输出模式（S2S）

完整流程：音频 -> 说话人分离（Diarization）-> ASR 识别 -> LLM 文本+语音生成
输出：带有说话人标签和时间戳的转写文本 + LLM 的文本回复 + 语音回复音频

Usage:
    python scripts/offline_pipeline_s2s.py --audio <wav> --funasr-model <path> --funaudiochat-model <path>

Example:
    python scripts/offline_pipeline_s2s.py \
        --audio examples/2speakers_example.wav \
        --funasr-model /data/models/Voice/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
        --funaudiochat-model /data/models/Voice/FunAudioLLM/Fun-Audio-Chat-8B \
        --output-dir AudioChat_saves
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ====== MUST be before importing audiochat / third_party modules ======
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "third_party"))
sys.path.insert(0, str(root / "third_party" / "CosyVoice"))
sys.path.insert(0, str(root / "third_party" / "Matcha-TTS"))
# ================================================================

from audiochat.audio_io import ensure_mono_16k, slice_waveform
from audiochat.asr.funasr_asr import FunASRTranscriber
from audiochat.diarization.diarizer_3dspeaker import ThreeDSpeakerDiarizer
from audiochat.llm.funaudiochat_infer import infer_s2t_or_s2s
from audiochat.prompting import build_llm_instruction


def main() -> None:


    parser = argparse.ArgumentParser(
        description="Offline: 3D-Speaker diarization -> FunASR ASR -> Fun-Audio-Chat S2S (audio-in, text+audio-out)"
    )
    parser.add_argument("--audio", required=True, help="Input audio path")
    parser.add_argument("--output-dir", default="AudioChat_saves")

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

    parser.add_argument(
        "--funaudiochat-model",
        default="/data/models/Voice/FunAudioLLM/Fun-Audio-Chat-8B",
    )
    parser.add_argument("--llm-device", default="cuda:0")
    parser.add_argument(
        "--instruction", default="请按 speaker 分组输出对话稿，并生成语音回复。"
    )
    parser.add_argument("--max-lines", type=int, default=400)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 统一成 16kHz mono：后续 diarization / ASR / LLM 都用这一份"原始完整音频"
    audio = ensure_mono_16k(args.audio)

    # 分支 A：完整音频 -> diarization -> 按段切片 -> ASR（产出结构化"证据文本"）
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

    # 从音频路径提取文件名作为中间文件前缀，避免不同音频的处理结果覆盖
    audio_basename = Path(args.audio).stem
    diarization_path = os.path.join(
        args.output_dir, f"{audio_basename}_diarization.json"
    )
    asr_utterances_path = os.path.join(
        args.output_dir, f"{audio_basename}_asr_utterances.json"
    )
    llm_reply_path = os.path.join(
        args.output_dir, f"{audio_basename}_llm_reply_s2s.json"
    )

    with open(diarization_path, "w", encoding="utf-8") as f:
        json.dump(
            {"audio": args.audio, "segments": [seg.__dict__ for seg in diar_segments]},
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(asr_utterances_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "audio": args.audio,
                "utterances": [u.__dict__ for u in utterances],
                "raw": raw_asr_items,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    instruction = build_llm_instruction(
        utterances=utterances,
        user_instruction=args.instruction,
        max_lines=args.max_lines,
    )

    result = infer_s2t_or_s2s(
        funaudiochat_model_path=args.funaudiochat_model,
        audio_16k_mono=audio.waveform,
        mode="s2s",
        instruction=instruction,
        device=args.llm_device,
        output_dir=args.output_dir,
    )

    with open(llm_reply_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
