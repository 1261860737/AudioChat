from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torchaudio.functional as F
import soundfile as sf
import numpy as np

TARGET_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class SpeakerSegment:
    index: int
    speaker: str
    start_ms: int
    end_ms: int
    text: str


def load_resample_mono(
    audio_path: str,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> tuple[torch.Tensor, int]:
    """Load an audio file, convert to mono, and resample.

    Returns:
        waveform: shape [1, num_samples] float32
        sample_rate: target_sample_rate
    """
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)  # [T, C]
    audio = audio.T  # -> [C, T]
    waveform = torch.from_numpy(audio)  # float32 tensor

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != target_sample_rate:
        waveform = F.resample(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate

    # 统一输出为 [1, T]，方便后续 FunASR 直接使用
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    return waveform.contiguous(), sample_rate


def _ensure_mono_batch(waveform: torch.Tensor) -> torch.Tensor:
    """确保 waveform 形状为 [1, T]。

    兼容调用方传入：
    - [T]
    - [1, T]
    - [C, T]（会做 mean downmix）
    """

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform ndim=1 or 2, got {waveform.ndim}")

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if waveform.shape[0] != 1:
        raise ValueError(f"Expected mono waveform [1, T], got {tuple(waveform.shape)}")

    return waveform.contiguous()


def funasr_diarize_and_transcribe(
    waveform_16k_mono: torch.Tensor,
    *,
    model: str,
    vad_model: str = "fsmn-vad",
    punc_model: str = "ct-punc",
    spk_model: str = "cam++",
    device: str = "cpu",
    preset_spk_num: Optional[int] = None,
    batch_size_s: int = 300,
    batch_size_threshold_s: int = 60,
    disable_update: bool = True,
) -> tuple[list[SpeakerSegment], dict[str, Any]]:
    """Run FunASR diarization+ASR on 16kHz mono audio.

    Notes:
    - Speaker diarization relies on `punc_model` and ASR timestamps.
    - `preset_spk_num` can stabilize clustering for MVP (e.g. 2 speakers).

    Args:
        waveform_16k_mono: shape [1, T], sample_rate must be 16k.
        model: ModelScope ID or local model directory.

    Returns:
        segments: normalized sentence-level segments with speaker labels
        raw_item: the first raw FunASR result item
    """
    # 兼容调用方传入 1D waveform（[T]），统一成 [1, T]
    waveform_16k_mono = _ensure_mono_batch(waveform_16k_mono)

    try:
        from funasr import AutoModel  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: funasr. Install with `pip install -U funasr modelscope`."
        ) from exc

    diar_model = AutoModel(
        model=model,
        vad_model=vad_model,
        punc_model=punc_model,
        spk_model=spk_model,
        device=device,
        disable_update=disable_update,
    )

    # FunASR 的前端（torchaudio.compliance.kaldi.fbank）期望输入是一维 waveform（[T]）。
    # 如果直接传 [1, T]，kaldi 里用 len(waveform) 会得到 1，从而触发：
    #   AssertionError: choose a window size 400 that is [2, 1]
    waveform_for_funasr = waveform_16k_mono.squeeze(0)

    res = diar_model.generate(
        input=waveform_for_funasr,
        batch_size_s=batch_size_s,
        batch_size_threshold_s=batch_size_threshold_s,
        preset_spk_num=preset_spk_num,
    )

    if not isinstance(res, list) or len(res) == 0 or not isinstance(res[0], dict):
        raise RuntimeError(f"Unexpected FunASR output: {type(res)}")

    item: dict[str, Any] = res[0]
    sentence_info = item.get("sentence_info")
    if not isinstance(sentence_info, list):
        raise RuntimeError(
            "FunASR output missing `sentence_info`. "
            "Check `punc_model` is enabled and the ASR model supports timestamps."
        )

    segments: list[SpeakerSegment] = []
    for idx, s in enumerate(sentence_info, start=1):
        if not isinstance(s, dict):
            continue

        text = str(s.get("text") or "").strip()
        if not text:
            continue

        start_ms = int(float(s.get("start") or 0))
        end_ms = int(float(s.get("end") or start_ms))
        spk = s.get("spk")
        speaker = f"spk{int(spk)}" if spk is not None else "spk?"

        segments.append(
            SpeakerSegment(
                index=idx,
                speaker=speaker,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
            )
        )

    return segments, item
