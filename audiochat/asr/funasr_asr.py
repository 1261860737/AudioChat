"""
ASR (自动语音识别) 模块

提供基于 FunASR 的语音识别功能，将音频转换为带时间戳的文本。
主要用途：识别说话人切片后的音频，输出带有说话人标签和时间戳的文本片段。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass(frozen=True)
class Utterance:
    """识别结果单元，表示一段带有标签的文本。

    Attributes:
        speaker: 说话人标识（如 "spk0"）
        start_ms: 开始时间（毫秒）
        end_ms: 结束时间（毫秒）
        text: 识别出的文本内容
    """

    speaker: str
    start_ms: int
    end_ms: int
    text: str


def _to_waveform_1d(waveform_16k_mono: torch.Tensor) -> torch.Tensor:
    """将 2D 波形 [1, T] 转换为 1D 波形 [T]，适配 FunASR 输入格式。"""
    if waveform_16k_mono.ndim == 2:
        return waveform_16k_mono.squeeze(0)
    return waveform_16k_mono


class FunASRTranscriber:
    """FunASR 语音识别封装类。

    接收 diarization 后的音频片段，输出带有时间戳和说话人标签的 ASR 结果。
    与 DiarizationSegment 配合使用，实现"谁在什么时候说了什么"的完整转写。
    """

    def __init__(
        self,
        *,
        model: str,
        device: str = "cpu",
        vad_model: Optional[str] = "fsmn-vad",
        punc_model: Optional[str] = "ct-punc",
        disable_update: bool = True,
    ):
        """初始化 FunASR 模型。

        Args:
            model: FunASR 模型路径或 ModelScope ID
            device: 计算设备（"cuda:0", "cpu"）
            vad_model: VAD 模型标识，默认 "fsmn-vad"
            punc_model: 标点模型标识，默认 "ct-punc"
            disable_update: 是否禁用模型自动更新
        """
        try:
            from funasr import AutoModel  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency: funasr. Install with `pip install -U funasr modelscope`."
            ) from exc

        self.model = AutoModel(
            model=model,
            vad_model=vad_model,
            punc_model=punc_model,
            device=device,
            disable_update=disable_update,
        )

    def generate(
        self,
        waveform_1d: torch.Tensor,
        *,
        pred_timestamp: bool = True,
        sentence_timestamp: bool = True,
        en_post_proc: bool = True,
        return_raw_text: bool = True,
        batch_size_s: int = 300,
        batch_size_threshold_s: int = 60,
    ) -> dict[str, Any]:
        """调用 FunASR 模型进行语音识别。

        Args:
            waveform_1d: 1D 波形数据 [T]
            pred_timestamp: 是否预测时间戳
            sentence_timestamp: 是否返回句子级时间戳
            en_post_proc: 是否启用后处理
            return_raw_text: 是否返回原始文本
            batch_size_s: 批次大小（秒）
            batch_size_threshold_s: 批次阈值（秒）

        Returns:
            包含识别结果和时间戳的字典
        """
        res = self.model.generate(
            input=waveform_1d,
            cache={},
            pred_timestamp=pred_timestamp,
            sentence_timestamp=sentence_timestamp,
            en_post_proc=en_post_proc,
            return_raw_text=return_raw_text,
            batch_size_s=batch_size_s,
            batch_size_threshold_s=batch_size_threshold_s,
        )
        if not isinstance(res, list) or len(res) == 0 or not isinstance(res[0], dict):
            raise RuntimeError(f"Unexpected FunASR output: {type(res)}")
        return res[0]

    def parse_sentence_info(
        self,
        item: dict[str, Any],
        *,
        speaker: str,
        segment_start_ms: int,
    ) -> list[Utterance]:
        sentence_info = item.get("sentence_info")
        if not isinstance(sentence_info, list):
            text = str(item.get("text") or "").strip()
            if not text:
                return []
            return [
                Utterance(
                    speaker=speaker,
                    start_ms=segment_start_ms,
                    end_ms=segment_start_ms,
                    text=text,
                )
            ]

        utterances: list[Utterance] = []
        for s in sentence_info:
            if not isinstance(s, dict):
                continue
            text = str(s.get("text") or "").strip()
            if not text:
                continue
            start_ms = int(float(s.get("start") or 0)) + segment_start_ms
            end_ms = int(float(s.get("end") or start_ms)) + segment_start_ms
            utterances.append(
                Utterance(
                    speaker=speaker,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=text,
                )
            )
        return utterances

    def transcribe_segment(
        self,
        waveform_16k_mono: torch.Tensor,
        *,
        speaker: str,
        segment_start_ms: int,
        pred_timestamp: bool = True,
        sentence_timestamp: bool = True,
        en_post_proc: bool = True,
        return_raw_text: bool = True,
        batch_size_s: int = 300,
        batch_size_threshold_s: int = 60,
    ) -> tuple[list[Utterance], dict[str, Any]]:
        """识别一个说话人切片音频，返回 Utterance 列表。

        这是 FunASRTranscriber 的核心方法，封装了：
        1. 波形格式转换（2D -> 1D）
        2. 调用 FunASR 模型识别
        3. 解析结果为 Utterance 列表

        Args:
            waveform_16k_mono: 16kHz 单声道波形 [1, T]
            speaker: 说话人标签（如 "spk0"）
            segment_start_ms: 切片在整段音频中的开始时间（毫秒）
            其他参数同 generate()

        Returns:
            tuple: (utterances 列表, 原始模型输出)
        """
        waveform_1d = _to_waveform_1d(waveform_16k_mono)
        item = self.generate(
            waveform_1d,
            pred_timestamp=pred_timestamp,
            sentence_timestamp=sentence_timestamp,
            en_post_proc=en_post_proc,
            return_raw_text=return_raw_text,
            batch_size_s=batch_size_s,
            batch_size_threshold_s=batch_size_threshold_s,
        )
        utterances = self.parse_sentence_info(
            item,
            speaker=speaker,
            segment_start_ms=segment_start_ms,
        )
        return utterances, item
