from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union
import numpy as np
import torch
from scipy import signal as scipy_signal
import soundfile as sf  # 使用 soundfile 替代 torchaudio（兼容 ARM64）

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


@dataclass(frozen=True)
class AudioData:
    """
    音频数据结构体，包含波形数据和采样率。

    Attributes:
        waveform: 音频波形，形状为 [1, T] 的 float32 Tensor
        sample_rate: 采样率（Hz）
    """

    waveform: torch.Tensor  # [1, T], float32
    sample_rate: int


def load_audio_mono(path: str) -> AudioData:
    """
    加载音频文件并转换为单声道。

    使用 soundfile 读取音频，支持多种格式（wav, mp3, flac 等）。
    读取后自动将多声道混合为单声道，并转换为 float32 类型。

    Args:
        path: 音频文件路径

    Returns:
        AudioData: 包含单声道波形和采样率的数据结构

    Raises:
        RuntimeError: 音频加载失败时抛出
    """
    # 使用 soundfile 加载音频
    waveform_np, sample_rate = sf.read(path, dtype="float32")

    # [T] -> [1, T]
    if waveform_np.ndim == 1:
        waveform_np = waveform_np[np.newaxis, :]

    # 多声道混合为单声道
    if waveform_np.shape[0] > 1:
        waveform_np = waveform_np.mean(axis=0, keepdims=True)

    waveform = torch.from_numpy(waveform_np)
    return AudioData(waveform=waveform.contiguous(), sample_rate=sample_rate)


def resample(audio: AudioData, target_sample_rate: int = 16000) -> AudioData:
    """
    重采样音频到目标采样率。

    使用 scipy.signal 进行重采样，将音频转换为指定的采样率。
    后续 diarization/ASR/LLM 处理流程均使用 16kHz，因此该函数主要用于统一采样率。

    Args:
        audio: 输入的 AudioData 对象
        target_sample_rate: 目标采样率，默认 16000 Hz

    Returns:
        AudioData: 重采样后的音频数据
    """
    

    if audio.sample_rate == target_sample_rate:
        return audio

    # 计算重采样比例
    ratio = target_sample_rate / audio.sample_rate

    # 使用 scipy 进行重采样
    waveform_np = audio.waveform.squeeze(0).numpy()
    waveform_np = scipy_signal.resample(waveform_np, int(len(waveform_np) * ratio))

    # [T] -> [1, T]
    waveform_np = waveform_np[np.newaxis, :]
    waveform = torch.from_numpy(waveform_np).to(torch.float32)

    return AudioData(waveform=waveform.contiguous(), sample_rate=target_sample_rate)


def ensure_mono_16k(path: str) -> AudioData:
    """
    加载音频并统一转换为 16kHz 单声道。

    这是 pipeline 的入口工具函数：
    1. 加载音频文件
    2. 转换为单声道（如原音频为多声道）
    3. 重采样至 16kHz

    后续所有模块（diarization、ASR、LLM）均假设输入为 16kHz mono。

    Args:
        path: 音频文件路径

    Returns:
        AudioData: 16kHz 单声道的音频数据
    """
    return resample(load_audio_mono(path), 16000)


def slice_waveform(
    waveform_16k_mono: torch.Tensor,
    start_s: float,
    end_s: float,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """
    从波形中截取指定时间范围的片段。

    用于 diarization 完成后，按说话人时间段切片音频，送入 ASR 识别。

    Args:
        waveform_16k_mono: 16kHz 单声道波形，形状 [1, T]
        start_s: 片段开始时间（秒）
        end_s: 片段结束时间（秒）
        sample_rate: 采样率，默认 16000

    Returns:
        torch.Tensor: 截取后的波形片段，形状 [1, T_segment]

    Raises:
        ValueError: 输入波形维度不为 [1, T] 时抛出
    """
    if waveform_16k_mono.ndim != 2 or waveform_16k_mono.shape[0] != 1:
        raise ValueError(
            f"Expected waveform [1, T], got {tuple(waveform_16k_mono.shape)}"
        )
    start = max(0, int(round(start_s * sample_rate)))
    end = max(start, int(round(end_s * sample_rate)))
    end = min(end, waveform_16k_mono.shape[1])
    return waveform_16k_mono[:, start:end].contiguous()


def to_numpy_1d(waveform: Union[torch.Tensor, "np.ndarray"]) -> "np.ndarray":
    """
    将波形转换为 1 维 float32 numpy 数组。

    用于将 PyTorch tensor 转换为 numpy，以便：
    - 与非 PyTorch 库（如 scipy、librosa）兼容
    - 保存为音频文件或进行数值分析

    Args:
        waveform: torch.Tensor 或 np.ndarray，支持 [1, T] 或 [T] 形状

    Returns:
        np.ndarray: 1 维 float32 数组

    Raises:
        ValueError: 输入维度不是 1D 或 2D 时抛出
    """
    if isinstance(waveform, np.ndarray):
        arr = waveform
    else:
        # 将 PyTorch tensor 移至 CPU 并转为 numpy
        arr = waveform.detach().cpu().numpy()
    if arr.ndim == 2:
        # [1, T] 或 [T, 1] 降维为 [T]
        arr = arr.squeeze()
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D audio, got shape={arr.shape}")
    return arr.astype(np.float32, copy=False)
