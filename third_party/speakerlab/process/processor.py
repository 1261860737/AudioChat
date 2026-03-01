# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import random
import pickle
import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi

from speakerlab.process.augmentation import NoiseReverbCorrupter
from speakerlab.utils.fileio import load_data_csv


class WavReader(object):
    """音频读取器，用于加载和预处理音频文件

    支持以下功能：
    - 加载 WAV 格式的音频文件
    - 采样率检查和转换
    - 速度扰动（可选）
    - 随机裁剪或填充到固定时长

    属性说明：
        sample_rate: 目标采样率，默认 16000 Hz
        duration: 音频片段时长，默认 3.0 秒
        speed_pertub: 是否启用速度扰动
        lm: 是否使用左-右速度扰动（当 speed_pertub 为 True 时有效）
    """

    def __init__(
        self,
        sample_rate=16000,
        duration: float = 3.0,
        speed_pertub: bool = False,
        lm: bool = True,
    ):
        """初始化音频读取器

        Args:
            sample_rate: 目标采样率，默认 16000 Hz
            duration: 音频片段时长，默认 3.0 秒
            speed_pertub: 是否启用速度扰动，默认为 False
            lm: 是否使用左-右速度扰动，默认为 True
        """
        self.duration = duration
        self.sample_rate = sample_rate
        self.speed_pertub = speed_pertub
        self.lm = lm

    def __call__(self, wav_path):
        """读取并预处理音频文件

        处理流程：
        1. 加载音频文件
        2. 检查采样率
        3. 应用速度扰动（可选）
        4. 随机裁剪或填充到固定时长

        Args:
            wav_path: 音频文件路径

        Returns:
            wav: 处理后的音频波形（一维张量）
            speed_idx: 速度扰动索引（0=正常, 1=0.9倍速, 2=1.1倍速）
        """


class SpkLabelEncoder(object):
    """说话人标签编码器，用于将说话人标签转换为整数索引

    该编码器支持速度扰动标签扩展，即同一个说话人在不同速度下的音频
    会被分配不同的标签索引。

    属性说明：
        lab2ind: 标签到索引的映射字典
        ind2lab: 索引到标签的映射字典
        starting_index: 当前分配的最后一个索引
    """

    def __init__(self, data_file):
        """初始化说话人标签编码器

        Args:
            data_file: 包含说话人信息的 CSV 文件路径
        """
        self.lab2ind = {}
        self.ind2lab = {}
        self.starting_index = -1
        self.load_from_csv(data_file)

    def __call__(self, spk, speed_idx=0):
        """将说话人标签转换为编码索引

        支持速度扰动标签扩展：同一个说话人在不同速度下的音频
        会被分配不同的标签索引。

        Args:
            spk: 说话人标签（字符串）
            speed_idx: 速度扰动索引（0=正常, 1=慢速, 2=快速）

        Returns:
            编码后的说话人索引（整数）
        """
        spkid = self.lab2ind[spk]
        spkid = spkid + len(self.lab2ind) * speed_idx
        return spkid

    def load_from_csv(self, path):
        """从 CSV 文件加载说话人数据

        读取 CSV 文件中的说话人信息，并构建标签映射。

        Args:
            path: CSV 文件路径
        """
        self.data = load_data_csv(path)
        for key in self.data:
            self.add(self.data[key]["spk"])

    def add(self, label):
        """添加说话人标签到编码器

        如果标签已存在，则不重复添加。

        Args:
            label: 说话人标签（字符串）
        """
        if label in self.lab2ind:
            return
        index = self._next_index()
        self.lab2ind[label] = index
        self.ind2lab[index] = label

    def _next_index(self):
        """获取下一个可用的标签索引

        Returns:
            下一个可用的整数索引
        """
        self.starting_index += 1
        return self.starting_index

    def __len__(self):
        """返回编码器中标签的数量

        Returns:
            说话人标签的数量（整数）
        """
        return len(self.lab2ind)

    def save(self, path, device=None):
        """保存标签编码器到文件

        将 lab2ind 字典保存为 pickle 格式的文件。

        Args:
            path: 输出文件路径
            device: 设备参数（当前未使用，保留兼容性）
        """
        with open(path, "wb") as f:
            pickle.dump(self.lab2ind, f)

    def load(self, path, device=None):
        """从文件加载标签编码器

        从 pickle 格式的文件中加载 lab2ind 字典，
        并自动构建 ind2lab 逆向映射。

        Args:
            path: 输入文件路径
            device: 设备参数（当前未使用，保留兼容性）
        """
        self.lab2ind = {}
        self.ind2lab = {}
        with open(path, "rb") as f:
            self.lab2ind = pickle.load(f)
        for label in self.lab2ind:
            self.ind2lab[self.lab2ind[label]] = label


class SpkVeriAug(object):
    """说话人验证数据增强器，用于对音频进行数据增强

    支持的增强类型：
    - 添加噪声
    - 添加混响
    - 同时添加噪声和混响

    属性说明：
        aug_prob: 数据增强的概率（0.0-1.0）
        augmentations: 增强器列表
    """

    def __init__(
        self,
        aug_prob: float = 0.0,
        noise_file: str = None,
        reverb_file: str = None,
    ):
        """初始化数据增强器

        Args:
            aug_prob: 数据增强的概率，默认为 0.0（不进行增强）
            noise_file: 噪声文件路径，默认为 None
            reverb_file: 混响文件路径，默认为 None
        """
        self.aug_prob = aug_prob
        if aug_prob > 0:
            self.add_noise = NoiseReverbCorrupter(
                noise_prob=1.0,
                noise_file=noise_file,
            )
            self.add_rir = NoiseReverbCorrupter(
                reverb_prob=1.0,
                reverb_file=reverb_file,
            )
            self.add_rir_noise = NoiseReverbCorrupter(
                noise_prob=1.0,
                reverb_prob=1.0,
                noise_file=noise_file,
                reverb_file=reverb_file,
            )

            self.augmentations = [self.add_noise, self.add_rir, self.add_rir_noise]

    def __call__(self, wav):
        """对音频应用数据增强

        根据 aug_prob 概率随机选择是否进行数据增强，
        如果进行增强，则随机选择一种增强方式。

        Args:
            wav: 输入音频波形（一维张量）

        Returns:
            增强后的音频波形（如果进行了增强），否则返回原始音频
        """
        sample_rate = 16000
        if self.aug_prob > random.random():
            aug = random.choice(self.augmentations)
            wav = aug(wav, sample_rate)

        return wav


class FBank(object):
    """FBank 特征提取器，用于从音频波形中提取 Filter Bank 特征

    使用 Kaldi 的 fbank 函数计算梅尔滤波器组特征。

    属性说明：
        n_mels: 梅尔滤波器组的滤波器数量
        sample_rate: 采样率
        mean_nor: 是否进行均值归一化
    """

    def __init__(
        self,
        n_mels,
        sample_rate,
        mean_nor: bool = False,
    ):
        """初始化 FBank 特征提取器

        Args:
            n_mels: 梅尔滤波器组的滤波器数量
            sample_rate: 采样率
            mean_nor: 是否进行均值归一化，默认为 False
        """
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mean_nor = mean_nor

    def __call__(self, wav, dither=0):
        """从音频波形中提取 FBank 特征

        处理流程：
        1. 检查采样率
        2. 处理多通道音频（选择单通道）
        3. 提取 FBank 特征
        4. 可选的均值归一化

        Args:
            wav: 输入音频波形（一维或二维张量）
            dither: 抖动参数，用于特征提取，默认为 0

        Returns:
            FBank 特征矩阵，形状为 [时间步, n_mels]
        """
        sr = 16000
        assert sr == self.sample_rate
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        # 选择单通道
        if wav.shape[0] > 1:
            wav = wav[0, :]
            wav = wav.unsqueeze(0)
        assert len(wav.shape) == 2 and wav.shape[0] == 1
        feat = Kaldi.fbank(
            wav, num_mel_bins=self.n_mels, sample_frequency=sr, dither=dither
        )
        # 特征矩阵：[时间步, 滤波器数量]
        if self.mean_nor:
            feat = feat - feat.mean(0, keepdim=True)
        return feat
