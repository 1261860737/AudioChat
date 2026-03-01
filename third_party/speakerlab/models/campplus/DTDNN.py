# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from speakerlab.models.campplus.layers import (
    DenseLayer,
    StatsPool,
    TDNNLayer,
    CAMDenseTDNNBlock,
    TransitLayer,
    BasicResBlock,
    get_nonlinear,
)


class FCM(nn.Module):
    """特征压缩模块（Feature Compression Module）

    用于将 FBank 特征压缩为紧凑的表示。
    通过多层卷积和残差块进行特征提取和压缩。

    属性说明：
        in_planes: 当前输入通道数
        conv1: 初始卷积层
        bn1: 批归一化层
        layer1, layer2: 残差块组成的层
        conv2: 压缩卷积层
        bn2: 批归一化层
        out_channels: 输出通道数
    """

    def __init__(
        self, block=BasicResBlock, num_blocks=[2, 2], m_channels=32, feat_dim=80
    ):
        """初始化特征压缩模块

        Args:
            block: 残差块类型，默认为 BasicResBlock
            num_blocks: 每个层的残差块数量，默认为 [2, 2]
            m_channels: 中间通道数，默认为 32
            feat_dim: 输入特征维度，默认为 80
        """
        super(FCM, self).__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(
            1, m_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[1], stride=2)

        self.conv2 = nn.Conv2d(
            m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        """创建残差块层

        构建由多个残差块组成的层。

        Args:
            block: 残差块类型
            planes: 输出通道数
            num_blocks: 残差块数量
            stride: 第一个残差块的步长

        Returns:
            由残差块组成的序贯层
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播

        处理流程：
        1. 添加通道维度
        2. 通过第一个卷积块
        3. 通过两个残差层
        4. 通过压缩卷积
        5. 展平特征图

        Args:
            x: 输入特征，形状为 [B, T, F]（批次, 时间步, 特征维度）

        Returns:
            压缩后的特征，形状为 [B, C*T, F']
        """
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))

        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])
        return out


class CAMPPlus(nn.Module):
    """CAM++ 说话人嵌入模型

    基于 TDNN 和 CAM Dense Block 的深度说话人嵌入提取网络。
    用于从语音特征中提取说话人身份向量。

    模型架构：
    1. 特征压缩模块（FCM）
    2. TDNN 层
    3. 多个 CAM Dense TDNN 块 + 过渡层
    4. 统计池化层
    5. 全连接层（输出说话人嵌入）

    属性说明：
        head: 特征压缩模块
        xvector: 主干网络（TDNN + CAM Dense Blocks + StatsPool + DenseLayer）
    """

    def __init__(
        self,
        feat_dim=80,
        embedding_size=512,
        growth_rate=32,
        bn_size=4,
        init_channels=128,
        config_str="batchnorm-relu",
        memory_efficient=True,
    ):
        """初始化 CAM++ 模型

        Args:
            feat_dim: 输入特征维度，默认为 80（FBank 特征维度）
            embedding_size: 说话人嵌入向量维度，默认为 512
            growth_rate: DenseNet 生长率，默认为 32
            bn_size: 批归一化瓶颈通道数乘数，默认为 4
            init_channels: 初始通道数，默认为 128
            config_str: 非线性层配置字符串，默认为 'batchnorm-relu'
            memory_efficient: 是否使用内存高效模式，默认为 True
        """
        super(CAMPPlus, self).__init__()

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels

        self.xvector = nn.Sequential(
            OrderedDict(
                [
                    (
                        "tdnn",
                        TDNNLayer(
                            channels,
                            init_channels,
                            5,
                            stride=2,
                            dilation=1,
                            padding=-1,
                            config_str=config_str,
                        ),
                    ),
                ]
            )
        )
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(
            zip((12, 24, 16), (3, 3, 3), (1, 2, 2))
        ):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.xvector.add_module("block%d" % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                "transit%d" % (i + 1),
                TransitLayer(
                    channels, channels // 2, bias=False, config_str=config_str
                ),
            )
            channels //= 2

        self.xvector.add_module("out_nonlinear", get_nonlinear(config_str, channels))

        self.xvector.add_module("stats", StatsPool())
        self.xvector.add_module(
            "dense", DenseLayer(channels * 2, embedding_size, config_str="batchnorm_")
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """前向传播

        处理流程：
        1. 维度转换：[B, T, F] → [B, F, T]
        2. 通过特征压缩模块
        3. 通过主干网络提取嵌入

        Args:
            x: 输入 FBank 特征，形状为 [B, T, F]

        Returns:
            说话人嵌入向量，形状为 [B, embedding_size]
        """
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        x = self.xvector(x)
        return x
