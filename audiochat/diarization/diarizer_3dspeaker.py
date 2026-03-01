# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from speakerlab.utils.builder import build
from speakerlab.utils.config import Config
from speakerlab.utils.fileio import load_audio
from speakerlab.utils.utils import (
    circle_pad,
    download_model_from_modelscope,
    silent_print,
)


@dataclass(frozen=True)
class DiarizationSegment:
    start_s: float
    end_s: float
    speaker: int


def get_speaker_embedding_model(
    device: torch.device | None = None,
    cache_dir: str | None = None,
    local_model_path: str | None = None,
):
    """
    获取说话人嵌入模型。

    Args:
        device: 计算设备
        cache_dir: ModelScope 模型缓存目录
        local_model_path: 本地模型路径（.pt 文件）。如果指定，则跳过 ModelScope 下载，
                         直接从此路径加载模型权重。
    """
    conf = {
        "model_id": "iic/speech_campplus_sv_zh_en_16k-common_advanced",
        "revision": "v1.0.0",
        "model_ckpt": "campplus_cn_en_common.pt",
        "embedding_model": {
            "obj": "speakerlab.models.campplus.DTDNN.CAMPPlus",
            "args": {
                "feat_dim": 80,
                "embedding_size": 192,
            },
        },
        "feature_extractor": {
            "obj": "speakerlab.process.processor.FBank",
            "args": {
                "n_mels": 80,
                "sample_rate": 16000,
                "mean_nor": True,
            },
        },
    }

    if local_model_path is not None:
        # 使用本地模型权重文件
        pretrained_model_path = local_model_path
    else:
        # 从 ModelScope 下载
        cache_dir = download_model_from_modelscope(
            conf["model_id"], conf["revision"], cache_dir
        )
        pretrained_model_path = f"{cache_dir}/{conf['model_ckpt']}"

    config = Config(conf)
    feature_extractor = build("feature_extractor", config)
    embedding_model = build("embedding_model", config)

    pretrained_state = torch.load(pretrained_model_path, map_location="cpu")
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.eval()
    if device is not None:
        embedding_model.to(device)
    return embedding_model, feature_extractor


def get_cluster_backend():
    conf = {
        "cluster": {
            "obj": "speakerlab.process.cluster.CommonClustering",
            "args": {
                "cluster_type": "spectral",
                "mer_cos": 0.8,
                "min_num_spks": 1,
                "max_num_spks": 15,
                "min_cluster_size": 4,
                "oracle_num": None,
                "pval": 0.012,
            },
        }
    }
    config = Config(conf)
    return build("cluster", config)


def get_voice_activity_detection_model(
    device: torch.device | None = None,
    cache_dir: str | None = None,
    local_model_path: str | None = None,
):
    """
    获取 VAD（语音活动检测）模型。

    Args:
        device: 计算设备
        cache_dir: ModelScope 模型缓存目录
        local_model_path: 本地模型路径。如果指定，则跳过 ModelScope 下载，
                         直接从此路径加载模型。
    """
    conf = {
        "model_id": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "revision": "v2.0.4",
    }

    if local_model_path is not None:
        # 使用本地模型目录
        model_dir = local_model_path
    else:
        # 从 ModelScope 下载
        model_dir = download_model_from_modelscope(
            conf["model_id"], conf["revision"], cache_dir
        )

    with silent_print():
        vad_pipeline = pipeline(
            task=Tasks.voice_activity_detection,
            model=model_dir,
            device="cpu"
            if device is None
            else f"{device.type}:{device.index}"
            if device.index is not None
            else device.type,
            disable_pbar=True,
            disable_update=True,
        )
    return vad_pipeline


def compressed_seg(seg_list: list[list[float | int]]) -> list[list[float | int]]:
    new_seg_list: list[list[float | int]] = []
    for i, seg in enumerate(seg_list):
        seg_st, seg_ed, cluster_id = seg
        seg_st = float(seg_st)
        seg_ed = float(seg_ed)
        cluster_id = int(cluster_id)
        if i == 0:
            new_seg_list.append([seg_st, seg_ed, cluster_id])
        elif cluster_id == new_seg_list[-1][2]:
            if seg_st > new_seg_list[-1][1]:
                new_seg_list.append([seg_st, seg_ed, cluster_id])
            else:
                new_seg_list[-1][1] = seg_ed
        else:
            if seg_st < new_seg_list[-1][1]:
                p = (new_seg_list[-1][1] + seg_st) / 2
                new_seg_list[-1][1] = p
                seg_st = p
            new_seg_list.append([seg_st, seg_ed, cluster_id])
    return new_seg_list


def normalize_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    if isinstance(device, str):
        return torch.device(device)
    return device


class Diarization3Dspeaker:
    def __init__(
        self,
        device: str | torch.device | None = None,
        speaker_num: int | None = None,
        model_cache_dir: str | None = None,
        speaker_embedding_model_path: str | None = None,
        vad_model_path: str | None = None,
    ):
        """
        说话人分离模型初始化。

        Args:
            device: 计算设备（"cuda:0", "cpu" 等）
            speaker_num: 预设说话人数量，None 表示自动检测
            model_cache_dir: ModelScope 模型缓存目录
            speaker_embedding_model_path: 本地说话人嵌入模型路径（.pt 文件）。
                                          例如：/data/models/Voice/iic/speech_campplus_sv_zh-cn_16k-common/xxx.pt
            vad_model_path: 本地 VAD 模型目录路径。
                           例如：/data/models/Voice/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch
        """
        self.device = normalize_device(device)
        self.embedding_model, self.feature_extractor = get_speaker_embedding_model(
            self.device, model_cache_dir, local_model_path=speaker_embedding_model_path
        )
        self.vad_model = get_voice_activity_detection_model(
            self.device, model_cache_dir, local_model_path=vad_model_path
        )
        self.cluster = get_cluster_backend()

        self.batchsize = 64
        self.fs = int(self.feature_extractor.sample_rate)
        self.speaker_num = speaker_num

    def __call__(
        self,
        wav: str | np.ndarray | torch.Tensor,
        wav_fs: int | None = None,
        speaker_num: int | None = None,
    ) -> list[list[float | int]]:
        wav_data = load_audio(wav, wav_fs, self.fs)

        vad_results = self.vad_model(wav_data[0])[0]
        vad_time = [[t0 / 1000.0, t1 / 1000.0] for t0, t1 in vad_results["value"]]

        chunks = [c for (st, ed) in vad_time for c in self.chunk(st, ed)]
        if len(chunks) == 0:
            return []

        embeddings = self.do_emb_extraction(chunks, wav_data)

        cluster_labels = self.cluster(
            embeddings,
            speaker_num=speaker_num if speaker_num is not None else self.speaker_num,
        )
        output_field_labels = [
            [i[0], i[1], int(j)] for i, j in zip(chunks, cluster_labels)
        ]
        return compressed_seg(output_field_labels)

    def chunk(self, st: float, ed: float, dur: float = 1.5, step: float = 0.75):
        chunks: list[list[float]] = []
        subseg_st = float(st)
        while subseg_st + dur < float(ed) + step:
            subseg_ed = min(subseg_st + dur, float(ed))
            chunks.append([subseg_st, subseg_ed])
            subseg_st += step
        return chunks

    def do_emb_extraction(
        self, chunks: list[list[float]], wav: torch.Tensor
    ) -> np.ndarray:
        wavs = [wav[0, int(st * self.fs) : int(ed * self.fs)] for st, ed in chunks]
        max_len = max(int(x.shape[0]) for x in wavs)
        wavs = [circle_pad(x, max_len) for x in wavs]
        wavs = torch.stack(wavs).unsqueeze(1)

        embeddings: list[torch.Tensor] = []
        batch_st = 0
        with torch.no_grad():
            while batch_st < len(chunks):
                wavs_batch = wavs[batch_st : batch_st + self.batchsize].to(self.device)
                feats_batch = torch.vmap(self.feature_extractor)(wavs_batch)
                emb_batch = self.embedding_model(feats_batch).cpu()
                embeddings.append(emb_batch)
                batch_st += self.batchsize
        return torch.cat(embeddings, dim=0).numpy()


class ThreeDSpeakerDiarizer:
    def __init__(
        self,
        *,
        device: Optional[str] = None,
        model_cache_dir: Optional[str] = None,
        speaker_num: Optional[int] = None,
        speaker_embedding_model_path: Optional[str] = None,
        vad_model_path: Optional[str] = None,
    ):
        """
        说话人分离器封装类（3D-Speaker 框架）。

        支持使用本地模型路径，绕过 ModelScope 下载。

        Args:
            device: 计算设备（"cuda:0", "cuda", "cpu" 等）
            model_cache_dir: ModelScope 模型缓存目录
            speaker_num: 预设说话人数量，None 表示自动检测
            speaker_embedding_model_path: 本地说话人嵌入模型路径（.pt 文件）。
                                          例如：/data/models/Voice/iic/speech_campplus_sv_zh-cn_16k-common/campplus_cn_en_common.pt
            vad_model_path: 本地 VAD 模型目录路径。
                           例如：/data/models/Voice/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch

        Example:
            # 使用本地模型
            diarizer = ThreeDSpeakerDiarizer(
                device="cuda:0",
                speaker_embedding_model_path="/data/models/Voice/iic/speech_campplus_sv_zh-cn_16k-common/campplus_cn_en_common.pt",
                vad_model_path="/data/models/Voice/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            )
        """
        self.impl = Diarization3Dspeaker(
            device=device,
            speaker_num=speaker_num,
            model_cache_dir=model_cache_dir,
            speaker_embedding_model_path=speaker_embedding_model_path,
            vad_model_path=vad_model_path,
        )

    def diarize(
        self,
        wav: str | np.ndarray | torch.Tensor,
        wav_fs: Optional[int] = None,
        speaker_num: Optional[int] = None,
    ) -> list[DiarizationSegment]:
        out = self.impl(wav, wav_fs=wav_fs, speaker_num=speaker_num)
        return [
            DiarizationSegment(start_s=float(st), end_s=float(ed), speaker=int(spk))
            for st, ed, spk in out
        ]
