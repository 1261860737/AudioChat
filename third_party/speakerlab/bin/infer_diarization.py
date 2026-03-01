# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
这是一个基于预训练模型的说话人分离推理脚本。
使用方法：
    1. python infer_diarization.py --wav [wav_list OR wav_path] --out_dir [out_dir]
    2. python infer_diarization.py --wav [wav_list OR wav_path] --out_dir [out_dir] --include_overlap --hf_access_token [hf_access_token]
    3. python infer_diarization.py --wav [wav_list OR wav_path] --out_dir [out_dir] --include_overlap --hf_access_token [hf_access_token] --nprocs [n]
"""

import os
import sys
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from scipy import optimize
import json

import torch
import torch.multiprocessing as mp

try:
    from speakerlab.utils.config import Config
except ImportError:
    sys.path.append("%s/../.." % os.path.dirname(os.path.abspath(__file__)))
    from speakerlab.utils.config import Config

from speakerlab.utils.builder import build
from speakerlab.utils.utils import (
    merge_vad,
    silent_print,
    download_model_from_modelscope,
    circle_pad,
)
from speakerlab.utils.fileio import load_audio

os.environ["MODELSCOPE_LOG_LEVEL"] = "40"
warnings.filterwarnings("ignore")

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from pyannote.audio import Inference, Model

parser = argparse.ArgumentParser(description="Speaker diarization inference.")
parser.add_argument("--wav", type=str, required=True, help="Input wavs")
parser.add_argument("--out_dir", type=str, required=True, help="Out results dir")
parser.add_argument(
    "--out_type",
    choices=["rttm", "json"],
    default="rttm",
    type=str,
    help="Results format, rttm or json",
)
parser.add_argument(
    "--include_overlap", action="store_true", help="Include overlapping region"
)
parser.add_argument(
    "--hf_access_token",
    type=str,
    help="hf_access_token for pyannote/segmentation-3.0 model. It's required if --include_overlap is specified",
)
parser.add_argument(
    "--diable_progress_bar", action="store_true", help="Close the progress bar"
)
parser.add_argument("--nprocs", default=None, type=int, help="Num of procs")
parser.add_argument(
    "--speaker_num", default=None, type=int, help="Oracle num of speaker"
)


def get_speaker_embedding_model(device: torch.device = None, cache_dir: str = None):
    """获取说话人嵌入模型

    从 ModelScope 下载预训练的 CAM++ 说话人嵌入模型，用于提取音频片段的说话人特征向量。

    Args:
        device: 计算设备（CPU/GPU），默认为 None
        cache_dir: 模型缓存目录，默认为 None

    Returns:
        embedding_model: 说话人嵌入模型
        feature_extractor: 特征提取器（FBank）
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

    cache_dir = download_model_from_modelscope(
        conf["model_id"], conf["revision"], cache_dir
    )
    pretrained_model_path = os.path.join(cache_dir, conf["model_ckpt"])
    config = Config(conf)
    feature_extractor = build("feature_extractor", config)
    embedding_model = build("embedding_model", config)

    # 加载预训练模型
    pretrained_state = torch.load(pretrained_model_path, map_location="cpu")
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.eval()
    if device is not None:
        embedding_model.to(device)
    return embedding_model, feature_extractor


def get_cluster_backend():
    """获取聚类后端

    创建并返回一个谱聚类（Spectral Clustering）后端，用于对说话人嵌入进行聚类分析。

    Returns:
        CommonClustering: 聚类器实例，用于将嵌入向量分组为不同的说话人
    """
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
    device: torch.device = None, cache_dir: str = None
):
    """获取语音活动检测模型

    从 ModelScope 下载并加载语音活动检测（VAD）模型，用于检测音频中的语音段落。

    Args:
        device: 计算设备，默认为 None
        cache_dir: 模型缓存目录，默认为 None

    Returns:
        vad_pipeline: VAD 推理管道，返回语音时间段列表
    """
    conf = {
        "model_id": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "revision": "v2.0.4",
    }
    cache_dir = download_model_from_modelscope(
        conf["model_id"], conf["revision"], cache_dir
    )
    with silent_print():
        vad_pipeline = pipeline(
            task=Tasks.voice_activity_detection,
            model=cache_dir,
            device="cpu"
            if device is None
            else "%s:%s" % (device.type, device.index)
            if device.index
            else device.type,
            disable_pbar=True,
            disable_update=True,
        )
    return vad_pipeline


def get_segmentation_model(use_auth_token, device: torch.device = None):
    """获取语音分割模型

    从 Hugging Face 下载并加载 pyannote 分割模型，用于检测重叠语音段。
    该模型用于处理多人同时说话的场景。

    Args:
        use_auth_token: Hugging Face 访问令牌
        device: 计算设备，默认为 None

    Returns:
        Inference: pyannote 分割推理对象
    """
    segmentation_params = {
        "segmentation": "pyannote/segmentation-3.0",
        "segmentation_batch_size": 32,
        "use_auth_token": use_auth_token,
    }
    model = Model.from_pretrained(
        segmentation_params["segmentation"],
        use_auth_token=segmentation_params["use_auth_token"],
        strict=False,
    )
    segmentation = Inference(
        model,
        duration=model.specifications.duration,
        step=0.1 * model.specifications.duration,
        skip_aggregation=True,
        batch_size=segmentation_params["segmentation_batch_size"],
        device=device,
    )
    return segmentation


class Diarization3Dspeaker:
    """
    说话人分离处理类，用于识别和分割音频中的说话人身份。

    参数说明：
        device (str, 默认为 None): 模型运行的设备
        include_overlap (bool, 默认为 False): 是否在分离结果中包含重叠语音段。
            当多个说话人同时说话时会产生重叠语音。
        hf_access_token (str, 默认为 None): Hugging Face 访问令牌，当 include_overlap 为 True 时需要提供。
            该令牌用于访问 Hugging Face 上的 pyannote 分割模型，以处理重叠语音。
        speaker_num (int, 默认为 None): 预设的说话人数量
        model_cache_dir (str, 默认为 None): 如果指定，预训练模型将下载到该目录；仅支持从 ModelScope 下载的预训练模型

    使用示例：
        diarization_pipeline = Diarization3Dspeaker(device, include_overlap, hf_access_token)
        output = diarization_pipeline(input_audio) # input_audio 可以是 WAV 文件路径、NumPy 数组或 PyTorch 张量
        print(output) # 输出格式：[[1.1, 2.2, 0], [3.1, 4.1, 1], ..., [st_n, ed_n, speaker_id]]
        diarization_pipeline.save_diar_output('audio.rttm') # 或保存为 audio.json
    """

    def __init__(
        self,
        device=None,
        include_overlap=False,
        hf_access_token=None,
        speaker_num=None,
        model_cache_dir=None,
    ):
        if include_overlap and hf_access_token is None:
            raise ValueError(
                "hf_access_token is required when include_overlap is True."
            )

        self.device = self.normalize_device(device)
        self.include_overlap = include_overlap

        self.embedding_model, self.feature_extractor = get_speaker_embedding_model(
            self.device, model_cache_dir
        )
        self.vad_model = get_voice_activity_detection_model(
            self.device, model_cache_dir
        )
        self.cluster = get_cluster_backend()

        if include_overlap:
            self.segmentation_model = get_segmentation_model(
                hf_access_token, self.device
            )

        self.batchsize = 64
        self.fs = self.feature_extractor.sample_rate
        self.output_field_labels = None
        self.speaker_num = speaker_num

    def __call__(self, wav, wav_fs=None, speaker_num=None):
        """执行说话人分离流程

        完整的说话人分离管道，包含以下步骤：
        1. 语音活动检测（VAD）
        2. 语音分割（可选，处理重叠语音）
        3. 子段落生成
        4. 说话人嵌入提取
        5. 聚类分析
        6. 后处理（可选，包含重叠结果）

        Args:
            wav: 输入音频，可以是文件路径、NumPy 数组或 PyTorch 张量
            wav_fs: 音频采样率，如果为 None 则从文件读取
            speaker_num: 预设的说话人数量，默认为 None（自动检测）

        Returns:
            output_field_labels: 分离结果列表，每个元素为 [开始时间, 结束时间, 说话人ID]
        """
        wav_data = load_audio(wav, wav_fs, self.fs)

        # 阶段 1-1: 执行语音活动检测
        vad_time = self.do_vad(wav_data)
        if self.include_overlap:
            # 阶段 1-2: 执行语音分割
            segmentations, count = self.do_segmentation(wav_data)
            valid_field = get_valid_field(count)
            vad_time = merge_vad(vad_time, valid_field)

        # 阶段 2: 准备子段落
        chunks = [c for (st, ed) in vad_time for c in self.chunk(st, ed)]

        # 阶段 3: 提取嵌入
        embeddings = self.do_emb_extraction(chunks, wav_data)

        # 阶段 4: 聚类
        speaker_num, output_field_labels = self.do_clustering(
            chunks, embeddings, speaker_num
        )

        if self.include_overlap:
            # 阶段 5: 包含重叠结果
            binary = self.post_process(
                output_field_labels, speaker_num, segmentations, count
            )
            timestamps = [
                count.sliding_window[i].middle for i in range(binary.shape[0])
            ]
            output_field_labels = self.binary_to_segs(binary, timestamps)

        self.output_field_labels = output_field_labels
        return output_field_labels

    def do_vad(self, wav):
        """执行语音活动检测

        使用 VAD 模型检测音频中的语音段落，过滤掉静音和非语音部分。

        Args:
            wav: 音频波形数据，形状为 [1, T]

        Returns:
            vad_time: 语音时间段列表，每个元素为 [开始时间(秒), 结束时间(秒)]
        """
        # 音频波形数据：[1, T]
        vad_results = self.vad_model(wav[0])[0]
        vad_time = [
            [vad_t[0] / 1000, vad_t[1] / 1000] for vad_t in vad_results["value"]
        ]
        return vad_time

    def do_segmentation(self, wav):
        """执行语音分割

        使用 pyannote 分割模型检测音频中的重叠语音段。
        该功能用于处理多人同时说话的复杂场景。

        Args:
            wav: 音频波形数据

        Returns:
            segmentations: 分割结果张量
            count: 聚合后的帧计数信息
        """
        segmentations = self.segmentation_model(
            {"waveform": wav, "sample_rate": self.fs}
        )
        frame_windows = self.segmentation_model.model.receptive_field

        count = Inference.aggregate(
            np.sum(segmentations, axis=-1, keepdims=True),
            frame_windows,
            hamming=False,
            missing=0.0,
            skip_average=False,
        )
        count.data = np.rint(count.data).astype(np.uint8)
        return segmentations, count

    def chunk(self, st, ed, dur=1.5, step=0.75):
        """将时间段分割为重叠的子段落

        将一个大的时间段分割成多个固定长度的小段落，用于后续的嵌入提取。
        子段落之间有重叠（step < dur），以确保边界信息不被遗漏。

        Args:
            st: 开始时间（秒）
            ed: 结束时间（秒）
            dur: 每个子段落的时长（秒），默认为 1.5
            step: 子段落的步长（秒），默认为 0.75

        Returns:
            chunks: 子段落列表，每个元素为 [开始时间, 结束时间]
        """
        chunks = []
        subseg_st = st
        while subseg_st + dur < ed + step:
            subseg_ed = min(subseg_st + dur, ed)
            chunks.append([subseg_st, subseg_ed])
            subseg_st += step
        return chunks

    def do_emb_extraction(self, chunks, wav):
        """执行说话人嵌入提取

        对每个子段落提取说话人嵌入向量，用于后续的聚类分析。

        Args:
            chunks: 子段落列表，每个元素为 [开始时间, 结束时间]
            wav: 原始音频波形数据

        Returns:
            embeddings: 说话人嵌入向量矩阵，形状为 [num_chunks, embedding_size]
        """
        # 子段落列表：[[开始时间1, 结束时间1]...]
        # 音频波形数据：[1, T]
        wavs = [wav[0, int(st * self.fs) : int(ed * self.fs)] for st, ed in chunks]
        max_len = max([x.shape[0] for x in wavs])
        wavs = [circle_pad(x, max_len) for x in wavs]
        wavs = torch.stack(wavs).unsqueeze(1)

        embeddings = []
        batch_st = 0
        with torch.no_grad():
            while batch_st < len(chunks):
                wavs_batch = wavs[batch_st : batch_st + self.batchsize].to(self.device)
                feats_batch = torch.vmap(self.feature_extractor)(wavs_batch)
                embeddings_batch = self.embedding_model(feats_batch).cpu()
                embeddings.append(embeddings_batch)
                batch_st += self.batchsize
        embeddings = torch.cat(embeddings, dim=0).numpy()
        return embeddings

    def do_clustering(self, chunks, embeddings, speaker_num=None):
        """执行说话人聚类

        使用谱聚类算法将说话人嵌入向量分组到不同的说话人簇中。

        Args:
            chunks: 子段落列表
            embeddings: 说话人嵌入向量矩阵
            speaker_num: 预设的说话人数量，默认为 None

        Returns:
            speaker_num: 检测到的说话人数量
            output_field_labels: 带说话人标签的时间段列表
        """
        cluster_labels = self.cluster(
            embeddings,
            speaker_num=speaker_num if speaker_num is not None else self.speaker_num,
        )
        speaker_num = cluster_labels.max() + 1
        output_field_labels = [
            [i[0], i[1], int(j)] for i, j in zip(chunks, cluster_labels)
        ]
        output_field_labels = compressed_seg(output_field_labels)
        return speaker_num, output_field_labels

    def post_process(self, output_field_labels, speaker_num, segmentations, count):
        """后处理重叠语音

        将聚类结果与分割模型的输出进行融合，处理重叠语音段。
        使用匈牙利算法进行最优标签分配。

        Args:
            output_field_labels: 原始聚类结果
            speaker_num: 说话人数量
            segmentations: 分割模型输出
            count: 帧计数信息

        Returns:
            binary: 二进制激活矩阵，形状为 [num_frames, speaker_num]
        """
        num_frames = len(count)
        cluster_frames = np.zeros((num_frames, speaker_num))
        frame_windows = count.sliding_window
        for i in output_field_labels:
            cluster_frames[
                frame_windows.closest_frame(
                    i[0] + frame_windows.duration / 2
                ) : frame_windows.closest_frame(i[1] + frame_windows.duration / 2),
                i[2],
            ] = 1.0

        activations = np.zeros((num_frames, speaker_num))
        num_chunks, num_frames_per_chunk, num_classes = segmentations.data.shape
        for i, (c, data) in enumerate(segmentations):
            # data: [每块的帧数, 类别数]
            # chunk_cluster_frames: [每块的帧数, 说话人数量]
            start_frame = frame_windows.closest_frame(
                c.start + frame_windows.duration / 2
            )
            end_frame = start_frame + num_frames_per_chunk
            chunk_cluster_frames = cluster_frames[start_frame:end_frame]
            align_chunk_cluster_frames = np.zeros((num_frames_per_chunk, speaker_num))

            # 根据 "data" 与 "chunk_cluster_frames" 之间的重叠帧数，
            # 为 "data" 的每个维度分配标签
            cost_matrix = []
            for j in range(num_classes):
                if sum(data[:, j]) > 0:
                    num_of_overlap_frames = [
                        (data[:, j].astype("int") & d.astype("int")).sum()
                        for d in chunk_cluster_frames.T
                    ]
                else:
                    num_of_overlap_frames = [-1] * speaker_num
                cost_matrix.append(num_of_overlap_frames)
            cost_matrix = np.array(cost_matrix)  # (类别数, 说话人数量)
            row_index, col_index = optimize.linear_sum_assignment(-cost_matrix)
            for j in range(len(row_index)):
                r = row_index[j]
                c = col_index[j]
                if cost_matrix[r, c] > 0:
                    align_chunk_cluster_frames[:, c] = np.maximum(
                        data[:, r], align_chunk_cluster_frames[:, c]
                    )
            activations[start_frame:end_frame] += align_chunk_cluster_frames

        # 根据计数数据校正激活值
        sorted_speakers = np.argsort(-activations, axis=-1)
        binary = np.zeros_like(activations)
        for t, ((_, c), speakers) in enumerate(zip(count, sorted_speakers)):
            cur_max_spk_num = min(speaker_num, c.item())
            for i in range(cur_max_spk_num):
                if activations[t, speakers[i]] > 0:
                    binary[t, speakers[i]] = 1.0

        supplement_field = (binary.sum(-1) == 0) & (cluster_frames.sum(-1) != 0)
        binary[supplement_field] = cluster_frames[supplement_field]
        return binary

    def binary_to_segs(self, binary, timestamps, threshold=0.5):
        """将二进制激活矩阵转换为时间段列表

        将后处理后的二进制激活矩阵转换为带说话人标签的时间段格式。

        Args:
            binary: 二进制激活矩阵，形状为 [num_frames, speaker_num]
            timestamps: 时间戳列表
            threshold: 激活阈值，默认为 0.5

        Returns:
            output_field_labels: 时间段列表，按开始时间排序
        """
        output_field_labels = []
        # binary: [帧数, 类别数]
        # timestamps: [T_1, ..., T_帧数]
        for k, k_scores in enumerate(binary.T):
            start = timestamps[0]
            is_active = k_scores[0] > threshold

            for t, y in zip(timestamps[1:], k_scores[1:]):
                if is_active:
                    if y < threshold:
                        output_field_labels.append([round(start, 3), round(t, 3), k])
                        start = t
                        is_active = False
                else:
                    if y > threshold:
                        start = t
                        is_active = True

            if is_active:
                output_field_labels.append([round(start, 3), round(t, 3), k])
        return sorted(output_field_labels, key=lambda x: x[0])

    def save_diar_output(self, out_file, wav_id=None, output_field_labels=None):
        """保存说话人分离结果

        将分离结果保存为 RTTM 或 JSON 格式的文件。

        Args:
            out_file: 输出文件路径（支持 .rttm 和 .json 格式）
            wav_id: 音频 ID，默认为 None（使用文件名作为 ID）
            output_field_labels: 分离结果列表，默认为 None（使用当前实例的结果）

        Raises:
            ValueError: 当没有可保存的结果时
            ValueError: 当文件格式不支持时
        """
        if output_field_labels is None and self.output_field_labels is None:
            raise ValueError("No results can be saved.")
        if output_field_labels is None:
            output_field_labels = self.output_field_labels

        wav_id = "default" if wav_id is None else wav_id
        if out_file.endswith("rttm"):
            line_str = "SPEAKER {} 0 {:.3f} {:.3f} <NA> <NA> {:d} <NA> <NA>\n"
            with open(out_file, "w") as f:
                for seg in output_field_labels:
                    seg_st, seg_ed, cluster_id = seg
                    f.write(
                        line_str.format(wav_id, seg_st, seg_ed - seg_st, cluster_id)
                    )
        elif out_file.endswith("json"):
            out_json = {}
            for seg in output_field_labels:
                seg_st, seg_ed, cluster_id = seg
                item = {
                    "start": seg_st,
                    "stop": seg_ed,
                    "speaker": cluster_id,
                }
                segid = (
                    wav_id + "_" + str(round(seg_st, 3)) + "_" + str(round(seg_ed, 3))
                )
                out_json[segid] = item
            with open(out_file, mode="w") as f:
                json.dump(out_json, f, indent=2)
        else:
            raise ValueError(
                "The supported output file formats are currently limited to RTTM and JSON."
            )

    def normalize_device(self, device=None):
        """标准化设备参数

        将设备参数转换为统一的 torch.device 对象。

        Args:
            device: 设备参数，可以是 None、字符串或 torch.device 对象

        Returns:
            torch.device: 标准化的设备对象
        """
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        elif isinstance(device, str):
            device = torch.device(device)
        else:
            assert isinstance(device, torch.device)
        return device


def get_valid_field(count):
    """获取有效的语音活动字段

    从分割模型的计数结果中提取连续的语音活动时间段。
    用于合并 VAD 结果和分割结果。

    Args:
        count: pyannote 分割模型的聚合计数结果

    Returns:
        valid_field: 有效时间段列表，每个元素为 [开始时间, 结束时间]
    """
    valid_field = []
    start = None
    for i, (c, data) in enumerate(count):
        if data.item() == 0 or i == len(count) - 1:
            if start is not None:
                end = c.middle
                valid_field.append([start, end])
                start = None
        else:
            if start is None:
                start = c.middle
    return valid_field


def compressed_seg(seg_list):
    """压缩合并相邻的同说话人段落

    将连续的同说话人时间段合并为一个段落，减少冗余输出。

    Args:
        seg_list: 时间段列表，每个元素为 [开始时间, 结束时间, 说话人ID]

    Returns:
        new_seg_list: 压缩后的时间段列表
    """
    new_seg_list = []
    for i, seg in enumerate(seg_list):
        seg_st, seg_ed, cluster_id = seg
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


def main_process(rank, nprocs, args, wav_list):
    """多进程主处理函数

    每个进程独立处理分配的音频文件，执行说话人分离流程并保存结果。

    Args:
        rank: 进程编号
        nprocs: 总进程数
        args: 命令行参数
        wav_list: 待处理的音频文件列表
    """
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        ngpus = torch.cuda.device_count()
        device = torch.device("cuda:%d" % (rank % ngpus))
    diarization = Diarization3Dspeaker(
        device, args.include_overlap, args.hf_access_token, args.speaker_num
    )

    wav_list = wav_list[rank::nprocs]
    if rank == 0 and (not args.diable_progress_bar):
        wav_list = tqdm(wav_list, desc=f"Rank 0 processing")
    for wav_path in wav_list:
        ouput = diarization(wav_path)
        # 写入文件
        wav_id = os.path.basename(wav_path).rsplit(".", 1)[0]
        if args.out_dir is not None:
            out_file = os.path.join(args.out_dir, wav_id + ".%s" % args.out_type)
        else:
            out_file = "%s.%s" % (wav_path.rsplit(".", 1)[0], args.out_type)
        diarization.save_diar_output(out_file, wav_id)


def main():
    """主函数

    说话人分离的入口函数，负责：
    1. 解析命令行参数
    2. 下载并加载预训练模型
    3. 准备音频文件列表
    4. 启动多进程进行说话人分离

    支持的输入格式：
    - 单个 WAV 文件
    - 包含多个 WAV 文件路径的文本文件

    支持的输出格式：
    - RTTM 格式（默认）
    - JSON 格式
    """
    args = parser.parse_args()
    if args.include_overlap and args.hf_access_token is None:
        parser.error(
            "--hf_access_token is required when --include_overlap is specified."
        )

    get_speaker_embedding_model()
    get_voice_activity_detection_model()
    get_cluster_backend()
    if args.include_overlap:
        get_segmentation_model(args.hf_access_token)
    print(f"[INFO]: Model downloaded successfully.")

    if args.wav.endswith(".wav"):
        # 输入是 WAV 文件
        wav_list = [args.wav]
    else:
        try:
            # 输入应该是 WAV 文件列表
            with open(args.wav, "r") as f:
                wav_list = [i.strip() for i in f.readlines()]
        except:
            raise Exception("[ERROR]: Input should be a wav file or a wav list.")
    assert len(wav_list) > 0

    if args.nprocs is None:
        ngpus = torch.cuda.device_count()
        if ngpus > 0:
            print(f"[INFO]: Detected {ngpus} GPUs.")
            args.nprocs = ngpus
        else:
            print("[INFO]: No GPUs detected.")
            args.nprocs = 1

    args.nprocs = min(len(wav_list), args.nprocs)
    print(f"[INFO]: Set {args.nprocs} processes to extract embeddings.")

    # 输出目录
    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)

    mp.spawn(main_process, nprocs=args.nprocs, args=(args.nprocs, args, wav_list))


if __name__ == "__main__":
    main()
