# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""speakerlab.utils.fileio

本模块提供常用的文件读写与音频加载工具函数，主要用于：
- 读取 YAML/CSV/JSON 等配置与数据文件
- 读写 diarization 相关的 wav.scp、trans7time 等格式
- 加载音频并进行必要的重采样、归一化与通道处理

约定：
- 音频数据统一返回 torch.Tensor，形状为 [1, T]
- 时间单位通常为秒（float）
"""

import csv
import yaml
import codecs
import json
import torch
import torchaudio
import numpy as np


def load_yaml(yaml_path):
    """加载 YAML 配置文件

    Args:
        yaml_path: YAML 文件路径

    Returns:
        dict: 解析后的配置字典
    """
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def load_data_csv(fpath):
    """读取 CSV 数据文件为字典

    CSV 文件必须包含 `ID` 字段，且每行的 ID 必须唯一。
    读取后返回的结构为：
        {
            "<ID>": {<其他列名>: <值>, ...},
            ...
        }

    Args:
        fpath: CSV 文件路径

    Returns:
        dict: 以 ID 为键的字典

    Raises:
        KeyError: CSV 缺少 `ID` 字段
        ValueError: 存在重复 ID
    """
    with open(fpath, newline="") as f:
        result = {}
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            if "ID" not in row:
                raise KeyError(
                    "CSV file has to have an 'ID' field, with unique ids for all data points."
                )

            data_id = row["ID"]
            del row["ID"]

            if data_id in result:
                raise ValueError(f"Duplicate id: {data_id}")
            result[data_id] = row
    return result


def load_data_list(fpath):
    """读取文本文件为字典列表

    将文件每一行（strip 后）按行号编号，返回：
        {0: line0, 1: line1, ...}

    Args:
        fpath: 文本文件路径

    Returns:
        dict: 行号到内容的映射
    """
    with open(fpath) as f:
        rows = [i.strip() for i in f.readlines()]
        result = {idx: row for idx, row in enumerate(rows)}
    return result


def load_wav_scp(fpath):
    """读取 Kaldi 风格的 wav.scp 文件

    wav.scp 格式通常为：
        <utt_id> <wav_path>

    Args:
        fpath: wav.scp 文件路径

    Returns:
        dict: utt_id -> wav_path 的映射
    """
    with open(fpath) as f:
        rows = [i.strip() for i in f.readlines()]
        result = {i.split()[0]: i.split()[1] for i in rows}
    return result


def load_json_file(json_file):
    """读取 JSON 文件

    Args:
        json_file: JSON 文件路径

    Returns:
        dict: JSON 解析后的字典
    """
    with codecs.open(json_file, "r", encoding="utf-8") as fr:
        data_dict = json.load(fr)
    return data_dict


def load_trans7time_list(filename):
    """读取 trans7time 格式文件

    trans7time 每行格式：
        spk_id start_time end_time [content]

    - 当一行只有 3 列时，content 为空字符串
    - 当一行超过 3 列时，将第 4 列及之后拼接为 content

    Returns 的每个元素为：
        (spk_id, st, ed, content)

    Args:
        filename: 文件路径

    Returns:
        list: (spk_id, st, ed, content) 的列表

    Raises:
        ValueError: 行格式不合法（列数 <= 2）
    """
    with open(filename, "r") as fr:
        trans7time_list = []
        lines = fr.readlines()
        for line in lines:
            trans7time_list.append(line.strip().split())
        result_trans7time_list = []
    for index, item in enumerate(trans7time_list):
        if len(item) <= 2:
            raise ValueError(f"filename {filename}: item - {index} = {item}")
        if len(item) == 3:
            st = float(item[1])
            ed = float(item[2])
            result_trans7time_list.append((item[0], st, ed, ""))
        else:
            result_trans7time_list.append(
                (item[0], float(item[1]), float(item[2]), "".join(item[3:]))
            )
    return result_trans7time_list


def write_json_file(json_file, data):
    """写入 JSON 文件

    Args:
        json_file: 输出 JSON 文件路径
        data: 可被 json 序列化的数据
    """
    assert str(json_file).endswith(".json") or str(json_file).endswith(".JSON")
    with codecs.open(json_file, "w", encoding="utf-8") as fw:
        json.dump(data, fw, indent=2, ensure_ascii=False)


def write_wav_scp(fpath, wav_scp):
    """写入 wav.scp 文件

    Args:
        fpath: 输出文件路径
        wav_scp: dict，utt_id -> wav_path
    """
    with open(fpath, "w") as f:
        for key, value in wav_scp.items():
            f.write(f"{key} {value}\n")


def write_trans7time_list(fpath, trans7time_list):
    """写入 trans7time 格式文件

    trans7time_list 每个元素格式：
        (spk_id, start_time, end_time, text)

    写入时会去掉 text 中的换行符，保证一条记录占一行。

    Args:
        fpath: 输出文件路径
        trans7time_list: 记录列表
    """
    with open(fpath, "w") as fw:
        for spk_id, start_time, end_time, text in trans7time_list:
            text = text.replace("\n", "").replace("\r", "")
            fw.write(f"{spk_id} {start_time} {end_time} {text}\n")


def load_audio(input, ori_fs=None, obj_fs=None):
    """加载音频并统一为 torch.Tensor

    支持三种输入：
    1) str：音频文件路径（使用 torchaudio.load 读取）
    2) np.ndarray：波形数组
    3) torch.Tensor：波形张量

    处理逻辑：
    - 多通道音频会做平均，转换为单通道
    - 整型音频会转换为 float32 并除以 32768 做归一化
    - 支持按需重采样（ori_fs -> obj_fs）

    Args:
        input: 音频输入（路径/np.ndarray/torch.Tensor）
        ori_fs: 原始采样率（当 input 为数组/张量时使用）
        obj_fs: 目标采样率（需要重采样时指定）

    Returns:
        torch.Tensor: 形状为 [1, T] 的单通道波形
    """
    if isinstance(input, str):
        wav, fs = torchaudio.load(input)
        # 多通道转单通道
        wav = wav.mean(dim=0, keepdim=True)
        # 按需重采样
        if obj_fs is not None and fs != obj_fs:
            wav = torchaudio.functional.resample(wav, orig_freq=fs, new_freq=obj_fs)
        return wav
    elif isinstance(input, np.ndarray) or isinstance(input, torch.Tensor):
        wav = torch.from_numpy(input) if isinstance(input, np.ndarray) else input
        # 整型音频归一化到 [-1, 1] 左右
        if wav.dtype in (torch.int16, torch.int32, torch.int64):
            wav = wav.type(torch.float32)
            wav = wav / 32768
        wav = wav.type(torch.float32)
        assert wav.ndim <= 2
        # 处理二维输入（可能是 [T, C] 或 [C, T]）
        if wav.ndim == 2:
            if wav.shape[0] > wav.shape[1]:
                wav = torch.transpose(wav, 0, 1)
            wav = wav.mean(dim=0, keepdim=True)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        # 按需重采样
        if ori_fs is not None and obj_fs is not None and ori_fs != obj_fs:
            wav = torchaudio.functional.resample(wav, orig_freq=ori_fs, new_freq=obj_fs)
        return wav
    else:
        return input
