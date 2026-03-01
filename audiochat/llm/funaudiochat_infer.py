"""
LLM (大语言模型) 模块 - S2S 推理模式

提供基于 Fun-Audio-Chat 的语音到语音（Speech-to-Speech）推理功能。
同时生成文本回复和语音响应，输出包含文本和音频文件路径。
"""

from __future__ import annotations

import os
import uuid

import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor
from utils.cosyvoice_detokenizer import get_audio_detokenizer, token2wav
from funaudiochat.register import register_funaudiochat
import numpy as np
import soundfile as sf
from utils.constant import (
    AUDIO_TEMPLATE,
    DEFAULT_S2M_GEN_KWARGS,
    DEFAULT_SP_GEN_KWARGS,
    DEFAULT_S2T_PROMPT,
    SPOKEN_S2M_PROMPT,
)

register_funaudiochat()


def infer_s2t_or_s2s(
    *,
    funaudiochat_model_path: str,
    audio_16k_mono: torch.Tensor,
    mode: str,
    instruction: str,
    device: str,
    output_dir: str,
) -> dict:
    """执行 Fun-Audio-Chat 推理（S2T 或 S2S 模式）。

    Args:
        funaudiochat_model_path: Fun-Audio-Chat 模型目录路径
        audio_16k_mono: 16kHz 单声道音频波形 [1, T]
        mode: 推理模式，"s2t"（文本输出）或 "s2s"（语音输出）
        instruction: 用户指令
        device: 计算设备
        output_dir: 输出目录（用于保存生成的音频文件）

    Returns:
        dict: 推理结果
            - mode: 模式类型
            - text: 生成的文本
            - audio_path: 生成的音频文件路径（仅 S2S 模式）
    """
    config = AutoConfig.from_pretrained(funaudiochat_model_path)
    processor = AutoProcessor.from_pretrained(funaudiochat_model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        funaudiochat_model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    if mode == "s2t":
        model.sp_gen_kwargs.update(
            {
                "text_greedy": True,
                "disable_speech": True,
            }
        )
        system_prompt = DEFAULT_S2T_PROMPT
    elif mode == "s2s":
        sp_gen_kwargs = DEFAULT_SP_GEN_KWARGS.copy()
        sp_gen_kwargs["text_greedy"] = True
        model.sp_gen_kwargs.update(sp_gen_kwargs)
        system_prompt = SPOKEN_S2M_PROMPT
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    audio_np = audio_16k_mono.squeeze(0).cpu().numpy()
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": AUDIO_TEMPLATE + "\n" + instruction},
    ]

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=text,
        audio=[audio_np],
        return_tensors="pt",
        return_token_type_ids=False,
    ).to(model.device)

    if mode == "s2t":
        generate_ids, _ = model.generate(**inputs)
        generate_ids = generate_ids[:, inputs.input_ids.size(1) :]
        generate_text = processor.decode(generate_ids[0], skip_special_tokens=True)
        return {
            "mode": "s2t",
            "text": generate_text,
        }

    gen_kwargs = DEFAULT_S2M_GEN_KWARGS.copy()
    gen_kwargs["max_new_tokens"] = 2048

    generate_ids, audio_ids = model.generate(**inputs, **gen_kwargs)
    generate_ids = generate_ids[:, inputs.input_ids.size(1) :]
    generate_text = processor.decode(generate_ids[0], skip_special_tokens=True)

    

    token_for_cosyvoice = [
        code
        for code in audio_ids[0].tolist()
        if isinstance(code, int) and 0 <= code < 6561
    ]

    os.makedirs(output_dir, exist_ok=True)
    output_uuid = str(uuid.uuid4())
    output_path = os.path.join(output_dir, f"output_audio_{output_uuid}.wav")

    cosyvoice_model = get_audio_detokenizer()
    speech = token2wav(
        cosyvoice_model,
        token_for_cosyvoice,
        embedding=None,
        token_hop_len=25 * 30,
        pre_lookahead_len=3,
    )



    audio_np_out = speech.squeeze(0).detach().cpu().numpy().astype(np.float32)
    sf.write(output_path, audio_np_out, cosyvoice_model.sample_rate)

    return {
        "mode": "s2s",
        "text": generate_text,
        "audio_path": output_path,
    }
