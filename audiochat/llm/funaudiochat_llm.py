"""
LLM (大语言模型) 模块 - 文本输出模式

提供基于 Fun-Audio-Chat 的文本生成功能。
用于 S2T（Speech-to-Text）模式：根据 ASR 转写结果生成文本回复。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor

from funaudiochat.register import register_funaudiochat
from utils.constant import DEFAULT_S2T_PROMPT


@dataclass
class FunAudioChatTextResult:
    """LLM 文本生成结果。

    Attributes:
        text: 生成的文本内容
    """

    text: str


class FunAudioChatLLM:
    """Fun-Audio-Chat 文本生成模型封装类。

    用途：接收结构化的 ASR 结果和用户指令，生成文本回复。
    适用于 S2T（Speech-to-Text）场景，不需要生成语音。
    """

    def __init__(
        self,
        *,
        model_path: str,
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """初始化 Fun-Audio-Chat 模型。

        Args:
            model_path: 模型目录路径
            device: 计算设备，默认自动选择 cuda:0 或 cpu
            torch_dtype: 模型权重数据类型，默认 bfloat16
        """
        register_funaudiochat()
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.config = AutoConfig.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            config=self.config,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        # S2T 模式：只生成文本，禁用语音生成
        self.model.sp_gen_kwargs.update({"text_greedy": True, "disable_speech": True})

    def generate_text(
        self, *, instruction: str, system_prompt: str = DEFAULT_S2T_PROMPT
    ) -> FunAudioChatTextResult:
        """根据指令生成文本回复。

        Args:
            instruction: 用户指令（包含 ASR 转写结果）
            system_prompt: 系统提示词

        Returns:
            FunAudioChatTextResult: 包含生成文本的结果对象
        """
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.model.device)

        generate_ids, _ = self.model.generate(**inputs)
        generate_ids = generate_ids[:, inputs.input_ids.size(1) :]
        generate_text = self.processor.decode(generate_ids[0], skip_special_tokens=True)
        return FunAudioChatTextResult(text=generate_text)
