"""
提示词（Prompting）模块

提供 ASR 结果格式化和 LLM 指令构建功能。
将 Utterance 列表转换为易于 LLM 理解的格式。
"""

from __future__ import annotations

from dataclasses import dataclass

from audiochat.asr.funasr_asr import Utterance


def format_ms(ms: int) -> str:
    """将毫秒转换为 MM:SS.mmm 格式。"""
    if ms < 0:
        ms = 0
    seconds = ms / 1000.0
    minutes = int(seconds // 60)
    seconds = seconds - minutes * 60
    return f"{minutes:02d}:{seconds:06.3f}"


def format_utterances(utterances: list[Utterance], *, max_lines: int = 400) -> str:
    """将 Utterance 列表格式化为易读的文本。

    格式示例：
        [00:01.234-00:05.678] spk0: 你好
        [00:05.890-00:08.123] spk1: 大家好

    Args:
        utterances: Utterance 列表
        max_lines: 最大输出行数

    Returns:
        格式化后的文本字符串
    """
    lines: list[str] = []
    for u in utterances[:max_lines]:
        lines.append(
            f"[{format_ms(u.start_ms)}-{format_ms(u.end_ms)}] {u.speaker}: {u.text}".strip()
        )
    if len(utterances) > max_lines:
        lines.append(f"... (truncated, total_lines={len(utterances)})")
    return "\n".join(lines)


def build_llm_instruction(
    *, utterances: list[Utterance], user_instruction: str, max_lines: int = 400
) -> str:
    """构建发送给 LLM 的完整指令。

    将 ASR 转写结果与用户指令整合，生成 LLM 可理解的完整提示词。

    Args:
        utterances: ASR 识别出的 Utterance 列表
        user_instruction: 用户要求的任务描述
        max_lines: 最大保留的转写行数

    Returns:
        完整的 LLM 指令字符串
    """
    diar_text = format_utterances(utterances, max_lines=max_lines)
    return (
        "以下是对输入音频的分角色转写结果（speaker diarization + ASR，供你参考）：\n"
        f"{diar_text}\n\n"
        "请基于以上分角色转写结果完成用户要求。\n"
        f"用户要求：{user_instruction}"
    )
