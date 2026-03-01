from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]


def get_beijing_now() -> datetime:
    """Get current Beijing time (Asia/Shanghai)."""
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo("Asia/Shanghai"))
        except Exception:
            pass
    # Fallback when zoneinfo data is unavailable.
    return datetime.now(timezone(timedelta(hours=8)))


def get_beijing_date_str() -> str:
    return get_beijing_now().strftime("%Y-%m-%d")


def _normalize_participants(raw_participants: object) -> list[str]:
    if raw_participants is None:
        return []

    if isinstance(raw_participants, str):
        candidates = [p for p in re.split(r"[，,、;/\s]+", raw_participants) if p]
    elif isinstance(raw_participants, (list, tuple, set)):
        candidates = [str(p).strip() for p in raw_participants if str(p).strip()]
    else:
        return []

    unique: list[str] = []
    for name in candidates:
        if name and name not in unique:
            unique.append(name)
    return unique


def _normalize_action_items(raw_action_items: object) -> list[str]:
    items: list[str] = []

    def append_line(owner: str, task: str) -> None:
        owner = (owner or "").strip()
        task = (task or "").strip()
        if not task:
            return
        items.append(f"{owner}：{task}" if owner else task)

    if isinstance(raw_action_items, list):
        for item in raw_action_items:
            if isinstance(item, str):
                line = item.strip()
                if line:
                    items.append(line)
                continue
            if isinstance(item, dict):
                owner = str(item.get("owner") or item.get("assignee") or item.get("name") or "").strip()
                task = str(item.get("task") or item.get("todo") or item.get("content") or "").strip()
                if task:
                    append_line(owner, task)
    elif isinstance(raw_action_items, dict):
        for owner, tasks in raw_action_items.items():
            owner_str = str(owner).strip()
            if isinstance(tasks, str):
                append_line(owner_str, tasks)
                continue
            if isinstance(tasks, list):
                for task in tasks:
                    if isinstance(task, str):
                        append_line(owner_str, task)
    return items


def parse_meeting_result_from_llm(
    llm_text: str,
    *,
    default_date: str,
) -> dict[str, Any]:
    """Parse and normalize meeting_result JSON from full LLM text output."""
    text = (llm_text or "").strip()
    if not text:
        raise ValueError("LLM output is empty")

    candidates: list[str] = [text]
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    for block in fenced:
        block = block.strip()
        if block:
            candidates.append(block)
    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        candidates.append(obj_match.group(0).strip())

    parsed_obj: dict[str, Any] | None = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            parsed_obj = payload
            break
    if parsed_obj is None:
        raise ValueError("LLM output is not valid JSON object")

    meeting_info_raw = parsed_obj.get("meeting_info")
    meeting_info = meeting_info_raw if isinstance(meeting_info_raw, dict) else {}

    title = str(meeting_info.get("title") or parsed_obj.get("title") or "会议纪要").strip()
    date = str(meeting_info.get("date") or parsed_obj.get("date") or default_date).strip()
    participants_raw = meeting_info.get("participants")
    if participants_raw is None:
        participants_raw = parsed_obj.get("participants")
    participants = _normalize_participants(participants_raw)

    action_items = _normalize_action_items(parsed_obj.get("action_items"))
    if not action_items:
        raise ValueError("meeting_result.action_items is empty")

    return {
        "meeting_info": {
            "title": title or "会议纪要",
            "date": date or default_date,
            "participants": participants,
        },
        "action_items": action_items,
    }


def render_meeting_result_markdown(meeting_result: dict[str, Any]) -> str:
    """Render normalized meeting_result JSON into markdown text for TTS/UI."""
    meeting_info_raw = meeting_result.get("meeting_info")
    meeting_info = meeting_info_raw if isinstance(meeting_info_raw, dict) else {}

    title = str(meeting_info.get("title") or "会议纪要").strip()
    date = str(meeting_info.get("date") or "").strip()
    participants = _normalize_participants(meeting_info.get("participants"))
    action_items = _normalize_action_items(meeting_result.get("action_items"))

    participants_text = "、".join(participants) if participants else "（未识别）"

    lines: list[str] = [
        f"# {title}",
        "",
        f"- 会议时间：{date or '（未提供）'}",
        f"- 参会人员：{participants_text}",
        "",
        "## 行动项",
    ]

    if action_items:
        for idx, item in enumerate(action_items, start=1):
            lines.append(f"{idx}. {item}")
    else:
        lines.append("- 暂无行动项")

    return "\n".join(lines)
