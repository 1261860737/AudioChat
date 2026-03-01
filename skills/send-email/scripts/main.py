from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
import re


# 允许直接执行 `python skills/send-email/scripts/main.py`
# 把仓库根目录加入 sys.path，确保脚本内 import 可用
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("GITEA_DRY_RUN", "0")

from dispatch_to_gitea import dispatch_master_issue
from meeting_result_parser import get_beijing_date_str, parse_meeting_result_from_llm
import re


def _normalize_meeting_result(meeting_minutes: dict[str, Any]) -> dict[str, Any]:
    """Convert flexible meeting_minutes schema into dispatch_master_issue format."""
    title = str(meeting_minutes.get("title") or meeting_minutes.get("meeting_title") or "会议纪要").strip()
    date = str(meeting_minutes.get("date") or meeting_minutes.get("meeting_date") or "").strip()
    action_items = meeting_minutes.get("action_items") or []

    if isinstance(action_items, str):
        action_items = [action_items]
    elif isinstance(action_items, list):
        action_items = [str(item).strip() for item in action_items if str(item).strip()]
    else:
        action_items = []

    return {
        "meeting_info": {
            "title": title or "会议纪要",
            "date": date or None,
        },
        "action_items": action_items,
    }


def _default_owner_to_gitea(meeting_result: dict[str, Any]) -> dict[str, str]:
    """Build a minimal owner->gitea mapping from action lines (best-effort)."""
    mapping: dict[str, str] = {}
    for line in meeting_result.get("action_items") or []:
        if "：" in line:
            owner = line.split("：", 1)[0].strip()
        elif ":" in line:
            owner = line.split(":", 1)[0].strip()
        else:
            owner = ""
        if owner and owner not in mapping:
            mapping[owner] = owner
    return mapping


def _build_meeting_result_from_content(subject: str, content: str) -> dict[str, Any]:
    """Fallback: package subject/content into a meeting_result without action items."""
    return {
        "meeting_info": {
            "title": subject.strip() or "会议纪要",
            "date": None,
        },
        "action_items": [],
        "raw_content": content.strip(),
    }

def _extract_action_items_from_markdown(text: str | None) -> list[str]:
    """Extract action items from markdown text."""
    if not text:
        return []
    items: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^[-*]\s*\[(?: |x|X)\]\s+(.*)$", line)
        if m:
            item = m.group(1).strip()
            if item:
                items.append(item)
            continue
        m = re.match(r"^[-*]\s+(.*)$", line)
        if m:
            item = m.group(1).strip()
            if item:
                items.append(item)
    return items
def _extract_action_items_from_markdown(text: str) -> list[str]:
    """Best-effort extraction of action items from Markdown.
    Supports patterns like:
      - **spk0** - task...
      1. **spk0** - task...
      - spk0：task...
      - spk0: task...
    Returns normalized lines: 'owner：task'
    """
    if not isinstance(text, str):
        return []
    s = text.replace("\r\n", "\n")
    # Try to isolate the Action Items section
    sec_match = re.search(r"(##+\s*(行动项|Action Items)[\s\S]*?)(\n##+\s|\Z)", s, re.IGNORECASE)
    section = sec_match.group(1) if sec_match else s

    items: list[str] = []
    for line in section.split("\n"):
        line = line.strip()
        if not line:
            continue
        # remove list markers
        line = re.sub(r"^[-*]\s+", "", line)
        line = re.sub(r"^\d+\.\s+", "", line)

        # **owner** - task
        m = re.match(r"\*\*(.+?)\*\*\s*[-:：]\s*(.+)", line)
        if m:
            owner = m.group(1).strip()
            task = m.group(2).strip()
            if owner and task:
                items.append(f"{owner}：{task}")
            continue

        # owner: task / owner：task
        m = re.match(r"([^:：]{1,32})\s*[:：]\s*(.+)", line)
        if m:
            owner = m.group(1).strip()
            task = m.group(2).strip()
            if owner and task:
                items.append(f"{owner}：{task}")
            continue

    # de-dup
    dedup = []
    seen = set()
    for it in items:
        if it not in seen:
            seen.add(it)
            dedup.append(it)
    return dedup



def main(
    *,
    meeting_result: dict[str, Any] | None = None,
    owner_to_gitea: dict[str, str] | None = None,
    meeting_minutes: dict[str, Any] | str | None = None,
    llm_text: str | None = None,
    to: str | None = None,
    subject: str | None = None,
    body: str | None = None,
    content: str | None = None,
    attachments: list[str] | None = None,
    upload_to_gitea: bool | None = None,
) -> str:
    """Dispatch a master Gitea issue from meeting results."""
    if meeting_result is None and llm_text:
        try:
            meeting_result = parse_meeting_result_from_llm(
                llm_text,
                default_date=get_beijing_date_str(),
            )
        except ValueError:
            meeting_result = _build_meeting_result_from_content("会议纪要", llm_text)

    # If parser failed or produced no action items, try extracting from Markdown
    if isinstance(meeting_result, dict) and not (meeting_result.get("action_items") or []):
        extracted = _extract_action_items_from_markdown(llm_text)
        if extracted:
            meeting_result["action_items"] = extracted

    if meeting_result is None and meeting_minutes is not None:
        if isinstance(meeting_minutes, str):
            meeting_result = parse_meeting_result_from_llm(
                meeting_minutes,
                default_date=get_beijing_date_str(),
            )
        else:
            meeting_result = _normalize_meeting_result(meeting_minutes)

    if meeting_result is None and (subject or body or content):
        content_text = body or content or ""
        meeting_result = _build_meeting_result_from_content(subject or "", content_text)

    if meeting_result is None:
        raise ValueError("meeting_result or meeting_minutes or subject/content is required")

    owner_to_gitea = owner_to_gitea or _default_owner_to_gitea(meeting_result)
    print("[GITEA_ISSUE_INPUT] meeting_result:", meeting_result)
    print("[GITEA_ISSUE_INPUT] owner_to_gitea:", owner_to_gitea)
    url = dispatch_master_issue(meeting_result, owner_to_gitea)
    print("✅ 群发完成，Issue 链接：", url)
    return url


def execute(**kwargs: Any) -> str:
    return main(**kwargs)


def run(**kwargs: Any) -> str:
    return main(**kwargs)


if __name__ == "__main__":
    sample_meeting_result = {
        "meeting_info": {
            "title": "周会行动项（自动派发）",
            "date": "2026-01-29",
        },
        "action_items": [
            "张三：完成接口联调",
            "李四：整理会议纪要并同步",
            "王五：下周一前输出测试结论",
            "张三：跟进服务部署状态",
        ],
    }

    # 中文名 -> Gitea 用户名映射
    sample_owner_to_gitea = {
        "张三": "zhangsan",
        "李四": "lisi",
        "王五": "wangwu",
    }

    main(meeting_result=sample_meeting_result, owner_to_gitea=sample_owner_to_gitea)
