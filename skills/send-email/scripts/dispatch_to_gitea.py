from __future__ import annotations
from typing import Any

from qwen_action_parser import parse_action_items
from gitea_plan import group_by_owner, render_master_issue_body
from gitea_sender import create_issue


def dispatch_master_issue(
    meeting_result: dict[str, Any],
    owner_to_gitea: dict[str, str],
) -> str:
    """把会议结果中的行动项汇总成一个“总 Issue”，并发到 Gitea。

    期望 `meeting_result` 结构（宽松取值，不强校验）：
    - `meeting_info.title`: 会议标题
    - `meeting_info.date`: 会议日期（可选）
    - `action_items`: 行动项文本行列表（每行类似 “张三：做xxx”）

    返回：
    - 新建 issue 的 URL（来自 Gitea 返回值）
    """
    meeting_title = meeting_result.get("meeting_info", {}).get("title") or "会议纪要"
    meeting_date = meeting_result.get("meeting_info", {}).get("date")
    action_lines = meeting_result.get("action_items") or []

    # 会议纪要里的文本行动项 -> 结构化 ActionItem
    action_items = parse_action_items(action_lines)
    grouped = group_by_owner(action_items)
    body = render_master_issue_body(meeting_title, meeting_date, grouped, owner_to_gitea=owner_to_gitea)

    title = f"[Action Items] {meeting_title}" + (f" {meeting_date}" if meeting_date else "")
    res = create_issue(title=title, body=body)

    return res.url
