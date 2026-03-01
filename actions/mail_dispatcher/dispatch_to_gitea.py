from __future__ import annotations
from typing import Any

from actions.mail_dispatcher.qwen_action_parser import parse_action_items
from actions.mail_dispatcher.gitea_plan import group_by_owner, render_master_issue_body
from actions.mail_dispatcher.gitea_sender import create_issue


def dispatch_master_issue(
    meeting_result: dict[str, Any],
    owner_to_gitea: dict[str, str],
) -> str:
    meeting_title = meeting_result.get("meeting_info", {}).get("title") or "会议纪要"
    meeting_date = meeting_result.get("meeting_info", {}).get("date")
    action_lines = meeting_result.get("action_items") or []

    action_items = parse_action_items(action_lines)
    grouped = group_by_owner(action_items)
    body = render_master_issue_body(meeting_title, meeting_date, grouped, owner_to_gitea=owner_to_gitea)

    title = f"[Action Items] {meeting_title}" + (f" {meeting_date}" if meeting_date else "")
    res = create_issue(title=title, body=body)

    return res.url