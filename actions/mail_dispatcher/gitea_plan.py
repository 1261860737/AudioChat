from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from actions.mail_dispatcher.schema import ActionItem


def normalize_owner(name: str) -> str:
    return name.strip()


def group_by_owner(items: Iterable[ActionItem]) -> dict[str, list[ActionItem]]:
    grouped: dict[str, list[ActionItem]] = defaultdict(list)
    for it in items:
        owner = normalize_owner(it.owner) if it.owner else "UNKNOWN"
        grouped[owner].append(it)
    return dict(grouped)


def render_master_issue_body(
    meeting_title: str,
    meeting_date: str | None,
    grouped: dict[str, list[ActionItem]],
    owner_to_gitea: dict[str, str] | None = None,
) -> str:
    """
    One master issue body:
    - per owner section
    - checklist tasks
    - @mention if mapping provided
    """
    owner_to_gitea = owner_to_gitea or {}

    lines: list[str] = []
    lines.append(f"## 会议行动项：{meeting_title}")
    if meeting_date:
        lines.append(f"- 日期：{meeting_date}")
    lines.append("")
    lines.append("> 本 Issue 由 meeting skill 自动生成；如需修正责任人或任务描述，请在评论中反馈。")
    lines.append("")

    # stable order: owner name sort
    for owner in sorted(grouped.keys()):
        guser = owner_to_gitea.get(owner)
        mention = f"@{guser}" if guser else ""
        header = f"### {owner} {mention}".strip()
        lines.append(header)

        for it in grouped[owner]:
            meta = []
            if it.priority:
                meta.append(f"优先级:{it.priority}")
            if it.due:
                meta.append(f"截止:{it.due}")
            meta_str = f"（{'，'.join(meta)}）" if meta else ""
            lines.append(f"- [ ] {it.task}{meta_str}")
            if it.context:
                lines.append(f"  - 背景：{it.context}")
        lines.append("")

    if "UNKNOWN" in grouped:
        lines.append("### 未识别责任人（请补充映射）")
        for it in grouped["UNKNOWN"]:
            lines.append(f"- [ ] {it.task}")
        lines.append("")

    return "\n".join(lines)