from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from schema import ActionItem, EmailDraft, Participant, PlanResult


def _normalize_name(name: str) -> str:
    """规范化人名用于匹配（去首尾空格并转小写）。"""
    return name.strip().lower()


def build_name_to_email(participants: Iterable[Participant]) -> dict[str, str]:
    """构建 `{规范化姓名 -> 邮箱}` 的映射表。"""
    return {_normalize_name(p.name): p.email.strip() for p in participants}


def group_action_items_by_email(
    action_items: Iterable[ActionItem],
    name_to_email: dict[str, str],
) -> tuple[dict[str, list[ActionItem]], list[str]]:
    """把行动项按“最终收件邮箱”分组，并返回无法解析邮箱的责任人列表。

    解析邮箱的优先级：
    1) `ActionItem.owner_email`（行动项已明确给邮箱）
    2) `name_to_email[_normalize_name(ActionItem.owner)]`（用参会人名单补齐）

    返回：
    - grouped: `{email: [ActionItem, ...]}`
    - unresolved_unique: `[owner, ...]`（去重，保留首次出现顺序）
    """
    grouped: dict[str, list[ActionItem]] = defaultdict(list)
    unresolved: list[str] = []

    for it in action_items:
        email = (it.owner_email or "").strip()
        if not email:
            email = name_to_email.get(_normalize_name(it.owner), "")
        if not email:
            unresolved.append(it.owner)
            continue
        grouped[email].append(it)

    # 去重但保留可读顺序
    unresolved_unique = []
    seen = set()
    for o in unresolved:
        if o not in seen:
            unresolved_unique.append(o)
            seen.add(o)

    return dict(grouped), unresolved_unique


def render_email_draft(
    meeting_title: str,
    meeting_date: str | None,
    to_email: str,
    items: list[ActionItem],
    cc: tuple[str, ...] = (),
    bcc: tuple[str, ...] = (),
) -> EmailDraft:
    """把某个邮箱名下的一组行动项渲染成邮件草稿（主题 + 正文）。"""
    date_part = f" {meeting_date}" if meeting_date else ""
    subject = f"[行动项] {meeting_title}{date_part}"

    lines: list[str] = []
    lines.append(f"你好，\n\n这是会议《{meeting_title}》的待办汇总：")
    if meeting_date:
        lines.append(f"会议日期：{meeting_date}")
    lines.append("")
    for i, it in enumerate(items, 1):
        # meta 信息只在存在时展示，避免正文噪音
        meta = []
        if it.priority:
            meta.append(f"优先级: {it.priority}")
        if it.due:
            meta.append(f"截止: {it.due}")
        meta_str = f"（{'，'.join(meta)}）" if meta else ""
        lines.append(f"{i}. {it.task}{meta_str}")
        if it.context:
            lines.append(f"   - 背景：{it.context}")
    lines.append("\n如有理解偏差请直接回复修正。谢谢！")

    return EmailDraft(
        to=to_email,
        subject=subject,
        body="\n".join(lines),
        cc=cc,
        bcc=bcc,
    )


def plan_email_drafts(
    meeting_title: str,
    meeting_date: str | None,
    action_items: Iterable[ActionItem],
    participants: Iterable[Participant] = (),
    cc: tuple[str, ...] = (),
    bcc: tuple[str, ...] = (),
) -> PlanResult:
    """生成“按人发”的邮件草稿规划（默认只生成，不发送）。

    常见用法：
    - 上游从会议纪要里抽出行动项（ActionItem 列表）
    - 可选传入 participants（参会人名单）补齐邮箱
    - 返回 PlanResult：你可以检查 `unresolved_owners` 再决定是否发送
    """
    name_to_email = build_name_to_email(participants)
    grouped, unresolved = group_action_items_by_email(action_items, name_to_email)

    drafts: list[EmailDraft] = []
    for email, items in grouped.items():
        drafts.append(render_email_draft(meeting_title, meeting_date, email, items, cc=cc, bcc=bcc))

    return PlanResult(drafts=tuple(drafts), unresolved_owners=tuple(unresolved))
