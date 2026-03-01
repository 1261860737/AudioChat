from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ActionItem:
    """会议行动项（Action Item）的标准结构。

    该结构的目标是把“会议纪要里一条待办”抽象成可被下游输出（邮件/Gitea）的数据：
    - `owner`: 责任人（通常是姓名/花名）
    - `task`: 待办内容（尽量是可执行、可检查的描述）
    - 其余字段为可选元信息，用于更好地生成邮件正文或 issue checklist。
    """

    owner: str
    task: str
    owner_email: Optional[str] = None
    due: Optional[str] = None  # YYYY-MM-DD
    context: Optional[str] = None
    priority: Optional[str] = None  # P0/P1/P2


@dataclass(frozen=True)
class Participant:
    """参会人信息：用于把“人名”映射到“邮箱”。

    在生成邮件草稿时，如果 `ActionItem.owner_email` 缺失，会用这里的映射来补齐。
    """

    name: str
    email: str


@dataclass(frozen=True)
class EmailDraft:
    """邮件草稿（不一定发送）。

    这个结构只描述“要发什么”，不负责“怎么发”（SMTP 配置等由 sender 模块处理）。
    """

    to: str
    subject: str
    body: str
    cc: tuple[str, ...] = ()
    bcc: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanResult:
    """规划结果：草稿 + 未能解析到邮箱的责任人列表。"""

    drafts: tuple[EmailDraft, ...]
    unresolved_owners: tuple[str, ...] = ()
