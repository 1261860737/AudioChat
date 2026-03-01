from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ActionItem:
    owner: str
    task: str
    owner_email: Optional[str] = None
    due: Optional[str] = None  # YYYY-MM-DD
    context: Optional[str] = None
    priority: Optional[str] = None  # P0/P1/P2


@dataclass(frozen=True)
class Participant:
    name: str
    email: str


@dataclass(frozen=True)
class EmailDraft:
    to: str
    subject: str
    body: str
    cc: tuple[str, ...] = ()
    bcc: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanResult:
    drafts: tuple[EmailDraft, ...]
    unresolved_owners: tuple[str, ...] = ()