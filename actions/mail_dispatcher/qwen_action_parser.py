from __future__ import annotations
import re
from typing import Iterable
from actions.mail_dispatcher.schema import ActionItem

_PAT = re.compile(r"^(?P<owner>[^，,：:\s]{1,12}).{0,6}(?P<task>.+)$")

def parse_action_items(lines: Iterable[str]) -> list[ActionItem]:
    items: list[ActionItem] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        s = re.sub(r"^\s*(?:[-•]\s+|\d+[.)、]\s+)", "", s).strip()
        m = _PAT.match(s)
        if m:
            owner = m.group("owner").strip()
            task = s[len(owner):].strip()
            items.append(ActionItem(owner=owner, task=task if task else s))
        else:
            items.append(ActionItem(owner="UNKNOWN", task=s))
    return items