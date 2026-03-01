from __future__ import annotations
import re
from typing import Iterable
from schema import ActionItem

# 这是一个“弱约束”的启发式匹配：
# - 先抓取行首的“可能是责任人的短字符串”（最多 12 字符，且不包含常见分隔符/空白）
# - 再允许中间出现少量分隔（例如“：”“-”“说”“负责”等），最后把剩余当作 task
_PAT = re.compile(r"^(?P<owner>[^，,：:\s]{1,12}).{0,6}(?P<task>.+)$")

def parse_action_items(lines: Iterable[str]) -> list[ActionItem]:
    """把上游的“行动项文本行”解析成 `ActionItem` 列表。

    设计假设（尽量兼容会议纪要常见写法）：
    - 每行大致是“责任人 + 任务描述”，例如：
      - `Alice：整理接口文档`
      - `Bob 修复服务端报错`
    - 允许行首是列表样式（`-`/`•`/`1.`/`1、` 等），会先剥离

    解析失败时：
    - 责任人会被标记为 `UNKNOWN`，便于下游提示用户补齐/修正
    """
    items: list[ActionItem] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        # 去掉常见列表前缀，减少对上游格式的依赖
        s = re.sub(r"^\s*(?:[-•]\s+|\d+[.)、]\s+)", "", s).strip()
        m = _PAT.match(s)
        if m:
            owner = m.group("owner").strip()
            # 这里用“行首 owner 后的剩余”作为任务，避免 regex 对 task 的截断偏差
            task = s[len(owner):].strip()
            # 去掉 owner 后常见连接符，避免出现 `：完成xxx` 这类前导符号
            task = re.sub(r"^[：:，,\-—\s]+", "", task)
            items.append(ActionItem(owner=owner, task=task if task else s))
        else:
            items.append(ActionItem(owner="UNKNOWN", task=s))
    return items
