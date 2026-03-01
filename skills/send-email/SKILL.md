---
name: send-email
description: 生成会议行动项目电子邮件草稿，并可选择派发Gitea主问题。
---

# Send Email Skill (会议纪要 → 邮件草稿 / Gitea 主问题)


## 你能用它做什么

- 从会议纪要中提取 **Action Items**
- 按负责人分组，生成 **邮件草稿**
- （可选）在 **Gitea** 创建/更新一个用于跟踪的 **Master Issue**
- 支持 `GITEA_DRY_RUN=1` 进行无副作用验证

## 入口与调用

入口文件：`skills/send-email/scripts/main.py`  
暴露函数：`execute / main / run`

工具调用 JSON 结构（示例）：
```json
{"tool":"send-email","args":{ "...": "..." }}
```

---

## LLM 使用规范（必须遵守，否则会导致“前端不显示会议纪要 / skill 解析失败”）

### 1) 必须先生成“可直接展示”的会议纪要 Markdown

在触发工具调用之前，先生成一份完整会议纪要 Markdown（给用户展示），至少包含：

- 标题 / 日期 / 参与者（不确定可写“未提供”）
- 议题要点（分点）
- 结论（分点）
- **行动项 Action Items（必须有）**：建议格式 `负责人：事项（截止时间可选）`

### 2) 若需要调用本工具，args 必须携带“完整会议纪要”或“结构化结果”

**推荐（更稳）：直接传结构化 `meeting_result`**
```json
{
  "tool": "send-email",
  "args": {
    "meeting_result": {
      "meeting_info": {"title": "...", "date": "...", "attendees": ["..."]},
      "summary_points": ["..."],
      "decisions": ["..."],
      "action_items": ["张三：...", "李四：..."]
    }
  }
}
```

**兼容（但易错）：传 `llm_text`，但必须是“完整会议纪要 Markdown 原文”**
```json
{
  "tool": "send-email",
  "args": {
    "llm_text": "### 会议纪要\n...（完整 Markdown，包含行动项）"
  }
}
```

  禁止只传状态句，例如：
- “会议纪要已生成并上传至 Gitea。”
- “已完成派发。”

这会导致脚本无法解析行动项，从而出现“只发了一个空壳 Issue / 前端没有会议纪要正文”。

### 3) 工具执行完成后，最终回复必须再次输出会议纪要 Markdown + 链接

工具返回的链接（例如 Gitea issue URL）必须附在会议纪要之后。

---

## Gitea 环境变量（可选）

如果启用 Gitea 派发，请设置：

- `GITEA_BASE_URL`
- `GITEA_TOKEN`
- `GITEA_OWNER`
- `GITEA_REPO`

建议开发/调试时：
- `GITEA_DRY_RUN=1`（避免真实创建）

## 备注

- 本技能的 Gitea 派发逻辑位于 `skills/send-email/scripts/dispatch_to_gitea.py`（若存在）
- 若你希望“真正发邮件”，需要显式启用 SMTP 并配置对应 env（实现以你仓库为准）
