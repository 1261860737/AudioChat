# mail_dispatcher

根据会议 action items 按负责人邮箱生成邮件草稿（默认不发送）。

## Usage (Python)

```python
from actions.mail_dispatcher.schema import ActionItem, Participant
from actions.mail_dispatcher.dispatcher import plan_email_drafts

result = plan_email_drafts(
  meeting_title="周例会",
  meeting_date="2026-01-28",
  action_items=[
    ActionItem(owner="Alice", owner_email="alice@xx.com", task="整理接口文档", due="2026-01-30", priority="P1"),
    ActionItem(owner="Bob", task="修复服务端报错"),  # 可通过 participants 映射补齐邮箱
  ],
  participants=[Participant(name="Bob", email="bob@xx.com")],
)

print(result.unresolved_owners)
for d in result.drafts:
  print(d.to, d.subject)