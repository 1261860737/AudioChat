from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage

from schema import EmailDraft


def send_email_smtp(draft: EmailDraft) -> None:
    """
    真发送邮件：依赖环境变量
    - SMTP_HOST
    - SMTP_PORT (default 587)
    - SMTP_USER
    - SMTP_PASS
    - SMTP_TLS (default 1)
    - MAIL_FROM (default SMTP_USER)

    说明：
    - 本仓库的默认策略是“生成草稿，不自动发送”，因此该函数通常由你在确认无误后显式调用。
    - `draft.cc`/`draft.bcc` 都会参与最终收件人列表（`to_addrs`），但 `Bcc` 不会写入邮件头。
    """
    host = os.environ.get("SMTP_HOST")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    pwd = os.environ.get("SMTP_PASS")
    use_tls = os.environ.get("SMTP_TLS", "1") == "1"
    mail_from = os.environ.get("MAIL_FROM") or user

    if not host or not user or not pwd or not mail_from:
        raise RuntimeError(
            "Missing SMTP env vars. Need SMTP_HOST/SMTP_USER/SMTP_PASS; MAIL_FROM optional."
        )

    msg = EmailMessage()
    msg["From"] = mail_from
    msg["To"] = draft.to
    if draft.cc:
        msg["Cc"] = ", ".join(draft.cc)
    msg["Subject"] = draft.subject
    msg.set_content(draft.body)

    # `send_message` 的 to_addrs 会决定实际投递对象（包括 bcc）
    recipients = [draft.to, *draft.cc, *draft.bcc]

    with smtplib.SMTP(host, port, timeout=30) as server:
        if use_tls:
            server.starttls()
        server.login(user, pwd)
        server.send_message(msg, from_addr=mail_from, to_addrs=recipients)
