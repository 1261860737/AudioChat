from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage

from actions.mail_dispatcher.schema import EmailDraft


def send_email_smtp(draft: EmailDraft) -> None:
    """
    真发送邮件：依赖环境变量
    - SMTP_HOST
    - SMTP_PORT (default 587)
    - SMTP_USER
    - SMTP_PASS
    - SMTP_TLS (default 1)
    - MAIL_FROM (default SMTP_USER)
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

    recipients = [draft.to, *draft.cc, *draft.bcc]

    with smtplib.SMTP(host, port, timeout=30) as server:
        if use_tls:
            server.starttls()
        server.login(user, pwd)
        server.send_message(msg, from_addr=mail_from, to_addrs=recipients)