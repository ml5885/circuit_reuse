"""Send a short notification by email (SMTP) and/or ntfy.sh.

Configuration is read from the environment so the same script works on any pod:
    NOTIFY_EMAIL_TO     recipient address
    NOTIFY_SMTP_USER    SMTP login (for Gmail: your address; use an app password)
    NOTIFY_SMTP_PASS    SMTP password
    NOTIFY_SMTP_HOST    default smtp.gmail.com
    NOTIFY_SMTP_PORT    default 465 (SSL)
    NOTIFY_NTFY_TOPIC   optional ntfy.sh topic for push notifications

Usage: python notify.py SUBJECT [BODY_FILE]   (body from stdin if BODY_FILE is -)
"""
import os
import smtplib
import socket
import sys
import urllib.request
from email.message import EmailMessage


def send(subject: str, body: str) -> None:
    host = socket.gethostname()
    subject = f"[{host}] {subject}"
    to = os.environ.get("NOTIFY_EMAIL_TO")
    user = os.environ.get("NOTIFY_SMTP_USER")
    password = os.environ.get("NOTIFY_SMTP_PASS")
    if to and user and password:
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = user, to, subject
        msg.set_content(body)
        with smtplib.SMTP_SSL(os.environ.get("NOTIFY_SMTP_HOST", "smtp.gmail.com"),
                              int(os.environ.get("NOTIFY_SMTP_PORT", "465")), timeout=30) as smtp:
            smtp.login(user, password)
            smtp.send_message(msg)
    topic = os.environ.get("NOTIFY_NTFY_TOPIC")
    if topic:
        req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=body.encode()[:4000],
                                     headers={"Title": subject.encode("ascii", "ignore").decode()})
        urllib.request.urlopen(req, timeout=30).read()
    if not (to and user and password) and not topic:
        print(f"[notify] no NOTIFY_* configured; would have sent: {subject}", file=sys.stderr)


if __name__ == "__main__":
    subject = sys.argv[1]
    body = ""
    if len(sys.argv) > 2:
        body = sys.stdin.read() if sys.argv[2] == "-" else open(sys.argv[2]).read()
    send(subject, body)
