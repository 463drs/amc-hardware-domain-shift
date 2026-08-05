import time
import urllib.parse
import urllib.request
from typing import Optional

def send_telegram_notification(bot_token: Optional[str], chat_id: Optional[str], text: str) -> None:
    """Sending message via telegram bot to the given chat."""
    if not bot_token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
    }).encode("utf-8")

    try:
        req = urllib.request.Request(url, data=payload)
        with urllib.request.urlopen(req, timeout=5):
            pass
    except Exception as e:
        print(f"[Telegram Warning]: {e}")