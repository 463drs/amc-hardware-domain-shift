import html
import os
import time
import urllib.parse
import urllib.request
from typing import Optional, Tuple


def load_telegram_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Bot token + chat id from Kaggle secrets, else a .env file, else the environment.

    setdefault, not assignment: an already-exported value wins, so a caller can blank the pair
    to silence notifications. Missing dotenv is not an error -- notifications just no-op.
    """
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        secrets = UserSecretsClient()
        os.environ.setdefault("TELEGRAM_BOT_TOKEN", secrets.get_secret("TELEGRAM_BOT_TOKEN"))
        os.environ.setdefault("TELEGRAM_CHAT_ID", secrets.get_secret("TELEGRAM_CHAT_ID"))
    except Exception:
        try:
            from dotenv import load_dotenv
            load_dotenv()          # does not override already-set variables
        except ImportError:
            pass
    return os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")


def format_elapsed(seconds: float) -> str:
    """Seconds -> HH:MM:SS, for notification text."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def send_telegram_notification(bot_token: Optional[str], chat_id: Optional[str], text: str) -> None:
    """Sending message via telegram bot to the given chat."""
    if not bot_token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": chat_id,
        # Escaped because parse_mode is HTML: a "<" from an exception repr would otherwise make
        # Telegram reject the whole message, precisely when the message matters most.
        "text": html.escape(text),
        "parse_mode": "HTML",
    }).encode("utf-8")

    try:
        req = urllib.request.Request(url, data=payload)
        with urllib.request.urlopen(req, timeout=5):
            pass
    except Exception as e:
        print(f"[Telegram Warning]: {e}")