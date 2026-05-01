"""
Sistema de alertas: envia notificacoes por email
quando drift ou degradacao de performance sao detectados.
"""

import json
import logging
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

logger = logging.getLogger(__name__)

ALERTS_DIR = Path(__file__).resolve().parent.parent / "logs" / "alerts"
ALERTS_DIR.mkdir(parents=True, exist_ok=True)

EMAIL_CONFIG = {
    "enabled": False,
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender": "",
    "password": "",
    "recipient": "",
}


def send_alert(subject: str, message: str, level: str = "WARNING"):
    alert = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "subject": subject,
        "message": message,
    }

    alert_path = ALERTS_DIR / f"alert_{datetime.now():%Y-%m-%d_%H%M%S}.json"
    with open(alert_path, "w") as f:
        json.dump(alert, f, indent=2)

    logger.warning(f"ALERTA [{level}]: {subject}")

    if EMAIL_CONFIG["enabled"]:
        try:
            _send_email(subject, message)
            logger.info("Email de alerta enviado")
        except Exception as e:
            logger.error(f"Falha ao enviar email: {e}")


def _send_email(subject: str, body: str):
    msg = MIMEText(body)
    msg["Subject"] = f"[ML Trading Alert] {subject}"
    msg["From"] = EMAIL_CONFIG["sender"]
    msg["To"] = EMAIL_CONFIG["recipient"]

    with smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"]) as server:
        server.starttls()
        server.login(EMAIL_CONFIG["sender"], EMAIL_CONFIG["password"])
        server.send_message(msg)


def get_recent_alerts(n: int = 10) -> list[dict]:
    alerts = []
    alert_files = sorted(ALERTS_DIR.glob("alert_*.json"), reverse=True)

    for path in alert_files[:n]:
        with open(path) as f:
            alerts.append(json.load(f))

    return alerts
