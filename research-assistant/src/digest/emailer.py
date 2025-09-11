import os
from typing import Iterable, List, Optional
from dotenv import load_dotenv
from utils.logging import configure_logging, get_logger

load_dotenv()
configure_logging()
log = get_logger(__name__)

FROM_EMAIL = os.getenv("FROM_EMAIL")
# Support TO_EMAILS (comma-separated) or legacy TO_EMAIL
TO_EMAILS_RAW = os.getenv("TO_EMAILS") or os.getenv("TO_EMAIL")
DRY_RUN_DEFAULT = (os.getenv("DRY_RUN", "false").strip().lower() in {"1", "true", "yes", "on"})


def _parse_recipients(value: Optional[object]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [e.strip() for e in str(value).split(",") if e.strip()]


def send_email(
    subject: str,
    html_content: str,
    to: Optional[Iterable[str]] = None,
    plain_text: Optional[str] = None,
    dry_run: Optional[bool] = None,
):
    """
    Send an email using SendGrid.

    Args:
        subject: Email subject.
        html_content: HTML body.
        to: Iterable of recipient emails or comma-separated string. Fallback to TO_EMAILS/TO_EMAIL env.
        plain_text: Optional plain-text alternative content.
        dry_run: If True, do not send; just log. Defaults to DRY_RUN env.
    """
    recipients = _parse_recipients(to if to is not None else TO_EMAILS_RAW)
    is_dry_run = DRY_RUN_DEFAULT if dry_run is None else dry_run

    if not FROM_EMAIL:
        raise ValueError("FROM_EMAIL is not set in environment")
    if not recipients:
        raise ValueError("No recipients provided. Set TO_EMAILS or pass 'to'.")

    if is_dry_run:
        log.info("[dry-run] Would send email from %s to %s with subject: %s", FROM_EMAIL, recipients, subject)
        return None

    SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
    if not SENDGRID_API_KEY:
        raise ValueError("SENDGRID_API_KEY is not set in environment")

    # Import sendgrid lazily to avoid requiring it for dry-run or other usage
    try:
        from sendgrid import SendGridAPIClient  # type: ignore
        from sendgrid.helpers.mail import Mail  # type: ignore
    except Exception as e:
        log.error("sendgrid import failed: %s", e)
        raise ImportError("sendgrid package is required to send email. Install with 'pip install sendgrid'.") from e

    message = Mail(
        from_email=FROM_EMAIL,
        to_emails=recipients,
        subject=subject,
        plain_text_content=plain_text,
        html_content=html_content,
    )
    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        log.info("email sent: status=%s", getattr(response, 'status_code', 'unknown'))
        return response
    except Exception as e:
        status = getattr(e, "status_code", None)
        body = getattr(e, "body", None)
        log.error("email error status=%s body=%s exc=%s", status, body, e)
        raise
