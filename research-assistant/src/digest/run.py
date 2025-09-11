import os
from dotenv import load_dotenv
from .compute_digest import build_digest
from .emailer import send_email

load_dotenv()


def main():
    limit = int(os.getenv("DIGEST_LIMIT", "5"))
    subject = os.getenv("DIGEST_SUBJECT", "Research Assistant Digest")
    to_env = os.getenv("TO_EMAILS") or os.getenv("TO_EMAIL")
    html, text = build_digest(limit=limit)
    send_email(subject=subject, html_content=html, plain_text=text, to=to_env)


if __name__ == "__main__":
    main()
