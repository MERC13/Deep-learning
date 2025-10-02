import os
from dotenv import load_dotenv
from .compute_digest import build_digest, build_summarized_digest
from .emailer import send_email

load_dotenv()


def main():
    limit = int(os.getenv("DIGEST_LIMIT", "5"))
    subject = os.getenv("DIGEST_SUBJECT", "Research Assistant Digest")
    to_env = os.getenv("TO_EMAILS") or os.getenv("TO_EMAIL")
    query = os.getenv("DIGEST_QUERY", "latest research")
    # Prefer summarized digest with overall + per-paper; fallback to simple if errors
    try:
        html, text = build_summarized_digest(
            query=query,
            top_k=limit,
            per_paper_sentences=int(os.getenv("DIGEST_PER_PAPER_SENTENCES", "3")),
            overall_sentences=int(os.getenv("DIGEST_OVERALL_SENTENCES", "6")),
        )
    except Exception:
        html, text = build_digest(limit=limit)
    send_email(subject=subject, html_content=html, plain_text=text, to=to_env)


if __name__ == "__main__":
    main()
