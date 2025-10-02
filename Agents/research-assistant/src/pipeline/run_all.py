import os
import time
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from utils.logging import configure_logging, get_logger

load_dotenv()
configure_logging()
log = get_logger(__name__)

# Local imports
from scrape.arxiv_scraper import run_once as scrape_once
from parsing.parse_pdf import extract_text_with_pymupdf, save_raw_text
from parsing.chunker import write_chunks_to_files
from ingest.embed_upsert import load_chunks_from_txts, upsert_chunks
from digest.compute_digest import build_digest_html, build_digest_text, build_summarized_digest
from digest.emailer import send_email


def retrieve_top_k_from_pinecone(query: str, top_k: int = 5) -> List[Tuple[str, str]]:
    """Query Pinecone using an embedding of `query` and return [(id, text_preview)].

    Falls back to reading local chunks if Pinecone config is missing.
    """
    try:
        from ingest.embed_upsert import embed_text, _get_pinecone_index, _embedding_dimension
    except Exception:
        # Pinecone not available; return local chunks
        items = [(r["id"], r["text"]) for r in load_chunks_from_txts()[:top_k]]
        return items

    try:
        dim = _embedding_dimension()
        index = _get_pinecone_index(dimension=dim)
        qvec = embed_text(query)
        res = index.query(vector=qvec, top_k=top_k, include_metadata=True)
        out = []
        for m in res.matches:
            cid = m.id
            text = (m.metadata or {}).get("text") or ""
            # If text not stored in metadata, best-effort: derive path and read
            src = (m.metadata or {}).get("source_file")
            if not text and src and Path(src).exists():
                try:
                    text = Path(src).read_text(encoding="utf-8")
                except Exception:
                    text = ""
            out.append((cid, text))
        return out
    except Exception:
        # On any error, gracefully fallback to local chunks
        items = [(r["id"], r["text"]) for r in load_chunks_from_txts()[:top_k]]
        return items


def run_pipeline():
    corpus_dir = Path(os.getenv("CORPUS_DIR", "data/raw_pdfs"))
    chunks_dir = Path(os.getenv("CHUNKS_DIR", "data/chunks"))
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # 1) Scrape PDFs (arXiv RSS)
    log.info("Step 1/6: scraping PDFs...")
    try:
        scrape_once()
    except Exception as e:
        log.warning("scrape step failed: %s", e)

    # 2) Parse PDFs to text
    log.info("Step 2/6: parsing PDFs to text...")
    for pdf in corpus_dir.glob("*.pdf"):
        try:
            txt_path = save_raw_text(pdf, extract_text_with_pymupdf(str(pdf)))
            # 3) Chunk to files
            write_chunks_to_files(txt_path, out_dir=chunks_dir)
        except Exception as e:
            log.warning("parse/chunk failed for %s: %s", pdf, e)
        time.sleep(0.1)

    # 4) Upsert to Pinecone
    log.info("Step 4/6: embedding and upserting chunks...")
    recs = load_chunks_from_txts()
    if recs:
        try:
            upsert_chunks(recs)
        except Exception as e:
            log.warning("upsert step failed: %s", e)
    else:
        log.info("No chunks found to upsert.")

    # 5) Build summarized digest (overall + per-paper)
    log.info("Step 5/6: building summarized digest...")
    query = os.getenv("DIGEST_QUERY", "latest research")
    try:
        html, text = build_summarized_digest(
            query=query,
            top_k=int(os.getenv("DIGEST_LIMIT", "5")),
            per_paper_sentences=int(os.getenv("DIGEST_PER_PAPER_SENTENCES", "3")),
            overall_sentences=int(os.getenv("DIGEST_OVERALL_SENTENCES", "6")),
        )
    except Exception as e:
        log.warning("summarized digest failed (%s); falling back to simple digest", e)
        top = retrieve_top_k_from_pinecone(query, top_k=int(os.getenv("DIGEST_LIMIT", "5")))
        html = build_digest_html([(i, t) for i, t in top])
        text = build_digest_text([(i, t) for i, t in top])

    # 6) Send email
    log.info("Step 6/6: sending email...")
    subject = os.getenv("DIGEST_SUBJECT", "Research Assistant Digest")
    to_env = os.getenv("TO_EMAILS") or os.getenv("TO_EMAIL")
    send_email(subject=subject, html_content=html, plain_text=text, to=to_env)
    log.info("Pipeline complete.")


if __name__ == "__main__":
    run_pipeline()
