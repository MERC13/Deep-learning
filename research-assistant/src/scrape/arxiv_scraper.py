import os, time, json, hashlib
from pathlib import Path
import feedparser
import requests
from urllib.parse import urljoin
from dotenv import load_dotenv

load_dotenv()
CORPUS_DIR = Path(os.getenv("CORPUS_DIR", "data/raw_pdfs"))
CORPUS_DIR.mkdir(parents=True, exist_ok=True)
ARXIV_RSS = "http://export.arxiv.org/rss/cs.CL"  # change category or rotate

def hash_text(s):
    return hashlib.sha1(s.encode()).hexdigest()

def fetch_rss_entries(rss_url):
    feed = feedparser.parse(rss_url)
    return feed.entries

def download_pdf(url, target_path):
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    with open(target_path, "wb") as f:
        for chunk in resp.iter_content(1024*16):
            f.write(chunk)

def normalize_arxiv_entry(entry):
    # Extract arXiv id from link or id field
    # entry.link often like https://arxiv.org/abs/2401.12345
    arxiv_id = entry.get("id") or entry.get("link")
    title = entry.get("title", "").strip()
    authors = [a.get("name") if isinstance(a, dict) else a for a in entry.get("authors", [])]
    summary = entry.get("summary", "")
    published = entry.get("published", "")
    pdf_link = None
    for l in entry.get("links", []):
        if l.get("type") == "application/pdf":
            pdf_link = l.get("href")
    if not pdf_link:
        pdf_link = entry.get("link", "").replace("/abs/", "/pdf/") + ".pdf"
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": authors,
        "summary": summary,
        "published": published,
        "pdf_url": pdf_link
    }

def save_metadata_and_pdf(meta):
    safe_name = hash_text(meta["pdf_url"])[:10]
    pdf_path = CORPUS_DIR / f"{safe_name}.pdf"
    meta_path = CORPUS_DIR / f"{safe_name}.json"
    if pdf_path.exists():
        print("already downloaded:", pdf_path)
        return meta_path, pdf_path
    try:
        download_pdf(meta["pdf_url"], pdf_path)
    except Exception as e:
        print("download error", e)
        return None, None
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta_path, pdf_path

def run_once():
    entries = fetch_rss_entries(ARXIV_RSS)
    for e in entries:
        meta = normalize_arxiv_entry(e)
        _, pdfp = save_metadata_and_pdf(meta)
        if pdfp:
            print("saved", pdfp)
        time.sleep(1)

if __name__ == "__main__":
    run_once()
