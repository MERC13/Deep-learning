# src/parsing/parse_pdf.py
import json, os
from pathlib import Path
import fitz  # PyMuPDF
from dotenv import load_dotenv
from utils.logging import configure_logging, get_logger

load_dotenv()
configure_logging()
log = get_logger(__name__)
CHUNKS_DIR = Path(os.getenv("CHUNKS_DIR", "data/chunks"))
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

def extract_text_with_pymupdf(pdf_path):
    log.info("Extracting text: %s", pdf_path)
    doc = fitz.open(pdf_path)
    pages = []
    for p in doc:
        text = p.get_text("text")
        pages.append(text)
    return "\n\n".join(pages)

def save_raw_text(pdf_path, text):
    name = Path(pdf_path).stem
    out = CHUNKS_DIR / f"{name}.txt"
    if out.exists():
        log.debug("Text already exists, skipping write: %s", out)
        return out
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    return out

if __name__ == "__main__":
    import sys
    pdf = sys.argv[1]
    txt = extract_text_with_pymupdf(pdf)
    outpath = save_raw_text(pdf, txt)
    log.info("wrote %s", outpath)
