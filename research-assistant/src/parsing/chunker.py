import os, json
from pathlib import Path
import tiktoken
from dotenv import load_dotenv
from utils.logging import configure_logging, get_logger
load_dotenv()
configure_logging()
log = get_logger(__name__)

CHUNKS_DIR = Path(os.getenv("CHUNKS_DIR", "data/chunks"))
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
ENC = tiktoken.get_encoding("cl100k_base")

def chunk_text(text, max_tokens=700, overlap=100):
    tokens = ENC.encode(text)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = ENC.decode(chunk_tokens)
        chunks.append(chunk_text)
        i += max_tokens - overlap
    return chunks

def chunk_file(txt_path, out_prefix=None):
    text = Path(txt_path).read_text(encoding="utf-8")
    chunks = chunk_text(text)
    base = out_prefix or Path(txt_path).stem
    out_records = []
    for idx, c in enumerate(chunks):
        rec = {
            "id": f"{base}_chunk_{idx:03d}",
            "text": c,
            "source_file": str(txt_path),
            "chunk_idx": idx
        }
        out_records.append(rec)
    return out_records


def write_chunks_to_files(txt_path, out_dir: Path | None = None):
    """Chunk a .txt file and write each chunk as its own .txt file next to the input or in out_dir.

    Output files are named: <base>_chunk_XXX.txt
    """
    p = Path(txt_path)
    out_dir = Path(out_dir) if out_dir else p.parent
    recs = chunk_file(txt_path)
    log.info("Chunking file to chunks: %s", txt_path)
    for rec in recs:
        outp = out_dir / f"{rec['id']}.txt"
        if outp.exists():
            log.debug("Chunk exists, skipping: %s", outp)
            continue
        outp.write_text(rec["text"], encoding="utf-8")
        log.debug("Wrote chunk: %s", outp)
    return [out_dir / f"{rec['id']}.txt" for rec in recs]

if __name__ == "__main__":
    import sys
    txtfile = sys.argv[1]
    recs = chunk_file(txtfile)
    print(f"created {len(recs)} chunks")
