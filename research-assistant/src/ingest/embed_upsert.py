import os, time, json
from pathlib import Path
from dotenv import load_dotenv
from utils.dedupe import filter_new

load_dotenv()

# --- Constants ---
EMBED_MODEL_DEFAULT = "text-embedding-3-large"
EMBED_DIM = 3072

# --- Directories ---
CHUNKS_DIR = Path(os.getenv("CHUNKS_DIR", "data/chunks"))
SEEN_PATH = Path(os.getenv("SEEN_PATH", "data/state/embedded.jsonl"))


def _get_openai_client():
    """Create OpenAI client if API key is set; otherwise raise ValueError."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    try:
        import openai  # type: ignore
    except Exception as e:
        raise ImportError("openai package is required. Install with 'pip install openai'.") from e
    return openai.OpenAI(api_key=api_key)


def _get_pinecone_index():
    """Return Pinecone index object, creating the index if needed."""
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    region = os.getenv("PINECONE_ENVIRONMENT")
    if not api_key or not index_name:
        raise ValueError("PINECONE_API_KEY and PINECONE_INDEX_NAME must be set")
    try:
        from pinecone import Pinecone, ServerlessSpec  # type: ignore
    except Exception as e:
        raise ImportError("pinecone-client is required. Install with 'pip install pinecone-client'.") from e
    pc = Pinecone(api_key=api_key)
    existing = [idx["name"] for idx in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            index_name,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=region or "us-east-1"),
        )
        # small delay to ensure index is ready
        time.sleep(2)
    return pc.Index(index_name)


# --- Functions ---
def embed_text(text: str, model: str | None = None):
    """Embed text using OpenAI embeddings API."""
    client = _get_openai_client()
    embed_model = model or os.getenv("OPENAI_EMBED_MODEL", EMBED_MODEL_DEFAULT)
    resp = client.embeddings.create(model=embed_model, input=text)
    return resp.data[0].embedding

def load_chunks_from_txts():
    """Load chunk .txt files.

    If any files match *_chunk_*.txt, only those are loaded.
    Otherwise, every .txt is treated as a single chunk.
    """
    all_txts = list(CHUNKS_DIR.glob("*.txt"))
    chunk_txts = [p for p in all_txts if "_chunk_" in p.stem]
    target_files = chunk_txts if chunk_txts else all_txts

    recs = []
    for p in target_files:
        base = p.stem
        text = p.read_text(encoding="utf-8")
        if "_chunk_" in base:
            chunk_id = base
            try:
                chunk_idx = int(base.split("_chunk_")[-1])
            except Exception:
                chunk_idx = 0
        else:
            chunk_id = base + "_chunk_000"
            chunk_idx = 0
        recs.append({
            "id": chunk_id,
            "text": text,
            "source_file": str(p),
            "chunk_idx": chunk_idx
        })
    return recs

def upsert_chunks(recs, batch_size=10):
    index = _get_pinecone_index()
    vectors = []
    for i, r in enumerate(recs):
        emb = embed_text(r["text"])
        meta = {
            "source_file": r.get("source_file"),
            "chunk_idx": r.get("chunk_idx"),
            "source_id": r.get("id"),
            # Storing text can be helpful for quick retrieval (consider size/cost); comment out if undesired
            "text": r.get("text"),
        }
        vectors.append((r["id"], emb, meta))
        if len(vectors) >= batch_size:
            index.upsert(vectors)
            print("upserted batch", i)
            vectors = []
            time.sleep(0.5)
    if vectors:
        index.upsert(vectors)
        print("upserted final batch")

# --- Entry point ---
if __name__ == "__main__":
    try:
        recs = load_chunks_from_txts()
        # Avoid re-embedding already processed IDs
        recs = filter_new(recs, key_fn=lambda r: r["id"], seen_path=str(SEEN_PATH))
        upsert_chunks(recs)
        print("Done.")
    except Exception as e:
        print("Error:", e)
        raise
