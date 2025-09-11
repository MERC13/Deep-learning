import os, time, json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Constants ---
EMBED_MODEL_DEFAULT = "text-embedding-3-large"
EMBED_DIM = 3072

# --- Directories ---
CHUNKS_DIR = Path(os.getenv("CHUNKS_DIR", "data/chunks"))


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
    """Load plain text files as single chunks (for testing)"""
    recs = []
    for p in CHUNKS_DIR.glob("*.txt"):
        base = p.stem
        text = p.read_text(encoding="utf-8")
        recs.append({
            "id": base + "_chunk_000",
            "text": text,
            "source_file": str(p),
            "chunk_idx": 0
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
        upsert_chunks(recs)
        print("Done.")
    except Exception as e:
        print("Error:", e)
        raise
