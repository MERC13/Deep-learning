import os, time
from pathlib import Path
from dotenv import load_dotenv
from utils.dedupe import filter_new
from utils.logging import configure_logging, get_logger

load_dotenv()
configure_logging()
log = get_logger(__name__)

# --- Constants / Config ---
# Single embedding model (Sentence-Transformers)
SBERT_MODEL_DEFAULT = os.getenv("SBERT_MODEL", "all-MiniLM-L6-v2")  # ~384 dims

# --- Directories ---
CHUNKS_DIR = Path(os.getenv("CHUNKS_DIR", "data/chunks"))
SEEN_PATH = Path(os.getenv("SEEN_PATH", "data/state/embedded.jsonl"))

_SBERT_MODEL_CACHE = None


def _get_sbert_model():
    """Load a Sentence-Transformers model lazily and cache it."""
    global _SBERT_MODEL_CACHE
    if _SBERT_MODEL_CACHE is not None:
        return _SBERT_MODEL_CACHE
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise ImportError("sentence-transformers is required for EMBED_BACKEND=sbert. Install with 'pip install sentence-transformers'.") from e
    model_name = SBERT_MODEL_DEFAULT
    model = SentenceTransformer(model_name)
    _SBERT_MODEL_CACHE = model
    return model


def _embedding_dimension() -> int:
    """Return embedding dimension for the configured SBERT model."""
    model = _get_sbert_model()
    try:
        return int(model.get_sentence_embedding_dimension())
    except Exception:
        vec = model.encode(["dim probe"], normalize_embeddings=False)[0]
        return len(vec)


def _get_pinecone_index(dimension: int):
    """Return Pinecone index object, creating the index if needed with provided dimension."""
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
        log.info("Creating Pinecone index %s (dim=%d)", index_name, dimension)
        pc.create_index(
            index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=region or "us-east-1"),
        )
        # small delay to ensure index is ready
        time.sleep(2)
    log.info("Using Pinecone index: %s", index_name)
    return pc.Index(index_name)


# --- Functions ---
def embed_text(text: str):
    """Embed text using Sentence-Transformers only."""
    sbert = _get_sbert_model()
    vec = sbert.encode([text], normalize_embeddings=False)[0]
    return vec.tolist() if hasattr(vec, "tolist") else list(vec)

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
    # Ensure index has the proper dimension for the SBERT model
    dim = _embedding_dimension()
    index = _get_pinecone_index(dimension=dim)
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
            log.info("Upserted batch at i=%d, batch_size=%d", i, len(vectors))
            vectors = []
            time.sleep(0.5)
    if vectors:
        index.upsert(vectors)
        log.info("Upserted final batch, size=%d", len(vectors))

# --- Entry point ---
if __name__ == "__main__":
    try:
        recs = load_chunks_from_txts()
        # Avoid re-embedding already processed IDs
        recs = filter_new(recs, key_fn=lambda r: r["id"], seen_path=str(SEEN_PATH))
        upsert_chunks(recs)
        log.info("Done embedding+upsert.")
    except Exception as e:
        log.exception("Error in embed_upsert: %s", e)
        raise
