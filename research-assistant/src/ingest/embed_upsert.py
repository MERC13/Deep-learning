import os, time, json
from pathlib import Path
from dotenv import load_dotenv
import openai
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# --- OpenAI setup ---
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
EMBED_DIM = 3072

# --- Pinecone setup ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=os.getenv("PINECONE_ENVIRONMENT"))
    )

index = pc.Index(INDEX_NAME)

# --- Directories ---
CHUNKS_DIR = Path(os.getenv("CHUNKS_DIR", "data/chunks"))

# --- Functions ---
def embed_text(text: str):
    """Embed text using OpenAI embeddings API"""
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
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
    recs = load_chunks_from_txts()
    upsert_chunks(recs)
    print("Done.")
