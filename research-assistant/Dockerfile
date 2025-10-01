# syntax=docker/dockerfile:1

FROM python:3.11-slim

# Install minimal OS dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Configure huggingface/transformers cache to a project-local path
ENV HF_HOME=/app/.cache \
    TRANSFORMERS_CACHE=/app/.cache

# Copy and install Python deps first for better layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && mkdir -p /app/.cache \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('SBERT model cached to /app/.cache')"

# Copy source
COPY src ./src

# Create a non-root user and data directory
RUN useradd -m appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /app /data

USER appuser

# Default envs; override via ECS task definition or docker run -e
ENV PYTHONPATH=/app/src \
    CORPUS_DIR=/data/raw_pdfs \
    CHUNKS_DIR=/data/chunks \
    SEEN_RSS_PATH=/data/state/arxiv_seen.jsonl \
    SEEN_PATH=/data/state/embedded.jsonl \
    DIGEST_USE_LLM=false \
    DRY_RUN=true

# Run the full weekly pipeline by default
CMD ["python", "-m", "pipeline.run_all"]
