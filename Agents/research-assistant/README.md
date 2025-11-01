# Research Assistant

Ingest, parse, embed, and summarize research PDFs into an email-ready digest. The pipeline can run locally (console output) or as a container in a scheduled environment. Typical stack: local Ollama for summaries, Pinecone for vector retrieval, SendGrid for email delivery.

Large data artifacts live under `data/` and are git-ignored by default.

## Features

- PDF parsing and chunking (PyMuPDF)
- Embedding and vector-store upsert (sentence-transformers + Pinecone)
- Semantic retrieval and LLM-powered summaries
- Email digest delivery (SendGrid)

## Repo structure

- `src/`
	- `scrape/` – arXiv scraper (RSS)
	- `parsing/` – PDF parsing and chunking
	- `ingest/` – embedding and Pinecone upsert
	- `digest/` – digest building and sending
	- `utils/` – logging, LLM client, helpers
	- `tools/print_digest.py` – CLI to print digest to console
- `data/raw_pdfs/` – PDFs and parsed JSON metadata (ignored)
- `data/chunks/` – preprocessed text chunks (ignored)

## Quick start (local, PowerShell)

```powershell
# 1) Create and activate a virtual environment
python -m venv .venv; . .venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) Copy env sample and edit as needed
Copy-Item .env_sample .env

# 4) (Optional) Start Ollama and pull a model, e.g.
# ollama pull llama3.1:8b-instruct

# 5) Print a digest to console (no email)
$env:PYTHONPATH = 'src'
python -m tools.print_digest
```

## Running the full pipeline

From the repo root (PowerShell):

```powershell
cd Agents/research-assistant/src
python -m pipeline.run_all
```

## Containerization & scheduling (high level)

Build the container from the repo root and push to your registry, then run on ECS/Fargate or another scheduler. Example build (PowerShell):

```powershell
# from repo root
docker build -t research-assistant:local -f Agents/research-assistant/Dockerfile .
# tag & push to your registry as needed
```

When creating a task definition, provide required secrets (SendGrid/Pinecone keys) via Secrets Manager or your cloud provider's secret store. Consider mounting persistent storage if you need to retain downloads or outputs.

## Notes & troubleshooting

- The container defaults `DIGEST_USE_LLM=false` to avoid requiring Ollama for CI runs. Enable LLMs if you have an accessible Ollama endpoint.
- Ollama connection errors: verify `http://localhost:11434` is reachable and the model is present (`ollama pull <model>`).
- If digest output is empty: ensure raw PDFs and metadata exist in `data/raw_pdfs/`.

## Development

Run tests with pytest (from repo root or the project folder):

```powershell
pytest
```

Configurations (ruff, black, pytest) are in `pyproject.toml`.
