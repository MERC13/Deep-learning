# Research Assistant

Ingest, parse, embed, and summarize research PDFs into an email-ready digest. Uses Ollama (local LLM) for summaries, Pinecone for vector retrieval, and SendGrid for email.

Large data artifacts live under `data/` and are git-ignored by default.

## Features

- PDF parsing and chunking (PyMuPDF)
- Embedding and vector store upsert (Sentence-Transformers + Pinecone)
- Retrieval by semantic query
- LLM-powered summaries via Ollama:
	- Overall literature overview
	- Per-paper bullet summaries
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

## Quick start

```powershell
# 1) Create and activate a virtual environment
python -m venv .venv; . .venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) Copy env sample and edit as needed
Copy-Item .env_sample .env

# 4) Start Ollama (https://ollama.com/) and pull a model, e.g.
#   ollama pull llama3.1:8b-instruct

# 5) Print a digest to console (no email)
$env:PYTHONPATH = 'src'
python -m tools.print_digest
```

## Running the pipeline

```powershell
cd src
python -m pipeline.run_all
```

## Containerize and run weekly on AWS (ECS Fargate)

This repo includes a Dockerfile for running the weekly assistant as a container. High-level steps:

1) Build and push the image

```powershell
# from repo root
docker build -t research-assistant:local .
# Optionally tag for ECR: <acct>.dkr.ecr.<region>.amazonaws.com/research-assistant:latest
```

2) Create an ECR repository and push the image (AWS Console or AWS CLI)

3) Create an ECS task definition (Fargate)
- CPU/memory: e.g., 1 vCPU / 2GB
- Command: default from image (python -m pipeline.run_all)
- Working dir: /app
- Environment variables: see `.env.sample` for required/optional vars
- Mount ephemeral storage for data: add an ephemeral volume mounted at `/data` (or leave as default 20GB)
- Networking: public subnet with outgoing internet access (NAT/IGW) so it can fetch PDFs and call Pinecone/SendGrid
- Secrets: store `SENDGRID_API_KEY` (and `PINECONE_API_KEY`) in AWS Secrets Manager and reference them in the task definition

4) Schedule weekly runs
- Use EventBridge (CloudWatch) schedule to trigger the ECS task (e.g., cron(0 13 ? * MON *))
- Target: your ECS cluster and the task definition revision
- Configure retry policy and dead-letter queue (optional)

Notes
- The image sets `PYTHONPATH=/app/src` and defaults `DIGEST_USE_LLM=false` to avoid needing an Ollama server. Enable LLM if you run an accessible Ollama endpoint.
- Outputs are ephemeral unless you back them up; consider mounting an EFS volume to `/data` for persistence across runs.
- Set `DRY_RUN=false` to actually send emails; with `true`, it logs only.


## Development

Formatting, linting, and tests:

```powershell
# Run tests
pytest
```

Configurations are in `pyproject.toml` (ruff, black, pytest) and `.editorconfig`.

## Troubleshooting

- Import warnings in editor: ensure your venv is activated and dependencies installed.
- Ollama connection errors: verify the server is running (`http://localhost:11434`) and the model exists (`ollama pull <model>`).
- No metadata in digest: ensure corresponding `<stem>.json` exists in `data/raw_pdfs/` for each chunk.
