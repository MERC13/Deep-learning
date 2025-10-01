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
