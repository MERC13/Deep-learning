# research-assistant

WIP: data ingestion, parsing, and utilities for research PDFs. Contains large data artifacts under `data/` which are git-ignored by default.

## Structure

- `src/`: code modules for ingest, parse, digest, and utils
- `data/raw_pdfs`: PDFs and parsed JSON (ignored)
- `data/chunks`: preprocessed text chunks (ignored)

## Setup

```powershell
python -m venv .venv; . .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Populate `data/raw_pdfs` as needed.
