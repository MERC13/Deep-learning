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

Populate `data/raw_pdfs` as needed or run the pipeline.

## End-to-end pipeline (dry-run)

```powershell
$env:PYTHONPATH = 'src'
$env:DRY_RUN = 'true'
$env:FROM_EMAIL = 'from@example.com'
$env:TO_EMAILS = 'to@example.com'
cd source
python -m pipeline.run_all
```

Notes:
- The pipeline uses simple de-duplication via JSONL state files under `data/state/` to avoid re-scraping and re-embedding the same items.
- To actually send email, set `DRY_RUN=false` and `SENDGRID_API_KEY`.
