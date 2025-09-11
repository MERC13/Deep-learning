import os
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

CHUNKS_DIR = Path(os.getenv("CHUNKS_DIR", "data/chunks"))


def load_recent_chunks(limit: int = 10) -> List[Tuple[str, str]]:
	"""Load up to `limit` chunk .txt files and return list of (name, text).

	For now we treat each .txt in CHUNKS_DIR as one document.
	"""
	if not CHUNKS_DIR.exists():
		return []
	txts = sorted(CHUNKS_DIR.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
	items = []
	for p in txts[:limit]:
		try:
			items.append((p.stem, p.read_text(encoding="utf-8")))
		except Exception:
			continue
	return items


def build_digest_html(items: List[Tuple[str, str]]) -> str:
	parts = [
		"<html><body>",
		"<h2>Research Assistant Digest</h2>",
	]
	for name, text in items:
		preview = (text[:800] + "…") if len(text) > 800 else text
		parts.append(f"<h3>{name}</h3>")
		parts.append(f"<pre style='white-space:pre-wrap;font-family:inherit'>{preview}</pre>")
		parts.append("<hr/>")
	parts.append("</body></html>")
	return "\n".join(parts)


def build_digest_text(items: List[Tuple[str, str]]) -> str:
	parts = ["Research Assistant Digest\n"]
	for name, text in items:
		preview = (text[:800] + "…") if len(text) > 800 else text
		parts.append(f"\n## {name}\n{preview}\n")
	return "\n".join(parts)


def build_digest(limit: int = 10):
	items = load_recent_chunks(limit=limit)
	return build_digest_html(items), build_digest_text(items)


if __name__ == "__main__":
	html, text = build_digest(limit=int(os.getenv("DIGEST_LIMIT", "5")))
	print("Built digest with lengths:", len(html), len(text))
