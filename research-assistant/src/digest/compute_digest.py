import os
import re
import html
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from dotenv import load_dotenv
from utils.logging import configure_logging, get_logger
from utils.llm import try_build_client

load_dotenv()
configure_logging()
log = get_logger(__name__)

CHUNKS_DIR = Path(os.getenv("CHUNKS_DIR", "data/chunks"))
RAW_PDFS_DIR = Path(os.getenv("RAW_PDFS_DIR", "data/raw_pdfs"))


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


def _read_json_metadata(stem: str) -> Optional[Dict]:
	"""Load metadata JSON for a given document stem from RAW_PDFS_DIR.

	Accepts stems like abcd1234 or abcd1234_chunk_003 (chunk suffix stripped).
	Returns a dict if found and parsed, else None.
	"""
	try:
		base = stem.split("_chunk_")[0] if "_chunk_" in stem else stem
		path = RAW_PDFS_DIR / f"{base}.json"
		if not path.exists():
			return None
		import json
		with path.open("r", encoding="utf-8") as f:
			data = json.load(f)
		return data if isinstance(data, dict) else None
	except Exception:
		return None


def _parse_authors(meta: Optional[Dict]) -> List[str]:
	if not meta:
		return []
	raw = meta.get("authors")
	if raw is None:
		return []
	authors: List[str] = []
	if isinstance(raw, str):
		authors = [a.strip() for a in raw.split(",") if a.strip()]
	elif isinstance(raw, list):
		tmp: List[str] = []
		for item in raw:
			if isinstance(item, str):
				tmp.append(item)
			elif isinstance(item, dict):
				name = item.get("name") or item.get("full_name") or item.get("author")
				if isinstance(name, str):
					tmp.append(name)
		if len(tmp) == 1 and "," in tmp[0]:
			authors = [a.strip() for a in tmp[0].split(",") if a.strip()]
		else:
			authors = [a.strip() for a in tmp if a and a.strip()]
	# Dedup preserve order
	seen = set(); out: List[str] = []
	for a in authors:
		if a and a not in seen:
			seen.add(a); out.append(a)
	return out


def _build_item_header(name: str, meta: Optional[Dict]) -> Tuple[str, str]:
	title = (meta or {}).get("title") if meta else None
	title = title.strip() if isinstance(title, str) else None
	title = title or name
	authors = _parse_authors(meta)
	published = (meta or {}).get("published") if meta else None
	byline_parts: List[str] = []
	if authors:
		byline_parts.append(", ".join(authors))
	if isinstance(published, str) and published.strip():
		byline_parts.append(published.strip())
	byline = " \u2022 ".join(byline_parts) if byline_parts else ""
	return title, byline


def _retrieve_top_k(query: str, top_k: int = 5) -> List[Tuple[str, str]]:
	"""Retrieve top_k (id, text) using Pinecone if configured; fallback to local chunks."""
	try:
		from ingest.embed_upsert import embed_text as _embed_text, _get_pinecone_index, _embedding_dimension
		dim = _embedding_dimension()
		index = _get_pinecone_index(dimension=dim)
		qvec = _embed_text(query)
		res = index.query(vector=qvec, top_k=top_k, include_metadata=True)
		out: List[Tuple[str, str]] = []
		for m in getattr(res, "matches", []) or []:
			cid = getattr(m, "id", "")
			mmeta = getattr(m, "metadata", None) or {}
			text = mmeta.get("text") or ""
			src = mmeta.get("source_file")
			if not text and src and Path(src).exists():
				try:
					text = Path(src).read_text(encoding="utf-8")
				except Exception:
					text = ""
			out.append((cid, text))
		if out:
			return out[:top_k]
	except Exception:
		pass

	# Fallbacks
	try:
		from ingest.embed_upsert import load_chunks_from_txts
		recs = load_chunks_from_txts()
		return [(r["id"], r["text"]) for r in recs[:top_k]]
	except Exception:
		return load_recent_chunks(limit=top_k)


def _sentence_split(text: str, max_sentences: int = 200) -> List[str]:
	if not text:
		return []
	t = re.sub(r"\s+", " ", text.strip())
	parts = re.split(r"(?<=[\.!?])\s+", t)
	sents: List[str] = []
	for p in parts:
		for seg in re.split(r"\n+", p):
			seg = seg.strip()
			if seg:
				sents.append(seg)
			if len(sents) >= max_sentences:
				return sents[:max_sentences]
	return sents[:max_sentences]


def _cosine_matrix(a, b):
	import numpy as np
	a = np.array(a, dtype=float)
	b = np.array(b, dtype=float)
	a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
	b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
	return a_norm @ b_norm.T


def _embed_sentences(sentences: Sequence[str]):
	try:
		from ingest.embed_upsert import _get_sbert_model
		model = _get_sbert_model()
		vecs = model.encode(list(sentences), normalize_embeddings=False)
		return vecs
	except Exception as e:
		raise ImportError("sentence-transformers is required for summarization. Install with 'pip install sentence-transformers'.") from e


def _select_summary_sentences(text: str, query: str, max_sentences: int = 3) -> List[str]:
	sents = _sentence_split(text, max_sentences=200)
	if not sents:
		return []
	sents = [s for s in sents if 40 <= len(s) <= 400] or sents
	candidates = sents[:80]
	qvec = _embed_sentences([query])[0]
	svecs = _embed_sentences(candidates)
	scores = _cosine_matrix(svecs, [qvec]).ravel()
	order = list(sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True))
	chosen: List[str] = []
	chosen_vecs = []
	import numpy as np
	for idx in order:
		sent = candidates[idx]
		vec = svecs[idx]
		if chosen_vecs:
			sims = _cosine_matrix([vec], chosen_vecs).ravel()
			if float(np.max(sims)) > 0.9:
				continue
		chosen.append(sent)
		chosen_vecs.append(vec)
		if len(chosen) >= max_sentences:
			break
	return chosen or candidates[:max_sentences]


def _overall_summary(per_paper_summaries: List[List[str]], query: str, max_sentences: int = 6) -> List[str]:
	pool = [s for group in per_paper_summaries for s in group]
	if not pool:
		return []
	seen = set(); unique: List[str] = []
	for s in pool:
		if s not in seen:
			seen.add(s); unique.append(s)
	qvec = _embed_sentences([query])[0]
	svecs = _embed_sentences(unique)
	scores = _cosine_matrix(svecs, [qvec]).ravel()
	order = list(sorted(range(len(unique)), key=lambda i: float(scores[i]), reverse=True))
	summary: List[str] = []
	summary_vecs = []
	import numpy as np
	for idx in order:
		sent = unique[idx]
		vec = svecs[idx]
		if summary_vecs:
			sims = _cosine_matrix([vec], summary_vecs).ravel()
			if float(np.max(sims)) > 0.9:
				continue
		summary.append(sent)
		summary_vecs.append(vec)
		if len(summary) >= max_sentences:
			break
	return summary


def build_digest_html(items: List[Tuple[str, str]]) -> str:
	parts = [
		"<html><body>",
		"<h2>Research Assistant Digest</h2>",
	]
	for name, text in items:
		meta = _read_json_metadata(name)
		title, byline = _build_item_header(name, meta)
		pdf_url = (meta or {}).get("pdf_url") if isinstance(meta, dict) else None
		preview_raw = (text[:800] + "…") if len(text) > 800 else text
		preview = html.escape(preview_raw).replace("\n", "<br/>")
		safe_title = html.escape(title)
		if isinstance(pdf_url, str) and pdf_url.startswith("http"):
			title_html = f"<a href=\"{html.escape(pdf_url)}\" target=\"_blank\">{safe_title}</a>"
		else:
			title_html = safe_title
		parts.append(f"<h3 style='margin-bottom:4px'>{title_html}</h3>")
		if byline:
			parts.append(f"<div style='color:#555;margin-bottom:8px'>{html.escape(byline)}</div>")
		parts.append("<div style='white-space:pre-wrap;font-family:inherit'>" + preview + "</div>")
		parts.append("<hr/>")
	parts.append("</body></html>")
	return "\n".join(parts)


def build_digest_text(items: List[Tuple[str, str]]) -> str:
	parts = ["Research Assistant Digest\n"]
	for name, text in items:
		meta = _read_json_metadata(name)
		title, byline = _build_item_header(name, meta)
		preview = (text[:800] + "…") if len(text) > 800 else text
		header = f"## {title}" if title else f"## {name}"
		if byline:
			header += f"\n{byline}"
		parts.append(f"\n{header}\n{preview}\n")
	return "\n".join(parts)


def build_digest(limit: int = 10):
	items = load_recent_chunks(limit=limit)
	return build_digest_html(items), build_digest_text(items)


def build_summarized_digest(
	query: str,
	top_k: int = 5,
	per_paper_sentences: int = 3,
	overall_sentences: int = 6,
) -> Tuple[str, str]:
	"""Build a digest with an overall summary and individual summaries for top-k relevant papers.

	Returns (html, text).
	"""
	# Retrieve potentially multiple chunks; group them by base document id
	raw_items = _retrieve_top_k(query=query, top_k=max(top_k * 2, top_k + 2))
	grouped: Dict[str, List[str]] = {}
	ordered_bases: List[str] = []
	for cid, text in raw_items:
		base = cid.split("_chunk_")[0] if "_chunk_" in cid else cid
		if base not in grouped:
			grouped[base] = []
			ordered_bases.append(base)
		if text:
			grouped[base].append(text)
		if len(ordered_bases) >= top_k and all(grouped[b] for b in ordered_bases):
			# We have top_k unique papers with some text each
			pass

	# Keep only top_k unique papers in original order
	selected_bases = ordered_bases[:top_k]

	per_items = []  # [(id, title, byline, meta, [bullets])]
	all_summaries: List[List[str]] = []

	# LLM client (optional)
	USE_LLM = (os.getenv("DIGEST_USE_LLM", "true").strip().lower() in {"1", "true", "yes", "on"})
	llm = try_build_client() if USE_LLM else None
	for base in selected_bases:
		# Merge text from multiple chunks (preserve order)
		combined_text = "\n\n".join(grouped.get(base, [])).strip()
		meta = _read_json_metadata(base)
		title, byline = _build_item_header(base, meta)
		# Build per-paper bullets
		summary_sents: List[str] = []
		if llm:
			try:
				prompt = [
					{"role": "system", "content": "You are an expert research assistant. Summarize papers succinctly as bullet points."},
					{"role": "user", "content": (
						"Query: " + query + "\n\n" +
						"Paper Title: " + (title or base) + "\n" +
						("Authors: " + byline.split(" • ")[0] + "\n" if byline else "") +
						"Instructions: Provide " + str(per_paper_sentences) + " concise bullet points capturing the key contributions and findings relevant to the query. Use '-' bullets, avoid fluff.\n\n" +
						"Paper Text (may be long, focus on the most relevant parts):\n" + combined_text[:12000]
					)}
				]
				resp = llm.chat(prompt, max_tokens=512, temperature=0.2)
				# Parse bullets (lines starting with '-' or '*')
				lines = [ln.strip() for ln in resp.splitlines()]
				summary_sents = [ln[1:].strip() if ln.startswith("-") or ln.startswith("*") else ln for ln in lines if ln and (ln.startswith("-") or ln.startswith("*"))]
				if not summary_sents:
					# fallback: take first non-empty lines as bullets
					summary_sents = [ln for ln in lines if ln][:per_paper_sentences]
				summary_sents = summary_sents[:per_paper_sentences]
			except Exception:
				summary_sents = []
		if not summary_sents:
			try:
				summary_sents = _select_summary_sentences(combined_text, query=query, max_sentences=per_paper_sentences)
			except Exception:
				summary_sents = _sentence_split(combined_text, max_sentences=per_paper_sentences)
		per_items.append((base, title, byline, meta or {}, summary_sents))
		all_summaries.append(summary_sents)

	# Build overall summary
	overall: List[str] = []
	if llm:
		try:
			flat_points = [s for group in all_summaries for s in group]
			joined = "\n- " + "\n- ".join(flat_points[:60]) if flat_points else ""
			prompt = [
				{"role": "system", "content": "You are an expert research analyst. Write a concise literature overview."},
				{"role": "user", "content": (
					f"Query: {query}\n\n" +
					f"Bullet Points from selected papers:\n{joined}\n\n" +
					f"Instructions: Synthesize an overall summary in {overall_sentences} bullets capturing themes, consensus, and notable results. Use '-' bullets."
				)}
			]
			resp = llm.chat(prompt, max_tokens=600, temperature=0.3)
			lines = [ln.strip() for ln in resp.splitlines()]
			overall = [ln[1:].strip() if ln.startswith("-") or ln.startswith("*") else ln for ln in lines if ln and (ln.startswith("-") or ln.startswith("*"))]
			if not overall:
				overall = [ln for ln in lines if ln][:overall_sentences]
			overall = overall[:overall_sentences]
		except Exception:
			overall = []
	if not overall:
		try:
			overall = _overall_summary(all_summaries, query=query, max_sentences=overall_sentences)
		except Exception:
			overall = [s for group in all_summaries for s in group][:overall_sentences]

	html_parts = [
		"<html><body>",
		"<h2>Research Assistant Digest</h2>",
		f"<div style='color:#555;margin-bottom:8px'>Query: {html.escape(query)}</div>",
		"<h3>Overall Summary</h3>",
	]
	if overall:
		html_parts.append("<ul>" + "".join([f"<li>{html.escape(s)}</li>" for s in overall]) + "</ul>")
	else:
		html_parts.append("<p>No overall summary available.</p>")

	html_parts.append("<hr/>")
	html_parts.append("<h3>Selected Papers</h3>")
	for name, title, byline, meta, summary_sents in per_items:
		pdf_url = meta.get("pdf_url") if isinstance(meta, dict) else None
		safe_title = html.escape(title)
		if isinstance(pdf_url, str) and pdf_url.startswith("http"):
			title_html = f"<a href=\"{html.escape(pdf_url)}\" target=\"_blank\">{safe_title}</a>"
		else:
			title_html = safe_title
		html_parts.append(f"<h4 style='margin-bottom:4px'>{title_html}</h4>")
		if byline:
			html_parts.append(f"<div style='color:#555;margin-bottom:6px'>{html.escape(byline)}</div>")
		if summary_sents:
			html_parts.append("<ul>" + "".join([f"<li>{html.escape(s)}</li>" for s in summary_sents]) + "</ul>")
		else:
			html_parts.append("<p><em>No summary available.</em></p>")
		html_parts.append("<hr/>")
	html_parts.append("</body></html>")
	html_out = "\n".join(html_parts)

	text_parts = [
		"Research Assistant Digest",
		f"Query: {query}",
		"",
		"## Overall Summary",
	]
	if overall:
		text_parts.extend([f"- {s}" for s in overall])
	else:
		text_parts.append("(No overall summary available)")
	text_parts.extend(["", "## Selected Papers"])
	for name, title, byline, meta, summary_sents in per_items:
		text_parts.append(f"\n### {title}")
		if byline:
			text_parts.append(byline)
		if summary_sents:
			text_parts.extend([f"- {s}" for s in summary_sents])
		else:
			text_parts.append("(No summary available)")
	text_out = "\n".join(text_parts) + "\n"

	return html_out, text_out


if __name__ == "__main__":
	html, text = build_digest(limit=int(os.getenv("DIGEST_LIMIT", "5")))
	log.info("Built digest with lengths: html=%d text=%d", len(html), len(text))
