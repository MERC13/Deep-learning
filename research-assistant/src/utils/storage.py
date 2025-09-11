import json
from pathlib import Path
from typing import Iterable, Set


class SeenStore:
	"""Tiny persistent set for IDs using JSONL file format."""

	def __init__(self, path: str | Path):
		self.path = Path(path)
		self.path.parent.mkdir(parents=True, exist_ok=True)
		self._seen: Set[str] = set()
		self._loaded = False

	def load(self):
		if self._loaded:
			return
		if self.path.exists():
			with open(self.path, "r", encoding="utf-8") as f:
				for line in f:
					try:
						obj = json.loads(line)
						val = obj.get("id") if isinstance(obj, dict) else str(obj).strip()
						if val:
							self._seen.add(val)
					except Exception:
						continue
		self._loaded = True

	def contains(self, key: str) -> bool:
		self.load()
		return key in self._seen

	def add_all(self, keys: Iterable[str]):
		self.load()
		with open(self.path, "a", encoding="utf-8") as f:
			for k in keys:
				k = str(k)
				if k and k not in self._seen:
					self._seen.add(k)
					f.write(json.dumps({"id": k}) + "\n")
