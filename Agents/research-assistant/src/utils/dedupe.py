from typing import Iterable, Callable, List, Any
from .storage import SeenStore


def filter_new(items: Iterable[Any], key_fn: Callable[[Any], str], seen_path: str) -> List[Any]:
	store = SeenStore(seen_path)
	out = []
	new_ids = []
	for it in items:
		k = key_fn(it)
		if not store.contains(k):
			out.append(it)
			new_ids.append(k)
	if new_ids:
		store.add_all(new_ids)
	return out
