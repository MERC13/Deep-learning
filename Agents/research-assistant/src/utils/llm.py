import os
import json
from typing import Dict, List, Optional

import requests

from .logging import get_logger

log = get_logger(__name__)


class LLMClient:
    """Minimal client for Ollama's chat API.

    Env vars:
      - OLLAMA_BASE_URL: base URL to the Ollama server (default: http://localhost:11434)
      - OLLAMA_MODEL: chat model to use
    """

    def __init__(self,
                 base_url: Optional[str] = None,
                 model: Optional[str] = None,
                 timeout: float = 120.0):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "gpt-oss")
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 512, temperature: float = 0.2) -> str:
        """Send chat messages to Ollama using /api/chat with stream=false.

        Notes:
        - Ollama's token control is `num_predict`.
        - Options are under the `options` field.
        """
        url = self.base_url + "/api/chat"
        headers = {"Content-Type": "application/json"}
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens) if max_tokens else -1,
            },
        }
        resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=self.timeout)
        if resp.status_code != 200:
            log.error("Ollama chat failed: status=%s body=%s", resp.status_code, resp.text[:500])
            raise RuntimeError(f"Ollama chat failed: {resp.status_code}")
        data = resp.json()
        try:
            # For stream=false, Ollama returns a single JSON with `message` containing the assistant reply
            msg = data.get("message", {})
            content = (msg or {}).get("content", "")
            return (content or "").strip()
        except Exception:
            log.error("Unexpected Ollama response: %s", data)
            raise RuntimeError("Unexpected Ollama response shape")


def try_build_client() -> Optional[LLMClient]:
    try:
        return LLMClient()
    except Exception as e:
        log.debug("LLM client not available: %s", e)
        return None
