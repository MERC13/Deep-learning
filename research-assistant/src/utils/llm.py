import os
import json
from typing import Any, Dict, List, Optional

import requests

from .logging import get_logger

log = get_logger(__name__)


class LLMClient:
    """Minimal HTTP-based LLM client (OpenAI-compatible chat API).

    Env vars:
      - OPENAI_API_KEY: API key (required for provider=openai)
      - OPENAI_BASE_URL: Optional; defaults to https://api.openai.com/v1
      - OPENAI_MODEL: Chat model (default: gpt-4o-mini)
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: Optional[str] = None,
                 timeout: float = 60.0):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for LLM summarization")

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 512, temperature: float = 0.2) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=self.timeout)
        if resp.status_code != 200:
            log.error("LLM chat failed: status=%s body=%s", resp.status_code, resp.text[:500])
            raise RuntimeError(f"LLM chat failed: {resp.status_code}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            log.error("Unexpected LLM response: %s", data)
            raise RuntimeError("Unexpected LLM response shape")


def try_build_client() -> Optional[LLMClient]:
    try:
        return LLMClient()
    except Exception as e:
        log.debug("LLM client not available: %s", e)
        return None
