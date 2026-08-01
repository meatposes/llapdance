"""Generic OpenAI-compatible client for the LLM-judge utility (SPEC.md §5).

Deliberately has no knowledge of any particular backend project - the base
URL, model name, and API key all come from config supplied by whoever
deploys the harness. Used only for coherence-judge fallback and cross-run
analysis summaries, never for orchestration mechanics.
"""
from __future__ import annotations

from typing import Any

import httpx


class OpenAICompatibleClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout_s: float = 60) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._timeout = timeout_s

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                f"{self._base_url}/v1/chat/completions",
                headers=self._headers,
                json={"model": self._model, "messages": messages, **kwargs},
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
