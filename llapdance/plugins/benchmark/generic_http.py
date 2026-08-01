"""Reference benchmark adapter: a plain OpenAI-compatible chat-completions
prober measuring TTFT and throughput. Works against any backend that
satisfies `endpoint_contract: openai-compatible` (SPEC.md §4), which is why
this - not a specific tool - is the adapter guaranteed to work out of the box.
See llama_benchy.py for why that integration is a stub, not this one.
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from llapdance.core.result import BenchmarkResult
from llapdance.plugins.base import BenchmarkAdapter
from llapdance.plugins.registry import register


class GenericHttpBenchmark(BenchmarkAdapter):
    name = "generic-http"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._default_config = config or {}

    def run(self, endpoint: str, config: dict[str, Any]) -> BenchmarkResult:
        cfg = {**self._default_config, **config}
        prompt = cfg.get("prompt", "Explain what a hash table is in two sentences.")
        model = cfg.get("model", "default")
        max_tokens = int(cfg.get("max_tokens", 128))
        num_requests = int(cfg.get("num_requests", 3))
        timeout = float(cfg.get("timeout_s", 60))

        url = endpoint.rstrip("/") + "/v1/chat/completions"
        ttfts_ms: list[float] = []
        totals_ms: list[float] = []
        tokens_per_sec: list[float] = []
        raw_responses: list[dict[str, Any]] = []

        with httpx.Client(timeout=timeout) as client:
            for _ in range(num_requests):
                start = time.perf_counter()
                first_byte_at: float | None = None
                completion_tokens = 0
                with client.stream(
                    "POST",
                    url,
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "stream": True,
                    },
                ) as resp:
                    resp.raise_for_status()
                    for chunk in resp.iter_lines():
                        if not chunk:
                            continue
                        if first_byte_at is None:
                            first_byte_at = time.perf_counter()
                        completion_tokens += 1  # coarse: one SSE line ~= one token event
                end = time.perf_counter()

                ttft = (first_byte_at or end) - start
                total = end - start
                ttfts_ms.append(ttft * 1000)
                totals_ms.append(total * 1000)
                if total > 0 and completion_tokens > 0:
                    tokens_per_sec.append(completion_tokens / total)
                raw_responses.append({"ttft_s": ttft, "total_s": total, "tokens": completion_tokens})

        def avg(xs: list[float]) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        return BenchmarkResult(
            adapter=self.name,
            metrics={
                "avg_ttft_ms": avg(ttfts_ms),
                "avg_total_ms": avg(totals_ms),
                "avg_tokens_per_sec": avg(tokens_per_sec),
                "requests": float(num_requests),
            },
            raw={"requests": raw_responses},
        )


register("benchmark", GenericHttpBenchmark.name, GenericHttpBenchmark)
