"""Reference benchmark adapter: a plain OpenAI-compatible chat-completions
prober measuring TTFT and throughput. Works against any backend that
satisfies `endpoint_contract: openai-compatible` (SPEC.md §4), which is why
this - not a specific tool - is the adapter guaranteed to work out of the box.
See llama_benchy.py for why that integration is a stub, not this one.
"""
from __future__ import annotations

import json
import time
from typing import Any

import httpx

from llapdance.core.result import BenchmarkResult
from llapdance.plugins.base import BenchmarkAdapter
from llapdance.plugins.registry import register


def _completion_token_count(chunks: list[dict[str, Any]], sse_line_count: int) -> tuple[int, str]:
    """Real bug, found validating Arcaine (see VALIDATION.md): counting one
    non-empty SSE line as one token assumes autoregressive one-token-per-
    chunk streaming. Arcaine's diffusion decoding emits the ENTIRE
    completion as a SINGLE chunk, so that heuristic undercounted its
    throughput by roughly 10x. Several engines embed an authoritative count
    somewhere in their chunks (not always the standard OpenAI `usage`
    field, which is commonly null during streaming unless a client asks
    for it) - prefer those, in order, before falling back to the crude
    per-line count.
    """
    for chunk in reversed(chunks):  # authoritative counts land on the final chunk
        usage = chunk.get("usage") or {}
        if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
            return int(usage["completion_tokens"]), "usage.completion_tokens"
        metrics = chunk.get("metrics") or {}
        if isinstance(metrics, dict) and metrics.get("new_token") is not None:
            return int(metrics["new_token"]), "metrics.new_token"  # Arcaine, OpenArc
        timings = chunk.get("timings") or {}
        if isinstance(timings, dict) and timings.get("predicted_n") is not None:
            return int(timings["predicted_n"]), "timings.predicted_n"  # llama.cpp
    return sse_line_count, "sse_line_count (fallback - no authoritative count found in any chunk)"


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
        headers = {"Authorization": f"Bearer {cfg['api_key']}"} if "api_key" in cfg else cfg.get("headers", {})

        url = endpoint.rstrip("/") + "/v1/chat/completions"
        ttfts_ms: list[float] = []
        totals_ms: list[float] = []
        tokens_per_sec: list[float] = []
        raw_responses: list[dict[str, Any]] = []

        with httpx.Client(timeout=timeout, headers=headers) as client:
            for _ in range(num_requests):
                start = time.perf_counter()
                first_byte_at: float | None = None
                sse_line_count = 0
                parsed_chunks: list[dict[str, Any]] = []
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
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        if first_byte_at is None:
                            first_byte_at = time.perf_counter()
                        sse_line_count += 1
                        payload = line[len("data: ") :] if line.startswith("data: ") else line
                        if payload.strip() == "[DONE]":
                            continue
                        try:
                            parsed_chunks.append(json.loads(payload))
                        except json.JSONDecodeError:
                            pass  # non-JSON line - fall back to sse_line_count for this request
                end = time.perf_counter()

                completion_tokens, counted_via = _completion_token_count(parsed_chunks, sse_line_count)

                ttft = (first_byte_at or end) - start
                total = end - start
                ttfts_ms.append(ttft * 1000)
                totals_ms.append(total * 1000)
                if total > 0 and completion_tokens > 0:
                    tokens_per_sec.append(completion_tokens / total)
                raw_responses.append(
                    {"ttft_s": ttft, "total_s": total, "tokens": completion_tokens, "counted_via": counted_via}
                )

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
