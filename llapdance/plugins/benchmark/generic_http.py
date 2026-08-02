"""Reference benchmark adapter: a plain OpenAI-compatible chat-completions
prober measuring TTFT and throughput. Works against any backend that
satisfies `endpoint_contract: openai-compatible` (SPEC.md §4), which is why
this - not a specific tool - is the adapter guaranteed to work out of the box.
See llama_benchy.py for why that integration is a stub, not this one.

PP/TG split (added after direct user feedback that this adapter only ever
reported one blended tokens/sec number, mixing prefill and decode - a real
gap confirmed by auditing every stored result: 33 of 34 had no PP/TG split
at all, only the one run that happened to use llama-benchy instead). Two
sources, preferred in order, both real - never a guess at what a backend
might report:

1. `timings.prompt_per_second` / `timings.predicted_per_second` - llama.cpp
   server's OWN authoritative split (confirmed via this project's local
   llama.cpp checkout, tools/server/server-task.cpp: `timings.prompt_n`,
   `timings.predicted_n`, `timings.prompt_per_second`,
   `timings.predicted_per_second` are all computed server-side and pushed
   onto the final chunk unconditionally, streaming or not - this adapter
   already reads `timings.predicted_n` from the same object for token
   counting, so this is the same chunk, just more of it). Most accurate:
   measured inside the server, excludes client/network overhead entirely.
2. `usage.prompt_tokens` + `usage.completion_tokens` (both present, e.g.
   qxmx - confirmed via its own source, tools/qxmx_serve.cpp) with no
   server-side timing split: derive PP from `prompt_tokens / ttft` and TG
   from `(completion_tokens - 1) / (total - ttft)`, the same TTFT-based
   approximation llama-bench-style tools use (prefill ends when the first
   token is emitted; decode covers every token after that one). Less
   accurate than (1) - includes client-observed network/queueing time -
   but real, not fabricated.

If neither is present (Arcaine, OpenArc, and vLLM were not confirmed to
expose prompt-token counts on this project's real captured responses),
PP/TG are left OUT of the metrics dict entirely - `avg_tokens_per_sec`
(blended) is still reported, same as before this change. No metric is ever
invented for a backend that didn't provide enough signal to compute it.
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


def _prompt_token_count(chunks: list[dict[str, Any]]) -> tuple[int | None, str]:
    """Same idea as `_completion_token_count` but for the PROMPT side - only
    two real sources confirmed so far (see module docstring), no crude
    fallback exists for this one (there's no equivalent of "count SSE
    lines" for a prompt that was never streamed back). Returns (None, ...)
    rather than a guess when neither is present."""
    for chunk in reversed(chunks):
        usage = chunk.get("usage") or {}
        if isinstance(usage, dict) and usage.get("prompt_tokens") is not None:
            return int(usage["prompt_tokens"]), "usage.prompt_tokens"
        timings = chunk.get("timings") or {}
        if isinstance(timings, dict) and timings.get("prompt_n") is not None:
            return int(timings["prompt_n"]), "timings.prompt_n"  # llama.cpp
    return None, "unavailable (no authoritative prompt-token count found in any chunk)"


def _server_reported_pp_tg(chunks: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    """llama.cpp's own `timings` object separates prefill from decode
    server-side (`prompt_per_second` / `predicted_per_second`) - the most
    accurate PP/TG source available, since it excludes client/network
    overhead entirely. Confirmed real via this project's own local
    llama.cpp checkout (tools/server/server-task.cpp), not guessed at.
    Returns (None, None) for any backend that doesn't emit this (qxmx,
    Arcaine, OpenArc, vLLM - none of which carry a `timings` object in
    their real captured responses this project has seen)."""
    for chunk in reversed(chunks):
        timings = chunk.get("timings") or {}
        if isinstance(timings, dict) and timings.get("prompt_per_second") is not None and timings.get("predicted_per_second") is not None:
            return float(timings["prompt_per_second"]), float(timings["predicted_per_second"])
    return None, None


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
        # Real gotcha found comparing this adapter's new PP number against a
        # live llama-benchy run on the same real production backend
        # (llama-cpp-bonsai): llama.cpp's server has prompt caching on by
        # default (`cache_n` in its timings object) - a short, repeated
        # prompt like this adapter's own default reuses almost the entire
        # prompt from cache (confirmed live: only 4 of 18 prompt tokens
        # were freshly processed on a repeat request), which reports a
        # tiny, meaningless "prompt_per_second" for a handful of tokens,
        # nothing like a real batch-prefill number. `request_extra` lets a
        # suite pass through raw fields (e.g. `{"cache_prompt": false}`,
        # llama.cpp-specific but harmless/ignored elsewhere) to get an
        # honest PP measurement - confirmed live: with caching disabled
        # and a real ~7400-token prompt, this adapter's PP (290 tok/s) and
        # TG (16.3 tok/s) landed close to the same server's llama-benchy
        # numbers (335.7 / 17.3 tok/s) - see VALIDATION.md.
        request_extra = cfg.get("request_extra", {})

        url = endpoint.rstrip("/") + "/v1/chat/completions"
        ttfts_ms: list[float] = []
        totals_ms: list[float] = []
        tokens_per_sec: list[float] = []
        pp_tokens_per_sec: list[float] = []
        tg_tokens_per_sec: list[float] = []
        pp_tg_sources: set[str] = set()
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
                        **request_extra,
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

                # PP/TG split - see module docstring for the two real
                # sources tried, in order, and why a backend with neither
                # simply contributes nothing here rather than a guess.
                pp_tps, tg_tps = _server_reported_pp_tg(parsed_chunks)
                pp_tg_source = "server_timings" if pp_tps is not None else None
                if pp_tps is None:
                    prompt_tokens, _ = _prompt_token_count(parsed_chunks)
                    if prompt_tokens is not None and prompt_tokens > 0 and ttft > 0:
                        pp_tps = prompt_tokens / ttft
                        decode_s = total - ttft
                        if completion_tokens > 1 and decode_s > 0:
                            tg_tps = (completion_tokens - 1) / decode_s
                        pp_tg_source = "ttft_split"
                if pp_tps is not None:
                    pp_tokens_per_sec.append(pp_tps)
                if tg_tps is not None:
                    tg_tokens_per_sec.append(tg_tps)
                if pp_tg_source is not None:
                    pp_tg_sources.add(pp_tg_source)

                raw_responses.append(
                    {
                        "ttft_s": ttft,
                        "total_s": total,
                        "tokens": completion_tokens,
                        "counted_via": counted_via,
                        "pp_tokens_per_sec": pp_tps,
                        "tg_tokens_per_sec": tg_tps,
                        "pp_tg_source": pp_tg_source or "unavailable",
                    }
                )

        def avg(xs: list[float]) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        metrics = {
            "avg_ttft_ms": avg(ttfts_ms),
            "avg_total_ms": avg(totals_ms),
            "avg_tokens_per_sec": avg(tokens_per_sec),
            "requests": float(num_requests),
        }
        # Only added when at least one request produced a real number -
        # never backfilled with 0.0 (that would look like a measured zero
        # throughput, not "not measured").
        if pp_tokens_per_sec:
            metrics["avg_pp_tokens_per_sec"] = avg(pp_tokens_per_sec)
        if tg_tokens_per_sec:
            metrics["avg_tg_tokens_per_sec"] = avg(tg_tokens_per_sec)

        return BenchmarkResult(
            adapter=self.name,
            metrics=metrics,
            raw={"requests": raw_responses, "pp_tg_sources": sorted(pp_tg_sources)},
        )


register("benchmark", GenericHttpBenchmark.name, GenericHttpBenchmark)
