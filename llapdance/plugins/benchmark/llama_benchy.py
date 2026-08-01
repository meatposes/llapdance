"""llama-benchy adapter - INTENTIONAL STUB.

While building this out, llama-benchy (hellohal2064/llama-benchy) was found
to expose only a Flask web dashboard (GET /, static assets) with no
documented/discoverable REST API - `curl` against plausible routes
(/api, /openapi.json) returned 404, and the container logs show no JSON
endpoints being hit. Wrapping it correctly requires either its actual API
(if one exists undocumented) or scripting its web UI, neither of which
should be guessed at.

Registered anyway so a suite config referencing `adapter: llama-benchy`
fails with this explanation instead of a bare KeyError. Replace `run()`
once the real integration point is confirmed.
"""
from __future__ import annotations

from typing import Any

from llapdance.core.result import BenchmarkResult
from llapdance.plugins.base import BenchmarkAdapter
from llapdance.plugins.registry import register


class LlamaBenchyBenchmark(BenchmarkAdapter):
    name = "llama-benchy"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def run(self, endpoint: str, config: dict[str, Any]) -> BenchmarkResult:
        raise NotImplementedError(
            "llama-benchy adapter is not implemented: no documented REST API was "
            "found on the running container (only a Flask dashboard on '/'). "
            "Use the 'generic-http' adapter, or confirm llama-benchy's real API "
            "and implement this adapter against it."
        )


register("benchmark", LlamaBenchyBenchmark.name, LlamaBenchyBenchmark)
