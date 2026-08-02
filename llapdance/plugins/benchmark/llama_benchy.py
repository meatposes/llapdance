"""llama-benchy adapter (hellohal2064/llama-benchy). Was long documented as
an intentional stub here - a prior session's `curl` probing against guessed
routes (`/api`, `/openapi.json`) 404'd, and concluded (wrongly) that the
running container exposed no API. Re-investigated by reading the real
container's own source (`docker exec llama-benchy-web grep -n '@app.route'
/app/web/app.py`) instead of guessing at routes - it's a genuine Flask app
with a real, if undocumented, JSON API wrapping the `llama-benchy` CLI as a
subprocess:

  - `POST /api/start` `{base_url, model, tokenizer, test_group,
    custom_config}` -> `{run_id}`. `base_url`/`model` are passed straight to
    the CLI's own `--base-url`/`--model` (confirmed in `web/engine.py`'s
    `_build_command()`) - genuinely arbitrary OpenAI-compatible endpoint,
    not llama-benchy-specific.
  - `GET /api/run/<run_id>/stream` - SSE progress
    (`{"status","progress","current_test","new_lines","done","error"}`),
    the only way to know when a run finished (no plain polling endpoint).
  - `GET /api/results/<run_id>/export/json` - the raw structured result
    (`{"model","benchmarks":[{"context_size","prompt_size","response_size",
    "concurrency","pp_throughput":{"mean","std"},
    "tg_throughput":{"mean","std"},"peak_throughput":{...},"ttfr":{...}},
    ...]}`), confirmed via `web/engine.py::format_result_rows()`.

Distinct from every other benchmark adapter here: this is a genuinely
ASYNC job API (start -> poll/stream -> fetch), not a synchronous
request/response prober, and it needs a SEPARATE endpoint from the model
server under test - `dashboard_url` (this Flask app, e.g.
`http://localhost:5059`) is not the same thing as `endpoint` (the model
server `base_url` this adapter tells llama-benchy to hit).

Presets (`test_group`, real values read from `web/engine.py::TEST_GROUPS`,
not invented): `quick_check` (default, ~3 min, PP2048+TG128 at concurrency
1), `baseline`, `shallow_context`, `deep_context`, `all_leaderboard`
(~90 min). `test_group: "custom"` + a `custom_config` dict bypasses the
presets entirely (same shape as a `TEST_GROUPS` entry - `pp`/`tg`/`depth`/
`concurrency`/`enable_prefix_caching`/`runs`).
"""
from __future__ import annotations

import json
from typing import Any

import httpx

from llapdance.core.result import BenchmarkResult
from llapdance.plugins.base import BenchmarkAdapter
from llapdance.plugins.registry import register


class LlamaBenchyBenchmark(BenchmarkAdapter):
    name = "llama-benchy"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def run(self, endpoint: str, config: dict[str, Any]) -> BenchmarkResult:
        cfg = {**self._config, **config}
        dashboard_url = cfg.get("dashboard_url")
        if not dashboard_url:
            raise ValueError(
                "llama-benchy adapter requires 'dashboard_url' - the llama-benchy web "
                "dashboard's OWN endpoint (e.g. 'http://localhost:5059'), distinct from "
                "`endpoint` (the model server under test, passed through as base_url)."
            )
        model = cfg.get("model", "default")

        # Real bug found live running an actual overnight sweep: for a
        # local-docker backend, `endpoint` is always `http://127.0.0.1:
        # <published-port>` (see local_docker.py's start()) - correct for
        # every OTHER benchmark adapter here, which runs in-process on the
        # host, where 127.0.0.1 genuinely means "this host". This adapter
        # is the one exception (see module docstring): it hands `base_url`
        # to a SEPARATE llama-benchy-web CONTAINER, which then makes its
        # OWN outbound requests - 127.0.0.1 inside that container means
        # itself, not the host, so every request silently failed to
        # connect. llama-benchy's own per-request error handling doesn't
        # surface that as a top-level error - it just records null for
        # every measurement, which is why this went unnoticed until the
        # aggregated metrics turned out to be all zero. `model_host_override`
        # lets a suite author supply whatever address IS reachable from the
        # llama-benchy-web container in their own deployment (e.g. a docker
        # bridge gateway IP) - deliberately not auto-guessed, since the
        # right address depends on which docker network llama-benchy-web
        # itself is on, which this adapter has no way to know.
        base_url = endpoint.rstrip("/")
        model_host_override = cfg.get("model_host_override")
        if model_host_override:
            base_url = str(httpx.URL(base_url).copy_with(host=model_host_override))

        start_payload: dict[str, Any] = {
            "base_url": base_url,
            "model": model,
            "tokenizer": cfg.get("tokenizer", model),
            "test_group": cfg.get("test_group", "quick_check"),
        }
        if "custom_config" in cfg:
            start_payload["custom_config"] = cfg["custom_config"]

        stream_timeout = cfg.get("timeout", 1800)

        with httpx.Client(base_url=dashboard_url.rstrip("/"), timeout=60) as client:
            resp = client.post("/api/start", json=start_payload)
            resp.raise_for_status()
            run_id = resp.json()["run_id"]

            with client.stream("GET", f"/api/run/{run_id}/stream", timeout=stream_timeout) as stream:
                for line in stream.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[len("data: "):])
                    if event.get("done"):
                        if event.get("error"):
                            raise RuntimeError(f"llama-benchy run {run_id} failed: {event['error']}")
                        break

            export = client.get(f"/api/results/{run_id}/export/json")
            export.raise_for_status()
            result_data = export.json()

        pp_means = [b["pp_throughput"]["mean"] for b in result_data.get("benchmarks", []) if b.get("pp_throughput")]
        tg_means = [b["tg_throughput"]["mean"] for b in result_data.get("benchmarks", []) if b.get("tg_throughput")]
        ttfr_means = [b["ttfr"]["mean"] for b in result_data.get("benchmarks", []) if b.get("ttfr")]

        def avg(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        return BenchmarkResult(
            adapter=self.name,
            metrics={
                "avg_pp_throughput": avg(pp_means),
                "avg_tg_throughput": avg(tg_means),
                "avg_ttfr_ms": avg(ttfr_means),
                "benchmark_count": float(len(result_data.get("benchmarks", []))),
            },
            raw={"run_id": run_id, "result": result_data},
        )


register("benchmark", LlamaBenchyBenchmark.name, LlamaBenchyBenchmark)
