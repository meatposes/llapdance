import json

import httpx
import pytest

from llapdance.plugins.benchmark.llama_benchy import LlamaBenchyBenchmark

RESULT_DATA = {
    "model": "m",
    "benchmarks": [
        {"pp_throughput": {"mean": 100.0, "std": 1.0}, "tg_throughput": {"mean": 50.0, "std": 0.5}, "ttfr": {"mean": 200.0}},
        {"pp_throughput": {"mean": 120.0, "std": 1.0}, "tg_throughput": {"mean": 60.0, "std": 0.5}, "ttfr": {"mean": 220.0}},
    ],
}


def _handler(events):
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/start":
            return httpx.Response(200, json={"run_id": "abc123"})
        if request.url.path == "/api/run/abc123/stream":
            body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})
        if request.url.path == "/api/results/abc123/export/json":
            return httpx.Response(200, json=RESULT_DATA)
        raise AssertionError(f"unexpected request: {request.url}")

    return handle


def _patch_client(monkeypatch, handler):
    real_init = httpx.Client.__init__
    monkeypatch.setattr(
        httpx.Client, "__init__",
        lambda self, *a, **kw: real_init(self, *a, **{**kw, "transport": httpx.MockTransport(handler)}),
    )


def test_requires_dashboard_url():
    adapter = LlamaBenchyBenchmark({})
    with pytest.raises(ValueError, match="dashboard_url"):
        adapter.run("http://fake", {"model": "m"})


def test_real_result_shape_on_success(monkeypatch):
    _patch_client(monkeypatch, _handler([{"done": True}]))
    adapter = LlamaBenchyBenchmark({"dashboard_url": "http://localhost:5059"})
    result = adapter.run("http://127.0.0.1:8001", {"model": "m"})

    assert result.adapter == "llama-benchy"
    assert result.metrics["avg_pp_throughput"] == 110.0
    assert result.metrics["avg_tg_throughput"] == 55.0
    assert result.metrics["avg_ttfr_ms"] == 210.0
    assert result.metrics["benchmark_count"] == 2.0
    assert result.raw["run_id"] == "abc123"


def test_raises_on_reported_error(monkeypatch):
    _patch_client(monkeypatch, _handler([{"done": True, "error": "model crashed"}]))
    adapter = LlamaBenchyBenchmark({"dashboard_url": "http://localhost:5059"})
    with pytest.raises(RuntimeError, match="model crashed"):
        adapter.run("http://127.0.0.1:8001", {"model": "m"})


def test_start_payload_passes_endpoint_as_base_url(monkeypatch):
    seen = {}

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/start":
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"run_id": "abc123"})
        if request.url.path == "/api/run/abc123/stream":
            return httpx.Response(200, content="data: {\"done\": true}\n\n", headers={"content-type": "text/event-stream"})
        if request.url.path == "/api/results/abc123/export/json":
            return httpx.Response(200, json=RESULT_DATA)
        raise AssertionError(request.url)

    _patch_client(monkeypatch, handle)
    adapter = LlamaBenchyBenchmark({"dashboard_url": "http://localhost:5059"})
    adapter.run("http://127.0.0.1:8001/", {"model": "my-model", "test_group": "baseline"})
    assert seen["payload"]["base_url"] == "http://127.0.0.1:8001"
    assert seen["payload"]["model"] == "my-model"
    assert seen["payload"]["test_group"] == "baseline"


def test_model_host_override_rewrites_unreachable_127_0_0_1(monkeypatch):
    # Real bug found running an actual overnight sweep: for a local-docker
    # backend, `endpoint` is always http://127.0.0.1:<port> (see
    # local_docker.py's start()) - correct for every OTHER benchmark
    # adapter, which runs in-process on the host. This adapter hands
    # base_url to a SEPARATE llama-benchy-web container, where 127.0.0.1
    # means itself, not the host - every request silently failed to
    # connect, and llama-benchy's own per-request error handling recorded
    # null for every measurement rather than raising. Confirmed live
    # against a real container: swapping in the real docker bridge gateway
    # (172.28.0.1 in this deployment) fixed it end to end.
    seen = {}

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/start":
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"run_id": "abc123"})
        if request.url.path == "/api/run/abc123/stream":
            return httpx.Response(200, content="data: {\"done\": true}\n\n", headers={"content-type": "text/event-stream"})
        if request.url.path == "/api/results/abc123/export/json":
            return httpx.Response(200, json=RESULT_DATA)
        raise AssertionError(request.url)

    _patch_client(monkeypatch, handle)
    adapter = LlamaBenchyBenchmark({"dashboard_url": "http://localhost:5059"})
    adapter.run("http://127.0.0.1:54321", {"model": "my-model", "model_host_override": "172.28.0.1"})
    assert seen["payload"]["base_url"] == "http://172.28.0.1:54321"


def test_no_override_leaves_endpoint_unchanged(monkeypatch):
    seen = {}

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/start":
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"run_id": "abc123"})
        if request.url.path == "/api/run/abc123/stream":
            return httpx.Response(200, content="data: {\"done\": true}\n\n", headers={"content-type": "text/event-stream"})
        if request.url.path == "/api/results/abc123/export/json":
            return httpx.Response(200, json=RESULT_DATA)
        raise AssertionError(request.url)

    _patch_client(monkeypatch, handle)
    adapter = LlamaBenchyBenchmark({"dashboard_url": "http://localhost:5059"})
    adapter.run("http://127.0.0.1:54321", {"model": "my-model"})
    assert seen["payload"]["base_url"] == "http://127.0.0.1:54321"
