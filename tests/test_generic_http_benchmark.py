import json

import httpx

from llapdance.plugins.benchmark.generic_http import (
    GenericHttpBenchmark,
    _completion_token_count,
    _prompt_token_count,
    _server_reported_pp_tg,
)


def test_prefers_standard_usage_field():
    chunks = [{"choices": []}, {"usage": {"completion_tokens": 42}}]
    count, via = _completion_token_count(chunks, sse_line_count=5)
    assert count == 42
    assert via == "usage.completion_tokens"


def test_falls_back_to_arcaine_openarc_metrics_field():
    # real bug (see VALIDATION.md): Arcaine's diffusion decoding emits the
    # WHOLE completion as one SSE chunk, so sse_line_count (here: 3) badly
    # undercounts against the real token count (14) - metrics.new_token is
    # the authoritative source for both Arcaine and OpenArc.
    chunks = [{"choices": [{"delta": {"role": "assistant"}}]}, {"metrics": {"new_token": 14}}]
    count, via = _completion_token_count(chunks, sse_line_count=3)
    assert count == 14
    assert via == "metrics.new_token"


def test_falls_back_to_llama_cpp_timings_field():
    chunks = [{"timings": {"predicted_n": 12}}]
    count, via = _completion_token_count(chunks, sse_line_count=12)
    assert count == 12
    assert via == "timings.predicted_n"


def test_falls_back_to_sse_line_count_when_nothing_else_present():
    count, via = _completion_token_count([{"choices": []}], sse_line_count=7)
    assert count == 7
    assert via.startswith("sse_line_count")


def test_final_chunk_takes_priority_over_earlier_chunks():
    # a middle chunk with a stale/partial usage shouldn't win over the
    # real final one
    chunks = [{"usage": {"completion_tokens": 1}}, {"usage": {"completion_tokens": 99}}]
    count, _ = _completion_token_count(chunks, sse_line_count=2)
    assert count == 99


def test_prompt_token_count_prefers_usage_field():
    chunks = [{"usage": {"prompt_tokens": 100, "completion_tokens": 31}}]
    count, via = _prompt_token_count(chunks)
    assert count == 100
    assert via == "usage.prompt_tokens"


def test_prompt_token_count_falls_back_to_llama_cpp_timings():
    chunks = [{"timings": {"prompt_n": 50, "predicted_n": 20}}]
    count, via = _prompt_token_count(chunks)
    assert count == 50
    assert via == "timings.prompt_n"


def test_prompt_token_count_unavailable_when_neither_present():
    # real state of Arcaine/OpenArc/vLLM responses captured this project -
    # no prompt-token count anywhere, must not be guessed at
    count, via = _prompt_token_count([{"metrics": {"new_token": 14}}])
    assert count is None
    assert via.startswith("unavailable")


def test_server_reported_pp_tg_reads_llama_cpp_timings():
    chunks = [{"timings": {"prompt_n": 50, "predicted_n": 20, "prompt_per_second": 250.0, "predicted_per_second": 40.0}}]
    pp, tg = _server_reported_pp_tg(chunks)
    assert pp == 250.0
    assert tg == 40.0


def test_server_reported_pp_tg_absent_when_no_timings_object():
    pp, tg = _server_reported_pp_tg([{"usage": {"prompt_tokens": 100, "completion_tokens": 31}}])
    assert pp is None
    assert tg is None


def _mock_stream_client(body: bytes):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    real_init = httpx.Client.__init__
    httpx.Client.__init__ = lambda self, *a, **kw: real_init(self, *a, **{**kw, "transport": httpx.MockTransport(handler)})
    return real_init


def test_run_uses_llama_cpp_server_reported_pp_tg_directly(monkeypatch):
    # llama.cpp's own timings object is authoritative - used as-is, no
    # client-side derivation, regardless of how long the request actually
    # took on the test machine.
    body = (
        b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n'
        b'data: {"choices":[{"delta":{}}],"timings":{"prompt_n":50,"predicted_n":20,'
        b'"prompt_per_second":250.0,"predicted_per_second":40.0}}\n'
        b'data: [DONE]\n'
    )
    real_init = _mock_stream_client(body)
    try:
        result = GenericHttpBenchmark().run("http://fake", {"num_requests": 1})
    finally:
        httpx.Client.__init__ = real_init

    assert result.metrics["avg_pp_tokens_per_sec"] == 250.0
    assert result.metrics["avg_tg_tokens_per_sec"] == 40.0
    assert result.raw["pp_tg_sources"] == ["server_timings"]


def test_run_derives_pp_tg_from_ttft_when_only_usage_counts_present(monkeypatch):
    # qxmx's real shape: usage.prompt_tokens + usage.completion_tokens,
    # no timings object at all - PP/TG must be derived from TTFT/total,
    # not left blended-only, since real prompt/completion counts exist.
    body = (
        b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n'
        b'data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":100,"completion_tokens":31}}\n'
        b'data: [DONE]\n'
    )
    real_init = _mock_stream_client(body)

    # httpx's own internals call time.perf_counter() once too (between our
    # `start` and `first_byte_at`, confirmed by tracing real call order) -
    # the 4 values below are [our start, httpx internal (unused), our
    # first_byte_at -> ttft=0.5s, our end -> total=2.0s].
    times = iter([0.0, 0.0, 0.5, 2.0])
    monkeypatch.setattr("llapdance.plugins.benchmark.generic_http.time.perf_counter", lambda: next(times))

    try:
        result = GenericHttpBenchmark().run("http://fake", {"num_requests": 1})
    finally:
        httpx.Client.__init__ = real_init

    # pp = prompt_tokens / ttft = 100 / 0.5 = 200
    assert result.metrics["avg_pp_tokens_per_sec"] == 200.0
    # tg = (completion_tokens - 1) / (total - ttft) = 30 / 1.5 = 20
    assert result.metrics["avg_tg_tokens_per_sec"] == 20.0
    assert result.raw["pp_tg_sources"] == ["ttft_split"]


def test_run_omits_pp_tg_when_no_prompt_signal_present():
    # Arcaine/OpenArc-shaped response: only metrics.new_token (completion
    # count), no prompt-token count anywhere - PP/TG must be left OUT of
    # metrics entirely, never backfilled with a guess or a fake 0.0.
    body = (
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n'
        b'data: {"choices":[{"delta":{}}],"metrics":{"new_token":14}}\n'
        b'data: [DONE]\n'
    )
    real_init = _mock_stream_client(body)
    try:
        result = GenericHttpBenchmark().run("http://fake", {"num_requests": 1})
    finally:
        httpx.Client.__init__ = real_init

    assert "avg_pp_tokens_per_sec" not in result.metrics
    assert "avg_tg_tokens_per_sec" not in result.metrics
    assert "avg_tokens_per_sec" in result.metrics  # blended metric unaffected


def test_request_extra_is_merged_into_the_request_body():
    # Real gotcha found live: llama.cpp's prompt caching makes a repeated/
    # short prompt report a near-meaningless PP number. request_extra lets
    # a suite disable it (`{"cache_prompt": false}`) for an honest PP
    # measurement - confirmed here that the field actually reaches the
    # request body, not just accepted and dropped.
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        body = b'data: {"choices":[{"delta":{}}],"metrics":{"new_token":1}}\ndata: [DONE]\n'
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    real_init = httpx.Client.__init__
    httpx.Client.__init__ = lambda self, *a, **kw: real_init(self, *a, **{**kw, "transport": httpx.MockTransport(handler)})
    try:
        GenericHttpBenchmark().run("http://fake", {"num_requests": 1, "request_extra": {"cache_prompt": False}})
    finally:
        httpx.Client.__init__ = real_init

    assert captured["body"]["cache_prompt"] is False
