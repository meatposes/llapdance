from llapdance.plugins.benchmark.generic_http import _completion_token_count


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
