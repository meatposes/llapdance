import json

import httpx

from llapdance.plugins.coherence.fixed_questions import FixedQuestionCoherence


def test_default_max_tokens_is_64_unchanged_from_before():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["max_tokens"])
        return httpx.Response(200, json={"choices": [{"message": {"content": "42"}}]})

    adapter = FixedQuestionCoherence({})
    real_init = httpx.Client.__init__
    httpx.Client.__init__ = lambda self, *a, **kw: real_init(self, *a, **{**kw, "transport": httpx.MockTransport(handler)})
    try:
        adapter.run("http://fake", {"questions": [{"prompt": "x", "expected_keywords": ["42"]}]})
    finally:
        httpx.Client.__init__ = real_init
    assert seen == [64]


def test_max_tokens_is_configurable_for_reasoning_models():
    # real gap found sweeping OpenVINO/Qwen3-0.6B-int4-ov (see VALIDATION.md):
    # a hardcoded 64 truncates <think> traces before an answer is reached
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["max_tokens"])
        return httpx.Response(200, json={"choices": [{"message": {"content": "42"}}]})

    adapter = FixedQuestionCoherence({})
    real_init = httpx.Client.__init__
    httpx.Client.__init__ = lambda self, *a, **kw: real_init(self, *a, **{**kw, "transport": httpx.MockTransport(handler)})
    try:
        adapter.run(
            "http://fake",
            {"max_tokens": 512, "questions": [{"prompt": "x", "expected_keywords": ["42"]}]},
        )
    finally:
        httpx.Client.__init__ = real_init
    assert seen == [512]


def test_falls_back_to_reasoning_field_when_content_is_null():
    # Real crash found live against vllm-urak's real Ornith-1.0-35B-int4-
    # AutoRound: content came back null, the whole answer sitting in a
    # `reasoning` field instead - same failure MODE as the llama-cpp-sycl
    # LLAMA_ARG_REASONING gotcha, different field name. Used to crash with
    # AttributeError: 'NoneType' object has no attribute 'lower'.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": None, "reasoning": "...the answer is 42."}}]},
        )

    adapter = FixedQuestionCoherence({})
    real_init = httpx.Client.__init__
    httpx.Client.__init__ = lambda self, *a, **kw: real_init(self, *a, **{**kw, "transport": httpx.MockTransport(handler)})
    try:
        result = adapter.run("http://fake", {"questions": [{"prompt": "x", "expected_keywords": ["42"]}]})
    finally:
        httpx.Client.__init__ = real_init
    assert result.passed == 1


def test_empty_string_when_no_content_or_reasoning_field_present():
    # A real "no answer" should be a gradeable failure, not a crash and
    # not a guess.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": None}}]})

    adapter = FixedQuestionCoherence({})
    real_init = httpx.Client.__init__
    httpx.Client.__init__ = lambda self, *a, **kw: real_init(self, *a, **{**kw, "transport": httpx.MockTransport(handler)})
    try:
        result = adapter.run("http://fake", {"questions": [{"prompt": "x", "expected_keywords": ["42"]}]})
    finally:
        httpx.Client.__init__ = real_init
    assert result.passed == 0
    assert result.failures[0]["answer"] == ""
