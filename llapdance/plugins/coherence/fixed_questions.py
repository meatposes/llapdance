"""Reference coherence adapter (SPEC.md §5, §11): a fixed question set,
graded by keyword match first, LLM-judge fallback for ambiguous cases.
Exists to catch fluent-looking garbage (driver bugs, numerical errors,
garbled tokens) that a throughput number alone won't reveal.
"""
from __future__ import annotations

from typing import Any

import httpx

from llapdance.core.result import CoherenceResult
from llapdance.llm.client import OpenAICompatibleClient
from llapdance.plugins.base import CoherenceAdapter
from llapdance.plugins.registry import register

DEFAULT_QUESTIONS: list[dict[str, Any]] = [
    {"prompt": "What is 12 + 30?", "expected_keywords": ["42"]},
    {"prompt": "Name the capital of France.", "expected_keywords": ["paris"]},
    {"prompt": "Spell the word 'cat' backwards.", "expected_keywords": ["tac"]},
    {"prompt": "What color do you get mixing blue and yellow?", "expected_keywords": ["green"]},
    {"prompt": "Is the sky, on a clear day, usually blue or red?", "expected_keywords": ["blue"]},
    {"prompt": "How many days are in a standard week?", "expected_keywords": ["7", "seven"]},
    {"prompt": "What is the chemical symbol for water?", "expected_keywords": ["h2o", "h₂o"]},
    {"prompt": "Complete: roses are red, violets are ___.", "expected_keywords": ["blue"]},
    {"prompt": "What is 9 multiplied by 9?", "expected_keywords": ["81"]},
    {"prompt": "Name the first month of the year.", "expected_keywords": ["january"]},
]


class FixedQuestionCoherence(CoherenceAdapter):
    name = "fixed-questions"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def run(self, endpoint: str, config: dict[str, Any]) -> CoherenceResult:
        cfg = {**self._config, **config}
        questions = cfg.get("questions", DEFAULT_QUESTIONS)
        model = cfg.get("model", "default")
        # Real gap found sweeping a reasoning model (OpenVINO/Qwen3-0.6B-int4-ov,
        # see VALIDATION.md): a hardcoded 64 truncates the <think> trace before
        # an answer is ever reached, producing a false coherence failure - not
        # a wrong answer, no answer at all. Configurable per-suite, still
        # defaults to 64 (unchanged behavior for every already-validated suite).
        max_tokens = cfg.get("max_tokens", 64)
        judge_cfg = cfg.get("llm_judge")
        judge = (
            OpenAICompatibleClient(
                base_url=judge_cfg["base_url"], model=judge_cfg["model"], api_key=judge_cfg.get("api_key")
            )
            if judge_cfg
            else None
        )

        headers = {"Authorization": f"Bearer {cfg['api_key']}"} if "api_key" in cfg else cfg.get("headers", {})

        passed = 0
        graded_by_match = 0
        graded_by_llm_judge = 0
        failures: list[dict[str, Any]] = []

        with httpx.Client(timeout=60, headers=headers) as client:
            for q in questions:
                answer = self._ask(client, endpoint, model, q["prompt"], max_tokens)
                lowered = answer.lower()
                matched = any(kw.lower() in lowered for kw in q["expected_keywords"])

                if matched:
                    graded_by_match += 1
                    passed += 1
                    continue

                if judge is None:
                    failures.append({"prompt": q["prompt"], "answer": answer, "graded_by": "none-no-judge-configured"})
                    continue

                graded_by_llm_judge += 1
                verdict = judge.chat(
                    [
                        {
                            "role": "system",
                            "content": "You grade whether a model's answer is coherent and correct for "
                            "the given question and expected keywords. Reply with exactly PASS or FAIL.",
                        },
                        {
                            "role": "user",
                            "content": f"Question: {q['prompt']}\nExpected keywords: {q['expected_keywords']}\n"
                            f"Model answer: {answer}",
                        },
                    ]
                )
                if verdict.strip().upper().startswith("PASS"):
                    passed += 1
                else:
                    failures.append({"prompt": q["prompt"], "answer": answer, "graded_by": "llm-judge"})

        return CoherenceResult(
            adapter=self.name,
            total=len(questions),
            passed=passed,
            graded_by_match=graded_by_match,
            graded_by_llm_judge=graded_by_llm_judge,
            failures=failures,
        )

    @staticmethod
    def _ask(client: httpx.Client, endpoint: str, model: str, prompt: str, max_tokens: int) -> str:
        resp = client.post(
            endpoint.rstrip("/") + "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},
        )
        resp.raise_for_status()
        message = resp.json()["choices"][0]["message"]
        # Real crash found live against vllm-urak's real Ornith-1.0-35B-int4-
        # AutoRound: `content` came back null, the entire answer sitting in a
        # `reasoning` field instead ({"content": null, "reasoning": "...42"},
        # finish_reason "length" - the whole token budget went into the
        # thinking trace, same failure MODE as the already-documented
        # llama-cpp-sycl LLAMA_ARG_REASONING gotcha, just a different field
        # name for this model/server). `.lower()` on a None crashed this
        # adapter outright rather than reporting a real coherence failure.
        # Falls back through every reasoning-field spelling seen so far
        # before giving up with an empty string (a real, gradeable "no
        # answer" failure, not a crash).
        return message.get("content") or message.get("reasoning_content") or message.get("reasoning") or ""


register("coherence", FixedQuestionCoherence.name, FixedQuestionCoherence)
