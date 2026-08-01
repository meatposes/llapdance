"""Result records. Every stored result carries full run context (SPEC.md §8) -
callers should never need to go back to the raw tool output to know what
produced a number."""
from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


class BenchmarkResult(BaseModel):
    adapter: str
    metrics: dict[str, float]
    raw: dict[str, Any] = Field(default_factory=dict)


class CoherenceResult(BaseModel):
    adapter: str
    total: int
    passed: int
    graded_by_match: int
    graded_by_llm_judge: int
    failures: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


class RunResult(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = Field(default_factory=time.time)
    backend_name: str
    backend_config: dict[str, Any]
    image_ref: str | None = None
    execution_target: dict[str, Any]
    device_target: dict[str, Any]
    benchmarks: list[BenchmarkResult] = Field(default_factory=list)
    coherence: list[CoherenceResult] = Field(default_factory=list)
