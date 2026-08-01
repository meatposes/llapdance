"""Plugin contracts (SPEC.md §5). Any third-party tool wraps one of these
ABCs; the orchestrator core never depends on a concrete adapter."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from llapdance.core.result import BenchmarkResult, CoherenceResult, RunResult


class RunningBackend(ABC):
    """Handle to a live backend instance returned by an ExecutionTargetAdapter."""

    @property
    @abstractmethod
    def endpoint(self) -> str:
        """Base URL the backend's OpenAI-compatible API is reachable at."""

    @abstractmethod
    def logs(self, tail: int = 200) -> str:
        ...


class ExecutionTargetAdapter(ABC):
    """Where backend containers actually run: local docker socket, or a
    remote host over SSH (SPEC.md §5, §6). Same backend-config definitions
    apply regardless of which one is active."""

    @abstractmethod
    def build(self, backend_config: dict[str, Any]) -> str:
        """Build (or resolve, for prebuilt) an image, return its image ref."""

    @abstractmethod
    def start(self, backend_config: dict[str, Any], image_ref: str, device_indices: list[int]) -> RunningBackend:
        ...

    @abstractmethod
    def stop(self, backend: RunningBackend) -> None:
        ...

    @abstractmethod
    def list_images(self, name_filter: str | None = None) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def remove_image(self, image_ref: str) -> None:
        ...


class BenchmarkAdapter(ABC):
    """Perf/throughput measurement plugin. Reference impls: generic-http,
    llama-benchy (stub, see README), guidellm."""

    name: str

    @abstractmethod
    def run(self, endpoint: str, config: dict[str, Any]) -> BenchmarkResult:
        ...


class CoherenceAdapter(ABC):
    """Correctness/output-quality plugin. Reference impl: fixed-question-set
    with string-match + LLM-judge fallback."""

    name: str

    @abstractmethod
    def run(self, endpoint: str, config: dict[str, Any]) -> CoherenceResult:
        ...


class StorageAdapter(ABC):
    """Result persistence plugin. Flat-file is the only default-on adapter
    (SPEC.md §8); everything else is opt-in and adapters can stack."""

    name: str

    @abstractmethod
    def write(self, result: RunResult) -> None:
        ...

    @abstractmethod
    def previous_for(self, backend_name: str, limit: int = 1) -> list[RunResult]:
        """Most recent prior result(s) for a backend, for delta computation."""
