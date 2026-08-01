"""Plugin contracts (SPEC.md §5). Any third-party tool wraps one of these
ABCs; the orchestrator core never depends on a concrete adapter."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from llapdance.core.probe import DeviceInfo
from llapdance.core.result import BenchmarkResult, CoherenceResult, RunResult, TelemetryResult


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


class TelemetryAdapter(ABC):
    """GPU/hardware telemetry captured alongside a benchmark/coherence run -
    a third concern distinct from throughput (BenchmarkAdapter) and output
    correctness (CoherenceAdapter): what the hardware was actually doing
    (utilization, power, memory bandwidth) while producing those numbers.
    Reference impl: xmxmon, validated against a real running instance.

    Brackets around the run rather than hitting the endpoint itself:
    `start()` is called before benchmark/coherence adapters run, `stop()`
    after, so the adapter can capture a window rather than a single point
    sample. `start()`/`stop()` take no endpoint - telemetry adapters watch
    hardware, not the OpenAI-compatible API surface the other two kinds do.
    """

    name: str

    @abstractmethod
    def start(self, config: dict[str, Any]) -> Any:
        """Begin capturing; returns an opaque handle passed to stop()."""

    @abstractmethod
    def stop(self, handle: Any) -> TelemetryResult:
        ...


@dataclass
class EngineInvocation:
    """What an EngineTranslator produces - merged onto a BackendConfig's
    raw command/env/devices, with any value the user set explicitly in
    config taking precedence over the generated one (SPEC.md's raw
    passthrough remains the escape hatch, this is the convenience layer
    on top of it, not a replacement for it)."""

    command: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    devices: list[str] = field(default_factory=list)
    post_start_requests: list[dict[str, Any]] = field(default_factory=list)
    """HTTP requests fired against the running backend's endpoint after the
    health check passes but before benchmark/coherence adapters run - for
    engines where starting the container and loading a model are two
    separate steps (found building the openarc translator: OpenArc starts
    with no model loaded, a model only becomes servable after POST
    /openarc/load - not every engine bakes 'load this model' into its
    start command/env). Each dict: {"method": str, "path": str, "json": dict}.
    A non-2xx response aborts the run rather than silently proceeding to
    benchmark a backend with no model loaded."""


class EngineTranslator(ABC):
    """Translates SPEC.md §4's normalized `params.shared` (context_size,
    batch_size, kv_cache_quant, parallel_slots - the set validated so far
    against real engines, see VALIDATION.md) plus a resolved model path and
    GPU device into a concrete container command/env/devices for one
    specific engine. This is the per-engine 'wrapper' the original spec
    envisioned - `BackendConfig.command`/`env`/`devices` remain valid raw
    passthrough for anything a translator doesn't cover or gets wrong.

    Reference impls: llama-cpp-sycl, qxmx (both validated against real
    hardware - see VALIDATION.md 'params translation layer' section for
    what does and doesn't map cleanly across the two).
    """

    name: str

    sweepable_params: dict[str, dict[str, Any]] = {}
    """Catalog of the params this translator reads out of the merged
    params dict (SPEC.md §10's 'cataloging what build switches we'd want
    to sweep') - a class attribute, not instance state, so it's
    introspectable without building a translator (`registry.get("engine",
    name).sweepable_params`). Each entry: {"type": str, "default": Any
    (optional), "values": list (optional, for enum-like params),
    "maps_to": str (the concrete flag/env var this becomes)}. This is
    documentation the code can serve up (`llapdance engines describe
    <name>`, the `describe_engine` MCP tool) - it does not validate or
    constrain what a suite's `sweep` axes can actually target; the dotted
    param path is still just a path into the raw config dict."""

    known_env_flags: dict[str, dict[str, Any]] = {}
    """Raw engine/library-level env var flags known to affect behavior
    (e.g. GGML/oneDNN runtime toggles) that the translator does NOT read
    or generate - these are swept directly via `env.<NAME>` on the raw
    backend config, the exact same generic dotted-path mechanism as
    sweepable_params, just a different section of the config (confirmed:
    the sweep expansion code has no special-casing per path prefix -
    `params.shared.x` and `env.X` go through identical logic, and both
    have been validated live against a real container). Listed here
    purely so `describe-engine`/`list_models`-adjacent tooling can surface
    them - this is NOT an exhaustive list, just the ones found by reading
    actual backend source rather than guessed at. Same entry shape as
    sweepable_params, plus a "source" note where the flag was found."""

    @abstractmethod
    def build(
        self,
        model_path: str,
        params: dict[str, Any],
        port: int,
        device: DeviceInfo | None,
    ) -> EngineInvocation:
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
