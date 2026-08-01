"""Name -> adapter class registry. Third-party adapters register here (either
by importing and calling `register()`, or, later, via a packaging entry-point
group) - core code only ever looks adapters up by the name in config."""
from __future__ import annotations

from typing import Type

from .base import (
    BenchmarkAdapter,
    CoherenceAdapter,
    EngineTranslator,
    ExecutionTargetAdapter,
    StorageAdapter,
    TelemetryAdapter,
)

_REGISTRIES: dict[str, dict[str, Type]] = {
    "benchmark": {},
    "coherence": {},
    "storage": {},
    "execution": {},
    "engine": {},
    "telemetry": {},
}

_KIND_BASE = {
    "benchmark": BenchmarkAdapter,
    "coherence": CoherenceAdapter,
    "storage": StorageAdapter,
    "execution": ExecutionTargetAdapter,
    "engine": EngineTranslator,
    "telemetry": TelemetryAdapter,
}


def register(kind: str, adapter_name: str, cls: Type) -> None:
    if kind not in _REGISTRIES:
        raise ValueError(f"unknown adapter kind: {kind!r}")
    base = _KIND_BASE[kind]
    if not issubclass(cls, base):
        raise TypeError(f"{cls!r} must subclass {base.__name__} for kind {kind!r}")
    _REGISTRIES[kind][adapter_name] = cls


def get(kind: str, adapter_name: str) -> Type:
    try:
        return _REGISTRIES[kind][adapter_name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRIES.get(kind, {}))) or "(none registered)"
        raise KeyError(
            f"no {kind} adapter named {adapter_name!r} registered. Available: {available}"
        ) from exc


def available(kind: str) -> list[str]:
    return sorted(_REGISTRIES.get(kind, {}))


def describe_engine(name: str) -> dict:
    """SPEC.md §10's 'catalog of build switches to sweep' - the sweepable
    params a specific EngineTranslator declares, without instantiating it
    (it's a class attribute)."""
    return dict(get("engine", name).sweepable_params)


def load_builtin_adapters() -> None:
    """Import the reference adapters so they self-register. Called once at
    CLI/TUI startup; safe to call multiple times."""
    from llapdance.plugins.benchmark import generic_http, guidellm, llama_benchy  # noqa: F401
    from llapdance.plugins.coherence import fixed_questions  # noqa: F401
    from llapdance.plugins.storage import flat_file  # noqa: F401
    from llapdance.plugins.storage import opensearch  # noqa: F401
    from llapdance.plugins.execution import local_docker, ssh_docker  # noqa: F401
    from llapdance.plugins.engine import arcaine, llama_cpp_sycl, openarc, qxmx  # noqa: F401
    from llapdance.plugins.telemetry import xmxmon  # noqa: F401
