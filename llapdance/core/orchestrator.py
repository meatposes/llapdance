"""Ties backend lifecycle + plugins + storage together (SPEC.md §5, §9, §11).
This is the only module that should know about *all* the plugin kinds at
once - individual adapters never talk to each other directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llapdance.config.models import BackendConfig, TestSuite
from llapdance.core.probe import discover_devices, free_vram_mb
from llapdance.core.result import RunResult
from llapdance.plugins import registry
from llapdance.plugins.base import StorageAdapter


class VramPreflightError(RuntimeError):
    """Raised when a targeted device's free VRAM can't be confirmed, or is
    too low - SPEC.md §7 requires failing closed here, never guessing."""


@dataclass
class RunOutcome:
    result: RunResult
    delta_against: RunResult | None


def _resolve_device_indices(suite: TestSuite) -> list[int]:
    target = suite.device_target
    if target.mode.value == "none":
        return []
    devices = discover_devices()
    discrete = [d for d in devices if not d.integrated]
    if target.mode.value == "indices":
        chosen = [d for d in discrete if d.index in target.indices]
        missing = set(target.indices) - {d.index for d in chosen}
        if missing:
            raise ValueError(f"requested device indices not found among discrete devices: {sorted(missing)}")
        return [d.index for d in chosen]
    return [d.index for d in discrete]  # all_discrete


def _vram_preflight(device_indices: list[int], min_free_mb: float, allow_unknown: bool) -> None:
    if not device_indices:
        return
    devices = {d.index: d for d in discover_devices()}
    for idx in device_indices:
        device = devices.get(idx)
        if device is None:
            raise VramPreflightError(f"device index {idx} not found during preflight re-probe")
        free = free_vram_mb(device)
        if free is None:
            if allow_unknown:
                continue
            raise VramPreflightError(
                f"cannot determine free VRAM for device {idx} ({device.name}); "
                "refusing to run rather than risk wedging the card. "
                "Set allow_unknown_vram: true in suite config to override."
            )
        if free < min_free_mb:
            raise VramPreflightError(
                f"device {idx} ({device.name}) has {free}MB free, below the {min_free_mb}MB minimum"
            )


def _build_storage_adapters(suite: TestSuite) -> list[StorageAdapter]:
    adapters: list[StorageAdapter] = [
        registry.get("storage", "flat-file")({"flat_file_dir": suite.storage.flat_file_dir})
    ]
    for ref in suite.storage.extra_adapters:
        adapters.append(registry.get("storage", ref.adapter)(ref.config))
    return adapters


def run_backend(suite: TestSuite, backend: BackendConfig) -> RunOutcome:
    execution = registry.get("execution", "local-docker")({})
    device_indices = _resolve_device_indices(suite)
    _vram_preflight(device_indices, min_free_mb=suite.min_free_vram_mb, allow_unknown=suite.allow_unknown_vram)

    image_ref = execution.build(backend.model_dump())
    running = execution.start(backend.model_dump(), image_ref, device_indices)
    try:
        benchmarks = []
        for ref in suite.benchmark_adapters:
            adapter = registry.get("benchmark", ref.adapter)(ref.config)
            benchmarks.append(adapter.run(running.endpoint, ref.config))

        coherence = []
        for ref in suite.coherence_adapters:
            adapter = registry.get("coherence", ref.adapter)(ref.config)
            coherence.append(adapter.run(running.endpoint, ref.config))
    finally:
        execution.stop(running)

    storages = _build_storage_adapters(suite)
    result = RunResult(
        backend_name=backend.name,
        backend_config=backend.model_dump(mode="json"),
        image_ref=image_ref,
        execution_target=suite.execution_target.model_dump(mode="json"),
        device_target={"mode": suite.device_target.mode.value, "resolved_indices": device_indices},
        benchmarks=benchmarks,
        coherence=coherence,
    )

    previous = storages[0].previous_for(backend.name, limit=1)
    for storage in storages:
        storage.write(result)

    return RunOutcome(result=result, delta_against=previous[0] if previous else None)


def run_suite(suite: TestSuite) -> list[RunOutcome]:
    return [run_backend(suite, backend) for backend in suite.backends]
