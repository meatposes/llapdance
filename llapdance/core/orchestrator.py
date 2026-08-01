"""Ties backend lifecycle + plugins + storage together (SPEC.md §5, §9, §11).
This is the only module that should know about *all* the plugin kinds at
once - individual adapters never talk to each other directly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from llapdance.config.models import BackendConfig, TestSuite
from llapdance.core.probe import DeviceInfo, discover_devices, free_vram_mb
from llapdance.core.result import RunResult
from llapdance.plugins import registry
from llapdance.plugins.base import StorageAdapter


class StartupTimeoutError(RuntimeError):
    """Backend never became healthy within startup_timeout_s."""


def _wait_until_ready(endpoint: str, health_path: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(endpoint.rstrip("/") + health_path, timeout=5)
            if resp.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(2)
    raise StartupTimeoutError(
        f"backend did not become healthy at {endpoint}{health_path} within {timeout_s}s "
        f"(last error: {last_error})"
    )


class VramPreflightError(RuntimeError):
    """Raised when a targeted device's free VRAM can't be confirmed, or is
    too low - SPEC.md §7 requires failing closed here, never guessing."""


@dataclass
class RunOutcome:
    result: RunResult
    delta_against: RunResult | None


def _resolve_devices(suite: TestSuite) -> list[DeviceInfo]:
    target = suite.device_target
    if target.mode.value == "none":
        return []
    discrete = [d for d in discover_devices() if not d.integrated]
    if target.mode.value == "indices":
        chosen = [d for d in discrete if d.index in target.indices]
        missing = set(target.indices) - {d.index for d in chosen}
        if missing:
            raise ValueError(f"requested device indices not found among discrete devices: {sorted(missing)}")
        return chosen
    return discrete  # all_discrete


def _apply_engine_translator(backend: BackendConfig, device: DeviceInfo | None) -> dict[str, Any]:
    """Layers an EngineTranslator's generated command/env/devices onto a
    backend's raw config, with anything the user set explicitly winning
    for that field (SPEC.md's raw passthrough stays the escape hatch)."""
    backend_dict = backend.model_dump()
    if not backend.engine:
        return backend_dict

    translator = registry.get("engine", backend.engine)()
    invocation = translator.build(
        model_path=backend.model_path or "",
        # shared + backend_specific merged into one flat namespace for the
        # translator - the shared/backend_specific split in config is for a
        # suite author's own organization (which knobs are cross-backend
        # concepts vs. this-engine-only), not something a translator needs
        # to care about when reading its own params back out.
        params={**backend.params.shared, **backend.params.backend_specific},
        port=backend.port,
        device=device,
    )
    if not backend_dict["command"]:
        backend_dict["command"] = invocation.command
    if not backend_dict["devices"]:
        backend_dict["devices"] = invocation.devices
    if not backend_dict["post_start_requests"]:
        backend_dict["post_start_requests"] = invocation.post_start_requests
    backend_dict["env"] = {**invocation.env, **backend_dict["env"]}  # user env wins on key conflicts
    return backend_dict


def _run_post_start_requests(endpoint: str, requests: list[dict[str, Any]]) -> None:
    for req in requests:
        method = req.get("method", "POST")
        path = req["path"]
        resp = httpx.request(method, endpoint.rstrip("/") + path, json=req.get("json"), timeout=120)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"post_start_request {method} {path} failed ({resp.status_code}): {resp.text[:500]}"
            )


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
    devices = _resolve_devices(suite)
    device_indices = [d.index for d in devices]
    _vram_preflight(device_indices, min_free_mb=suite.min_free_vram_mb, allow_unknown=suite.allow_unknown_vram)

    # Only the first resolved device is handed to an EngineTranslator - both
    # reference engines (llama.cpp, qxmx) only support pinning to one GPU
    # each; multi-GPU split is out of scope for this translation layer
    # (see VALIDATION.md).
    backend_dict = _apply_engine_translator(backend, devices[0] if devices else None)

    image_ref = execution.build(backend_dict)
    running = execution.start(backend_dict, image_ref, device_indices)
    try:
        _wait_until_ready(running.endpoint, backend.health_path, backend.startup_timeout_s)
        _run_post_start_requests(running.endpoint, backend_dict["post_start_requests"])

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
    # Record the FINAL command/env/devices (post-translator), not the
    # possibly-empty raw config - this is what actually ran.
    recorded_config = {
        **backend.model_dump(mode="json"),
        "command": backend_dict["command"],
        "env": backend_dict["env"],
        "devices": backend_dict["devices"],
        "post_start_requests": backend_dict["post_start_requests"],
    }
    result = RunResult(
        backend_name=backend.name,
        backend_config=recorded_config,
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
