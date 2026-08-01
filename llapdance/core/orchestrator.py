"""Ties backend lifecycle + plugins + storage together (SPEC.md §5, §9, §11).
This is the only module that should know about *all* the plugin kinds at
once - individual adapters never talk to each other directly.
"""
from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Any

import httpx

from llapdance.config.models import BackendConfig, ExecutionTargetConfig, TestSuite
from llapdance.config.sweep import expand_suite_sweep
from llapdance.core.probe import CommandRunner, DeviceInfo, LocalRunner, SSHRunner, discover_devices, free_vram_mb
from llapdance.core.result import RunResult
from llapdance.plugins import registry
from llapdance.plugins.base import RunningBackend, StorageAdapter


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


class _ExternalRunningBackend(RunningBackend):
    """Wraps an already-running, externally-managed endpoint (SPEC.md's
    'test a backend that's already loaded' case, e.g. through llm-proxy) -
    no container of ours exists, so there is nothing to build/start/stop
    and no logs to fetch."""

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def logs(self, tail: int = 200) -> str:
        return ""


def _make_runner(execution_target: ExecutionTargetConfig) -> CommandRunner:
    if execution_target.mode.value == "ssh":
        return SSHRunner(host=execution_target.host, user=execution_target.user, ssh_key_path=execution_target.ssh_key_path)
    return LocalRunner()


def _execution_adapter_name(execution_target: ExecutionTargetConfig) -> str:
    return "ssh-docker" if execution_target.mode.value == "ssh" else "local-docker"


def _resolve_devices(suite: TestSuite, runner: CommandRunner) -> list[DeviceInfo]:
    target = suite.device_target
    if target.mode.value == "none":
        return []
    discrete = [d for d in discover_devices(runner) if not d.integrated]
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


def _vram_preflight(device_indices: list[int], runner: CommandRunner, min_free_mb: float, allow_unknown: bool) -> None:
    if not device_indices:
        return
    devices = {d.index: d for d in discover_devices(runner)}
    for idx in device_indices:
        device = devices.get(idx)
        if device is None:
            raise VramPreflightError(f"device index {idx} not found during preflight re-probe")
        free = free_vram_mb(device, runner)
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


def _run_adapters_with_telemetry(suite: TestSuite, endpoint: str) -> tuple[list, list, list]:
    """Runs benchmark + coherence adapters bracketed by telemetry
    start()/stop() (SPEC.md §5 - telemetry is a third concern, distinct
    from throughput and correctness). Shared by both the normal and
    external-backend paths - telemetry doesn't care whether there's a
    container of ours behind the endpoint or not."""
    started = [
        (registry.get("telemetry", ref.adapter)(ref.config), ref.config) for ref in suite.telemetry_adapters
    ]
    handles = [(adapter, adapter.start(config)) for adapter, config in started]

    benchmarks = [
        registry.get("benchmark", ref.adapter)(ref.config).run(endpoint, ref.config)
        for ref in suite.benchmark_adapters
    ]
    coherence = [
        registry.get("coherence", ref.adapter)(ref.config).run(endpoint, ref.config)
        for ref in suite.coherence_adapters
    ]

    telemetry = [adapter.stop(handle) for adapter, handle in handles]
    return benchmarks, coherence, telemetry


def _build_storage_adapters(suite: TestSuite) -> list[StorageAdapter]:
    adapters: list[StorageAdapter] = [
        registry.get("storage", "flat-file")({"flat_file_dir": suite.storage.flat_file_dir})
    ]
    for ref in suite.storage.extra_adapters:
        adapters.append(registry.get("storage", ref.adapter)(ref.config))
    return adapters


def _device_target_result(suite: TestSuite, devices: list[DeviceInfo]) -> dict[str, Any]:
    """Full device identity, not just an index - this is what makes cross-run
    comparison across different physical GPUs (or GPU models) meaningful.
    `verified: True` means this came from actually probing hardware;
    `verified: False` (external backends) means it's whatever the suite
    author claimed, never to be trusted the same way."""
    return {
        "mode": suite.device_target.mode.value,
        "verified": True,
        "devices": [
            {
                "index": d.index,
                "vendor": d.vendor,
                "name": d.name,
                "pci_bus_id": d.pci_bus_id,
                "render_node": d.render_node,
            }
            for d in devices
        ],
    }


def run_backend(suite: TestSuite, backend: BackendConfig) -> RunOutcome:
    if backend.source.mode.value == "external":
        return _run_external_backend(suite, backend)

    runner = _make_runner(suite.execution_target)
    execution = registry.get("execution", _execution_adapter_name(suite.execution_target))(
        suite.execution_target.model_dump()
    )
    devices = _resolve_devices(suite, runner)
    device_indices = [d.index for d in devices]
    _vram_preflight(device_indices, runner, min_free_mb=suite.min_free_vram_mb, allow_unknown=suite.allow_unknown_vram)

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

        benchmarks, coherence, telemetry = _run_adapters_with_telemetry(suite, running.endpoint)
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
    execution_target_result = suite.execution_target.model_dump(mode="json")
    if suite.execution_target.mode.value == "local":
        # local mode never sets `host` in config - fill in the real
        # hostname so runs are still comparable/traceable across different
        # physical machines without requiring the suite author to name it.
        execution_target_result["host"] = socket.gethostname()

    result = RunResult(
        backend_name=backend.name,
        backend_config=recorded_config,
        image_ref=image_ref,
        execution_target=execution_target_result,
        device_target=_device_target_result(suite, devices),
        benchmarks=benchmarks,
        coherence=coherence,
        telemetry=telemetry,
    )

    previous = storages[0].previous_for(backend.name, limit=1)
    for storage in storages:
        storage.write(result)

    return RunOutcome(result=result, delta_against=previous[0] if previous else None)


def _run_external_backend(suite: TestSuite, backend: BackendConfig) -> RunOutcome:
    """No build/start/stop at all - SPEC.md's 'test a backend that's
    already loaded' case (e.g. through llm-proxy). No device probing/VRAM
    preflight either: this harness manages none of the GPU allocation for
    an already-running process, so there is nothing of ours to preflight.
    `backend.device_note` (unverified, free text) is the only device
    identity captured - see _device_target_result's `verified` flag."""
    running = _ExternalRunningBackend(backend.source.endpoint)
    benchmarks, coherence, telemetry = _run_adapters_with_telemetry(suite, running.endpoint)

    storages = _build_storage_adapters(suite)
    result = RunResult(
        backend_name=backend.name,
        backend_config=backend.model_dump(mode="json"),
        image_ref=None,
        execution_target={"mode": "external"},
        device_target={"mode": "external", "verified": False, "note": backend.device_note},
        benchmarks=benchmarks,
        coherence=coherence,
        telemetry=telemetry,
    )

    previous = storages[0].previous_for(backend.name, limit=1)
    for storage in storages:
        storage.write(result)

    return RunOutcome(result=result, delta_against=previous[0] if previous else None)


def run_suite(suite: TestSuite) -> list[RunOutcome]:
    # Expansion happens here, not at load time - get_suite/list_suites (CLI
    # and MCP) show the compact sweep-spec a suite author wrote; only an
    # actual run sees the expanded cartesian product (SPEC.md §10).
    expanded = expand_suite_sweep(suite)
    return [run_backend(expanded, backend) for backend in expanded.backends]
