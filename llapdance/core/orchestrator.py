"""Ties backend lifecycle + plugins + storage together (SPEC.md §5, §9, §11).
This is the only module that should know about *all* the plugin kinds at
once - individual adapters never talk to each other directly.
"""
from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from llapdance.config.models import BackendConfig, ExecutionTargetConfig, TestSuite
from llapdance.config.sweep import expand_suite_sweep
from llapdance.core.probe import CommandRunner, DeviceInfo, LocalRunner, SSHRunner, discover_devices, free_vram_mb
from llapdance.core.result import RunResult
from llapdance.plugins import registry
from llapdance.plugins.base import RunningBackend, StorageAdapter

# Real gap found building the TUI (SPEC.md §13): this module had ZERO
# progress visibility of any kind - a caller (CLI, TUI, MCP) could not tell
# "building image" from "waiting for health check" from "running
# coherence" without reading logs by hand. `on_event` is a plain string
# callback, fired at real stage transitions - opt-in (defaults to a no-op)
# so every existing caller is unaffected.
EventCallback = Callable[[str], None]


def _noop_event(_message: str) -> None:
    pass


class StartupTimeoutError(RuntimeError):
    """Backend never became healthy within startup_timeout_s."""


def _wait_until_ready(endpoint: str, health_path: str, timeout_s: float, on_event: EventCallback = _noop_event) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    attempt = 0
    on_event(f"waiting for health check at {endpoint}{health_path} (timeout {timeout_s:.0f}s)...")
    while time.monotonic() < deadline:
        attempt += 1
        try:
            resp = httpx.get(endpoint.rstrip("/") + health_path, timeout=5)
            if resp.status_code == 200:
                on_event(f"healthy after {attempt} attempt(s)")
                return
        except httpx.HTTPError as exc:
            last_error = exc
        if attempt % 5 == 0:
            on_event(f"still waiting for health check (attempt {attempt})...")
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


def _run_adapters_with_telemetry(
    suite: TestSuite, endpoint: str, on_event: EventCallback = _noop_event
) -> tuple[list, list, list]:
    """Runs benchmark + coherence adapters bracketed by telemetry
    start()/stop() (SPEC.md §5 - telemetry is a third concern, distinct
    from throughput and correctness). Shared by both the normal and
    external-backend paths - telemetry doesn't care whether there's a
    container of ours behind the endpoint or not."""
    started = [
        (registry.get("telemetry", ref.adapter)(ref.config), ref.config) for ref in suite.telemetry_adapters
    ]
    if started:
        on_event(f"starting telemetry: {', '.join(r.adapter for r in suite.telemetry_adapters)}")
    handles = [(adapter, adapter.start(config)) for adapter, config in started]

    benchmarks = []
    for ref in suite.benchmark_adapters:
        on_event(f"running benchmark: {ref.adapter}...")
        result = registry.get("benchmark", ref.adapter)(ref.config).run(endpoint, ref.config)
        on_event(f"benchmark {ref.adapter} done: {result.metrics}")
        benchmarks.append(result)

    coherence = []
    for ref in suite.coherence_adapters:
        on_event(f"running coherence: {ref.adapter}...")
        result = registry.get("coherence", ref.adapter)(ref.config).run(endpoint, ref.config)
        on_event(f"coherence {ref.adapter} done: {result.passed}/{result.total} passed")
        coherence.append(result)

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


def run_backend(suite: TestSuite, backend: BackendConfig, on_event: EventCallback = _noop_event) -> RunOutcome:
    if backend.source.mode.value == "external":
        return _run_external_backend(suite, backend, on_event)

    on_event("resolving device(s)...")
    runner = _make_runner(suite.execution_target)
    execution = registry.get("execution", _execution_adapter_name(suite.execution_target))(
        suite.execution_target.model_dump()
    )
    devices = _resolve_devices(suite, runner)
    for d in devices:
        on_event(f"device resolved: {d.name} (index {d.index}, render_node {d.render_node})")
    device_indices = [d.index for d in devices]
    on_event("checking free VRAM...")
    _vram_preflight(device_indices, runner, min_free_mb=suite.min_free_vram_mb, allow_unknown=suite.allow_unknown_vram)

    # Only the first resolved device is handed to an EngineTranslator - both
    # reference engines (llama.cpp, qxmx) only support pinning to one GPU
    # each; multi-GPU split is out of scope for this translation layer
    # (see VALIDATION.md).
    backend_dict = _apply_engine_translator(backend, devices[0] if devices else None)

    on_event(f"preparing image ({backend.source.mode.value})...")
    image_ref = execution.build(backend_dict)
    on_event(f"image ready: {image_ref}")
    on_event("starting container...")
    running = execution.start(backend_dict, image_ref, device_indices)
    try:
        _wait_until_ready(running.endpoint, backend.health_path, backend.startup_timeout_s, on_event)
        if backend_dict["post_start_requests"]:
            on_event("running post-start requests...")
        _run_post_start_requests(running.endpoint, backend_dict["post_start_requests"])

        benchmarks, coherence, telemetry = _run_adapters_with_telemetry(suite, running.endpoint, on_event)
    finally:
        on_event("stopping container...")
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
    on_event("done")

    return RunOutcome(result=result, delta_against=previous[0] if previous else None)


def _run_external_backend(
    suite: TestSuite, backend: BackendConfig, on_event: EventCallback = _noop_event
) -> RunOutcome:
    """No build/start/stop at all - SPEC.md's 'test a backend that's
    already loaded' case (e.g. through llm-proxy). No device probing/VRAM
    preflight either: this harness manages none of the GPU allocation for
    an already-running process, so there is nothing of ours to preflight.
    `backend.device_note` (unverified, free text) is the only device
    identity captured - see _device_target_result's `verified` flag."""
    running = _ExternalRunningBackend(backend.source.endpoint)
    benchmarks, coherence, telemetry = _run_adapters_with_telemetry(suite, running.endpoint, on_event)

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
    on_event("done")

    return RunOutcome(result=result, delta_against=previous[0] if previous else None)


def run_suite(suite: TestSuite, on_event: EventCallback = _noop_event) -> list[RunOutcome]:
    # Expansion happens here, not at load time - get_suite/list_suites (CLI
    # and MCP) show the compact sweep-spec a suite author wrote; only an
    # actual run sees the expanded cartesian product (SPEC.md §10).
    #
    # Real bug found from direct user feedback: this used to be a bare
    # list comprehension - if ANY one backend raised (a container crash,
    # a connection refused, a bad model load), the exception propagated
    # straight out and every backend's result was discarded, including
    # ones that already ran successfully before the failure. For a sweep
    # of N combinations, one bad combination meant zero results for all N,
    # not N-1 - no summary, no partial data, nothing. Each backend now
    # runs in its own try/except: a failure is reported via `on_event`
    # (visible in the TUI's live log / CLI's echoed events) and skipped,
    # the rest of the sweep continues.
    expanded = expand_suite_sweep(suite)
    outcomes: list[RunOutcome] = []
    for backend in expanded.backends:
        try:
            outcomes.append(run_backend(expanded, backend, on_event))
        except Exception as exc:
            on_event(f"[{backend.name}] FAILED, skipping: {exc}")
    return outcomes


def best_outcome(outcomes: list[RunOutcome]) -> tuple[float, str, int] | None:
    """Real user question, direct feedback: for a multi-backend/sweep run,
    how do you tell which combination had the best result? Shared here so
    the CLI's `run` command and the TUI's `RunScreen` (each building its
    own detailed per-outcome view - the CLI's includes telemetry, the
    TUI's uses colored verdicts) agree on what "best" means.

    Ranks by real per-backend throughput - preferring a genuine decode-
    only `avg_tg_tokens_per_sec` (see generic_http.py's PP/TG split) over
    the blended `avg_tokens_per_sec` when both exist, since TG is the more
    comparable number across a sweep of otherwise-identical configs. Only
    backends whose coherence check (if any) fully passed are eligible -
    a faster but wrong answer isn't a real win, just a faster wrong
    answer. Returns `(throughput, backend_name, comparable_count)`, or
    `None` when there's nothing to rank (a single outcome, or no
    comparable throughput metric anywhere)."""
    ranked: list[tuple[float, str]] = []
    for outcome in outcomes:
        result = outcome.result
        all_passed = all(c.passed == c.total for c in result.coherence) if result.coherence else None
        if all_passed is False:
            continue
        for b in result.benchmarks:
            throughput = b.metrics.get("avg_tg_tokens_per_sec", b.metrics.get("avg_tokens_per_sec"))
            if throughput is not None:
                ranked.append((throughput, result.backend_name))
    if len(outcomes) <= 1 or not ranked:
        return None
    best_throughput, best_name = max(ranked, key=lambda r: r[0])
    return best_throughput, best_name, len(ranked)
