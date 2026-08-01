# Validation run — 2026-08-01

First real end-to-end test of LLAPDANCE against actual hardware: build/start a
llama.cpp container on a real Intel Arc GPU, run the benchmark + coherence
adapters against it, store the result, confirm teardown, confirm config-only
modularity. Suite used: `examples/validation.suite.yaml`.

## Result: it works

```
=== bonsai-validation (12cea5493a10) ===
  [generic-http] {'avg_ttft_ms': 201.8, 'avg_total_ms': 1034.1, 'avg_tokens_per_sec': 36.5, 'requests': 3.0}
  [fixed-questions] 10/10 passed
```

- Container built (resolved from local cache since `mode: prebuilt`), started with real GPU access, health-polled, benchmarked, coherence-checked, torn down — cleanly, no leftover container (`docker ps -a` confirmed clean after every run).
- Ran it twice; the second run's `RunOutcome.delta_against` correctly pointed at the first run's `RunResult` — the flat-file storage adapter's delta lookup works, not just theoretical.
- Re-ran with `--set benchmark_adapters.0.config.num_requests=1 --set benchmark_adapters.0.config.prompt="Say hello."` and no file edits — result changed (`requests: 1.0`, faster total time) confirming config, not code, drives behavior. This also surfaced and fixed a real bug (see below).
- The already-running production `llama-cpp-bonsai` container was never touched — confirmed via its own healthcheck before/after.

## Bug found in our own code: `--set` couldn't index into lists

`_deep_merge` treated `--set benchmark_adapters.0.config.x=y` as `{"benchmark_adapters": {"0": {...}}}` and merged that dict directly onto the real list, which Pydantic then rejected (`Input should be a valid list`). Fixed in `llapdance/config/loader.py::_deep_merge`: when the override at a given key is a dict with all-digit keys and the base value is a list, it's now treated as index assignment. Covered by `tests/test_config.py::test_deep_merge_indexes_into_lists`.

**Breadcrumb:** this only surfaced by actually running the CLI against a real suite with a list-valued override — the existing unit tests only exercised dict/dict merging. Worth remembering when adding new config fields: exercise the actual CLI path, not just the Pydantic model, before calling an adapter "done."

## Bug found in the test suite itself, mid-build

Adding the new startup-readiness poll (`_wait_until_ready`, see below) broke the existing orchestrator unit tests — they hung for up to 120s because the fake test backend's endpoint (`http://fake:8000`) isn't a real host and the poll kept retrying until `startup_timeout_s`. Fixed by monkeypatching `_wait_until_ready` to a no-op in orchestrator unit tests (`tests/test_orchestrator.py`), and added a dedicated `tests/test_startup_wait.py` that mocks `httpx.get` directly to test the polling/timeout logic in isolation. **Breadcrumb:** any new blocking-network-call added to orchestrator core needs either a fake-friendly seam or its own isolated test — don't let it hide inside a test that's supposed to be about something else.

## Retracted claim: NOT a llama.cpp bug — invalid test setup on my part

An earlier version of this doc claimed `Ternary-Bonsai-27B-dspark-Q4_1.gguf` crashing on inference was a "real bug caught by the coherence check." That was wrong, and the correction matters: **`-dspark` names a speculative-decode draft/auxiliary artifact, not a standalone servable model.** Loading it alone as the main model was an invalid test configuration on my part, not a discovery about llama.cpp or this image. The SIGSEGV it produced (`GGML_ASSERT(tokenizer && ...)`) is the expected failure mode of feeding the server a file it was never meant to run standalone — there is no confirmed bug in `llama-cpp-bonsai:meat6-hardened` from this session. Left this section in, corrected, rather than deleting it silently, because the mistake (not checking what a `-dspark`-suffixed artifact actually is before treating it as a normal quant) is the thing worth remembering, not the false "bug found."

## What this proves about "modular enough to be configurable"

- Swapped nothing in code to point at a different model/image/GPU/port/benchmark-prompt — all suite YAML + one CLI `--set` override.
- The execution adapter, benchmark adapter, and coherence adapter never referenced each other directly; the orchestrator was the only thing that knew about all three, exactly per SPEC.md §5's intent.
- Registering a second `local-docker`-shaped SSH-based execution adapter, or a second benchmark adapter, requires only implementing the ABC in `llapdance/plugins/base.py` and calling `register()` — proven by how easily the test suite's `FakeExecutionTarget`/`FakeBenchmark`/`FakeCoherence` slotted in without touching orchestrator code.

## Gaps found that required real code changes (not just config)

These were missing entirely before this validation pass — the spec anticipated the categories but the v0.1 build hadn't implemented the mechanics yet:

1. **`BackendConfig.command`** (raw CLI args) — llama.cpp's SYCL server takes model path, GPU device, and context size as CLI flags (`-m`, `-dev`, `-c`...), not environment variables. The harness had no way to pass a container command at all before this. Added as a raw list-of-strings passthrough; there is still no translation from `params.shared` (the normalized cross-backend knobs from SPEC.md §4) into a concrete command — that per-engine "wrapper" (SPEC.md's original vision: "each engine wrapper knows what params it accepts and translates them") is the next real piece of work, not yet built. Breadcrumb for whoever builds the first real per-engine adapter: this is where that translation logic goes, likely one small module per engine family (`llama_cpp_sycl.py`, `vllm_xpu.py`, etc.) that takes `BackendParams` and emits the `command`/`env` a raw `BackendConfig` needs today.
2. **`BackendConfig.volumes`** — mounting the actual model file into the container. Read-only bind mounts only; no rw/tmpfs case has come up yet.
3. **`BackendConfig.devices`** — **this was the actual blocker**, not GPU selection env vars. Without `--device /dev/dri:/dev/dri` passed through, the container has zero GPU visibility and SYCL throws `No device of requested type available` — a *different* error than a bad device index would produce, and easy to misdiagnose as a device-selector problem instead of a missing-passthrough problem. Breadcrumb: **check device passthrough before debugging device *selection*** — the symptom (no device found) looks identical either way, but the fix is completely different (`--device` flag vs. an env var).
4. **`_wait_until_ready` / startup health-poll** — the orchestrator previously assumed a container was immediately ready to serve traffic the instant `docker run` returned. Real model loading takes real time (several seconds even for a small quant); every real benchmark/coherence request would have hit connection-refused without this. Added a polling wait against `backend.health_path` with `startup_timeout_s`. Breadcrumb: any future execution-target adapter (e.g. the not-yet-built SSH one) needs the same contract — `RunningBackend.endpoint` must be a URL the orchestrator can poll, readiness is not the execution adapter's job to guarantee before returning.

## GPU targeting: what's confirmed, what's still open

- `clinfo -l` (OpenCL) enumeration order and llama.cpp's SYCL/level-zero device order are **separate index spaces that are not guaranteed to correspond** — `core/probe.py`'s `discover_devices()` reports OpenCL-order indices (used for VRAM preflight / device-target recording), while the actual pinning that worked used `ONEAPI_DEVICE_SELECTOR=level_zero:N` supplied directly in `BackendConfig.env`, a completely separate enumeration. This is exactly SPEC.md §7's flagged open decision, now confirmed as a real problem rather than a theoretical one — the two numbering schemes must eventually be reconciled (or explicitly documented as unreconciled) before "GPU2" can mean one consistent thing across the probe layer and the actual pinning mechanism.
- Concretely confirmed today: `level_zero:0` = an Intel Arc Pro B70 with only ~3.7GB free at the time (something else was already using it) — still enough headroom for this small quant to load. `level_zero:1` is whatever the already-running production `llama-cpp-bonsai` container is pinned to (untouched, not re-verified today).
- Intel free-VRAM detection is still unimplemented (`probe.py::free_vram_mb` returns `None` for Intel, fail-closed per SPEC.md §7) — today's run only proceeded because `allow_unknown_vram: true` was set after manually eyeballing the numbers above, exactly the escape hatch the spec described and exactly why it defaults to `false`.

## Second validation run — qxmx backend, same model (2026-08-01)

Goal: prove the harness is portable across engines by config alone, not by
re-testing llama.cpp again. Used `examples/validation-qxmx.suite.yaml`:
same `Ternary-Bonsai-27B-Q2_0.gguf` model, different image (`qxmx:latest`,
a from-scratch custom Intel-Arc inference engine at `~/qxmx`), different
container name/port/device, run alongside the still-untouched production
`llama-cpp-bonsai` container.

```
=== qxmx-validation (f8f3b7c36f8d) ===
  [generic-http] {'avg_ttft_ms': 506.4, 'avg_total_ms': 2485.1, 'avg_tokens_per_sec': 21.9, 'requests': 3.0}
  [fixed-questions] 10/10 passed
```

Worked end to end, same as the llama.cpp run — but only after fixing a real gap and finding two new gotchas:

### Real capability added: Intel VRAM detection (was the last run's unresolved gap)

`xpumcli` (Intel's own GPU management CLI, `/usr/bin/xpumcli`) is installed and gives PCI bus address + exact free/total VRAM per device in one JSON call (`xpumcli discovery -j` / `xpumcli discovery -d N -j`). This directly closes the "Intel free-VRAM detection unimplemented" gap flagged in the first validation run. `core/probe.py` now prefers `xpumcli` for Intel discovery (falling back to `clinfo`, enumeration-only, when xpumcli isn't installed), and `free_vram_mb()` returns a real number for Intel devices discovered this way instead of `None`.

**Confirmed working for real**, not just unit-tested: re-ran the qxmx suite with `--set min_free_vram_mb=999999` and got a real `VramPreflightError` reporting the actual free VRAM (`~25.2GB free, below the 999999MB minimum`) — the preflight check now has teeth on Intel hardware, not just the documented fail-closed placeholder from the last run.

### GOTCHA: at least 4 non-corresponding GPU index spaces exist for the same hardware

Confirmed concretely, not theoretically: the same 4 physical GPUs enumerate as different indices depending on which tool asks — `clinfo` (OpenCL) order, `xpumcli`'s own `device_id` (1-4, not 0-based), llama.cpp's SYCL/level-zero index (separate again, used in the first validation run), and the kernel's DRM card/render node number. **The only thing that ties them together reliably is the PCI bus address.** `core/probe.py`'s `DeviceInfo` now carries `pci_bus_id` and `render_node` (resolved via `/sys/class/drm/renderD*/device` symlinks, vendor-agnostic, no extra tool needed) specifically so future code has one stable key to reconcile across tools — this is the concrete version of SPEC.md §7's "GPU2 identity" open decision. Switching `core/probe.py` to prefer xpumcli also means `DeviceInfo.index` now means something different (xpumcli's 1-based `device_id`) than it did in the first validation run (clinfo's 0-based enumeration) — harmless pre-1.0, but would be a breaking change to any suite config written against the old numbering.

### GOTCHA: whether `command` needs the binary name depends on ENTRYPOINT vs CMD

`llama-cpp-bonsai` sets `ENTRYPOINT ["/app/llama-server"]`, so `BackendConfig.command` only needed the flags. `qxmx:latest` sets no `ENTRYPOINT`, only `CMD ["./build/qxmx_serve"]` — supplying `command` in docker-py *replaces* the default `Cmd` entirely rather than appending to an entrypoint, so the binary path itself had to be `command[0]`. Both look identical from the suite-YAML author's side (just a list of strings) but silently do different things depending on the image. Breadcrumb: **check `docker inspect --format '{{.Config.Entrypoint}}'` on any new backend image before writing its `command` list** — this was caught fast here only because the qxmx binary printed its usage banner to stdout, a broken build with a quieter failure mode could take longer to diagnose.

### GPU choice for this run

Picked device index 3 (xpumcli numbering) / PCI `0000:8a:00.0` / `/dev/dri/renderD131` deliberately — confirmed via `probe.discover_devices()` to be the least-loaded discrete card (~32.6GB free at the time) before writing the suite, specifically to avoid contending with the already-running production `llama-cpp-bonsai` (on a different card) or the heavily-loaded B50 (`0000:84:00.0`, was at ~16.3/16.3GB used, i.e. essentially full). Passed through as a single render node (`--device /dev/dri/renderD131:/dev/dri/renderD131`), not the whole `/dev/dri` directory — qxmx has no device-selector flag or env var at all (its README says "only one GPU is supported" right now), so **which render node(s) are passed through *is* qxmx's entire GPU-pinning mechanism**, not a hint alongside some other selector like llama.cpp's `ONEAPI_DEVICE_SELECTOR`.

## Updated adapter status (see README.md, now reflects reality instead of aspiration)

| Adapter | Status |
|---|---|
| `local-docker` execution | Real, validated against two different engines (llama.cpp + qxmx) on real GPU hardware. |
| `generic-http` benchmark | Real, validated against two different real servers — llama.cpp and qxmx both produced real TTFT/throughput numbers. |
| `fixed-questions` coherence | Real, validated — 10/10 against two different working backends. (An earlier draft of this doc claimed it also caught a "tokenizer crash bug" — retracted, see above; that crash was my invalid test setup, not a finding.) |
| `flat-file` storage | Real, validated — write + delta-lookup both exercised, across two backends. |
| Intel VRAM preflight | Real, validated — `xpumcli`-backed, confirmed to actually reject an impossible VRAM requirement with a real free-memory number, not just documented as a placeholder. |
| `llama-benchy` benchmark | Still a stub — unrelated to this validation, no new information. |
| SSH execution target | Not built. `RunningBackend`/`ExecutionTargetAdapter` contract exercised only locally twice; nothing contradicts it working remotely, but it's unverified. |
| OpenSearch / embedded-DB / Prometheus storage | Not built. |
| `params.shared` → per-engine `command`/`env` translation | Still not built — both validation runs used raw `command`/`env`/`devices` passthrough. Two real backends now exist as worked examples (`examples/validation.suite.yaml` for llama.cpp-SYCL, `examples/validation-qxmx.suite.yaml` for qxmx) to build that translation layer against. |
