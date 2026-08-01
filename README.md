# LLAPDANCE

**LL**M **A**utomated **P**ipeline for **D**eployment, **A**nalysis a**N**d **C**oherence **E**valuation

Orchestrates building/starting/stopping LLM inference engine containers under different configurations, runs pluggable benchmark and coherence/quality checks against them, and stores results for cross-run comparison. Every piece — which engines, which benchmark/coherence tool, where results go, which GPU, which machine it runs on — is config, not hardcoded. Full design rationale: [SPEC.md](./SPEC.md).

Status: early build (v0.1) — thin vertical slice + TUI + a real per-engine params translation layer + a real OpenSearch storage adapter, validated end-to-end against real Intel Arc GPU hardware running **four** different backends (llama.cpp, qxmx, Arcaine, OpenArc) and a real git-clone-and-build-from-source run (2026-08-01). Not everything in SPEC.md is implemented yet; see "What's implemented" below and [VALIDATION.md](./VALIDATION.md) for the full writeup, bugs found, and breadcrumbs for building the remaining adapters.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Usage

```bash
llapdance adapters                       # list available plugin adapters
llapdance run examples/example.suite.yaml       # run a suite
llapdance run examples/example.suite.yaml --set backends.0.model=other-model
llapdance tui                            # browse *.suite.yaml in cwd, run interactively
```

Copy `examples/example.suite.yaml` and edit it for a real backend — nothing in it is a usable default, every value needs to be set for your environment. Real, working suites used for the 2026-08-01 validation runs (device/volume paths specific to that machine, but otherwise real references, not toy examples):

| Suite | Engine | What it specifically demonstrates |
|---|---|---|
| `examples/validation.suite.yaml` | llama.cpp/SYCL | `ENTRYPOINT`+flags, env-var GPU selection turned out unnecessary (render-node passthrough alone suffices) |
| `examples/validation-qxmx.suite.yaml` | qxmx (custom engine) | `CMD`+full command (no `ENTRYPOINT`), render-node-only GPU passthrough |
| `examples/validation-arcaine.suite.yaml` | Arcaine (diffusion Gemma MoE) | fully env-var-driven invocation, no CLI args or `/health` endpoint at all |
| `examples/validation-openarc.suite.yaml` | OpenArc (OpenVINO IR models) | `post_start_requests` — model loading is a separate HTTP call after the container starts, not baked into start command/env |
| `examples/validation-build-from-source.suite.yaml` | qxmx | `source.mode: build` — real git clone + docker build, not `prebuilt` |
| `examples/validation-opensearch.suite.yaml` | qxmx | `storage.extra_adapters` — OpenSearch as an additional write target alongside the always-on flat-file default |

## What's implemented (v0.1)

- Config schema (`llapdance/config/models.py`) matching SPEC.md §4/§6/§7/§8/§9 — backend-as-config, test suites, device targets, execution targets, storage config.
- Plugin contracts + registry (`llapdance/plugins/base.py`, `registry.py`) for benchmark, coherence, storage, and execution-target adapters.
- Reference adapters:
  - `local-docker` execution target (build from git or pull prebuilt, start/stop, image list/remove) — local docker socket only, no remote-SSH target yet. Supports `command` (raw CLI args), `volumes` (read-only bind mounts), and `devices` (host device passthrough, e.g. `/dev/dri` for GPU access) on `BackendConfig` — all three were added mid-build once a real GPU container turned out to need them (see VALIDATION.md).
  - `generic-http` benchmark adapter — TTFT/throughput prober against any OpenAI-compatible endpoint. This, not `llama-benchy`, is the one guaranteed to work out of the box (see below). **Validated with real numbers against a real llama.cpp server.**
  - `fixed-questions` coherence adapter — 10-question set, keyword match with LLM-judge fallback via a generic OpenAI-compatible client. **Validated (10/10) against a real server.**
  - `flat-file` storage adapter — the only always-on default per SPEC.md §8. **Validated**: write + prior-run delta lookup both exercised for real.
- **Reference engine translators** (`llapdance/plugins/engine/`) — the per-engine "wrapper" the spec originally envisioned: `EngineTranslator` plugin kind, generating `command`/`env`/`devices`/`post_start_requests` from `BackendConfig.engine` + merged `params` + the resolved GPU device, instead of hand-writing raw CLI args per suite. **Four** reference implementations, all validated end to end against real hardware:
  - `llama-cpp-sycl`, `qxmx` — see the per-engine table above.
  - `arcaine` — fully env-var-driven (`MODEL_PATH`, `MAX_SEQ`, `DEFAULT_MAX_TOKENS`, plus diffusion/MoE-specific `DENOISING_STEPS`/`DEFAULT_SEED`/`LAYER_PLACEMENT`/`EXPERT_PLACEMENT`), never emits `command`.
  - `openarc` — introduced `EngineInvocation.post_start_requests`: fires `POST /openarc/load` after the container's health check passes but before benchmark/coherence adapters run, since OpenArc starts with no model loaded at all (a fundamentally different lifecycle than the other three "bake the model into the start command" engines).
  - Normalized params covered so far: `context_size`, `batch_size` (llama.cpp only), `kv_cache_quant` (`f16`/`q8_0`/`f8`, translated to each engine's own value spelling), `parallel_slots`, `reasoning` (llama.cpp only — see VALIDATION.md for a real bug this caught). Anything set explicitly in `command`/`env`/`devices`/`post_start_requests` still overrides what a translator generates for that field.
- Orchestrator core tying the above together, including GPU hardware probing (`llapdance/core/probe.py` — Intel via `xpumcli` (preferred, gives real free-VRAM + stable PCI bus id) or `clinfo` (fallback, enumeration only), plus NVIDIA via `nvidia-smi`) with a **real, validated** VRAM preflight check (confirmed to reject an actual over-budget request with a real free-memory number on Intel hardware), and a startup health-poll (`_wait_until_ready`) so benchmark/coherence adapters don't fire before the model has finished loading.
- **`source.mode: build`, validated for real** — actual git clone (not `~/`-hardcoded, a real remote into a real scratch path) + `docker build`, not just `prebuilt`. Image tags now include the resolved git commit SHA (`llapdance/<name>:<ref>-<sha>`) for real build-version tracking, and `local_docker.py::build()` refuses to `git checkout` over an existing clone directory with uncommitted changes (a real safety gap found and fixed while validating this).
- **`opensearch` storage adapter** (`llapdance/plugins/storage/opensearch.py`) — opt-in (not a core dependency; `opensearch-py` is imported lazily and only required if a suite actually selects this adapter — install with `pip install -e ".[opensearch]"`). Validated end to end against a real local OpenSearch 3.7.0 instance, including catching a real bug: dynamic mapping defaults a Unix-epoch `timestamp` field to 32-bit `float`, which can't represent it precisely enough to distinguish two runs seconds apart — fixed with an explicit index mapping (`double`, plus explicit `keyword` types for the fields queried by delta lookup).
- CLI (`llapdance run/adapters/tui`) and a Textual TUI browsing `*.suite.yaml` files in the working directory.

## Known gaps / not yet built

- `llama-benchy` adapter is a **stub that raises `NotImplementedError`** — see `llapdance/plugins/benchmark/llama_benchy.py`. It exposes only a Flask dashboard with no discoverable REST API; wrapping it needs its real API confirmed first, not guessed at.
- No remote (SSH) execution target yet — `execution_target.mode: ssh` validates in config but has no adapter registered.
- No embedded-DB or Prometheus/Grafana storage adapters yet — `flat-file` (default) and `opensearch` (opt-in) are the only two.
- VRAM-free-memory detection now works for both NVIDIA (`nvidia-smi`) and Intel (`xpumcli`) — Intel only falls back to fail-closed (`allow_unknown_vram: true` required) when `xpumcli` itself isn't installed, in which case discovery falls back to `clinfo` (enumeration only, no VRAM reporting). AMD is still unimplemented.
- GPU index spaces don't correspond across tools — confirmed on real hardware, not theoretical (see VALIDATION.md). Now **four** non-corresponding spaces found (clinfo, xpumcli, SYCL/level-zero, DRM render-node, and OpenArc/OpenVINO's own `GPU`/`GPU.0`/`GPU.1`). `DeviceInfo.pci_bus_id` and `DeviceInfo.render_node` exist specifically to give something stable to reconcile against; `DeviceInfo.index` is only meaningful within whichever discovery source produced it.
- No image-catalog/labeling UI yet (SPEC.md §12).
- No web UI yet — TUI + CLI only.
- **No MCP integration yet** (SPEC.md §13) — noted explicitly as future work so agents can push suites/pull results programmatically, not just human operators via CLI/TUI. Not built this session; the orchestrator's `run_suite`/`run_backend` are the operations an MCP layer would wrap.

These are exactly the open items from SPEC.md §15, now reflected in code rather than left as prose.

## Development

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```
