# LLAPDANCE

**LL**M **A**utomated **P**ipeline for **D**eployment, **A**nalysis a**N**d **C**oherence **E**valuation

Orchestrates building/starting/stopping LLM inference engine containers under different configurations, runs pluggable benchmark and coherence/quality checks against them, and stores results for cross-run comparison. Every piece — which engines, which benchmark/coherence tool, where results go, which GPU, which machine it runs on — is config, not hardcoded. Full design rationale: [SPEC.md](./SPEC.md).

Status: early build (v0.1) — thin vertical slice + TUI + MCP server + a real per-engine params translation layer + a real OpenSearch storage adapter + a real telemetry harness + a real SSH remote execution target + an already-loaded/no-lifecycle backend mode + real sweep automation + a real image catalog + a real model catalog, validated end-to-end against real Intel Arc GPU hardware (local and remote) running four different backends (llama.cpp, qxmx, Arcaine, OpenArc), a real git-clone-and-build-from-source run, and a real already-loaded backend hit through the user's own OpenAI-compatible proxy (2026-08-01). See [SPEC_REVIEW.md](./SPEC_REVIEW.md) for an honest "are we still on track" assessment. See "What's implemented" below and [VALIDATION.md](./VALIDATION.md) for the full writeup, bugs found, and breadcrumbs for building the remaining adapters.

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
| `examples/validation-ssh-remote.suite.yaml` | llama.cpp/SYCL | `execution_target.mode: ssh` — real remote host, smaller GPU, `lspci`-fallback device discovery |
| `examples/validation-external.suite.yaml` | (already-loaded model) | `source.mode: external` — no container lifecycle at all, hits an already-running model through the user's own OpenAI-compatible proxy |

## What's implemented (v0.1)

- Config schema (`llapdance/config/models.py`) matching SPEC.md §4/§6/§7/§8/§9 — backend-as-config, test suites, device targets, execution targets, storage config.
- Plugin contracts + registry (`llapdance/plugins/base.py`, `registry.py`) for benchmark, coherence, storage, and execution-target adapters.
- Reference adapters:
  - `local-docker` execution target (build from git or pull prebuilt, start/stop, image list/remove). Supports `command` (raw CLI args), `volumes` (read-only bind mounts), and `devices` (host device passthrough, e.g. `/dev/dri` for GPU access) on `BackendConfig` — all three were added mid-build once a real GPU container turned out to need them (see VALIDATION.md).
  - `ssh-docker` execution target — real remote host over SSH, via raw `ssh`+`docker` CLI calls (not docker-py's `ssh://` transport, which requires `paramiko` and offers no clean way to pin a specific identity file — see VALIDATION.md). **Validated**: full build/start/benchmark/coherence/stop cycle against a real remote host, including a real hardware-probing gap found and fixed (see `probe.py` below). `source.mode: build` is **not** supported remotely yet — `prebuilt` only.
  - `generic-http` benchmark adapter — TTFT/throughput prober against any OpenAI-compatible endpoint. This, not `llama-benchy`, is the one guaranteed to work out of the box (see below). **Validated with real numbers against five different real servers/paths.** Token-counting was a real bug for a while (see below) — `_completion_token_count` now prefers authoritative per-request counts (`usage.completion_tokens`, engine-specific `metrics.new_token`/`timings.predicted_n`) over a raw SSE-line count, and records which method was actually used (`counted_via`) in the stored result.
  - `fixed-questions` coherence adapter — 10-question set, keyword match with LLM-judge fallback via a generic OpenAI-compatible client. **Validated (10/10) across five different real backends/paths.**
  - `pinbench` coherence adapter — shells out to a real external tool ([PinBench](https://github.com/ShadyHippo/PinBench), pinyin-to-character structured-output benchmark), not vendored - a user-supplied local checkout (`pinbench_dir`) is run via its own `run_benchmark.py`. **Validated live**, twice (direct adapter call + full CLI), against a real running backend.
  - `flat-file` storage adapter — the only always-on default per SPEC.md §8. **Validated**: write + prior-run delta lookup both exercised for real, including for `source.mode: external` backends.
- **Reference engine translators** (`llapdance/plugins/engine/`) — the per-engine "wrapper" the spec originally envisioned: `EngineTranslator` plugin kind, generating `command`/`env`/`devices`/`post_start_requests` from `BackendConfig.engine` + merged `params` + the resolved GPU device, instead of hand-writing raw CLI args per suite. **Five** reference implementations, all validated end to end against real hardware:
  - `llama-cpp-sycl`, `qxmx` — see the per-engine table above.
  - `arcaine` — fully env-var-driven (`MODEL_PATH`, `MAX_SEQ`, `DEFAULT_MAX_TOKENS`, plus diffusion/MoE-specific `DENOISING_STEPS`/`DEFAULT_SEED`/`LAYER_PLACEMENT`/`EXPERT_PLACEMENT`), never emits `command`.
  - `openarc` — introduced `EngineInvocation.post_start_requests`: fires `POST /openarc/load` after the container's health check passes but before benchmark/coherence adapters run, since OpenArc starts with no model loaded at all (a fundamentally different lifecycle than the other three "bake the model into the start command" engines).
  - `vllm` — built from a real running production container's (`vllm-urak`) actual `docker inspect` output, not guessed. Unlike every other engine here, needs the *whole* `/dev/dri` directory device-passed-through (a single render node crashes oneCCL's device enumeration) plus a separate `/dev/dri/by-path` bind mount - both found via real crashes during validation, not anticipated up front. **Validated**: clean run against a raw HF bf16 `gemma3` checkpoint no other engine here can load, using the *published* `intel/vllm:latest` image (not the unpublished fork the CLI/env shape was originally read from) to confirm the translator depends only on the shared contract.
  - Normalized params covered so far: `context_size`, `batch_size` (llama.cpp only), `kv_cache_quant` (`f16`/`q8_0`/`f8`, translated to each engine's own value spelling), `parallel_slots`, `reasoning` (llama.cpp only — see VALIDATION.md for a real bug this caught). Anything set explicitly in `command`/`env`/`devices`/`post_start_requests` still overrides what a translator generates for that field.
- Orchestrator core tying the above together, including GPU hardware probing (`llapdance/core/probe.py` — Intel via `xpumcli` (preferred, gives real free-VRAM + stable PCI bus id), `clinfo` (fallback, enumeration only), or `lspci` (last-resort fallback for a host with no compute-runtime tooling at all — identification only, structurally excludes non-Intel/non-NVIDIA chips like server-BMC graphics; confirmed necessary on a real remote host, see VALIDATION.md), plus NVIDIA via `nvidia-smi`) with a **real, validated** VRAM preflight check, and a startup health-poll (`_wait_until_ready`) so benchmark/coherence adapters don't fire before the model has finished loading. Every probing function now takes an explicit `CommandRunner` (local or SSH), so discovery works identically for a remote execution target — this is what makes the `ssh-docker` adapter above actually work.
- **Full GPU device identity tracking** — `RunResult.device_target` carries vendor/name/PCI-bus-id/render-node per device (not just an index), plus a `verified` flag distinguishing "actually probed" from "suite-author-claimed" (the latter only ever applies to `source.mode: external` backends' free-text `device_note`). Local runs also record the real hostname, not just `None`.
- **`source.mode: build`, validated for real** (local execution target only) — actual git clone (not `~/`-hardcoded, a real remote into a real scratch path) + `docker build`, not just `prebuilt`. Image tags now include the resolved git commit SHA (`llapdance/<name>:<ref>-<sha>`) for real build-version tracking, and `local_docker.py::build()` refuses to `git checkout` over an existing clone directory with uncommitted changes (a real safety gap found and fixed while validating this).
- **`source.mode: external`** — test an already-running, already-loaded backend with no container of ours built/started/stopped at all (e.g. through an existing OpenAI-compatible proxy). **Validated** against a real already-loaded model reached through the user's own proxy project.
- **`opensearch` storage adapter** (`llapdance/plugins/storage/opensearch.py`) — opt-in (not a core dependency; `opensearch-py` is imported lazily and only required if a suite actually selects this adapter — install with `pip install -e ".[opensearch]"`). Validated end to end against a real local OpenSearch 3.7.0 instance, including catching a real bug: dynamic mapping defaults a Unix-epoch `timestamp` field to 32-bit `float`, which can't represent it precisely enough to distinguish two runs seconds apart — fixed with an explicit index mapping (`double`, plus explicit `keyword` types for the fields queried by delta lookup).
- **`telemetry` plugin kind + `xmxmon` reference adapter** (`llapdance/plugins/telemetry/xmxmon.py`) — a genuinely different contract from benchmark: brackets `start()`/`stop()` around a run, watching hardware counters rather than hitting the endpoint. **Validated**, and caught a real "watching the wrong physical GPU" mismatch live — see VALIDATION.md.
- **MCP server** (`llapdance/mcp/server.py`, `llapdance mcp`) — 8 tools (`list_adapters`, `list_suites`, `get_suite`, `run_suite`, `get_results`, `describe_engine`, `list_images`/`label_image`/`remove_image`, `list_models`), all wrapping the same orchestrator/catalog functions the CLI uses. **Validated with a real stdio client** (the official `mcp` SDK) — install with `pip install -e ".[mcp]"`.
- **Sweep/parameter-matrix automation** (SPEC.md §10) — `BackendConfig.sweep` (a list of `{param, values}` axes) expands into the cartesian product of concrete backend configs at run time (`llapdance/config/sweep.py`). **Validated**: a real 2-value sweep produced 2 real automatically-generated container runs, both 10/10 coherence, distinct results. See `examples/validation-sweep.suite.yaml`. The mechanism has no special-casing per config section — `params.shared.x`, raw engine env flags (`env.X`), and even build-time cmake flags (`source.build.build_args.X`) all sweep identically. **Validated live** for a raw GGML runtime flag (`env.GGML_OP_OFFLOAD_MIN_BATCH`, confirmed via `docker exec` that a real container received its distinct value); build-arg sweeping is only unit-tested so far, not exercised against a real (slow) from-source rebuild.
- **Engine sweepable-params catalog** — `EngineTranslator.sweepable_params` (translator-consumed params) + `known_env_flags` (raw engine/library env vars, e.g. GGML/oneDNN toggles, swept via `env.<NAME>` — same generic mechanism), populated for all 4 reference engines by reading each engine's actual source, exposed via `llapdance describe-engine <name>` / the `describe_engine` MCP tool. `qxmx` has no oneDNN dependency at all (confirmed, from-scratch engine); `llama-cpp-sycl`'s oneDNN toggle (`GGML_SYCL_DNNL`) is a build-time cmake flag; `Arcaine`'s (`DIFF_ONEDNN_SDPA`) is a runtime env var — same underlying library, different toggle mechanism per engine. Cataloging `openarc` surfaced and fixed a real gap: its translator never forwarded `runtime_config` (OpenArc's actual OpenVINO-tuning surface) at all — fixed and **validated live** (a real `PERFORMANCE_HINT` override loaded successfully against the real server).
- **Image catalog & cleanup** (SPEC.md §12, `llapdance/core/catalog.py`, `llapdance images list/label/rm`) — wraps `ExecutionTargetAdapter.list_images()` (already implemented by both execution targets since session one, never previously consumed), enriched with labels + cross-referenced run history from flat-file storage. **Validated** against the real, still-growing local image sprawl — labeled a real validated image, confirmed persistence, confirmed the good-label removal safety check (refuses without `force=True`) against a disposable test tag, never touching the real sprawl.
- **Model catalog + format/backend compatibility** (new, not originally in SPEC.md; `llapdance/core/model_catalog.py`, `llapdance models <dir>...`) — scans directories for GGUF/OpenVINO-IR/HF-safetensors models, reports a quant hint (parsed from real `config.json`/`openvino_config.json` fields where present) and which registered engines could plausibly load each — a could-run-on signal, never a will-run guarantee. **Validated against ground truth**: all three previously-validated cross-format models matched exactly (Bonsai GGUF → llama-cpp-sycl/qxmx, Phi-4 OpenVINO-IR → openarc, diffusiongemma safetensors → arcaine).
- CLI (`llapdance run/adapters/describe-engine/images/models/tui/mcp`) and a Textual TUI browsing `*.suite.yaml` files in the working directory.

## Known gaps / not yet built

- `llama-benchy` and `guidellm` benchmark adapters are both **documented stubs that raise `NotImplementedError`** — real API/library limitations found for each (see `VALIDATION.md`), not guessed at.
- ~~No remote (SSH) execution target yet~~ — **built and validated**, `prebuilt` only (no remote build-from-source yet).
- No embedded-DB or Prometheus/Grafana storage adapters yet — `flat-file` (default) and `opensearch` (opt-in) are the only two. Image/model catalog labels also only live in flat-file for now, not mirrored into OpenSearch when that adapter is active.
- VRAM-free-memory detection works for NVIDIA (`nvidia-smi`) and Intel (`xpumcli`, falling back to `clinfo` enumeration-only, falling back to `lspci` identification-only on a host with no compute-runtime tooling at all — confirmed necessary on a real remote host). Any tier below `xpumcli` means free-VRAM is unknown and `allow_unknown_vram: true` is required. AMD is still unimplemented.
- GPU index spaces don't correspond across tools — confirmed on real hardware, not theoretical (see VALIDATION.md). **Five** non-corresponding spaces now found (clinfo, xpumcli, SYCL/level-zero, DRM render-node, OpenArc/OpenVINO's `GPU.N`, and `lspci`'s own bare enumeration order) — and the telemetry adapter adds a sixth in practice (xmxmon's own device numbering), with no automatic reconciliation between "which device the backend targets" and "which device telemetry watches" (caught live — see VALIDATION.md). `DeviceInfo.pci_bus_id` and `DeviceInfo.render_node` exist specifically to give something stable to reconcile against; `DeviceInfo.index` is only meaningful within whichever discovery source produced it.
- ~~No sweep/parameter-matrix automation~~ — **built and validated**, see above.
- ~~No image-catalog/labeling~~ — **built and validated**, see above.
- Sweeping is backend-param-only so far — device-target sweeping (e.g. "run this same config across every discovered GPU") isn't built.
- No web UI yet — TUI + CLI only.
- Multi-GPU expert/layer placement (Arcaine's `LAYER_PLACEMENT`/`EXPERT_PLACEMENT`) is raw-passthrough only — `EngineTranslator.build()` only ever resolves one device per backend.
- ~~No MCP integration yet~~ — **built and validated**, see above.

These are exactly the open items from SPEC.md §15, now reflected in code rather than left as prose. See [SPEC_REVIEW.md](./SPEC_REVIEW.md) for a full assessment of whether the project is still on track.

## Development

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```
