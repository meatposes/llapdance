# LLAPDANCE

**LL**M **A**utomated **P**ipeline for **D**eployment, **A**nalysis a**N**d **C**oherence **E**valuation

Orchestrates building/starting/stopping LLM inference engine containers under different configurations, runs pluggable benchmark and coherence/quality checks against them, and stores results for cross-run comparison. Every piece — which engines, which benchmark/coherence tool, where results go, which GPU, which machine it runs on — is config, not hardcoded. Full design rationale: [SPEC.md](./SPEC.md).

Status: early build (v0.1) — thin vertical slice + TUI, validated end-to-end against a real Intel Arc GPU running a real llama.cpp container (2026-08-01). Not everything in SPEC.md is implemented yet; see "What's implemented" below and [VALIDATION.md](./VALIDATION.md) for the full writeup, bugs found, and breadcrumbs for building the remaining adapters.

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

Copy `examples/example.suite.yaml` and edit it for a real backend — nothing in it is a usable default, every value needs to be set for your environment. `examples/validation.suite.yaml` is a real, working suite (the one used for the 2026-08-01 validation run) — closer to a working reference than the generic example, though its device/volume paths are specific to that machine.

## What's implemented (v0.1)

- Config schema (`llapdance/config/models.py`) matching SPEC.md §4/§6/§7/§8/§9 — backend-as-config, test suites, device targets, execution targets, storage config.
- Plugin contracts + registry (`llapdance/plugins/base.py`, `registry.py`) for benchmark, coherence, storage, and execution-target adapters.
- Reference adapters:
  - `local-docker` execution target (build from git or pull prebuilt, start/stop, image list/remove) — local docker socket only, no remote-SSH target yet. Supports `command` (raw CLI args), `volumes` (read-only bind mounts), and `devices` (host device passthrough, e.g. `/dev/dri` for GPU access) on `BackendConfig` — all three were added mid-build once a real GPU container turned out to need them (see VALIDATION.md).
  - `generic-http` benchmark adapter — TTFT/throughput prober against any OpenAI-compatible endpoint. This, not `llama-benchy`, is the one guaranteed to work out of the box (see below). **Validated with real numbers against a real llama.cpp server.**
  - `fixed-questions` coherence adapter — 10-question set, keyword match with LLM-judge fallback via a generic OpenAI-compatible client. **Validated (10/10) against a real server**, and indirectly caught a real tokenizer-crash bug in one model/build combination during earlier validation iterations.
  - `flat-file` storage adapter — the only always-on default per SPEC.md §8. **Validated**: write + prior-run delta lookup both exercised for real.
- Orchestrator core tying the above together, including GPU hardware probing (`llapdance/core/probe.py`, Intel-OpenCL + NVIDIA so far) with a hard-fail-closed VRAM preflight check, and a startup health-poll (`_wait_until_ready`) so benchmark/coherence adapters don't fire before the model has finished loading.
- CLI (`llapdance run/adapters/tui`) and a Textual TUI browsing `*.suite.yaml` files in the working directory.

## Known gaps / not yet built

- `llama-benchy` adapter is a **stub that raises `NotImplementedError`** — see `llapdance/plugins/benchmark/llama_benchy.py`. It exposes only a Flask dashboard with no discoverable REST API; wrapping it needs its real API confirmed first, not guessed at.
- No remote (SSH) execution target yet — `execution_target.mode: ssh` validates in config but has no adapter registered.
- No embedded-DB, OpenSearch, or Prometheus/Grafana storage adapters yet — only `flat-file`.
- VRAM-free-memory detection is NVIDIA-only right now (`nvidia-smi`); Intel devices are enumerated and integrated-vs-discrete classified, but free-VRAM reporting isn't wired up (`core/probe.py: free_vram_mb`), so Intel runs fail closed unless `allow_unknown_vram: true` is explicitly set in the suite.
- No image-catalog/labeling UI yet (SPEC.md §12).
- No web UI yet — TUI + CLI only.
- No translation from `params.shared` (normalized cross-backend knobs) into a concrete `command`/`env` per engine yet — `BackendConfig.command`/`env` are raw passthrough today. This is the next real piece of work; see VALIDATION.md "Gaps found" for where it plugs in.

These are exactly the open items from SPEC.md §15, now reflected in code rather than left as prose.

## Development

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```
