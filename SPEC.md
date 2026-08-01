# LLAPDANCE — Spec (v2)

**LLAPDANCE**: **LL**M **A**utomated **P**ipeline for **D**eployment, **A**nalysis a**N**d **C**oherence **E**valuation

Status: draft for build kickoff
Date: 2026-08-01

## 0. Design principle: portable by construction

This is not a spec for one machine. Nothing in this document should assume: a specific home directory, a specific GPU model/count/vendor, a specific user's private backend project, a specific container network name, or a fixed list of which test tools exist. Anywhere the harness needs to know something about the environment, it must **discover it at runtime or take it as config** — never hardcode it.

Concretely, that means:
- Hardware (GPUs, vendor, count, topology) is *probed*, not assumed.
- Home/working directories, git remotes, and image sources are *config*, not defaults tied to one account.
- Any backend/service the harness talks to is described generically by the *contract it satisfies* (e.g. "exposes an OpenAI-compatible endpoint"), never by one user's product name for their instance of that contract.
- Networking, storage, and remote-vs-local execution are all switches in config, defaulting to the least-coupled option (isolated/local/flat-file), with anything more integrated as explicit opt-in.

If moving to a different machine, swapping GPU vendors, adding/removing a GPU, changing users, or pointing at a remote host over SSH requires editing this spec or the harness's core code rather than editing a config file, that's a bug in the design.

## 1. Goal

A standalone, containerized software package (web UI + TUI) for orchestrating LLM inference engine testing: build/start/stop engine containers under different configurations, run pluggable benchmark and coherence/quality checks across them, and store results for cross-run comparison — all driven by config, not hardcoded per deployment.

## 2. Scope decision history (why this shape)

- Original idea: delegate orchestration to an existing agent-with-tool-access, to keep an interactive chat session token-light.
- Revised: the bulk of this work (container lifecycle, config sweeps, coherence scaffolding, bench parsing, image cataloging) is mechanical and repeatable — a poor fit for an agent reasoning loop (slow, nondeterministic, costly per action), a good fit for deterministic code with a UI.
- Final: build it as real software. Reserve LLM calls (via a generic OpenAI-compatible client) for the narrow judgment-requiring tasks only: coherence LLM-judge fallback, and cross-run analysis/fact-check summaries. Agent delegation is out of the primary path for this project.

## 3. Prior art (checked 2026-08-01, do not reinvent)

These are examples of tools that already solve a piece of this — reference points, not dependencies and not an exhaustive or privileged list. Any of them, or a user's own tool, should be pluggable the same way (see §5).

| Tool | Status | Relevance |
|---|---|---|
| [guidellm](https://github.com/vllm-project/guidellm) (vLLM project) | Active, Q3 2026 roadmap (new CLI/UI, eval support) | Backend-agnostic load/perf testing against OpenAI-compatible endpoints (TTFT, ITL, throughput). Tried integrating for real: its `synthetic_text` data source always resolves a tokenizer via `AutoTokenizer.from_pretrained(model_name)` with no override, which fails against any served model name that isn't a real HF Hub repo id — true of most of what this harness tests (custom quants, local builds, proxy-aliased names). Shipped as a documented stub, same pattern as llama-benchy, not a working adapter. |
| genai-perf (NVIDIA/Triton) | **Deprecated** — NVIDIA directs users to **AIPerf** | Don't build against genai-perf; AIPerf is the live successor if an NVIDIA-ecosystem workflow needs it. |
| [promptfoo](https://github.com/promptfoo) | Active, MIT, acquired by OpenAI Mar 2026 | Its `providers` + `assertions` YAML shape is solid prior art for adapter/coherence config schema — borrow the shape, not the dependency (corp-owned project, direction risk). |
| [PinBench](https://github.com/ShadyHippo/PinBench) | **Wired up and validated live** (2026-08-01) - `llapdance/plugins/coherence/pinbench.py`, external tool (not vendored), shells out to a local checkout's own `run_benchmark.py`. Real run against `llama-cpp-bonsai`'s `Ternary-Bonsai-27B-Q2_0.gguf` confirmed the full path end to end (see VALIDATION.md). | Structured-output/instruction-compliance benchmark, fixed case set, hits OpenAI-compatible/vLLM endpoints. Fits the coherence/quality slot, not the perf slot. |
| `llama-benchy` | In active local use | Perf/throughput benchmark tool — one pluggable option, not the built-in default. |
| `xmxmon` | In active local use | Telemetry/monitoring tool — **the reference telemetry adapter, built and validated** (see §5). Real GPU hardware-counter daemon; its capture-to-file endpoint isn't consumed (samples only ever hit a file inside its own container, never the API) — the adapter uses its rolling-window snapshot endpoint instead. |

None of these do the full stack this spec covers (container lifecycle + hardware-aware preflight + pluggable benchmark/coherence/telemetry + pluggable storage + remote execution + web/TUI). That gap is real; building it is not wasted motion.

## 4. Core concept: a "backend" is a configuration, not a fixed list

There is no built-in enum of engines. A **backend definition** is a config object describing one thing under test:

```yaml
backend:
  name: my-engine-config          # user-chosen label
  source:
    mode: build | prebuilt        # build from source, or use an existing image
    build:
      repo: <git url>             # only if mode: build
      ref: <branch/tag/commit>
      path: <clone target dir>    # configurable, no default tied to a user/home
      dockerfile: <path>
      build_args: {...}
    prebuilt:
      image: <image ref>          # only if mode: prebuilt
    external:
      endpoint: <base url>        # only if mode: external - already running, no lifecycle at all
  model:
    ref: <model id/path/url>
  params:
    shared: {context_size, batch_size, quant, gpu_split, ...}   # normalized cross-backend knobs
    backend_specific: {...}        # anything that doesn't map to a shared concept
  endpoint_contract: openai-compatible   # what the harness can assume to talk to it
```

Requirements:
- The harness must support testing **a single backend config** or **multiple backend configs** in one invocation.
- Any test run must be able to **pull prior result(s)** for the same or a related backend config and show a **delta** (perf and coherence), regardless of which storage adapter is active (works even flat-file-only).
- Build-vs-prebuilt-vs-external is a per-backend-config choice, not a global mode. `external` (built and validated) means "already running, no lifecycle of the harness's own at all" — no build/start/stop, and §7's hardware-preflight requirements below do not apply, since there is no container of the harness's own to preflight.

## 5. Architecture — plugin contracts, not hardcoded tools

Six plugin kinds (revised from the original four-box sketch once telemetry
was actually integrated and turned out not to behave like a benchmark
adapter at all — see below):

```
┌───────────────────────────────────────────────────────────────┐
│                       Web UI  /  TUI  /  MCP server              │
├───────────────────────────────────────────────────────────────┤
│                       Orchestrator Core                          │
│  - backend-config lifecycle (build/start/stop, or external)        │
│  - sweep / test-suite runner                                       │
│  - hardware probing & preflight (skipped for external backends)     │
│  - image catalog & labeling                                         │
│  - result delta lookup                                              │
├──────────┬──────────┬──────────┬──────────┬──────────┬───────────┤
│Execution │  Engine  │Benchmark │Coherence │Telemetry │  Storage    │
│ Target   │Translator│ Adapter  │ Adapter  │ Adapter  │ Adapter(s)  │
│ Adapter  │ (plugin: │ (plugin: │ (plugin: │ (plugin: │ (0+ active) │
│(local /  │per-engine│ llama-   │ fixed-Q/ │ xmxmon/  │(plugin:flat/│
│ remote   │ command/ │ benchy/  │ PinBench/│ custom - │embedded-db/ │
│ via SSH) │env/device│ guidellm/│ custom)  │ brackets │ opensearch/ │
│          │generator)│ custom)  │          │ a run,   │ prometheus/ │
│          │          │          │          │ doesn't  │ custom)     │
│          │          │          │          │ hit the  │             │
│          │          │          │          │ endpoint)│             │
└──────────┴──────────┴──────────┴──────────┴──────────┴───────────┘
                              │
                        LLM-judge utility
                (generic OpenAI-compatible client;
                 used only for coherence-judge fallback
                 and cross-run analysis summaries)
```

- **Execution target adapter**: where containers actually run. Options: local docker socket, or a remote host reachable over SSH (key-based auth). This makes "test on a different machine" a config change, not a redeploy. Same backend-config definitions apply regardless of target. Does not apply to `source.mode: external` backends (§4) — nothing of the harness's own runs there.
- **Engine translator**: the per-engine "wrapper" the original spec envisioned (§4's `params.shared` → concrete invocation) — a distinct plugin kind, not folded into the execution target. Generates `command`/`env`/`devices` from normalized params + the resolved GPU device; raw passthrough on the backend config itself remains the escape hatch for anything a translator doesn't cover.
- **Benchmark adapter**: plugin contract — "run this measurement against this endpoint with this config, return normalized metrics." llama-benchy, guidellm, are reference implementations (perf/throughput specifically); a user's own tool is a first-class option, not a fallback.
- **Coherence adapter**: same plugin contract shape, for correctness/output-quality checks rather than perf. Fixed-question-set-with-LLM-judge-fallback and PinBench-style structured-output grading are both reference implementations.
- **Telemetry adapter**: a genuinely different contract from benchmark, not a variant of it — it brackets `start()`/`stop()` around whatever benchmark/coherence adapters run, watching hardware (GPU utilization, power, memory bandwidth) rather than making requests against the endpoint itself. xmxmon is the reference implementation. The original architecture sketch folded this into "Telemetry/Benchmark" as one combined slot; that turned out to be wrong once a real telemetry tool was integrated, hence the split.
- **Storage adapter(s)**: see §8 — zero or more active at once, all opt-in beyond flat-file.
- **LLM-judge utility**: a thin client against *any* endpoint that satisfies the OpenAI-compatible contract. The spec does not name or assume a particular backend project for this — that's private config on whoever deploys the harness.

## 6. Deployment model

- Ships as its own container.
- Container networking is a config switch, applying **both** to the harness's own container and to any backend containers it spins up:
  - `disabled` — no extra network, default bridge only.
  - `enabled` + `network: <name>` — attach to a named existing network (e.g. to reach sibling services), name supplied by the deployer, never hardcoded.
  - `isolated` — explicitly no access beyond what's required for the harness's own operation.
- Requires access to its execution target (local docker socket, or SSH credentials for a remote target) — configured per §5, not assumed.

## 7. Hardware discovery & GPU management

No hardware topology is assumed. At startup (and on demand), the harness probes for available accelerators through whatever vendor-appropriate mechanism is available (e.g. vendor CLI/driver interfaces for Intel/NVIDIA/AMD), and builds a device inventory: vendor, model, index/bus-id, and — critically — whether a device is integrated or discrete.

Requirements:
- **Never target an integrated GPU.** This is enforced by classification during probing (integrated vs. discrete), not by a hardcoded device list — so it holds true regardless of which machine the harness runs on.
- GPU target is a **test parameter**, settable per backend-config or per test suite (§9) — single GPU, multiple GPUs, or "all discovered discrete GPUs." Nothing is hardcoded to a particular device.
- **VRAM preflight is mandatory** before any bench/coherence run starts *for backends the harness itself builds/starts* (`source.mode: build` or `prebuilt`, §4): confirm the targeted device(s) have free memory headroom before load is placed on them. Mechanism must be pluggable per vendor (different vendors expose this differently); if no mechanism is available for a given vendor/device, the harness must fail closed (refuse to run) rather than proceed blind — repeatedly running an inference workload against an already-saturated card has caused hangs in practice. Does not apply to `source.mode: external` backends — there is no container of the harness's own to preflight; the harness manages none of that GPU's allocation.
- Device pinning mechanism (e.g. environment variables, driver-specific selectors, `--gpus` equivalents) is chosen per vendor at runtime based on what the probed device requires — not fixed to one vendor's convention.
- Architecture must support GPUs changing over time (added, removed, swapped, different vendor) without editing core code — only the config/probe layer should notice the difference.
- Architecture must support a remote execution target having entirely different hardware than the local machine — probing happens against whichever execution target is active (§5).

## 8. Storage adapter(s)

**Default: flat file.** No database of any kind is required to use the harness — every run's results are always written to flat files (JSON/CSV) as the baseline, guaranteed path.

Everything beyond flat file is **opt-in**, and multiple can be enabled simultaneously:
- **Embedded lightweight DB** (e.g. SQLite or similar) — optional, for local querying without standing up an external service.
- **External document/search store** (e.g. OpenSearch) — optional, for cross-run/cross-model/cross-backend graphing and comparison at scale.
- **Metrics scrape endpoint** (e.g. Prometheus-scrapeable, feeding Grafana) — optional, for live dashboarding during a run.
- **Custom adapter** — the storage interface is documented/open so a user can write their own (a different DB, a different observability stack, etc.).

None of the non-flat options ship enabled by default. A user turns on what they want in their own config, based on what they already run. The harness must never assume OpenSearch, Prometheus, or any specific external service exists.

Every stored result (regardless of adapter) must carry full run context: backend config (source/build-or-prebuilt/model/params), execution target used, device(s) targeted, telemetry/benchmark tool + its raw output, coherence/quality result. Telemetry tools generally don't capture this context themselves — the harness is responsible for attaching it.

## 9. Test suites

A **test suite** is a saved, named, reusable definition combining:
- one or more backend configs (§4),
- GPU/device target(s) (§7),
- which telemetry/benchmark adapter(s) to run,
- which coherence/quality adapter(s) to run,
- which storage adapter(s) to write to.

Suites are how a repeatable "these backends, these models, on this GPU, with these bench/telemetry outputs" workflow gets defined once and re-run on demand — via config file for anything meant to be saved/repeated, with CLI-flag overrides for quick one-off variations. Suites are portable: nothing in a suite definition should reference a fixed path, device index, or network name that only makes sense on one machine — those are resolved from the environment at run time.

## 10. Sweep / parameter matrix

**Built and validated.** `BackendConfig.sweep` (a list of `{param, values}` axes, `llapdance/config/models.py`) expands into the cartesian product of concrete backend configs at `run_suite()` time (`llapdance/config/sweep.py`) — a real automated matrix, not a hand-authored suite file per variant (every comparison run before this was the latter). `param` is a dotted path into the backend's own config dict (e.g. `params.shared.context_size`); each combination gets a distinguishing name (`<backend>--<param>_<value>--...`) so results stay traceable per-variant. Validated live: one backend config, one axis (`context_size: [2048, 4096]`), two real container runs produced automatically, both 10/10 coherence, distinct stored results.

- Sweep dimensions include backend-specific parameters (not shared across all backends) and GPU/device target(s) — device selection is itself sweepable (test the same config across multiple discovered devices). Device-target sweeping specifically is not yet built — only backend-param axes are; device sweeping would need `TestSuite`/`DeviceTarget`-level axes, not just `BackendConfig`-level ones.
- Where a concept is shared across backends (context size, batch size, quant, GPU split, etc.), express it once in the normalized `params.shared` block (§4); backend-only params live in `params.backend_specific`. `llapdance describe-engine <name>` / the `describe_engine` MCP tool now catalog which params each registered `EngineTranslator` actually reads (SPEC.md's own "cataloging build switches to sweep") — built for all four reference engines.
- Rebuild/reconfigure is not limited to "pull latest source and build" — the same trigger path drives sweeps and truth tables (repeated variants of a config to compare effect on speed/quality) without necessarily touching source.

## 11. Unit of work

Per backend-config, per invocation:
1. **Telemetry/benchmark run** — via whichever benchmark adapter(s) are configured for the suite.
2. **Coherence/quality check** — via whichever coherence adapter(s) are configured; existing reference default is a fixed question set graded by string/keyword match first, LLM-judge fallback for ambiguous cases — specifically to catch cases where output is fluent-looking garbage (numerical errors, driver-bug artifacts, garbled tokens) that a throughput number alone won't reveal.

Granularity:
- Single backend-config invocation (build/start, run adapters, stop, report) is supported standalone.
- A test suite (§9) runs the full matrix across its configured backends/devices/adapters in one invocation.

## 12. Image catalog & cleanup

**Built and validated.** `llapdance/core/catalog.py` + `llapdance images list/label/rm` (CLI) + `list_images`/`label_image`/`remove_image` (MCP) — wraps `ExecutionTargetAdapter.list_images()` (already implemented by both `local-docker` and `ssh-docker`, never previously consumed anywhere), enriched with any label and cross-referenced against stored `RunResult.image_ref` history.

- The harness enumerates images it has built/knows about (per backend-config), and lets a user label/identify good vs. outdated/failed builds, and clean up the latter. Validated against the real, still-growing local image sprawl (`qxmx:*`, `llama-cpp-*`, `llapdance/*`) referenced in §16 — listed it, labeled a real validated image `good`, confirmed persistence, confirmed `remove_image` refuses a `good`-labeled image without `force=True` and succeeds with it (tested against a disposable tag created for the purpose, not the real sprawl).
- Every stored result retains enough metadata (image tag/digest, full param set) to trace a result back to the exact image that produced it — this is what makes "which build was actually good, and with what settings" answerable later, this is expected to get be a common problem as the number of experimental builds under test grows over time. Confirmed working: `_index_runs_by_image()` reads exactly this metadata back out of flat-file results.
- Labeling mechanism should follow whatever storage adapter(s) are active: if only flat-file is enabled, labels live in flat metadata; if a DB/search adapter is enabled, it can be the queryable source of truth; a lightweight docker-image-label mirror is optional convenience, not required. Built for flat-file only (a small `_image_labels.json` alongside a suite's results) — labels are not yet mirrored into OpenSearch when that adapter is active; flat-file remains the source of truth for labels regardless of what else is enabled.

## 12a. Model catalog & backend compatibility (new — not originally in this spec)

**Built and validated.** `llapdance/core/model_catalog.py` + `llapdance models <dir>...` (CLI) + `list_models` (MCP). Scans directories recursively for models (GGUF files, OpenVINO IR directories, HF-safetensors directories — correctly handling org/contributor-namespaced nesting, e.g. `OpenVINO/droans/qwen3.5-9B-int4-ov/`, found to be a real layout on this box, not assumed) and reports, per model: detected format, a best-effort quant hint (parsed from `config.json`'s real `quantization_config` field when present, filename/dirname regex fallback otherwise), and which registered `EngineTranslator`s could plausibly load it based on format alone.

**Explicitly a could-run-on signal, never a will-run guarantee** — a corrupt file, a quant an engine's translator actually rejects (e.g. `llama-cpp-sycl`'s `f8` KV-cache rejection), or a model too large for available VRAM all pass this check and still fail at runtime; format compatibility is necessary, not sufficient. Validated against all three cross-format models already validated as real backends in prior sessions, confirming the mapping matches reality exactly: `Ternary-Bonsai-27B-Q2_0.gguf` → gguf → `[llama-cpp-sycl, qxmx]`; `Phi-4-mini-instruct-int4-ov` → openvino_ir → `[openarc]`; `diffusiongemma-26B-A4B-it-NVFP4` → safetensors → `[arcaine]`.

## 13. UI

- **Web UI**: browse image catalog, build/edit test suites, trigger sweeps, view comparison graphs (when a graphing-capable storage adapter is enabled), review coherence results, view deltas against prior runs.
- **TUI**: same core operations (spin up/down, run adapters, trigger suite) for terminal-only use; graphing views are not required to be duplicated in TUI.
- **MCP integration — built and validated.** `llapdance/mcp/server.py`, 5 tools (`list_adapters`, `list_suites`, `get_suite`, `run_suite`, `get_results`), all calling straight into the same orchestrator functions the CLI uses. Validated with a real stdio client (the official `mcp` SDK), including a full `run_suite` execution pulled back afterward via `get_results`. Real gotcha for anyone building the next MCP tool: list-returning tools' results land in `structured_content["result"]`, not as JSON text in `content[0]`.

## 14. Out of scope for v1

- Delegating orchestration to a conversational agent (superseded by this design).
- Shipping a specific GPU vendor's tooling as a hard dependency — vendor support is added via the probing/pinning plugin layer (§7) as needed, starting with whatever hardware is available to validate against first, but the architecture itself is vendor-agnostic from the start.
- AIPerf integration specifically — noted as a future telemetry-adapter candidate, not built now.

## 15. Open decisions blocking full build-readiness

1. Embedded lightweight DB choice for the optional local-DB storage adapter (e.g. SQLite vs. an alternative) — needs a decision before that adapter is built.
2. ~~Exact VRAM-preflight mechanism per GPU vendor~~ — resolved for Intel (`xpumcli` → `clinfo` → `lspci` tiers, built and validated) and NVIDIA (`nvidia-smi`). AMD still open.
3. ~~Remote execution target auth/config shape~~ — resolved and built (`ssh-docker` adapter, explicit identity-file path, no reliance on `~/.ssh/config` or agent state). `source.mode: build` is not yet supported over that adapter, only `prebuilt`.
4. Exact plugin interface (function signatures / IPC boundary) for third-party telemetry, coherence, and storage adapters — needs a decision before the plugin contract can be documented for outside contributors.
5. ~~Sweep/parameter matrix automation (§10) is not built~~ — **resolved and built.** `BackendConfig.sweep` + `expand_suite_sweep()`, validated with a real 2-value sweep against a live engine.
6. ~~Image catalog & cleanup (§12) is not built at all~~ — **resolved and built.** `llapdance images list/label/rm`, validated against real image sprawl.

## 16. Reference deployment (example only — not part of the spec)

The following describes one specific environment used to validate this design during scoping. None of it should be assumed by the harness itself — it's here only as a concrete example / initial test target, kept separate from the spec on purpose so it doesn't leak into "how the harness must work."

- Local dev box has 3 discrete Intel Arc Pro GPUs (2×B70, 1×B50) plus one integrated GPU, discoverable via `clinfo -l`; the integrated device must be excluded by the classification rule in §7, not by name.
- A large number of pre-existing experimental image tags exist locally already (e.g. `llama-cpp-intel:*`, `llama-cpp-sycl:*`, `llama-cpp-bonsai:*`, `qxmx:*`), which is the motivating case for §12.
- OpenSearch is already running locally as one available storage-adapter target; Prometheus/Grafana-style scraping is a second option under consideration; flat-file remains the guaranteed default regardless.
- A private OpenAI-compatible backend proxy is used locally as "the" LLM endpoint for the LLM-judge utility — this is a deployer-side config value, not something the spec names or depends on.
- One user's own container-to-container network exists locally for reaching sibling AI services — an example of the `enabled` + `network: <name>` case in §6, not a default.
- 4 "arcane" containers/images were referenced during scoping but were not found in the current local `docker ps -a`/`docker images` snapshot — worth reconciling locally before relying on them as an existing artifact, but irrelevant to the spec itself.
