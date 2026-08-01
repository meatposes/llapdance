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
| [guidellm](https://github.com/vllm-project/guidellm) (vLLM project) | Active, Q3 2026 roadmap (new CLI/UI, eval support) | Backend-agnostic load/perf testing against OpenAI-compatible endpoints (TTFT, ITL, throughput). |
| genai-perf (NVIDIA/Triton) | **Deprecated** — NVIDIA directs users to **AIPerf** | Don't build against genai-perf; AIPerf is the live successor if an NVIDIA-ecosystem workflow needs it. |
| [promptfoo](https://github.com/promptfoo) | Active, MIT, acquired by OpenAI Mar 2026 | Its `providers` + `assertions` YAML shape is solid prior art for adapter/coherence config schema — borrow the shape, not the dependency (corp-owned project, direction risk). |
| [PinBench](https://github.com/ShadyHippo/PinBench) | Confirmed real, checked 2026-08-01 | Structured-output/instruction-compliance benchmark, fixed case set, hits OpenAI-compatible/vLLM endpoints. Fits the coherence/quality slot, not the perf slot. |
| `llama-benchy` | In active local use | Perf/throughput benchmark tool — one pluggable option, not the built-in default. |
| `xmxmon` | In active local use | Telemetry/monitoring tool — one pluggable option for the telemetry slot. |

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
- Build-vs-prebuilt is a per-backend-config choice, not a global mode.

## 5. Architecture — plugin contracts, not hardcoded tools

```
┌───────────────────────────────────────────────────┐
│                 Web UI  /  TUI                      │
├───────────────────────────────────────────────────┤
│                 Orchestrator Core                    │
│  - backend-config lifecycle (build/start/stop)        │
│  - sweep / test-suite runner                          │
│  - hardware probing & preflight                       │
│  - image catalog & labeling                           │
│  - result delta lookup                                │
├───────────┬───────────┬───────────┬─────────────────┤
│ Execution │ Telemetry/│ Coherence/│  Storage          │
│  Target   │ Benchmark │  Quality  │  Adapter(s)       │
│  Adapter  │  Adapter  │  Adapter  │  (0+ active)      │
│ (local /  │ (plugin:  │ (plugin:  │ (plugin: flat /   │
│  remote   │ llama-    │ fixed-Q/  │  embedded-db /    │
│  via SSH) │ benchy/   │ PinBench/ │  opensearch/      │
│           │ guidellm/ │ custom)   │  prometheus/      │
│           │ xmxmon/   │           │  custom)          │
│           │ custom)   │           │                   │
└───────────┴───────────┴───────────┴─────────────────┘
                    │
              LLM-judge utility
        (generic OpenAI-compatible client;
         used only for coherence-judge fallback
         and cross-run analysis summaries)
```

- **Execution target adapter**: where containers actually run. Options: local docker socket, or a remote host reachable over SSH (key-based auth). This makes "test on a different machine" a config change, not a redeploy. Same backend-config definitions apply regardless of target.
- **Telemetry/Benchmark adapter**: plugin contract — "run this measurement against this endpoint with this config, return normalized metrics." llama-benchy, guidellm, xmxmon are reference implementations; a user's own tool is a first-class option, not a fallback.
- **Coherence/Quality adapter**: same plugin contract, for correctness/output-quality checks rather than perf. Fixed-question-set-with-LLM-judge-fallback and PinBench-style structured-output grading are both reference implementations.
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
- **VRAM preflight is mandatory** before any bench/coherence run starts: confirm the targeted device(s) have free memory headroom before load is placed on them. Mechanism must be pluggable per vendor (different vendors expose this differently); if no mechanism is available for a given vendor/device, the harness must fail closed (refuse to run) rather than proceed blind — repeatedly running an inference workload against an already-saturated card has caused hangs in practice.
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

- Sweep dimensions include backend-specific parameters (not shared across all backends) and GPU/device target(s) — device selection is itself sweepable (test the same config across multiple discovered devices).
- Where a concept is shared across backends (context size, batch size, quant, GPU split, etc.), express it once in the normalized `params.shared` block (§4); backend-only params live in `params.backend_specific`.
- Rebuild/reconfigure is not limited to "pull latest source and build" — the same trigger path drives sweeps and truth tables (repeated variants of a config to compare effect on speed/quality) without necessarily touching source.

## 11. Unit of work

Per backend-config, per invocation:
1. **Telemetry/benchmark run** — via whichever benchmark adapter(s) are configured for the suite.
2. **Coherence/quality check** — via whichever coherence adapter(s) are configured; existing reference default is a fixed question set graded by string/keyword match first, LLM-judge fallback for ambiguous cases — specifically to catch cases where output is fluent-looking garbage (numerical errors, driver-bug artifacts, garbled tokens) that a throughput number alone won't reveal.

Granularity:
- Single backend-config invocation (build/start, run adapters, stop, report) is supported standalone.
- A test suite (§9) runs the full matrix across its configured backends/devices/adapters in one invocation.

## 12. Image catalog & cleanup

- The harness enumerates images it has built/knows about (per backend-config), and lets a user label/identify good vs. outdated/failed builds, and clean up the latter.
- Every stored result retains enough metadata (image tag/digest, full param set) to trace a result back to the exact image that produced it — this is what makes "which build was actually good, and with what settings" answerable later, this is expected to get be a common problem as the number of experimental builds under test grows over time.
- Labeling mechanism should follow whatever storage adapter(s) are active: if only flat-file is enabled, labels live in flat metadata; if a DB/search adapter is enabled, it can be the queryable source of truth; a lightweight docker-image-label mirror is optional convenience, not required.

## 13. UI

- **Web UI**: browse image catalog, build/edit test suites, trigger sweeps, view comparison graphs (when a graphing-capable storage adapter is enabled), review coherence results, view deltas against prior runs.
- **TUI**: same core operations (spin up/down, run adapters, trigger suite) for terminal-only use; graphing views are not required to be duplicated in TUI.
- **MCP integration (future, not built)**: this suite will need an MCP server surface so agents (not just human operators via web/TUI) can push new test suites/runs and pull back results programmatically. Noted here deliberately early — the orchestrator core (§5) and CLI already expose the operations an MCP server would wrap (`run_suite`/`run_backend`, adapter registry, storage query), so this should be a thin translation layer on top rather than a redesign, but it is explicitly out of scope for this build pass.

## 14. Out of scope for v1

- Delegating orchestration to a conversational agent (superseded by this design).
- Shipping a specific GPU vendor's tooling as a hard dependency — vendor support is added via the probing/pinning plugin layer (§7) as needed, starting with whatever hardware is available to validate against first, but the architecture itself is vendor-agnostic from the start.
- AIPerf integration specifically — noted as a future telemetry-adapter candidate, not built now.

## 15. Open decisions blocking full build-readiness

1. Embedded lightweight DB choice for the optional local-DB storage adapter (e.g. SQLite vs. an alternative) — needs a decision before that adapter is built.
2. Exact VRAM-preflight mechanism per GPU vendor (what's available without extra installs vs. what requires installing vendor tooling) — needs to be resolved per vendor as each is brought online, starting with whichever vendor is used for initial validation.
3. Remote execution target auth/config shape (SSH key path, host inventory format, per-host hardware caching vs. re-probe-every-run) — needs a decision before the remote execution-target adapter is built.
4. Exact plugin interface (function signatures / IPC boundary) for third-party telemetry, coherence, and storage adapters — needs a decision before the plugin contract can be documented for outside contributors.

## 16. Reference deployment (example only — not part of the spec)

The following describes one specific environment used to validate this design during scoping. None of it should be assumed by the harness itself — it's here only as a concrete example / initial test target, kept separate from the spec on purpose so it doesn't leak into "how the harness must work."

- Local dev box has 3 discrete Intel Arc Pro GPUs (2×B70, 1×B50) plus one integrated GPU, discoverable via `clinfo -l`; the integrated device must be excluded by the classification rule in §7, not by name.
- A large number of pre-existing experimental image tags exist locally already (e.g. `llama-cpp-intel:*`, `llama-cpp-sycl:*`, `llama-cpp-bonsai:*`, `qxmx:*`), which is the motivating case for §12.
- OpenSearch is already running locally as one available storage-adapter target; Prometheus/Grafana-style scraping is a second option under consideration; flat-file remains the guaranteed default regardless.
- A private OpenAI-compatible backend proxy is used locally as "the" LLM endpoint for the LLM-judge utility — this is a deployer-side config value, not something the spec names or depends on.
- One user's own container-to-container network exists locally for reaching sibling AI services — an example of the `enabled` + `network: <name>` case in §6, not a default.
- 4 "arcane" containers/images were referenced during scoping but were not found in the current local `docker ps -a`/`docker images` snapshot — worth reconciling locally before relying on them as an existing artifact, but irrelevant to the spec itself.
