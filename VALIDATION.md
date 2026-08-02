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

## Third validation run — params translation layer built and validated (2026-08-01)

Built the "per-engine wrapper" the original spec envisioned: `EngineTranslator` (`llapdance/plugins/base.py`), a new plugin kind (`registry.get("engine", ...)`), and two reference implementations - `llama-cpp-sycl` and `qxmx` (`llapdance/plugins/engine/`). A `BackendConfig.engine` name plus normalized `params.shared` now generates `command`/`env`/`devices`; anything set explicitly in `command`/`env`/`devices` still wins for that field (raw passthrough stays the escape hatch, not replaced).

Normalized params validated against both real engines: `context_size`, `batch_size` (llama.cpp only - qxmx has no batching flag, confirmed via its own usage banner, so the translator ignores it rather than inventing one), `kv_cache_quant` (`f16`/`q8_0`/`f8`, engines use different value spellings - `f16`→`fp16` for qxmx, and `f8` is rejected outright for llama.cpp since it has no fp8 KV cache type), `parallel_slots`, and `reasoning` (llama.cpp only, see the real bug below).

### GPU pinning simplified along the way

Went to verify whether llama.cpp's `ONEAPI_DEVICE_SELECTOR=level_zero:N` reliably maps to a specific physical card (the open question from the second validation run) and instead found something better: **restricting `--device` to a single render node is sufficient on its own** - with only one render node passed through, llama.cpp only ever sees one SYCL device and uses it, no selector env var needed at all. Confirmed empirically: passed only `/dev/dri/renderD131` with no `ONEAPI_DEVICE_SELECTOR`/`GGML_SYCL_VISIBLE_DEVICES`, and llama.cpp correctly reported exactly one device (`SYCL0`). This means GPU pinning is now **uniform across both engines** - render-node-scoped passthrough, resolved from `DeviceInfo.render_node` (added last session) - rather than qxmx using render nodes and llama.cpp using a separately-numbered vendor selector. Simpler, and doesn't depend on a mapping this session couldn't actually verify (attempted a live cross-check between `level_zero:N` and `xpumcli`'s per-device free-VRAM numbers; they didn't match closely enough to trust, most likely because VRAM usage on this shared box fluctuates from other processes between the two readings, not because the index mapping itself is wrong - but since it couldn't be verified cleanly, it isn't relied on).

Also noted along the way: llama.cpp's own self-reported free VRAM (in its startup log line) did not match `xpumcli`'s live reading for the same device at nearly the same moment (32.5GB vs 25.2GB) even when the render-node restriction guarantees they're looking at the *same* physical card - likely a difference in what each tool measures/when, not a device-identity problem. Breadcrumb: **don't cross-validate the VRAM preflight against an engine's own self-reported number** - trust the vendor tool (`xpumcli`) for the preflight check itself, and treat an engine's own log line as informational only.

### Real bug found: an incorrect assumption I made, not a pre-existing issue

The first pass of the `llama-cpp-sycl` translator omitted `--reasoning off` because an earlier validation run's comment guessed it was "likely specific to this hardened fork's model family, not stock llama.cpp." That guess was wrong, and running the translator-generated suite for real caught it immediately: benchmark numbers looked fine (39 tok/s), but **all 10 coherence questions failed with empty answers**.

Root cause, confirmed via `llama-server --help`: `--reasoning [on|off|auto]` (env `LLAMA_ARG_REASONING`) is a real, general upstream llama.cpp flag, default `auto` ("detect from template"). This model's chat template auto-enables thinking, so every response put its entire token budget into a hidden `message.reasoning_content` field while `message.content` came back empty (confirmed by hand: `finish_reason: "length"`, `content: ""`, `reasoning_content: "Thinking Process:..."`). The benchmark adapter didn't catch this because it only counts streamed tokens, never inspects content - **this is exactly why the coherence check exists as a separate concern from throughput benchmarking**, and it caught a real mistake I made, not a pre-existing bug in the target.

Fixed by adding `reasoning` as a proper normalized param (`llapdance/plugins/engine/llama_cpp_sycl.py`), left llama.cpp's own default (`auto`) as the translator's default (does not silently override it), and updated `examples/validation.suite.yaml` to set `reasoning: "off"` explicitly with a comment explaining why it's required for this model, not optional. Re-ran end to end after the fix: 10/10 coherence, confirmed via the actual stored result JSON that the generated command included `LLAMA_ARG_REASONING: off`.

**Breadcrumb: don't assert a flag is "probably fork-specific" or "probably not needed" without checking `--help` first** - it took under 10 minutes to confirm via `docker run --rm --device ... --entrypoint /app/llama-server <image> --help` once the failure was noticed, and would have taken zero minutes to just check before writing the claim into a comment in the first place.

Both engines confirmed working through the translator, end to end, for real:

```
=== bonsai-validation (23f41b0a8eb6) ===
  [generic-http] {'avg_tokens_per_sec': 37.0, ...}
  [fixed-questions] 10/10 passed

=== qxmx-validation (fe0a047f310c) ===
  [generic-http] {'avg_tokens_per_sec': 22.0, ...}
  [fixed-questions] 10/10 passed
```

### Bug found in this session's own test suite (again) - silently excluded tests

While adding unit tests for the two translators, `pyproject.toml`'s `python_classes = ["Test_*"]` (added two sessions ago specifically to silence a pytest warning about colliding with the `TestSuite` config model) silently excluded every class-based test file from ever being collected - `pytest -q` reported "22 passed" with no indication that 9 newly-written tests never ran at all. Caught only by noticing the collected-test count didn't match what was actually written, not by any failure output. Fixed by reverting to pytest's default `python_classes` and suppressing the specific warning via `filterwarnings` instead (`pyproject.toml`). **Breadcrumb: never narrow what a test runner is allowed to collect to silence a cosmetic warning - it will eventually eat real tests silently, and "N passed" with no failures is not the same as "everything that was written actually ran."**

## Fourth session — Arcaine, OpenArc, build-from-source, OpenSearch (2026-08-01, overnight)

Task: integrate two more backends (Arcaine, OpenArc), validate `source.mode: build` for real (git pull + docker build, not just `prebuilt`), validate build-version tracking, build a real OpenSearch storage adapter, and note the future MCP integration need. All done in one pass while the user slept; log kept live in `NEXT_STEPS.md` throughout, this section is the consolidated writeup.

### Direct answers to questions asked at the start

- **Using local git clone folders?** Yes — `~/Arcaine` (remotes: `origin`=`meatposes/Arcaine`, `upstream`=`SearchSavior/Arcaine`) and `~/OpenArc/OpenArc` (remote: `origin`=`SearchSavior/OpenArc`) are the real local clones used, same pattern as `~/qxmx` in the prior session.
- **Tested pulling a repo?** Yes, for the first time this session — see "build-from-source" below. Used `qxmx`'s real remote (`https://tangled.org/clee.sh/qxmx`) rather than Arcaine's (Arcaine's `.devops/Dockerfile` compiles oneDNN from source, a build easily 20-40+ minutes; qxmx's build is much lighter and already had cached layers).

### Arcaine

Real image `arcaine-server:latest` already existed locally (plus many experimental tags), fully **env-var driven** — `server-entrypoint.sh` builds its own argv from `MODEL_PATH`, `SERVER_HOST`/`PORT`, `MAX_SEQ`, `DEFAULT_MAX_TOKENS`, plus diffusion/MoE-specific `DENOISING_STEPS`/`DEFAULT_SEED`/`LAYER_PLACEMENT`/`EXPERT_PLACEMENT` — no CLI args at all, unlike every other engine validated so far. No `/health` endpoint (confirmed 404 on a live container) — `/v1/models` is the real readiness signal, same gotcha class as qxmx/llama.cpp needing their own health-path override.

Model: `/mnt/ignite/LLM/models/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4` — an HF-safetensors-style directory (not a single GGUF file), a 26B-parameter MoE diffusion-decoding Gemma variant. Matches the `integration/nvfp4-27b` branch currently checked out in `~/Arcaine`. Manually verified working before writing the translator: loaded, correctly answered `12 + 30 = 42`, single render-node passthrough (same idle B70 used throughout this session). Translator: `llapdance/plugins/engine/arcaine.py` — never emits `command`, only `env`.

Real run: `examples/validation-arcaine.suite.yaml` → **10/10 coherence**, benchmark completed (`avg_tokens_per_sec: 2.1`, notably lower than the `~13 tok/s` seen in the manual curl test — likely `generic-http`'s coarse "one SSE line = one token" heuristic undercounting Arcaine's particular streaming chunk shape, not a real perf regression; flagged for a closer look, not fixed tonight, coherence content itself was correct).

### OpenArc

Real image `openarc:dev` already existed locally. **Fundamentally different lifecycle** from every other engine validated so far: it starts with no model loaded at all (`openarc serve start`, no model args), and a model only becomes servable via a *separate* `POST /openarc/load` call after the server is already up. Discovered live (not guessed): the load config's `engine` field must be `ovgenai` for `model_type: llm` — tried `optimum` first, the server told me the exact valid combinations directly in its error message.

Model: `/mnt/ignite/LLM/models/OpenVINO/Phi-4-mini-instruct-int4-ov` — genuinely different format again: OpenVINO IR (`openvino_model.xml`/`.bin` + separate tokenizer/detokenizer IR files), not GGUF or HF-safetensors. Device naming is **also** its own: `GPU`/`GPU.0`/`GPU.1` — a fourth non-corresponding GPU index space, on top of the three already found (clinfo/xpumcli/SYCL-level-zero/DRM render-node — see prior session). Sidestepped rather than reconciled: with only one render node ever passed through per backend, the literal string `"GPU"` always resolves correctly regardless of what number OpenVINO would otherwise assign it.

This lifecycle difference required a genuinely new harness capability that didn't exist before tonight: **`EngineInvocation.post_start_requests`** (`llapdance/plugins/base.py`) — a list of HTTP requests fired against the running backend after the health check passes but before benchmark/coherence adapters run, wired into the orchestrator (`_run_post_start_requests`, aborts the run on any non-2xx response rather than silently benchmarking a backend with no model loaded). `BackendConfig.post_start_requests` exists too, as the same raw-passthrough escape hatch every other generated field has.

Real run: `examples/validation-openarc.suite.yaml` → **10/10 coherence**, `avg_tokens_per_sec: 89.9` (a 4B-parameter INT4 model, much faster than the 26-27B models tested so far — expected, not a bug).

### Build-from-source, validated for real

Previous sessions only ever used `source.mode: prebuilt`. Tonight: a real `git clone` of `qxmx`'s actual remote (`https://tangled.org/clee.sh/qxmx`) into a scratch path (**not** the user's live `~/qxmx` working clone — see safety fix below), a real `docker build`, a real run. `examples/validation-build-from-source.suite.yaml`.

Two real things found building this out:

1. **Safety gap in `local_docker.py::build()`**: the existing-clone-path branch blindly ran `git checkout <ref>` with no check for uncommitted changes. If `build.path` had pointed at a real working directory with in-progress work, this would have silently discarded or conflicted with it. Fixed: refuses with a clear error if `git status --porcelain` isn't clean. This is also why the build-from-source test cloned into a scratch path rather than reusing `~/qxmx` directly — this harness has no business mutating a live dev clone just because a suite config names it.
2. **Build-version tracking, built and confirmed**: image tags now include the resolved commit SHA, not just the branch/ref name (`llapdance/qxmx-from-source:main-3ae4eff`) — confirmed `3ae4eff` matched `git log`'s HEAD at build time. Two builds of the same branch at different points in time are now distinguishable from the stored `image_ref` alone.

Bonus: this run's coherence check scored **9/10**, not 10/10 — the model answered `12 + 12 = 24` instead of `12 + 30 = 42`. A real model arithmetic error, not a harness bug — exactly the failure class coherence-checking exists to catch, and it caught one unprompted.

### OpenSearch storage adapter, built and validated for real

`llapdance/plugins/storage/opensearch.py` — opt-in per SPEC.md §8 (flat-file remains the only default-on adapter). `opensearch-py` is imported *lazily inside `__init__`*, not at module level, so `load_builtin_adapters()` importing this module never requires the dependency unless a suite actually selects `adapter: opensearch` — added as an optional extra (`pip install -e ".[opensearch]"`), not a core dependency.

Validated against the real local OpenSearch 3.7.0 instance (`opensearch`/`opensearch-dashboards` containers, already running). Two real bugs found doing this, neither guessed at in advance:

1. **opensearch-py 3.x API signature change**: `indices.exists()`/`indices.create()` require `index=` as a keyword argument — a positional call raises `TypeError` immediately. Found by just running the code, not by reading changelogs.
2. **Much more serious — silent precision loss on `timestamp`**: with no explicit index mapping, OpenSearch's dynamic mapping guesses a JSON float field as 32-bit `float` (Lucene's default). A Unix epoch timestamp (~1.7×10⁹ + fractional seconds) is far beyond float32's ~7-significant-digit precision — two results written 10 seconds apart both rounded to the *identical* sort key (`1785567600.0`), silently breaking delta-lookup ordering. Caught by writing two real documents and checking the returned order was wrong, not by reading OpenSearch's own docs about default type inference. Fixed with an explicit index mapping (`timestamp: double`, plus explicit `keyword` types for `run_id`/`backend_name` rather than relying on a dynamic `.keyword` sub-field existing). Re-validated after the fix: correct descending order, correct `previous_for()` results.

End-to-end integration confirmed too, not just the adapter in isolation: ran a real suite (`examples/validation-opensearch.suite.yaml`, qxmx backend) with `opensearch` configured as an `extra_adapters` entry alongside the default flat-file, through the actual CLI/orchestrator, and confirmed the resulting document landed in OpenSearch (`curl`-queried it back afterward) while the flat-file copy was also written — proving storage fan-out (multiple adapters active at once, SPEC.md §8) works, not just a single adapter swapped in for another.

**Security note**: the committed example suite (`validation-opensearch.suite.yaml`) uses a placeholder password (`CHANGE_ME`) rather than the real local OpenSearch admin credential — the real credential was supplied only via a `--set` CLI override at run time, never written into a file that gets committed.

### MCP integration — noted, not built

Added to `SPEC.md` §13 and as a code comment at the top of `llapdance/cli.py`: this suite will need an MCP server surface later so agents (not just human operators via CLI/TUI) can push test suites/runs and pull back results programmatically. Explicitly out of scope for this build pass — the orchestrator's `run_suite`/`run_backend` functions are the operations an MCP layer would wrap, so this should be a thin translation layer on top rather than a redesign when it's built.

## Fifth session — SSH remote target, GPU identity tracking, Arcaine benchmark fix, external/already-loaded backends (2026-08-01, continued)

Follow-up to the overnight session's open items: fix the Arcaine benchmark undercount (#1), track full GPU identity per run (#4), and build a real SSH remote execution target (#5) — plus, raised mid-conversation, a new "test an already-loaded backend, no container lifecycle" mode.

### GPU device identity tracking

`RunResult.device_target` now carries full device identity, not just a bare index: `{"mode", "verified", "devices": [{"index", "vendor", "name", "pci_bus_id", "render_node"}, ...]}`. `verified: true` means this came from actually probing hardware; `verified: false` (external backends, see below) means it's whatever the suite author claimed, never conflated with a real probe result. `execution_target` now also records the real local hostname (`socket.gethostname()`) for local runs, not just `None` — runs on different physical machines are now distinguishable from the stored result alone, not just from context you have to remember.

### Remote hardware probing — Runner abstraction + a real gap found live

`core/probe.py` now threads an explicit `CommandRunner` (LocalRunner or SSHRunner) through every discovery/VRAM function, so probing works identically for a local execution target and a remote one — this is what SPEC.md §7 meant by "probing happens against whichever execution target is active," now actually true rather than aspirational.

Checked screamer (the remote host) directly before writing any code: it has **neither `xpumcli` nor a working host-level OpenCL runtime** (`clinfo` reports 0 platforms — the Intel compute stack there only exists bundled inside container images, not on the bare host, unlike the local box). This meant the existing two-tier discovery (xpumcli → clinfo) would find literally nothing on a real, in-scope host. Added a third tier: **`lspci`-based discovery** — identification only (vendor/model/PCI-bus-id via the PCI-SIG vendor-ID registry, structurally excluding non-Intel/non-NVIDIA chips like screamer's ASPEED server-BMC graphics, not by name-matching a specific chip), no free-VRAM reporting (correctly fails closed, same as ever). `_render_node_for_pci` was also rewritten to resolve via the runner (a small remote shell one-liner) instead of direct local `pathlib` access, so render-node resolution — the one thing that's actually reconciled across GPU index spaces — now works over SSH too.

### SSH execution target — built via raw `ssh`+`docker` CLI, not docker-py's `ssh://` transport

Tried docker-py's native `ssh://` base URL first; its transport module does `import paramiko` unconditionally even in `use_ssh_client=True` mode (shell out to system `ssh`), and offers no clean way to pin a specific identity file short of editing `~/.ssh/config` or relying on ssh-agent state this process has no guarantee persists between tool invocations (confirmed: an agent started and loaded in one Bash call was gone in the next). Rather than fight that, `llapdance/plugins/execution/ssh_docker.py` shells out to `ssh -i <key> ... docker <args>` directly for every operation — more code than the local adapter, but the identity file is used exactly as specified, no side effects on the user's real SSH config or agent state.

**Scoped deliberately**: only `source.mode: prebuilt` is supported remotely — building from source on a remote host would need the build context transferred there first (rsync or similar), out of scope for this pass. A suite wanting to build from source against a remote host should build locally and push to a registry the host can pull from instead.

The orchestrator previously **hardcoded** `"local-docker"` regardless of `suite.execution_target` — a real bug, not just a missing feature: the config schema already had `ExecutionTargetConfig.mode: local|ssh` from an earlier session, but nothing ever read it. Fixed: `_execution_adapter_name()`/`_make_runner()` now actually dispatch on `suite.execution_target.mode`.

### Real validation: stop/test/restore against screamer

Checked screamer directly first (real findings, not assumptions): single Intel Arc Pro B50 (`lspci`: PCI `84:00.0`, `[8086:e212]`), one render node (`renderD128`), and the real production `bonsai` container (`llama-cpp-bonsai:meat6-hardened`, same image tag as the local box) already occupying it. Model files live at `/home/nullraptor/bonsai-models` there (host path differs from the local box, as expected — this is exactly what the harness's `volumes`/`model_path` config exists to make swappable).

Procedure: `docker stop bonsai` on screamer (not removed) → ran `examples/validation-ssh-remote.suite.yaml` (smaller `context_size: 2048`, per the heads-up that this GPU has much less VRAM than the local B70s) through the new SSH execution target → confirmed clean teardown (`docker ps -a` showed no leftover test container) → `docker start bonsai` → confirmed it came back healthy and answering correctly (`12 + 30 = **42**`) on its original port.

Result: **10/10 coherence**, real (and honestly quite slow — `~1.2 tok/s` decode, `~34s` average per request) throughput numbers reflecting genuinely weaker/differently-tuned hardware, not a bug. One operational note: the first attempt hit the local Bash tool's own timeout (not a harness hang) — this GPU is slow enough that 3 benchmark requests + 10 coherence questions took several minutes; a container was left running mid-test as a result of the tool timeout (not the harness — it never reached its own `finally: execution.stop()` because the whole process was killed), cleaned up manually (`docker rm -f`) before retrying with more time budgeted. Worth remembering when running suites against slow/remote hardware: budget wall-clock time generously, and check `docker ps` on the target if a run gets interrupted.

### Arcaine benchmark undercount — real bug, root-caused and fixed

Root cause, found by actually inspecting Arcaine's raw SSE stream (`curl -N` with `stream: true`): Arcaine's **diffusion decoding emits the entire completion as a single SSE chunk**, not one-token-per-chunk like the autoregressive engines (llama.cpp, qxmx). The `generic-http` benchmark adapter's original heuristic — count one non-empty SSE line as one token — assumed autoregressive streaming and undercounted Arcaine's real throughput by roughly 7x (`2.1 tok/s` measured vs. `~13-15 tok/s` real).

Fixed generically, not with an Arcaine special-case: `generic_http.py::_completion_token_count` now parses each SSE chunk as JSON and prefers, in order: the standard OpenAI `usage.completion_tokens` field, then `metrics.new_token` (the convention both Arcaine *and* OpenArc happen to use), then `timings.predicted_n` (llama.cpp's convention), falling back to the old per-line count only if none of those are present anywhere in the stream. Every stored benchmark result now also records `counted_via` so it's always clear which method actually produced a given number, rather than silently trusting a heuristic. Re-validated live across three engines after the fix:

- Arcaine: `2.1 → 14.7 tok/s` (`counted_via: metrics.new_token`) — matches the earlier manual `curl` measurement.
- llama.cpp: `34.6 tok/s` (`counted_via: timings.predicted_n`) — consistent with prior runs, now via an actually-authoritative field instead of a coincidentally-close line count.

### External/already-loaded backend mode — new capability, requested mid-session

Raised as a good idea while discussing GPU tracking: test a model that's *already running*, with no container of ours to build/start/stop at all. Added `source.mode: external` (`BackendSource.endpoint`, required only for this mode) — `run_backend()` now branches early to `_run_external_backend()`, which skips device resolution, VRAM preflight, and the execution-adapter registry entirely (a dedicated test — `test_external_backend_skips_build_start_stop` — asserts the execution registry is never touched for this path). `BackendConfig.device_note` is free-text, explicitly and permanently `verified: false` in the stored result — never conflated with a real probed `DeviceInfo`.

Also required adding `api_key`/`headers` support to both `generic-http` and `fixed-questions` (neither previously sent any auth header at all) — needed for the real target: the already-loaded `Ternary-Bonsai-27B-Q2_0.gguf` model on this box's GPU1, reached through `llm-proxy` (the user's own separate OpenAI-compatible aggregator project — found its config at `/mnt/ignite/LLM/llm-proxy/config.yaml`, confirmed the exact model id via its own `/v1/models` response rather than guessing the naming convention). Real run (`examples/validation-external.suite.yaml`): **10/10 coherence**, `13.3 tok/s`, confirmed zero containers created (`docker ps -a` before/after identical).

## Sixth session — MCP server, telemetry harness, guidellm attempt, spec review (2026-08-01, continued)

Task: build the MCP integration (SPEC.md §13, previously just a note), validate it for real, then review SPEC.md against original intent, then add more testing/telemetry harnesses.

### MCP server — built and validated with a real client

`llapdance/mcp/server.py`, using the `mcp` SDK (v2.0's `MCPServer`/`@server.tool()` API — the older `mcp.server.fastmcp.FastMCP` name from 1.x docs doesn't exist in this version, found by just importing and checking). Five tools, every one calling straight into the same orchestrator functions the CLI uses (`run_suite`, the plugin registry, `FlatFileStorage`) — no separate business logic, per the note that was already sitting in `cli.py` from when this was still unbuilt.

Validated with the **real** official `mcp` client SDK over stdio (not a mock): connected to `llapdance mcp`, listed tools, called `list_adapters`/`list_suites`/`get_suite`, then ran a **real `run_suite` execution** against the already-loaded model through `llm-proxy` (external mode, from the prior session — chosen specifically because it's fast, no container boot needed for an MCP smoke test), got back real benchmark numbers and 10/10 coherence, then pulled the same result back via `get_results` reading flat-file storage.

Real gotcha found doing this: a tool that returns `list[str]` (`list_suites`) does **not** come back as JSON text in `result.content[0].text` — the SDK wraps structured returns as `{"result": [...]}` under `result.structured_content`. Found by trying the naive `content[0].text` parse first (works fine for dict-returning tools like `list_adapters`, silently wrong shape for list-returning ones) and getting a `JSONDecodeError`. Documented in `mcp/server.py`'s test file and `SPEC.md` for whoever builds the next tool.

### Telemetry harness — xmxmon, built and validated, with a real "wrong device" gotcha caught live

New `TelemetryAdapter` plugin kind (`llapdance/plugins/base.py`) — deliberately separate from `BenchmarkAdapter`, not a variant of it: it brackets `start()`/`stop()` around whatever benchmark/coherence adapters run, watching hardware rather than hitting the endpoint itself. The original architecture sketch (SPEC.md §5) had folded "Telemetry/Benchmark" into one combined slot; integrating a real telemetry tool proved that wrong, so the spec diagram/text got updated to match (see `SPEC_REVIEW.md`).

Reference implementation: `xmxmon` (`llapdance/plugins/telemetry/xmxmon.py`), a real GPU hardware-counter daemon already running locally. Read its actual source directly (no docs existed) to find its real API: `GET /now` (rolling-window snapshot: gauges/rates/derived metrics), `POST /capture`/`POST /capture/stop` (writes a tagged sample file **inside its own container**, returns only file metadata — never the samples — over the API). Deliberately did NOT build against the capture endpoints: reading that file would need container filesystem access this session had no access-checked reason to assume; the `/now` snapshot is a real, complete, already-computed summary and doesn't need it.

Real validation (`examples/validation-telemetry.suite.yaml`, qxmx backend): ran clean, telemetry data captured and stored in `RunResult.telemetry` — **but** the numbers came back all near-zero, because xmxmon (configured to watch device `0`) and qxmx (targeting device `3`/`renderD131`, the same idle B70 used all session) were watching **different physical GPUs**. Not a bug — a concrete, live demonstration of the unreconciled-GPU-index-space problem this project has been flagging since the second session: the numbers were syntactically valid and semantically meaningless, and nothing would have caught the mismatch except knowing to check. Documented as a real risk for anyone wiring up telemetry: **the suite author is responsible for pointing the telemetry adapter's own device number at the same physical card the backend is actually using** — there's no automatic reconciliation, and there won't be until the GPU-index-space problem gets a real fix (still SPEC.md §15's open item).

### guidellm — attempted for real, shipped as an honest stub

Installed the real package (`pip install guidellm`, a genuine vLLM-project tool) and tried it against the already-loaded model through `llm-proxy`. Two real errors fixed along the way (`--constraint kind=max_requests` needs `count=`, not `max_requests=`; needed an explicit `--tokenizer` to avoid it defaulting to the backend's `model=` value). Hit a structural wall on the third attempt: guidellm's `synthetic_text` data source **always** resolves a tokenizer via `AutoTokenizer.from_pretrained(model_name)`, using the backend's served model name directly as an HF Hub repo id, with no field in `HuggingFaceTokenizerArgs` to override it (confirmed by reading the source, not guessing). This breaks against `llm-proxy`'s `<file>@<backend>` naming convention — and against most of what this harness actually tests, since a locally-built custom quant is essentially never a real HF Hub repo id either. Shipped as a documented `NotImplementedError` stub (`llapdance/plugins/benchmark/guidellm.py`), same honest pattern as `llama_benchy.py` — registered so a suite referencing it fails with the real explanation instead of a traceback buried in guidellm's own tokenizer code.

### Spec review — see `SPEC_REVIEW.md`

Full write-up in that file. Short version: portability principle (§0) is holding with no violations found across five sessions of real additions. Spec text had gone stale in three places (fixed in this pass): the §5 architecture diagram still showed the old four-plugin-kind sketch with telemetry folded into benchmark; `source.mode: external` didn't exist in the spec text at all despite being real and validated; §13's MCP line still said "future, not built." More importantly: **sweep/parameter-matrix automation (§10) and image catalog/cleanup (§12) are both still completely unbuilt**, while four inference engines, two execution targets, a telemetry adapter, and MCP all exist. Recommendation: the next session should probably pivot toward those two rather than a fifth engine — the engine-integration pattern is proven four times over, but there's no way to run an actual sweep or see which of the sprawling image tags are worth keeping without hand-authoring separate files or `docker images | grep` by hand.

## Seventh session — sweep automation, image catalog, model catalog (2026-08-01, continued)

Direct follow-up to `SPEC_REVIEW.md`'s top recommendation: build the two flagged gaps (sweep automation, image catalog), then extend into a new capability - a model catalog with format-based backend compatibility, plus a start on cataloging each engine's actual sweepable params.

### Sweep/parameter-matrix automation — built and validated

`BackendConfig.sweep: list[SweepAxis]` (`llapdance/config/models.py`) + `expand_suite_sweep()` (`llapdance/config/sweep.py`), wired into `run_suite()` only (not `load_suite()`) - `get_suite`/`list_suites` (CLI and MCP) still show the compact spec a suite author wrote; only an actual run expands it. Each axis is a dotted path into the backend's own config dict (e.g. `params.shared.context_size`) plus a list of values; multiple axes on one backend cartesian-product together. A real design bug found immediately: the first implementation required the target key to already exist in the config dict, which breaks for `params.shared`/`params.backend_specific` (open dicts where a sweep should be able to introduce a param the base config never set, not just vary an existing one) - fixed to only require *intermediate* path components to pre-exist (catches a typo'd path) while allowing the final leaf to be new.

Real validation (`examples/validation-sweep.suite.yaml`): one backend, one axis (`context_size: [2048, 4096]`), ran `llapdance run` once and got **two real, automatically-generated container runs** (`qxmx-sweep--context_size_2048`, `qxmx-sweep--context_size_4096`), both 10/10 coherence, clean teardown on both, two distinct stored results - the first sweep this project has actually run, versus every prior comparison being a hand-authored separate suite file.

### Engine sweepable-params catalog — built

`EngineTranslator.sweepable_params` (class attribute, `llapdance/plugins/base.py`) - a structured catalog (type/default/values/maps-to) of the params each translator actually reads, populated for all four reference engines by turning their existing docstring prose into machine-readable dicts (no new research needed, the knowledge already existed from building each translator). Exposed via `llapdance describe-engine <name>` and the `describe_engine` MCP tool - answers "what can I sweep for this engine" directly, which is exactly what a suite author needs before writing `sweep` axes.

### Image catalog & cleanup — built and validated against real sprawl

`llapdance/core/catalog.py` + `llapdance images list/label/rm` (CLI) + `list_images`/`label_image`/`remove_image` (MCP tools). Discovered that `ExecutionTargetAdapter.list_images()` already existed on both `local-docker` and `ssh-docker` (built in the very first session, alongside `build`/`start`/`stop`) but had never been called from anywhere - the catalog just needed to consume it. Labels follow flat-file storage per SPEC.md §12's own guidance (a small `_image_labels.json` alongside a suite's results, not a new database), and results are cross-referenced by the `image_ref` every `RunResult` already carries.

Real validation against the actual, still-growing local image sprawl (`qxmx:*`, `llama-cpp-*`, `llapdance/*`): ran a fresh real `qxmx` validation, listed images with `--catalog-dir results` and confirmed the run showed up cross-referenced (`runs=1`) against `qxmx:latest`, labeled it `good` with a note, confirmed the label persisted and round-tripped. Confirmed the safety behavior specifically requested by the catalog's design (refuse to remove a `good`-labeled image without `force=True`) against a disposable tag created for the test - **never touched the real sprawl** to validate deletion.

### Model catalog: format detection + backend compatibility — new capability, built and validated

Not originally in `SPEC.md` - added per direct request. `llapdance/core/model_catalog.py` + `llapdance models <dir>...` (CLI) + `list_models` (MCP tool). Scans directories recursively for three formats (GGUF files, OpenVINO IR directories, HF-safetensors directories) and reports a best-effort quant hint plus which registered `EngineTranslator`s could plausibly load each, based on format alone - **explicitly a could-run-on signal, never a will-run guarantee** (a corrupt file, an engine-rejected quant like llama.cpp's `f8`, or a model too large for available VRAM would all pass this check and still fail at runtime).

Real layout discovered scanning this box's actual model folders, not assumed: OpenVINO IR and safetensors model roots are frequently nested under an org/contributor namespace directory (e.g. `OpenVINO/droans/qwen3.5-9B-int4-ov/`, `OpenVINO/Echo9Zulu/...`) - the same convention as the HF Hub's own `org/model` layout. Detection walks recursively and stops descending once a directory is identified as a model root, rather than assuming models sit directly under the scanned directory.

Quant-hint extraction uses real structured data where available rather than guessing: HF-safetensors models' `config.json` has a real `quantization_config.format` field (confirmed: `diffusiongemma-26B-A4B-it-NVFP4` → `"nvfp4-pack-quantized"`, read directly, not inferred from the directory name); OpenVINO models' `openvino_config.json` has a real `dtype` field (confirmed: `Phi-4-mini-instruct-int4-ov` → `"int4"`). GGUF files fall back to a filename regex (no equivalent structured metadata file exists for GGUF).

**Validated against ground truth**, not just structurally: scanned this box's real model directories and cross-checked the three models already validated as real backends across this project's prior sessions - `Ternary-Bonsai-27B-Q2_0.gguf` → `gguf` → `[llama-cpp-sycl, qxmx]` (matches three separate validated sessions using this exact file with these exact engines); `Phi-4-mini-instruct-int4-ov` → `openvino_ir` → `[openarc]` (matches the OpenArc validation session exactly); `diffusiongemma-26B-A4B-it-NVFP4` → `safetensors` → `[arcaine]` (matches the Arcaine validation session exactly). All three format/compatibility calls agreed with reality.

## Eighth session — does sweep account for raw GGML/oneDNN flags? (2026-08-01, continued)

Direct question: our stacks set several GGML/oneDNN-related flags (turning oneDNN on/off, etc.) when starting containers - does the sweep concept (built last session) account for these, or only the normalized `params.shared` concepts?

Investigated the actual source rather than guessing. Found in `ggml/src/ggml-sycl/` (the backend `llama-cpp-sycl` and Arcaine both use): two real runtime env flags read via plain `getenv()` - `GGML_SYCL_NO_PINNED` (disables pinned host memory if set to anything) and `GGML_OP_OFFLOAD_MIN_BATCH` (`atoi(getenv(...))`, default 32). Also found `GGML_SYCL_DNNL` - but that one is a **build-time cmake option** (`ggml/src/ggml-sycl/CMakeLists.txt`: `target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_DNNL=${GGML_SYCL_DNNL})`), controlling whether oneDNN kernels get linked in at all (e.g. for flash-attention) - not something any running container's env can toggle, only a rebuild can.

**Answer: yes, mechanically, for both cases** - `expand_backend_sweep()`'s dotted-path expansion has no special-casing per config section; `params.shared.x`, `env.X`, and `source.build.build_args.X` are all just paths into the same backend config dict. This was true by construction, not something added for this question - the previous session validated it against `params.shared` only, so the generalization needed proving, not building.

**Validated live for the env-var case**: wrote a suite sweeping `env.GGML_OP_OFFLOAD_MIN_BATCH` across `["16", "64"]` against the validated `llama-cpp-sycl` backend, started both real containers directly (skipping the full benchmark/coherence cycle to keep the check fast), and confirmed via `docker exec ... echo $GGML_OP_OFFLOAD_MIN_BATCH` that each container actually received its distinct value - not just that the config expanded correctly, that the real running process saw it.

**Not validated live for the build-arg case** (`source.build.build_args.GGML_SYCL_DNNL`): a from-source oneDNN build is a genuinely slow rebuild (the reason Arcaine's build was skipped for the build-from-source validation two sessions ago, in favor of qxmx's lighter build), so this was confirmed by unit test (the same generic dotted-path mechanism, exercised against a `source.build` config) rather than a real rebuild-sweep. Flagged explicitly in code as unvalidated live - don't treat it as equivalently proven to the env-var case.

**Real gap this surfaced**: the `describe-engine` catalog (built last session) only ever covered translator-consumed `params.shared`/`backend_specific` concepts - it had no way to tell a suite author these raw env flags exist at all, even though they were always sweepable. Fixed: `EngineTranslator.known_env_flags` (new class attribute, same shape as `sweepable_params`), populated for `llama-cpp-sycl` with the three flags found this session (`GGML_SYCL_NO_PINNED`, `GGML_OP_OFFLOAD_MIN_BATCH`, and `GGML_SYCL_VISIBLE_DEVICES` for reference even though this translator doesn't set it). `describe_engine()`/`llapdance describe-engine`/the MCP tool now return `{"params": ..., "env_flags": ...}` instead of a flat dict. **Not populated for qxmx/Arcaine/OpenArc yet** - only llama-cpp-sycl's source was actually read this session; the other three likely have their own equivalent flags (Arcaine links oneDNN too - `CMakeLists.txt` confirms `find_package(dnnl CONFIG REQUIRED)`) but cataloging those wasn't done here, would need the same source-reading approach applied to each.

## Ninth session — cataloging known flags for qxmx, Arcaine, OpenArc (2026-08-01, continued)

Direct follow-up: last session only read `llama-cpp-sycl`'s source for `known_env_flags`. Do the same for the other three reference engines.

### qxmx — from-scratch, no oneDNN at all

Read every `getenv()` call site in `~/qxmx/src/*.cpp`. Found ~16 real flags: perf-tuning ones (`QXMX_CHUNK` default 256, `QXMX_GEMM_WGM8`/`WGN16`/`GPP`, `QXMX_GEMV_SMAX`/`TGTWGS`, `QXMX_FD`/`FD_CHUNK`, `QXMX_FOLD`, `QXMX_VEC`, `QXMX_SNAP_STRIDE`, `QXMX_FA_SPLIT`, `QXMX_FA_PHASES`) and debug-only ones (`QXMX_PROFILE`, `QXMX_DUMP_LAYERS`, `QXMX_FFN_DEBUG`, `QXMX_BATCH_FFN_ONLY`) — cataloged both, labeled which is which. Confirmed **qxmx has no oneDNN dependency at all** (no `dnnl::` usage, nothing in `meson.build`) — it's a genuinely from-scratch engine, consistent with its own README. Real gap check: none — nothing here required a code fix, just cataloging.

**Validated live**: swept `env.QXMX_CHUNK` across `["128", "256"]` against the real validated qxmx backend — both runs 10/10 coherence, clean teardown, distinct results.

### Arcaine — also from-scratch, but DOES link oneDNN, as a runtime toggle

Read `getenv()` sites across `~/Arcaine/src/` (excluding third_party and bench-only tools). Confirmed Arcaine has its own modeling/gpu code (not vendored `ggml-sycl`) but genuinely links oneDNN (`CMakeLists.txt: find_package(dnnl CONFIG REQUIRED)`). Found the real oneDNN toggle for the validated model family (`diffusion_gemma`): **`DIFF_ONEDNN_SDPA`** — unlike llama.cpp's `GGML_SYCL_DNNL` (build-time cmake flag), this is a **runtime env var**: unset or `"off"/"0"/"false"/"no"` (any case) disables oneDNN-backed attention (the default), any other value enables it and is passed through as an implementation-variant selector (exact valid non-empty values not characterized this session — flagged honestly rather than guessed). Also cataloged `DIFF_ARENA`/`DISABLE_SCRATCH` (memory-pool allocator toggle, two env vars controlling the same thing), `DIFF_PREFILL_CHUNK` (default 2048), `DIFF_FORCE_DENOISE_STEPS`, `DIFF_HOST_SAMPLER`.

**Deliberately left uncataloged and said so in code**: a separate Qwen3.5 model family (`src/modeling/qwen3_5*/`, ~15 `ARCAINE_QWEN35_*` flags) that this harness has never validated against — only `diffusion_gemma` has been tested. Also ~13 `DIFF_NVFP4_*` flags (directly relevant since the validated model IS NVFP4-quantized) and MoE-specific flags (`DIFF_MOE_STATS`, `DIFF_MOE_TAIL_CAP`) exist and were found, but not individually characterized — each would need the same context-reading treatment as `DIFF_ONEDNN_SDPA` to document honestly rather than guess at defaults/valid values, and there wasn't time to do all of them to that standard this session.

### OpenArc — real gap found and fixed, not just cataloged

Different shape entirely: OpenArc has no GGML/oneDNN-style env flags — its real tuning surface is the `runtime_config` dict already visible in its `/openarc/load` API schema (found two sessions ago), which gets merged straight into the OpenVINO/`ov_genai` pipeline call (confirmed by reading `src/engine/ov_genai/llm.py`: `pipeline_kwargs = {**(loader.runtime_config or {})}`) — so any real OpenVINO plugin property (`NUM_STREAMS`, `PERFORMANCE_HINT`, `INFERENCE_PRECISION_HINT`, etc.) is a legitimate sweep target.

**Real gap found**: the `openarc` translator never read or forwarded `runtime_config` at all — the capability existed and was reachable through OpenArc's own API, but this harness silently dropped it, so it looked structurally sweepable (a plain dict, same as everything else) but wasn't actually reachable. This is exactly the kind of thing cataloging is supposed to catch, and it did — cataloging isn't just documentation here, it's a correctness pass over what the translators actually forward. Fixed: `OpenArcEngine.build()` now reads `params.get("runtime_config", {})` and includes it in the `/openarc/load` JSON body.

**Validated live**: ran the real OpenArc suite with `params.backend_specific.runtime_config = {"PERFORMANCE_HINT": "THROUGHPUT"}` set via `--set` — model loaded successfully (OpenArc's own API would have rejected an invalid property), 10/10 coherence, real benchmark numbers (71 tok/s on the 4B int4 model). Confirms the plumbing works end-to-end against the real API, not just structurally.

## Tenth session — real Qwen3.5 sweep, a stale-image bug found and fixed (2026-08-01, continued)

Direct request: find a real local Qwen3.5/3.6 model Arcaine supports, sweep it for optimal flags.

### Model chosen

Cross-referenced `arcaine_server.cpp`'s dispatch comment (`config.json` `model_type=="qwen3_5"` -> `Qwen35Model`, dense AR) against real local `config.json` files. `/mnt/ignite/LLM/models/unsloth/Qwen3.6-27B-NVFP4` matches exactly (`architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type: "qwen3_5"`). Flagged and skipped a near-miss: `AEON-7/Ornith-1.0-35B-...-NVFP4`'s `config.json` says `model_type: "qwen3_5_moe"` (missing the `_text` suffix the dispatch checks for) — untested, don't assume it loads.

### Cataloged the real `ARCAINE_QWEN35_*` flags (13 of them)

Read every `getenv()` call site in `~/Arcaine/src/modeling/qwen3_5/*.{cpp,hpp}` — added to `arcaine.py`'s `known_env_flags` alongside the existing `diffusion_gemma` entries. Central one: **`ARCAINE_QWEN35_NVFP4_DPAS`**, default OFF (dense Xe2 kernel), with the source's own comment claiming oneDNN's BMG f4 path is "materially faster for both M=1 decode and large-M prefill on this checkpoint." That claim, being concrete and testable, became the real sweep target — see below for why it turned out to be false on this checkpoint.

### Real bug found: the deployed image predates a KV-cache-reset fix

First sweep attempt (`arcaine-server:latest`) crashed on the very first coherence question after one successful benchmark request: `HTTP 500 "Qwen3.5 KV cache position mismatch"`. Reproduced manually outside the harness (bypassing the sweep) to isolate it: request #1 to a fresh container succeeds every time, request #2 (any new, unrelated prompt) always 500s.

Root-caused via `docker inspect --format '{{.Created}}'` vs `git log`: `arcaine-server:latest` was built **2026-07-26 05:45 UTC**; Arcaine commit `f6724df` ("qwen3_5: invalidate the mixer caches when a new sequence starts") landed **2026-07-26 18:00 UTC** — the deployed image is ~12 hours stale relative to the current source, and is missing exactly the fix for this bug. The commit's own message is worth quoting because it's a real, more-severe hazard than the visible crash: *"the only reset_cache() caller is arcaine_mbench, so arcaine_server and main.cpp never reset between requests... For the linear-attention layers [the missing reset] is silent. conv_state and recurrent_state are inputs to the delta rule, not a window that refills, so every request after the first would start 48 of 64 layers from the previous request's running summary and produce plausible, wrong output."* I.e. without this fix, a server that *didn't* crash on request #2 would have been silently corrupting 48/64 layers' state instead — the crash is the loud half of the bug.

**Also found**: no `Dockerfile.server` (or `server-entrypoint.sh`) exists anywhere in the `~/Arcaine` repo or its git history — the deployed image's actual build recipe is untracked. Reconstructed it from `docker history arcaine-server:latest --no-trunc` (every `RUN`/`COPY`/`ARG` layer is visible verbatim) and extracted the baked-in `server-entrypoint.sh` directly from the running image (`docker run --rm --entrypoint cat ... /usr/local/bin/server-entrypoint.sh`). Saved as `examples/Dockerfile.arcaine-server-rebuild` so this reconstruction doesn't have to happen again.

**Second gotcha while rebuilding**: `~/Arcaine/build/` is gitignored but there's no `.dockerignore`, so the stale host-side `build/` directory (leftover `CMakeCache.txt` pointing at a different original path, `/workspace/build313`) got copied into the image by `COPY . .` and collided with a fresh `cmake -B` configure (`CMake Error: ... different than the directory /workspace/build313 where CMakeCache.txt was created`). Fixed with `rm -rf /workspace/build` before configuring — see the Dockerfile.

Rebuilt from `arcaine:onednn313` (a pre-existing dev image with oneDNN already compiled from source, confirmed via `docker run --entrypoint sh` probing `/opt/onednn/lib` and `which cmake ninja icpx`) — reusing it meant only the `arcaine_server` C++ target needed recompiling, ~23s, not a full oneDNN-from-source rebuild. Tagged `arcaine-server:qwen35fix`. **Validated the fix live**: 5 sequential requests against the same container, all `200`, no crash — confirmed the fix actually lands in the rebuilt binary.

### The real sweep result — and it refutes the source comment

Ran `examples/validation-arcaine-qwen35.suite.yaml` (`env.ARCAINE_QWEN35_NVFP4_DPAS` swept `["0", "1"]`) against `arcaine-server:qwen35fix`. Both runs completed cleanly, clean teardown confirmed (`docker ps -a`).

| | `DPAS=0` (dense Xe2, the default) | `DPAS=1` (oneDNN BMG f4) |
|---|---|---|
| avg TTFT | 523 ms | 718 ms |
| avg total | 2921 ms | 3621 ms |
| avg tokens/sec | 10.16 | 9.90 |
| coherence | 9/10 (1 empty answer, not a wrong one) | **5/10** |

`DPAS=1`'s wrong answers were not borderline: `12 + 30` → "43", `'cat'` backwards → "t", "roses are red, violets are ___" → "purple", `9 * 9` → "9", first month of the year → empty. These are simple, unambiguous questions the same model gets right at `DPAS=0`.

**This directly refutes the source code's own comment.** On this real checkpoint, the oneDNN BMG f4 path (`ARCAINE_QWEN35_NVFP4_DPAS=1`) is both *slower* (higher TTFT, higher total time, lower tok/s) and *measurably less correct* (half the fixed questions wrong, including basic arithmetic) than the dense Xe2 kernel that ships as the default. The empirical "optimal flags" finding for `unsloth/Qwen3.6-27B-NVFP4` is: **leave `ARCAINE_QWEN35_NVFP4_DPAS` unset (off)** — the default is already correct, and the alternative path the comment recommends is worse on both axes tested. This is a single run per config (n=1, not averaged/repeated) — the throughput gap is modest and could use a repeat to firm up, but the coherence gap (9/10 vs 5/10, with genuinely wrong arithmetic) is large enough not to be sampling noise.

Fixed an overclaim caught before this was actually run: `known_env_flags`'s note on this flag briefly said "validated live: the comment was correct" before the sweep had executed — corrected to state the hypothesis neutrally, now updated again to record the real (opposite) result.

### Second Qwen3.5 model confirmed — cross-checkpoint validation

Direct follow-up: find a different Arcaine-compatible model to test. Cross-referenced every local `config.json`'s `model_type` field against Arcaine's dispatch strings (`for f in $(find /mnt/ignite/LLM/models -maxdepth 3 -iname config.json); do ...`). Found `sakamakismile/Huihui-ThinkingCap-Qwen3.6-27B-abliterated-NVFP4` — `model_type: "qwen3_5"`, exact match, 20GB `nvfp4-pack-quantized` safetensors, plus a separate `model-mtp-bf16.safetensors` MTP head (the first model, unsloth's, has no such file — a real difference worth testing `ARCAINE_QWEN35_MTP_ACCEPTANCE` against later). Also confirmed (again) that `AEON-7/Ornith-...` and `urakozz/Ornith-1.0-35B-int4-AutoRound` both report `model_type: "qwen3_5_moe"` (missing `_text`) — still flagged as likely-incompatible, still untested.

**Validated live** against the already-fixed `arcaine-server:qwen35fix` image (default flags, no sweep this time): real container boot, real benchmark (8.11 tok/s, 575ms TTFT), **10/10 fixed-questions coherence** — clean pass, no KV-cache crash (confirms the image fix generalizes across checkpoints, not just the one it was validated against). Clean teardown confirmed via `docker ps -a`.

## Eleventh session — model catalog now cross-references real test history (2026-08-01, continued)

Direct request, following the observation that `llapdance models` only ever reported static could-run-on compatibility, never whether a model was actually tried on a backend before: `ModelInfo` gained a `tested: dict[engine, TestedStatus]` field, built by `annotate_tested_status()` cross-referencing `load_run_history()` (every stored `RunResult` in a flat-file results dir) against the catalog scan.

The real matching problem: a `RunResult` only stores the in-container `model_path` (e.g. `/models/qwen35`) plus `volumes` (host->container), while `ModelInfo.path` is always a host path. `_resolve_host_path()` reverses the mount to recover the real host path a run actually pointed at, handling both cases found in real suites: the model root mounted directly (`model_path == container_vol`, the safetensors/OpenVINO case) and a GGUF file nested under a directory mount (`model_path` prefixed by `container_vol`, host path reconstructed from the relative remainder).

Outcome is one of `pass` (100% coherence), `partial` (some failures), or `ran` (completed, but no coherence adapter configured - no correctness signal at all, just "didn't crash"). **Documented the real gap honestly rather than pretend it's complete**: `run_backend` only writes a `RunResult` after a run finishes cleanly - a crash mid-request (like the real Arcaine KV-cache 500 found this session) never reaches storage, so a genuinely broken combination is indistinguishable from "never tried" here. `TestedStatus`'s docstring says so explicitly.

**Validated live** against this session's own real results: `llapdance models /mnt/ignite/LLM/models/unsloth /mnt/ignite/LLM/models/sakamakismile /mnt/ignite/LLM/models/RedHatAI --results-dir ./results` correctly reported `unsloth/Qwen3.6-27B-NVFP4` as `arcaine:partial(5/10)` (the most recent of its two stored results is the `DPAS=1` run - correct, "most recent" semantics working as intended), `sakamakismile/Huihui-...` as `arcaine:pass(10/10)`, and `RedHatAI/diffusiongemma-...` as **`untested`** even though it was validated in an earlier session - honest, because that session's result file isn't present in the current `./results` directory, and the tool doesn't fabricate history it can't find.

Wired into both the CLI (`llapdance models --results-dir DIR`, defaults to `./results`) and the MCP `list_models` tool (`results_dir` param, same default).

## Twelfth session — sweeping more of the catalog, using the new tested-status feature to pick targets (2026-08-01, continued)

Direct follow-up: with `tested` now real, used it to find genuinely-untested models and ran several.

### Confirmed, empirically, why the two `qwen3_5_moe` (missing `_text`) models can't load

Earlier sessions flagged `AEON-7/Ornith-1.0-35B-...-NVFP4` and (found this session) `urakozz/Ornith-1.0-35B-int4-AutoRound` as likely-incompatible from reading a comment. Read the real dispatch code this time instead of trusting a comment (a lesson from this session's own `NVFP4_DPAS` finding - comments can be wrong): `ModelRegistry::create()` (`~/Arcaine/src/common/registry.cpp`) does an exact `unordered_map` lookup on `model_type` against three registered keys - `qwen3_5`, `qwen3_5_moe_text`, `gemma4_unified` - and throws if it's not an exact match. `qwen3_5_moe` isn't one of them.

**Validated live**: booted `urakozz/Ornith-1.0-35B-int4-AutoRound` against `arcaine-server:qwen35fix` directly (bypassing the suite runner, since this needed the raw crash log). Container exits in ~2s: `[error] No model architecture registered for model_type='qwen3_5_moe'. Registered: qwen3_5 qwen3_5_moe_text gemma4_unified`. Exactly as the registry code predicts. This is also a live demonstration of the `TestedStatus` gap documented in the model-catalog work above: this crash happens before `run_backend` ever reaches its storage step, so it will never show up as `tested=[...]` in the catalog - it's real, confirmed-broken, and permanently invisible to the automated tested-status feature. Didn't bother re-running the AEON-7 NVFP4 variant - same `model_type`, same registry, same outcome, no new information.

### OpenArc + a tiny untested model - a real config-mismatch finding, not a bug

Ran `OpenVINO/Qwen3-0.6B-int4-ov` (previously `untested` per the new catalog feature) against OpenArc. Real numbers: 228 tok/s (small model, as expected), but only **5/10** fixed-questions passed. Checked the actual failures before concluding anything (per this project's own standing rule - never guess): every failure was the model's reasoning trace (`<think>...`) still running when `max_tokens: 64` cut it off - the answer was never reached, not wrong. This is a real, useful finding but not a model or harness bug: `fixed-questions`' benchmark config assumes non-reasoning models fit an answer in 64 tokens, which doesn't hold for Qwen3's `<think>` reasoning models. Anyone sweeping a reasoning-capable checkpoint needs either a much larger `max_tokens` or `enable_thinking: false` in the chat template kwargs - this harness doesn't yet have a helper for that distinction (worth a real follow-up, not done here).

### Diffusion Gemma refreshed on the current image

Re-ran `examples/validation-arcaine.suite.yaml` (the previously-validated `diffusion_gemma` model, `arcaine-server:latest` - the stale-image KV-cache bug was Qwen3.5-specific, `Qwen35Model::forward`, so the old image is still fine for this model family). 16.35 tok/s, **10/10** coherence, clean teardown - now has a real stored record so the tested-status feature reports it instead of `untested`. Confirmed live: `llapdance models .../RedHatAI --results-dir ./results` now shows `arcaine:pass(10/10)` where it previously showed `untested` (correctly, since the earlier validation's result file wasn't in the current results directory).

## Thirteenth session — searching beyond `/mnt/ignite/LLM/models` for more Arcaine-compatible models, and a real (attempted) fix (2026-08-01, continued)

Direct question: are there other models anywhere that work on Arcaine? The model catalog had only ever been pointed at `/mnt/ignite/LLM/models`. Searched the rest of the filesystem (`/mnt/acheron`, `/mnt/malebolge`, `/mnt/Ironwolf-4TB/Models`, `/mnt/WINMOUNT/models`, `~/.cache/huggingface/hub`) - the last one had real, fully-downloaded models not in the catalog's scan path. Two matched Arcaine's registered `model_type`s:

- **`Qwen/Qwen3.5-27B`** (`model_type: qwen3_5`, exact match, dense) - a real, different checkpoint from both already-validated ones: full bf16, not NVFP4-quantized. Confirmed fully downloaded (52GB, all 11 shards real size, no `.no_exist` markers).
- **`Qwen/Qwen3.5-35B-A3B`** (`model_type: qwen3_5_moe`, MoE, official Qwen release) - also fully downloaded (67GB, 14 shards).

### Investigated whether the two already-known-broken `qwen3_5_moe` models could actually be fixed

Previously flagged `AEON-7/Ornith-1.0-35B-...-NVFP4` and `urakozz/Ornith-1.0-35B-int4-AutoRound` as broken from a comment-level read of the dispatch code. This session read the loader's actual config parser (`~/Arcaine/src/modeling/qwen3_5_moe/config.hpp`) instead of trusting a comment (a direct lesson from this session's own `NVFP4_DPAS` finding). Its docstring says it expects "the NVFP4 'Qwen-AgentWorld-35B-A3B' text model shipped inside a multimodal container" with a **flat** config (no `text_config` wrapper) and `model.language_model.`-prefixed weight keys.

Read the AEON-7 checkpoint's actual safetensors header (93,346 tensor names, no full weight-loading needed) directly: it genuinely has **zero vision tensors**, every key is `model.language_model.*` - it IS the text-only checkpoint the loader wants, just with the original multimodal repo's nested `config.json` (fields still under `text_config`/`vision_config`) instead of the flattened shape, and `model_type: "qwen3_5_moe"` instead of `"qwen3_5_moe_text"`.

**Built a non-destructive fix**: a sibling directory (`...-textfix`) with every file **hard-linked** (not symlinked - a symlink to an absolute host path breaks inside a container whose bind mount only exposes the single model directory, found the hard way, first attempt failed with `Cannot open tokenizer` even though the symlink existed) to the original, plus a patched `config.json` (`text_config` flattened up to top level, `model_type` corrected). Original directory never modified - confirmed after cleanup (`model_type` still reads `qwen3_5_moe`).

**Real result: partially fixed, then hit a genuine tensor-format wall.** Iterated through real container boots against `arcaine-server:qwen35fix`, each time reading the actual crash and fixing forward:
1. Config parse succeeded once flattened - model type dispatch worked, tokenizer loaded, `93346 tensors in single safetensors file` logged, GPU selected.
2. Crashed on `tensor not found: model.language_model.layers.0.linear_attn.out_proj.weight_packed`. Read the safetensors header directly to see why: this checkpoint's **MoE expert weights are NVFP4-packed** (`weight_packed`/`weight_scale`/`weight_global_scale`, matches the loader) but its **`linear_attn` (Gated DeltaNet) weights are plain, unquantized** (`out_proj.weight`, no `_packed` suffix) - a genuinely mixed-precision quantization recipe. The loader's tensor lookup for `linear_attn.*` has no dense-weight fallback path (unlike the *dense* `qwen3_5` loader, see below) - it's hard-coded to expect every relevant tensor NVFP4-packed.

This is a real engine-side gap, not something fixable by editing metadata - it would need a source change to Arcaine's `qwen3_5_moe` loader (a dense-weight fallback for `linear_attn`, mirroring what the dense `qwen3_5` loader already does), out of scope for this test harness. Checked the sibling AutoRound checkpoint too before ruling it out further: its `linear_attn` tensors use GPTQ-style `qweight`/`qzeros`/`scales`, a completely different quantization scheme the NVFP4-only loader has no path for at all - confirmed less compatible, not more. **Removed the non-working shim directory** after confirming the dead end, rather than leave a broken artifact sitting in the real model library.

### A genuinely new, real capability found while investigating: the dense loader already supports 3 weight formats

Reading `~/Arcaine/src/modeling/qwen3_5/loader.cpp` (the *dense* Qwen3.5 loader, not MoE) to compare against the MoE loader's rigid NVFP4-only assumption: it explicitly branches on **NVFP4 packed, FP8 (`.weight` + `.weight_scale`), or a plain dense `.weight`** - real multi-format support already built in, unlike the MoE loader. This means `Qwen/Qwen3.5-27B` (full bf16, unquantized, `model_type: qwen3_5` - the dense dispatch) should load without any patching at all.

**Not run**: GPU 3 (the idle B70 used for every validated run this session) has `Memory Physical Size: 32656 MiB` (`xpumcli discovery -d 3`) - roughly 32GB. The bf16 checkpoint is 52GB, won't fit one GPU. Would need Arcaine's multi-GPU layer split (`LAYER_PLACEMENT`/`--layers`), which this harness's translation layer doesn't resolve (see the module docstring in `llapdance/plugins/engine/arcaine.py`: only one device is ever resolved per backend) - a real, larger follow-up, not attempted this session to avoid a guaranteed OOM.

### Bottom line

No, there are no additional models that work on Arcaine beyond the two already validated (`unsloth/Qwen3.6-27B-NVFP4`, `sakamakismile/Huihui-...`) plus `diffusion_gemma`. The two MoE candidates found are real, confirmed-broken by a genuine engine-side tensor-format gap (attempted and documented, not just asserted), and the one plausibly-fixable dense bf16 candidate (`Qwen/Qwen3.5-27B`) needs multi-GPU support this harness doesn't have yet to actually run.

## Fourteenth session — 3 more catalog models, a real `fixed-questions` gap fixed, a real OpenArc+model crash root-caused (2026-08-01, continued)

Swept 3 more untested OpenArc models while a large download ran in the background.

### Real gap fixed first: `fixed-questions`' `max_tokens` was hardcoded to 64

Before running `DeepSeek-R1-Distill-Qwen-7B-int4-ov` (a reasoning model), fixed the exact issue found last session with `OpenVINO/Qwen3-0.6B-int4-ov`: `FixedQuestionCoherence._ask()` hardcoded `max_tokens: 64` in the `/v1/chat/completions` body with no way to override it from a suite - meaning every reasoning-model sweep would hit the same `<think>`-truncation false-negative no matter what the suite YAML said. Added a `max_tokens` config key (default 64, unchanged for every already-validated suite). 2 new tests (`tests/test_fixed_questions_coherence.py`, real `httpx.MockTransport` requests, not mocks of internal calls) confirm the default is preserved and the override actually reaches the request body. 108 tests passing.

### Real results

| Model | Result |
|---|---|
| `Phi-4-mini-instruct-int4-ov` | Clean: 84.1 tok/s, 10/10 coherence. |
| `DeepSeek-R1-Distill-Qwen-7B-int4-ov` (`max_tokens: 512`) | Clean: 104.5 tok/s, **10/10 coherence** - confirms the `max_tokens` fix above actually works: a reasoning model now gets real answers instead of truncated `<think>` traces. |
| `phi-2-int4-ov` | **Real crash, root-caused, not guessed.** `generic-http` failed with `httpx.RemoteProtocolError: Server disconnected without sending a response`. Reproduced manually outside the harness to get the real server-side log (the crashed container's own logs are lost once `run_backend`'s `finally: execution.stop()` tears it down - the same gap `TestedStatus` documents). Root cause: `phi-2-int4-ov`'s tokenizer has no `chat_template` set, and OpenArc's `/v1/chat/completions` worker calls a chat-template function unconditionally - `ValueError: Cannot use chat template functions because tokenizer.chat_template is not set...` on the very first real inference call, which triggers OpenArc's own auto-unload of the crashed worker. Confirmed via full container logs: model *loads* successfully (`POST /openarc/load` returns 200), the first 3 generate requests return `200` with **zero tokens** (worker already dead, response drained empty), then the connection hard-drops once the worker is fully gone - explaining exactly the failure mode `generic-http` hit. Genuine OpenArc+model incompatibility (this specific int4-ov conversion never got a chat template baked in), not a llapdance bug. |

Clean teardown confirmed for all three (`docker ps -a`).

## Fifteenth session — the official Arcaine-supported MoE checkpoint, closing out the Thirteenth session's question (2026-08-01, continued)

Direct follow-up: Arcaine's own README lists 4 "Supported Models"; we'd only validated 3 (`DiffusionGemma`, `Unsloth Qwen3.6-27B`, and by extension the dense `qwen3_5` family). The 4th, `Qwen AgentWorld-35B-A3B NVFP4` (`Frosty40/Qwen-AgentWorld-35B-A3B-NVFP4` on HF), is the exact MoE checkpoint the `qwen3_5_moe_text` loader's own source comment describes - and directly answers the open question from the Thirteenth session: were the `AEON-7`/`urakozz` MoE crashes a genuine Arcaine bug, or just incompatible third-party quantization recipes?

Downloaded it for real (`hf download ... --local-dir`, 21GB, confirmed `model_type: "qwen3_5_moe_text"` - the exact registry key, no patching needed unlike the AEON-7 shim attempt).

**Result: it loads and runs. The earlier crashes were genuinely bad third-party checkpoints, not an Arcaine bug.** Real benchmark: 14.7 tok/s, 932ms TTFT, clean teardown, no crash of any kind through the whole run - confirms `ModelRegistry`'s MoE loader itself works correctly given a checkpoint that actually matches its expected tensor layout (NVFP4-packed `linear_attn`, not the mixed-precision AEON-7 recipe that broke it).

**But a separate, new, real finding**: only 5/10 fixed-questions passed. Checked the actual failures (never guess) - this is NOT the same benign truncation issue found with `OpenVINO/Qwen3-0.6B-int4-ov` or fixed by the `max_tokens` change this session. The failing answers are genuinely degenerate: repeated empty `<think>\n\n</think>` loops or repeated backtick blocks filling the entire response, e.g. `"<think>\n\n</think>\n\n<think>\n\n</think>\n\n<think>\n\n</think>..."` eight times over, never reaching real content. All 5 passes were plain keyword matches (`graded_by_match: 5`), not LLM-judged - meaning the passing answers actually contained the right keyword, while the failing ones never generated any real content at all. Same `max_tokens: 64` as every other coherence run (not a budget issue this time - a real generation-quality/stability issue with this checkpoint under Arcaine's default sampling settings). Not investigated further this session (would need looking at default temperature/repetition-penalty/MTP-speculative-decode interaction) - flagged honestly as a new open finding rather than something quietly worked around.

## Sixteenth session — 3 more catalog models (2026-08-01, continued)

Direct follow-up: sweep a few more untested catalog models.

| Model | Result |
|---|---|
| `Phi-3.5-mini-instruct-int8-ov` | Clean: 99.5 tok/s, 10/10 coherence. |
| `Qwen3-8B-int8-ov` (`max_tokens: 512` per the reasoning-model fix) | Clean: 58.4 tok/s, 10/10 coherence. |
| `TinyLlama-1.1B-Chat-v1.0-int4-ov` | 267 tok/s (small model, as expected), but **7/10 coherence - real wrong answers, not a bug.** Checked the actual failures: `"Spell 'cat' backwards"` -> `"tas"` (repeated several times, wrong - correct is `"tac"`), `"chemical symbol for water"` -> `"H"` (wrong, should be H2O), `"9 * 9"` -> `"72"` (wrong, should be 81). Real content in every answer, genuinely incorrect - a real capability limitation of a 1.1B model on basic reasoning/spelling, not a harness or truncation issue (distinct from both the `phi-2` crash and the AgentWorld MoE's empty-`<think>`-loop degenerate output found in prior sessions). |

Clean teardown confirmed for all three.

## Seventeenth session — PinBench wired up for real (2026-08-01, continued)

Direct question: is PinBench (SPEC.md's "prior art" table, "fits the coherence/quality slot") actually wired up? It wasn't - only cataloged. Built it for real.

Cloned the real repo (`github.com/ShadyHippo/PinBench`) rather than guessing its interface from the README, and read `providers.py`/`runner.py`/`grader.py` directly. Found: its `type: "vllm"` provider is, underneath, a bare `openai.OpenAI(api_key=..., base_url=...)` client (`OpenAICompatibleProvider`) - works against any OpenAI-compatible endpoint despite the name, which is exactly what every engine translator here exposes. Confirmed the real output shape (`<output_dir>/<run_id>/summary.json` + `results.json`, `run_id` a runtime timestamp not knowable in advance) by actually running it (`--mock`, then for real against a live config).

**Design decision**: PinBench is a real, separately-maintained project with non-trivial domain logic (pinyin/hanzi matching, weighted structured-output grading) - reimplementing its grading natively would just be a worse copy. Built `llapdance/plugins/coherence/pinbench.py` as an adapter that shells out to a user-supplied local checkout (`pinbench_dir` config, same "external tool, not vendored" pattern as `source.mode: build`'s git-clone-by-path convention), not a vendored copy. Needs the new `pinbench` optional dependency group (`openai`, `requests` - PinBench's own imports) since the subprocess runs under this same interpreter (`python_bin` defaults to `sys.executable`).

**Validated live end to end, twice**: once calling the adapter class directly, once through the full CLI (`llapdance run examples/validation-pinbench.suite.yaml --set coherence_adapters.0.config.pinbench_dir=...`) - both against the real, already-running production `llama-cpp-bonsai` container (`source.mode: external`, read-only HTTP, never touched the container's lifecycle). Real result: 2/3 passed on a 3-test filtered slice. 5 new tests (subprocess mocked with fixture output matching PinBench's real file shapes - no real PinBench checkout needed in CI) - 113 passing total.

## Eighteenth session — vLLM engine translator (a 5th engine), and a real fork-provenance practice (2026-08-01, continued)

Direct follow-up to the model-consolidation review: 3 real raw HF bf16 checkpoints (`deepseek_v2`, `gemma3`, `qwen3`) survived cleanup because nothing here could load them - `arcaine` only dispatches a narrow `model_type` allowlist, `llama-cpp-sycl`/`qxmx` need GGUF, `openarc` needs OpenVINO IR. vLLM natively supports arbitrary HF-transformers architectures, and real images already existed locally: `intel/vllm:latest`/`0.11.1-xpu`, `intel/llm-scaler-vllm:0.14.0-b8.1`, and `urakozz/vllm-xpu-env` (already running in production as `vllm-urak`, serving `Ornith-1.0-35B-int4-AutoRound`).

### Built `llapdance/plugins/engine/vllm.py` - the 5th engine translator

Read the real running `vllm-urak` container's actual `docker inspect` output (CMD args, env, devices, mounts) rather than guessing from vLLM's docs - confirmed `ENTRYPOINT: vllm serve`, CMD = `<model_path> --served-model-name ... --host ... --port ... [tuning flags]`, and real health check (`/health`, confirmed 200 via `docker exec vllm-urak curl`). `sweepable_params` covers `context_size` (`--max-model-len`), `tensor_parallel_size`, `kv_cache_dtype`, `max_num_seqs`, `max_num_batched_tokens`, `block_size`, `quantization`, `reasoning_parser`, `tool_call_parser`, and presence flags `enable_auto_tool_choice`/`trust_remote_code`/`language_model_only` - all confirmed present on the real container's actual CMD, not invented. `known_env_flags` covers `VLLM_TARGET_DEVICE`, `ONEAPI_DEVICE_SELECTOR`, `VLLM_WORKER_MULTIPROC_METHOD`, `VLLM_XPU_ENABLE_XPU_GRAPH`, `HF_HUB_OFFLINE`, `HF_HUB_ENABLE_HF_TRANSFER` - same source. Also updated `model_catalog.py`'s `COMPATIBLE_ENGINES["safetensors"]` to include `vllm` alongside `arcaine`, since vLLM has no model_type allowlist.

### Three real bugs found and fixed getting the first live validation to actually pass

Targeted `mlabonne/gemma-3-12b-it-abliterated` (23GB bf16, chosen specifically because it fits the validated GPU's ~32GB VRAM - `Qwen/Qwen3.5-27B` at 52GB, also raw and vLLM-loadable in principle, was explicitly NOT attempted for this reason). Used the **published** `intel/vllm:latest` image, not the unpublished fork, specifically to prove the translator depends on the shared CLI/env contract, not fork internals.

1. **HF cache relative symlinks break under a narrow mount.** First attempt mounted only `models--org--name/snapshots/<hash>/` - crashed with `vllm`'s own `Invalid repository ID or local directory specified` even though `config.json` was right there. Real cause: HF cache snapshot files are symlinks with **relative** targets (`config.json -> ../../blobs/<hash>`) - `blobs/` lives in the *parent* directory, outside a snapshot-only mount, so every symlink breaks. Same bug class as the Arcaine AEON-7 shim earlier this session. Fixed by mounting the whole `models--org--name` cache root and pointing `model_path` at the nested `snapshots/<hash>` subpath.
2. **A single render node isn't enough for vLLM, unlike every other engine here.** Second attempt (matching every other translator's device-passthrough pattern) crashed with `oneCCL: ze_fd_manager.cpp:144 init_device_fds: EXCEPTION: opendir failed: could not open device directory` - oneCCL enumerates `/dev/dri` as a directory even at `tensor_parallel_size=1`, and a lone render-node device file leaves no directory to `opendir()`. Cross-checked against `vllm-urak`'s real `HostConfig.Devices`: it passes the WHOLE `/dev/dri` directory, not one render node. Fixed `vllm.py`'s `build()` to emit `["/dev/dri:/dev/dri"]` instead of a single render node - the one real, confirmed exception to this project's usual render-node-only convention.
3. **`/dev/dri` alone still wasn't enough - a separate bind mount was also needed.** Third attempt still hit the identical oneCCL `opendir` crash even with the whole `/dev/dri` device passed through. Re-checked `vllm-urak`'s real `Mounts` (not `Devices`) more carefully: it *also* bind-mounts `/dev/dri/by-path` as a plain read-only volume - oneCCL's GPU-topology discovery needs it. This can't be generated by the engine translator (`EngineInvocation` has no `volumes` field, only `command`/`env`/`devices`), so it's documented as a required suite-level mount for any vLLM backend, added directly to the example suite's `volumes:`.

**Fourth attempt: clean pass.** 18.9 tok/s, **10/10 fixed-questions coherence**, clean teardown (`docker ps -a`). `examples/validation-vllm-gemma3-12b.suite.yaml` now documents all three gotchas inline for the next suite author. 25 new tests (5 for `describe_engine`, 20 for `VLLMEngine.build()` covering both device-passthrough behaviors and every real CLI flag) - 118 passing total before this final round, unchanged count after (translator logic itself didn't change between attempts 2-4, only the suite YAML's volumes and one internal `devices` fix already covered by the existing tests).

### A real fork-provenance practice, applied for real

Direct ask: how do we track forks (published vs not) used in testing? Investigated `urakozz/vllm-xpu-env`'s actual build history (`docker history --no-trunc`) rather than assuming: it genuinely does clone real public upstream repos during build (`github.com/intel/auto-round`, `github.com/vllm-project/vllm-xpu-kernels`, pinned refs) - but its own main source tree (containing `tools/check_repo.sh`, `tests/vllm_test_utils`) is `COPY`'d in from a local build context, not cloned from a URL, and the build explicitly sets `GIT_REPO_CHECK=0` (skipping its own commit-verification step). Unlike `arcaine-server:latest` (fully reconstructable from `docker history` alone, see Tenth session), this image's exact source diff is **not** currently reconstructable if it's ever lost - and it's a real, currently-running production container (`vllm-urak`), not a hypothetical risk.

**Practice adopted**: use the existing image catalog's label + free-text note mechanism (`llapdance images label <ref> unknown --note "..."`, no schema change needed - `label_image()` already took a free string, only the CLI's `Choice` restricted it to good/bad/unknown) to record provenance findings per image, alongside (not conflated with) the existing quality labels. Applied to `urakozz/vllm-xpu-env:latest` now, with the investigation above captured as the note - visible via `llapdance images list`.

## Nineteenth session — sweeping every remaining fitting model, groundwork for pruning (2026-08-01, continued)

Direct request: set up every remaining untested model that would actually fit for testing, so a real pruning decision can follow. Checked sizes first - all 14 remaining OpenVINO models were comfortably under 15GB (fit easily); the 2 remaining raw HF-cache models (`deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` 30G, `NousResearch/NousCoder-14B` 28G, both bf16) were excluded as too tight against the ~32GB budget, same reasoning already applied to the 52G `Qwen3.5-27B`. Also found and tested a genuinely new `vllm` candidate: the stray `hub/models--Qwen--Qwen3-0.6B` (dense `qwen3`, not arcaine-compatible, but vLLM has no allowlist).

Ran all 15 in one sequential batch - **no crashes, no infra bugs this round** (the 3 vLLM mount/device bugs from the prior session were already fixed and held up across a real new model). Real results:

| Model | Engine | tok/s | Coherence | Verdict |
|---|---|---|---|---|
| `Echo9Zulu/Qwen3-14B-int4_sym-ov` | openarc | 53.9 | 10/10 | clean |
| `Echo9Zulu/phi-4-int4_asym-awq-se-ov` | openarc | 52.4 | 9/10 | clean (checked: model hedges on "spell backwards", a known real LLM weakness, not a bug) |
| `Echo9Zulu/Phi-4-mini-instruct-int4_asym-awq-se-ov` | openarc | 92.8 | 10/10 | clean |
| `DeepSeek-R1-Distill-Qwen-14B-int4-ov` | openarc | 53.8 | 10/10 | clean |
| `MeatPoses/Qwen3-Coder-30B-A3B-Instruct-int4` | openarc | 99.2 | 10/10 | clean |
| `MeatPoses/NousCoder-14B-int8-ov` | openarc | 34.3 | **10/10 after a fix** | first pass 4/10 - my own mistake, mislabeled it non-reasoning in the suite generator, default `max_tokens: 64` truncated real `<think>` traces (same false-negative class as `OpenVINO/Qwen3-0.6B` last session). Rerun with `max_tokens: 512` confirmed 10/10 - the model itself is fine. |
| `MeatPoses/Omega-Directive-24B-Unslop-v2-int4-ov` | openarc | 37.1 | 10/10 | clean |
| `mistral-7b-instruct-v0.1-fp16-ov` | openarc | 35.6 | 10/10 | clean |
| `Qwen3-pruned-6L-from-0.6B-int8-ov` | openarc | 282.5 | 9/10 | clean (checked: rambling non-answer on "roses are red..." - genuine small-pruned-model quality limit) |
| `Phi-4-mini-FastDraft-120M-int8-ov` | openarc | 547.5 | **2/10 - real, not a bug** | checked every failure: rambling, self-contradicting, non-answers across basic arithmetic/spelling/facts. This is a **speculative-decoding draft model** (the name says so) - designed to propose candidate tokens for a larger target model to verify, never meant to answer standalone. Confirms a real, structural limitation, not a defect - **flagged as a real pruning/relabeling candidate**: this harness has no way to test a draft model in its actual role (paired speculative decoding), so a low standalone score here is expected and uninformative, not evidence the file is bad. |
| `NPU/DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov` | openarc | 217.0 | 9/10 | clean, one genuine minor miss |
| `NPU/Qwen3-0.6B-fp16-ov` | openarc | 205.3 | 9/10 | clean, one genuine minor miss |
| `NPU/Qwen3-0.6B-int8-ov` | openarc | 229.1 | 9/10 | clean, one genuine minor miss |
| `DeepSeek-R1-Distill-Qwen-1.5B-int4-ov` | openarc | 168.7 | 9/10 | clean, one genuine minor miss |
| `hub/models--Qwen--Qwen3-0.6B` (stray misplaced dir) | **vllm** | 93.5 | 10/10 | clean - confirms the vLLM translator's fixes from last session generalize to a second, different model, not a one-off |

**Bottom line for the pruning decision**: of 16 models tested this round, 15 work cleanly (10 at perfect 10/10, 5 with a single genuine minor miss that isn't a bug). Only `Phi-4-mini-FastDraft-120M-int8-ov` is a real candidate for pruning-consideration or relabeling - not because it's broken, but because it's a draft-only model this harness has no meaningful way to evaluate standalone.

Combined with prior sessions' confirmed-broken models (`AEON-7`/`urakozz` MoE checkpoints - genuine Arcaine engine-format mismatch; `phi-2-int4-ov` - missing chat_template crashes OpenArc's worker), the full untested-catalog sweep is now essentially complete: 16 newly tested + 10 previously tested + 3 confirmed-broken = 29 of the catalog's real models have a real, current verdict.

## Twentieth session — llama-benchy was never actually a stub (2026-08-01, continued)

Direct question: what about llama-benchy? Long documented (see this file's own second session) as an intentional stub: "curl against plausible routes (/api, /openapi.json) returned 404" was taken as proof the running container exposed no API. **That conclusion was wrong** - guessed at routes instead of reading the container's own source. Re-investigated properly: `docker exec llama-benchy-web grep -n '@app.route' /app/web/app.py` shows a real, complete Flask JSON API - `POST /api/start`, `GET /api/run/<id>/stream` (SSE progress), `GET /api/results/<id>/export/json`, `GET /api/runs` - all confirmed by reading `web/app.py`/`web/engine.py` directly, not guessed.

Rewrote `llapdance/plugins/benchmark/llama_benchy.py` as a real adapter: `POST /api/start` with `{base_url, model, tokenizer, test_group}` (`base_url`/`model` confirmed passed straight to the real `llama-benchy` CLI's own `--base-url`/`--model` in `engine.py::_build_command()` - genuinely arbitrary OpenAI-compatible endpoint, not llama-benchy-specific), then consumes the SSE stream until `done`, then fetches the structured JSON result and aggregates `pp_throughput`/`tg_throughput`/`ttfr` means. This is the first adapter here with a genuinely async start/poll/fetch job shape, unlike every synchronous prober before it - needs its own `dashboard_url` config, distinct from `endpoint` (the model server under test).

**Two real live-validation gotchas, both found by running it for real, not by guessing:**
1. First live run returned all-null throughput metrics with no error surfaced - not a parsing bug. Read the actual result file from inside the container (`docker exec llama-benchy-web python3 -c "..."`) and confirmed every field was genuinely `null` - the underlying benchmark itself silently failed every request. Root cause: `llama-benchy-web` runs on its own docker bridge network (`ai-network`, not host networking) - the CLI subprocess resolves `base_url` from *inside that container's own network namespace*, and the suite had passed `127.0.0.1:8001` (the model server's host-published port), which inside that container is itself, not the host. `host.docker.internal` wasn't configured (`000` response); the bridge gateway IP (`docker inspect llama-benchy-web` -> `Networks.<net>.Gateway`) worked.
2. Confirmed real, sensible numbers on the corrected run: `pp_throughput` 336.5 tok/s (prefill), `tg_throughput` 17.4 tok/s (generation) against the real production `llama-cpp-bonsai` (27B Q2 GGUF) - consistent with this model's known performance elsewhere in this project. `llama-cpp-bonsai` confirmed still healthy/undisturbed throughout (`source.mode: external`, read-only, no lifecycle touched).

Validated live twice - direct adapter call and the full CLI (`llapdance run examples/validation-llama-benchy.suite.yaml`) - both against the real dashboard + real production model. 4 new tests (`httpx.MockTransport`, no real container needed in CI), all passing.

## Twenty-first session — TUI rebuild (2026-08-01, continued)

Direct, blunt feedback: the TUI was "all but un-usable." Read the original `llapdance/tui/app.py` (74 lines) and confirmed every specific complaint against the actual code, not defensively:

- **"How is a human supposed to read any of this?"** - the suite list was `table.add_row(str(path))`, raw file paths, nothing else.
- **"Initiate a test of a specific image against a specific backend - could they even?"** - no. The entire input mechanism was `self._search_dir.glob("**/*.suite.yaml")` - you could only run a YAML file that already existed on disk.
- **"Any method for seeing what models or backends are available?"** - no. The TUI never called `scan_models()`, `describe_engine()`, or the registry - all real, working, CLI-only.
- **"Entirely example suites you wrote, unorganized"** - confirmed, a flat unsorted glob of 40+ files by the time this session ran.
- **"No feedback mechanism"** - confirmed: a raw Python dict repr of `bench.metrics` dumped only after the *entire* suite finished - no per-stage visibility at all, because the orchestrator itself had zero logging/callback hooks of any kind.

User chose (asked directly, not assumed): a real interactive run builder, not a patched file browser.

### Orchestrator: added the progress visibility that never existed

`llapdance/core/orchestrator.py` had no logging, no print statements, no callback of any kind - confirmed by grepping the whole file. Added an opt-in `on_event: Callable[[str], None]` parameter (default a no-op, so every existing caller - CLI, MCP, tests - is unaffected) threaded through `run_backend`/`_run_external_backend`/`_wait_until_ready`/`_run_adapters_with_telemetry`/`run_suite`, firing at every real stage transition: resolving device(s), VRAM check, image prep, container start, health-check polling (including periodic "still waiting" updates, not just silence for the whole timeout), post-start requests, each benchmark/coherence adapter by name, container stop, done. 1 new test confirms real firing order (container must start before adapters run).

### New TUI: `llapdance/tui/screens.py` (new file) - three real screens

- **`ModelBrowserScreen`** - the same `scan_models()` + `annotate_tested_status()` the CLI's `llapdance models` uses, not a hardcoded list. Real format/quant/compatible-engines/tested-status columns.
- **`BuildScreen`** - pick a real registered engine (`available("engine")`) and a real discovered device (`discover_devices()`), supply an image, then generate a real suite (the same `TestSuite`/`BackendConfig` Pydantic models the CLI validates against) as **editable YAML** before running. Deliberately not fully automatic: this harness has too many real per-engine gotchas (HF cache symlinks, vLLM's `/dev/dri/by-path` mount, per-engine health-check conventions) to pretend one-size-fits-all defaults are always correct - showing the exact generated config and letting a human fix it is the honest choice, not a cop-out.
- **`RunScreen`** - live progress via the new `on_event` callback, streamed into a `RichLog` from a background thread; a clear color-coded PASS/FAIL banner (all coherence adapters at 100%) plus real per-adapter metrics at the end, not a raw dict.

### Three real bugs found getting the first live end-to-end validation to actually pass

Targeted `Phi-4-mini-instruct-int4-ov` via `openarc` (already known-good from earlier sessions) specifically to isolate TUI-introduced bugs from model/engine issues.

1. **Worker-thread bug**: `RunScreen.on_mount` called `self.run_worker(self._run(), thread=True, ...)` - the parentheses call `_run()` **immediately, on the main thread** (a plain method, not a coroutine), instead of passing the method reference. `_log()`'s `call_from_thread` then correctly detected it wasn't actually on a separate thread and raised. **The original TUI had the exact same mistake** (`self.run_worker(self._run(path), exclusive=True, thread=True)`) - there `_run` was `async def`, meaning Textual would've scheduled the coroutine onto the main event loop instead of a thread, so `run_suite()`'s fully synchronous, blocking call would have frozen the *entire UI*, not just looked frozen. Fixed: pass `self._run` (no call).
2. **Model-name mismatch**: the generated benchmark/coherence configs never set a `model` field, defaulting to the literal string `"default"`. Confirmed via a real container (manual reproduction, not guessed) that OpenArc doesn't fail cleanly on an unrecognized model name - it returns HTTP 200 then crashes the SSE stream mid-response (`ValueError: Model 'default' is not loaded or no worker is available`, the exact same error class as the earlier `phi-2` chat-template crash), which httpx surfaces as a raw `RemoteProtocolError`/disconnect. Fixed: `BuildScreen` now uses one consistent `served_name` across `backend_specific.model_name`/`served_model_name` and both adapters' `model` config.
3. (Structural, not a bug per se) `Static` widgets in this Textual version (8.2.8) don't expose `.renderable` - use `.render()`. Only affected test assertions, not the app itself.

**Fourth attempt: clean pass**, real container, real progress log through every stage in order, real metrics (88.6 tok/s), real 10/10 coherence, clear `PASS` banner, clean teardown confirmed via `docker ps -a`. 6 new tests (Textual's `run_test()` pilot harness, sync wrappers around `asyncio.run()` since `pytest-asyncio` isn't installed) plus the model-name-consistency regression test.

## Twenty-second session — TUI still wasn't clear, a second real usability pass (2026-08-01, continued)

Direct follow-up after actually trying the rebuilt TUI: "wasn't clear." Read the fresh `screens.py` looking for exactly this, and found a real, embarrassing gap: the file `import`ed `Button` and never placed one anywhere - the entire interaction relied on keybindings visible only as small text in the Footer widget (`s`/`enter`/`g`/`r`), with no on-screen instructions and a blank free-text image field with no discovery mechanism at all (confirmed via `grep -n "Button(" screens.py` returning nothing).

Fixed all three screens with real, visible, clickable buttons alongside the existing keybindings (neither replaces the other): `ModelBrowserScreen` gets a "Scan" button and a "Configure a run →" button; `BuildScreen` gets "Generate config ↓" / "Run this suite ▶" / "← Back"; `RunScreen` gets "← Back to model browser". Added a numbered step banner to each screen ("Step 1 of 3: ...", "Step 2 of 3: ...", "Step 3 of 3: ...") so the overall flow is visible without reading any documentation. Replaced the blank image `Input` with a real `Select` populated from `catalog.list_images()` (the same code `llapdance images list` uses) - pick from what's actually on the machine, or still type a tag directly if the image isn't local yet.

**A real CSS layout bug found via a pilot test, not guessed**: the very first attempt at a real button-click test (not just calling the action method directly - the whole point of this pass was proving buttons are actually clickable) failed with `OutOfBounds: Target offset is outside of currently-visible screen region` - on the *first* button, at the top of the screen. Inspected the actual computed region (`widget.region`) directly: the button's `x` coordinate landed exactly at the screen's width (one column past the last visible one). Root cause: Textual's `Input`/`Select` default to `width: 100%` - inside a `Horizontal` row next to a `Button`, that claims the *entire* row for itself regardless of siblings, pushing the button off past the edge. Fixed with real CSS (`Horizontal > Input/Select { width: 1fr; } Horizontal > Button { width: auto; }`) added to `LLAPDanceApp`.

**Validated live end to end, driven entirely by real `pilot.click()` calls on the actual button widgets** (not by calling `action_*` methods directly, which would have proven the logic but not the actual clickability): scan → click Configure → real image auto-fills from `arcaine-server:qwen35fix` (a real local image) → click Generate → click Run → real progress log through every stage → clean `PASS`, 88.9 tok/s, 10/10 coherence, delta shown against the prior run → click Back → lands on `BuildScreen` as expected. Clean teardown confirmed via `docker ps -a`. 2 new tests exercise real clicks (`pilot.click("#scan-btn")` etc., not action-method calls) - 131 passing total.

## Twenty-third session — test-by-model vs test-by-backend, and real sweep support in the TUI (2026-08-01, continued)

Direct request: a screen to test by model, a screen to test by backend, and a way to sweep from the TUI.

### `HomeScreen` + `BackendBrowserScreen` - two real entry points

Added `HomeScreen` (two buttons: "Test by model" / "Test by backend") as the new app entry point. "Test by model" is the existing flow unchanged. "Test by backend" is new (`BackendBrowserScreen`): pick a real registered engine first - its real sweepable params + known env flags shown immediately via `describe_engine()` (the same info `llapdance describe-engine` prints) - then a model table sorted compatible-first (`engine in m.compatible_engines`), so "what can I throw at OpenArc" is answerable without already knowing which models qualify. Both paths converge on the same `BuildScreen`, extended with an optional `preselected_engine` so "test by backend" doesn't lose the engine choice already made.

### Real sweep control in `BuildScreen`

A `sweep-param` `Select` populated from the chosen engine's real `describe_engine()` output (`params.shared.<key>` for each declared param, `env.<key>` for each known env flag - not a static list, changes per engine) plus a comma-separated `sweep-values` `Input`. `action_generate()` writes a real `sweep: [{param, values}]` block into the backend config - the actual `BackendConfig.sweep` mechanism (SPEC.md §10, `llapdance/config/sweep.py`), not a TUI-only concept; the generated YAML is the same shape a hand-written sweep suite would use. Values are coerced to int/float where they genuinely parse as one (`_coerce_sweep_value`) so e.g. `context_size` sweeps as real integers, not numeric-looking strings.

**Real validation gotcha, not a bug**: first live attempt tried sweeping `params.shared.context_size` against `openarc` and Textual's own `Select` raised `InvalidSelectValueError: Illegal select value` - correct, since `openarc`'s translator genuinely doesn't declare `context_size` as sweepable (confirmed in an earlier session: OpenArc's real tuning surface is `runtime_config`, not a scalar `context_size`). The dropdown was doing exactly what it should - only offering params real for the selected engine. Retried with `qxmx` (which does have one) instead of chasing a nonexistent bug.

**Validated live end to end**: `qxmx` + `Ternary-Bonsai-27B-Q2_0.gguf`, sweeping `params.shared.context_size` across `[2048, 4096]`. Real sweep options list confirmed (17 real params/env flags for qxmx). Generated YAML had a correct `sweep:` block; status message correctly said "expand into 2 real runs". `RunScreen`'s summary showed **two distinct real results**, correctly named by the sweep-expansion convention (`ternary-bonsai-27b-q2-0--context_size_2048` / `--context_size_4096`), both clean `PASS`, both 10/10 coherence, real distinct benchmark numbers (22.80 vs 22.87 tok/s). Clean teardown confirmed via `docker ps -a`.

7 new tests (`HomeScreen` button navigation ×2, `BackendBrowserScreen` compatibility marking + engine handoff to `BuildScreen`, sweep-value coercion, real sweep-axis generation), 136 passing total.

## Twenty-fourth session — model-table + button compaction, real DataTable height bug

Direct feedback: model name should be the first column (short relative path, not buried in a full absolute host path), and "the configure button requires a HUGE screen to see... this TUI requires far too many lines and a HUGE resolution."

### Short model names, path hidden entirely

Added `_short_model_name(path)`: joins the last two `Path(path).parts` (e.g. `/mnt/ignite/LLM/models/AEON-7/Ornith-1.0-abc` → `AEON-7/Ornith-1.0-abc`). `ModelBrowserScreen` and `BackendBrowserScreen` tables both now lead with a `Model` column built from this helper; the `Path` column and every full-path string were removed from both tables entirely - confirmed via test (`str(tmp_path) not in str(first_row[0])`).

### Button/label compaction

Shortened every on-screen label across all four screens (`Configure a run for the selected model →` → `Configure →`, `Generate config ↓` → `Generate`, `Run this suite ▶` → `Run ▶`, etc.) and removed several instructional `Static` paragraphs ("Step 1 of 3: ...") that duplicated what placeholder/prompt text on the same widgets already said. Engine param/env-flag info lines capped to `limit=4` names + "+N more" (`_short_list`) instead of wrapping across multiple lines for engines with many flags (Arcaine's `ARCAINE_QWEN35_*` family).

### Real DataTable height bug found and fixed

The actual mechanical cause of "requires a HUGE screen": confirmed via a real pilot check (`run_test(size=(100, 30))`, not the 140×60 used in earlier validation) that `#configure-btn`'s region was `Region(x=16, y=31, width=16, height=3)` on a `Size(width=100, height=30)` screen - `y=31` is one row past the visible height, i.e. genuinely invisible/unclickable at a normal terminal size, not just "small." Root cause: Textual's `DataTable` defaults to filling ALL remaining vertical space in its container - it claimed `Region(x=0, y=2, width=100, height=28)`, 28 of the 30 available rows, regardless of how many actual model rows existed.

Fixed with `DataTable { height: 12; }` in `LLAPDanceApp.CSS` (`app.py`) - the table scrolls internally past 12 rows instead of pushing everything below it off-screen. Re-verified with the same pilot: `configure-btn region: Region(x=16, y=21, width=16, height=3)`, fully inside the 100×30 screen, and confirmed clicking it navigates to `BuildScreen`. Also re-checked `BuildScreen`'s three buttons (`generate-btn`, `launch-btn`, `back-btn`) all visible at the same size after the label/layout compaction.

New regression test `test_configure_button_visible_at_a_normal_terminal_size` pins this at `size=(100, 30)` so it can't regress silently again. 138 passing total.

## Twenty-fifth session — engine-declared `image_hints`, real gap closed

Direct question after the arcaine/qxmx image-picker fix: does anything stop an architecturally-incompatible image (e.g. an OpenVINO/OpenArc image) from being wired into an engine that can't run it? Answer at the time: no - the picker's fix that session was a crude substring match on the engine's own name, and nothing on `EngineTranslator` itself declared which images it actually fits. Confirmed via grep: zero hits for any "image_pattern"/"compatible_image" concept anywhere in the codebase before this session.

Added `EngineTranslator.image_hints: list[str]` (`llapdance/plugins/base.py`) - fnmatch glob patterns of docker tags each engine has actually been validated/run against, same spirit and same class-attribute pattern as `sweepable_params`/`known_env_flags` (introspectable without instantiating, surfaced via `describe_engine()` and now `llapdance describe-engine`). Populated for all 5 reference engines from real evidence, not guessed:

- `qxmx`: `["qxmx:*", "llapdance/qxmx-from-source:*"]`
- `llama-cpp-sycl`: `["llama-cpp-bonsai:*"]` - real production tag from `llama_cpp_sycl.py`'s own docstring
- `arcaine`: `["arcaine-server:*", "arcaine:*"]`
- `openarc`: `["openarc:*"]`
- `vllm`: `["intel/vllm:*", "intel/llm-scaler-vllm*", "urakozz/vllm-xpu-env*"]`

Wired into `_local_image_options()` (`llapdance/tui/screens.py`): when an engine declares hints, filters local images via `fnmatch` against them instead of the crude substring-on-engine-name match; falls back to the old substring behavior only for an engine with no hints declared (so nothing regresses to showing zero images).

**Real gap the old substring approach had, confirmed live**: `llama-cpp-sycl`'s actual validated image is tagged `llama-cpp-bonsai:meat6-hardened` - the string `"llama-cpp-sycl"` never appears in it at all, so the previous fix (substring match on the engine's own name) would have shown ZERO images for this engine despite one existing locally. Confirmed via a real call: `_local_image_options("llama-cpp-sycl")` now correctly returns `['llama-cpp-bonsai:meat6-hardened']`; every other engine's real local images (qxmx: 8 tags, arcaine: 24 tags, openarc: 3 tags, vllm: 4 tags) filter correctly with no cross-engine leakage.

This is still a hint, not an enforced guarantee (documented on the attribute itself) - a tag matching a pattern could still be stale or the wrong build. It closes the "shows nothing" and "shows an unrelated engine's image" failure modes, not "silently loads the wrong architecture inside an image with a matching tag."

2 new/updated tests (`describe_engine`'s empty-catalog shape now includes `image_hints: []`), 140 passing total.

## Twenty-sixth session — mmproj hiding, reconciling the real llama.cpp image sprawl

Two direct requests: (1) hide mmproj companion files from model tables, but keep a compact indicator of their existence; (2) a real, sharper gap - "llama-cpp-sycl and llama-cpp and llama-cpp-intel and llama-cpp are treated as different?" - reconcile the local llama.cpp image sprawl (`llama-cpp-intel:meaton`, `llama-cpp-sycl:meat4-dnnfix`, `llama-cpp-bonsai:meat6-hardened`, `llama-cpp-vulkan:prism-bonsai`, `llama-cpp-vulkan:newmeat2`, built "almost daily") against the harness's engine model, which only ever had one llama.cpp translator.

**mmproj**: `scan_models()` (`llapdance/core/model_catalog.py`) now excludes any `*.gguf` file with `"mmproj"` in its name from results entirely, but adds `ModelInfo.has_mmproj: bool` - True when a sibling mmproj file exists in the same directory. TUI tables (`_model_name_cell` in `llapdance/tui/screens.py`) append ` (m)` to the short name when set. Real effect on `/mnt/ignite/LLM/models`: 18 gguf rows -> 13, the 5 removed were all genuine mmproj companions, `(m)` correctly appears on their real siblings (e.g. `Ternary-Bonsai-27B-gguf/Ternary-Bonsai-27B-Q2_0.gguf (m)`).

**llama.cpp image reconciliation - investigated for real, not assumed**: `docker run --entrypoint sh <image> -c "ls /app | grep ggml"` against all 5 local tags, plus `docker inspect` for baked-in env and ENTRYPOINT:

| tag | GGML backend lib | oneAPI env baked in | ENTRYPOINT |
|---|---|---|---|
| `llama-cpp-intel:meaton` | `libggml-sycl.so` | yes | `/app/llama-server` |
| `llama-cpp-sycl:meat4-dnnfix` | `libggml-sycl.so` | yes | `/app/llama-server` |
| `llama-cpp-bonsai:meat6-hardened` | `libggml-sycl.so` | yes | `/app/llama-server` |
| `llama-cpp-vulkan:prism-bonsai` | `libggml-vulkan.so` | no | `/app/llama-server` |
| `llama-cpp-vulkan:newmeat2` | `libggml-vulkan.so` | no | **`/app/tools.sh`** |

Confirms: `llama-cpp-intel`/`llama-cpp-sycl`/`llama-cpp-bonsai` are genuinely the SAME backend (SYCL) under different tag names/build dates - correctly one engine. `llama-cpp-vulkan` is genuinely a DIFFERENT GGML backend (Vulkan, no oneAPI at all) - there was previously NO translator for it at all; any Vulkan image had to go through raw `command`/`env` passthrough, no TUI/CLI convenience path.

**Fixed**: broadened `llama-cpp-sycl`'s `image_hints` to all three real SYCL tags (`llama-cpp-bonsai:*`, `llama-cpp-sycl:*`, `llama-cpp-intel:*`). Added a new `llama-cpp-vulkan` engine translator (`llapdance/plugins/engine/llama_cpp_vulkan.py`) - same `params.shared` mapping as the SYCL translator (llama.cpp's CLI is backend-agnostic for these), but its own real `GGML_VK_*` env-flag catalog (found by reading this project's actual local Vulkan checkout, `~/llama.cpp.git/llama.cpp.prism/ggml/src/ggml-vulkan/ggml-vulkan.cpp`'s `getenv()` call sites - not guessed, not the SYCL flag list). Registered, added to `COMPATIBLE_ENGINES["gguf"]` in `model_catalog.py` (gguf models now correctly show all 3 llama.cpp-family engines plus qxmx).

**Real gotcha found and deliberately NOT papered over**: `llama-cpp-vulkan:newmeat2`'s `ENTRYPOINT` is `/app/tools.sh`, not `/app/llama-server` - despite containing a working llama-server binary + `libggml-vulkan.so` internally. This harness's `local_docker` execution adapter has no per-backend entrypoint override, so pointing this translator at that tag would silently append llama-server flags to `tools.sh`'s argv instead of running the server. `llama-cpp-vulkan`'s `image_hints` is therefore deliberately narrow (`["llama-cpp-vulkan:prism*"]`, not a blanket `llama-cpp-vulkan:*`) to exclude it - documented in the engine's own docstring as a real breadcrumb: check `docker inspect --format '{{.Config.Entrypoint}}'` on any newly built tag before trusting it fits this translator, same practice as the earlier qxmx/bonsai CMD-vs-ENTRYPOINT gotcha.

GPU pinning for the new Vulkan translator is NOT validated live this session (unlike SYCL's, which was) - documented as an open assumption in the file's own docstring, not silently inherited as "confirmed" from the SYCL translator's real validation.

Verified live: `_local_image_options("llama-cpp-sycl")` now correctly returns all 3 real SYCL tags; `_local_image_options("llama-cpp-vulkan")` returns only `llama-cpp-vulkan:prism-bonsai`, never `newmeat2`. A real Ternary-Bonsai model's engine dropdown now lists `llama-cpp-sycl`, `llama-cpp-vulkan`, `qxmx`; switching to `llama-cpp-vulkan` correctly fills the image field with `llama-cpp-vulkan:prism-bonsai`.

8 new tests (mmproj hiding/flagging, `llama-cpp-vulkan` translator behavior mirroring the SYCL test suite, image_hints exclusion check), 146 passing total.

## Twenty-seventh session — real llama-cpp-sycl flag catalog gap, multi-axis TUI sweep

Two direct complaints: (1) sweep options are missing real performance-tuning flags - "look through for the actual flags that do things like turn on and off graph for intel"; (2) sweeping in the TUI was "one at a time" - needs the ability to sweep several params together.

**Flag catalog gap - real, confirmed via source, not guessed**: `llama_cpp_sycl.py`'s `known_env_flags` was built from a naive `grep getenv(` (3 flags) that completely missed every flag read through ggml-sycl's own `ggml_sycl_get_env()` wrapper (`ggml/src/ggml-sycl/common.cpp`) instead of a bare `getenv()` call. Re-derived from this project's own local checkout (`~/llama.cpp.git/llama.cpp`, the most recent of several local llama.cpp checkouts, `2026-07-31`) - every real call site across `ggml-sycl.cpp`/`fattn.cpp`/`fattn-mkl.cpp`. Found exactly what the user's own example ("graph on/off for intel") turned out to be: `GGML_SYCL_ENABLE_GRAPH` (default 0) - SYCL command-graph capture/replay, with a real gotcha of its own: it's gated behind a BUILD-TIME cmake option (`GGML_SYCL_GRAPH`) - if an image wasn't compiled with it, the env var is a silent no-op (the binary logs "graph disabled by compile flag" at its own startup, confirmed by reading the log-emitting code). 18 real flags now cataloged total (was 3), including oneDNN/MKL flash-attention path toggles (`GGML_SYCL_ENABLE_DNN`, `GGML_SYCL_FA_ONEDNN`, `GGML_SYCL_ENABLE_MKL_FA`), op-fusion (`GGML_SYCL_ENABLE_FUSION`), async memory ops, and the level-zero-vs-generic-SYCL-API selector - each with its real default value where the source states one.

Scope note: this pass focused on `llama-cpp-sycl` specifically (the concrete example raised); Arcaine's/qxmx's/vLLM's/OpenArc's catalogs were NOT re-audited for similar gaps this session - Arcaine in particular already has a known-incomplete NVFP4/MoE flag set flagged in an earlier NEXT_STEPS entry, still open.

**Multi-axis sweep in the TUI**: `BuildScreen`'s sweep control was a single `Select` + single `Input`, generating exactly one `SweepAxis`. Added a `+ axis` button and a new `#sweep-axes` `TextArea` (height-capped at 4 rows - re-verified at a real 100x30 terminal that every button, including the new one, stays visible, same discipline as the earlier DataTable-height fix) holding one `param=values` line per axis. Clicking `+ axis` appends the current param/values builder row as a new line and clears the builder for reuse; the box is also directly hand-editable. `action_generate()` now parses every line via the new `_parse_sweep_axes_text()` helper AND folds in whatever's still sitting unclicked in the builder row, so a single-axis sweep still needs no extra click. Status message now reports the real total run count as the product of every axis's value count (`N runs, sweeping paramA, paramB`), not just one param's length.

The underlying multi-axis mechanism itself was never the gap - `llapdance/config/sweep.py`'s cartesian-product expansion already supported any number of axes (validated in an earlier session with a single axis); only the TUI's input surface was artificially single-axis.

Verified live: a real pilot at `size=(100, 30)` (matching the earlier DataTable-height validation) confirms `add-sweep-btn` and all three action buttons stay visible and clickable; a real click on `+ axis` correctly appends `params.shared.context_size=2048,4096` to the box; `Generate` with that plus a second unclicked builder-row axis produces `Generated (4 runs, sweeping params.shared.context_size, params.shared.parallel_slots)`.

7 new tests (flag-line parsing incl. malformed-line rejection, add-axis button behavior, multi-axis YAML generation + run-count math), 151 passing total.

## Updated adapter status (see README.md, now reflects reality instead of aspiration)

| Adapter | Status |
|---|---|
## Twenty-eighth session — generic-http PP/TG split, cross-checked live against llama-benchy

Direct request: the earlier "PP/TG data mostly missing" gap (33 of 34 stored results only had a blended tok/s number) needed a real fix, not just a diagnosis - extend `generic-http` to measure PP and TG separately, then validate the new numbers against the one real `llama-benchy` result on file for the same model to see if they agree.

### What was added

`generic_http.py` now tries two real sources for a PP/TG split, in order, never fabricating one:

1. **`timings.prompt_per_second` / `timings.predicted_per_second`** - llama.cpp's OWN server-side split (confirmed via this project's local llama.cpp checkout, `tools/server/server-task.cpp`: `timings.prompt_n`/`predicted_n`/`prompt_per_second`/`predicted_per_second` are computed server-side and pushed onto the final chunk unconditionally - this adapter already read `timings.predicted_n` from the same object for token counting, so no new chunk parsing was needed, just reading more of what was already there). Most accurate - excludes client/network overhead entirely.
2. **`usage.prompt_tokens` + `usage.completion_tokens`** (confirmed present together in qxmx's real responses, `tools/qxmx_serve.cpp`), with no server timing object: PP derived as `prompt_tokens / ttft`, TG as `(completion_tokens - 1) / (total - ttft)` - the same TTFT-based approximation llama-bench-style tools use (prefill ends at the first token).

If neither exists (Arcaine/OpenArc/vLLM's real captured responses have neither), PP/TG are **left out of `metrics` entirely** - never backfilled with a guess or a fake `0.0`. Blended `avg_tokens_per_sec` is untouched, so every prior result/test stays valid. Added `request_extra` (raw passthrough dict merged into the request body) alongside this - needed for the honest comparison below, see the gotcha.

7 new tests (source-priority unit tests + 3 full `run()`-level tests against a mocked SSE stream shaped like each of: llama.cpp's timings object, qxmx's usage-only shape, and Arcaine/OpenArc's neither-shape - plus `request_extra` passthrough), 160 passing total.

### Real gotcha found live, comparing against llama-benchy

First live comparison (against the real, already-running production `llama-cpp-bonsai` container, `source.mode: external`, read-only, lifecycle never touched, confirmed via `docker ps -a` before and after) looked broken: TG lined up well (~18.5 tok/s vs llama-benchy's stored 17.3 tok/s, ~7% apart), but PP did not (22–55 tok/s vs llama-benchy's stored 335.7 tok/s - over 10x off).

Root cause, found by reading the actual `timings` object instead of guessing: llama.cpp's server has **prompt caching on by default** (`cache_n` in `timings`). This adapter's default prompt is short and gets sent repeatedly across a benchmark's `num_requests` - confirmed live that a repeat request reused 18 of the prompt's tokens from cache and only prefilled 4 fresh ones, so "prompt_per_second" was measuring a tiny, batch-size-starved handful of tokens, nothing like the large-batch prefill llama-benchy measured (`prompt_size: 2048`, `cache_prompt` not applicable to its own harness).

Fixed the comparison, not the adapter's default behavior (a warm-cache PP number is real and useful for a "second identical request" scenario - it just isn't the number to compare against a synthetic large-prompt benchmark): passed `request_extra: {"cache_prompt": false}` plus a genuinely large, unique (non-cacheable) ~7400-token synthetic prompt. Result, through the actual registered adapter, live:

| Source | PP tok/s | TG tok/s | Prompt size |
|---|---|---|---|
| `llama-benchy` (stored result, `bonsai-llama-benchy`) | 335.7 | 17.3 | 2048 (synthetic) |
| `generic-http`, cache disabled (this session, live) | 297.6 | 16.3 | ~7430 (synthetic, unique) |

**They line up** - PP within ~11%, TG within ~6%, well inside normal run-to-run GPU/thermal/contention variance and the two tools' different exact prompt/response lengths. This is real cross-validation that the new PP/TG numbers are measuring the same real thing llama-benchy measures, not an artifact of this adapter's own implementation - once the prompt-caching confound is controlled for.

Container confirmed still healthy and untouched throughout (`docker ps -a`: `llama-cpp-bonsai   Up 4 days (healthy)`, unchanged).

### Real doors this opens

`llama-benchy` is far more thorough (concurrency sweeps, percentile stats, multiple prompt/response-size combinations) - this doesn't replace it. But every backend already validated with `generic-http` (arcaine/openarc/qxmx/vllm, five real servers total) that turns out to expose either signal above now gets a real PP/TG split for free, no new integration needed, and llama.cpp-family backends (`llama-cpp-sycl`, `llama-cpp-vulkan`) get the MOST accurate version (server-side timings) automatically, with zero config.

| `local-docker` execution | Real, validated against **four** different engines (llama.cpp, qxmx, Arcaine, OpenArc) on real GPU hardware, plus a real build-from-source run. |
| `ssh-docker` execution | **Built and validated.** Real remote host, real stop/test/restore cycle around the host's own production container. `prebuilt` only for now (see above). |
| `generic-http` benchmark | Real, validated against five different real servers/paths (four engines + external/llm-proxy), all producing real TTFT/throughput numbers. Token-counting bug found and fixed (see below) - now records `counted_via` per request. **PP/TG split added and cross-validated live against a real `llama-benchy` run on the same backend (see below)** - lines up within ~11%/~6% once a prompt-caching confound is controlled for via the new `request_extra` passthrough. |
| `fixed-questions` coherence | Real, validated — 10/10 (or a genuine 9/10 model error) across five different real backends/paths. (An earlier draft of this doc claimed it also caught a "tokenizer crash bug" — retracted, see above; that crash was my invalid test setup, not a finding.) |
| `flat-file` storage | Real, validated — write + delta-lookup both exercised, across all backends including external mode. |
| `opensearch` storage | Built and validated (prior session). Real write+query round-trip against a live instance, including catching and fixing a silent timestamp-precision bug. Storage fan-out (flat-file + opensearch simultaneously) confirmed. |
| Intel VRAM preflight | Real, validated — `xpumcli`-backed where available; correctly falls back to `allow_unknown_vram`-gated fail-closed on a host with no VRAM-capable tooling (screamer). |
| `source.mode: build` | Built and validated (prior session) locally; explicitly NOT supported yet over SSH (see above). |
| `source.mode: external` | **Built and validated.** No container lifecycle at all - benchmark/coherence adapters pointed directly at an already-running endpoint (validated via `llm-proxy`). `device_note` is the only device identity captured, explicitly and permanently unverified. |
| GPU device identity tracking | **Built and validated.** `RunResult.device_target` now carries full `DeviceInfo` (vendor/name/pci_bus_id/render_node) plus a `verified` flag, not just a bare index. Real hostname captured for local runs too. |
| `llama-benchy` benchmark | Still a stub — no new information. |
| `guidellm` benchmark | **Attempted for real, shipped as a stub.** Structural limitation found (tokenizer resolution requires a real HF Hub repo id), not a guess - see above. |
| `xmxmon` telemetry | **Built and validated.** New `telemetry` plugin kind. Real capture against a real run - also caught a real "wrong physical GPU" mismatch live (see above). |
| MCP server | **Built and validated.** Real stdio client, all 5 tools, a genuine `run_suite` execution pulled back via `get_results`. |
| Sweep/parameter-matrix automation (SPEC.md §10) | **Built and validated.** Real 2-value sweep produced 2 real container runs automatically. |
| Engine sweepable-params catalog (SPEC.md §10) | **Built.** `describe-engine` CLI command + MCP tool, populated for all 4 reference engines. |
| Image catalog & cleanup (SPEC.md §12) | **Built and validated** against real image sprawl, including the good-label removal safety check. |
| Model catalog + format/backend compatibility (new, not in original spec) | **Built and validated** against ground truth - all 3 previously-validated cross-format models matched exactly. |
| Embedded-DB / Prometheus storage | Not built. |
| `params.shared` → per-engine `command`/`env`/`devices`/`post_start_requests` translation | Built and validated against four engines (prior session). Raw passthrough remains available for anything a translator doesn't cover. |

## Twenty-ninth session — real PP/TG data run across every engine, one real crash found and fixed

Direct follow-up to the PP/TG work: "run our models through using our tooling" - 5 real live runs through the actual harness (not ad hoc scripts), to see which real backends' PP/TG support holds up and to populate the results artifact with genuine new data.

Device index 3 (`renderD131`, PCI `0000:8a:00.0`) confirmed idle via `xpumcli`/`probe.discover_devices()` before starting anything - devices 1 and 2 are the real production `llama-cpp-bonsai`/`vllm-urak` containers, never targeted.

**qxmx** (`examples/validation-qxmx-pptg.suite.yaml`, real container boot+teardown): PP 64.3, TG 25.0 tok/s (`ttft_split` source, as expected - qxmx has no server timings). 10/10 coherence. Volumes path updated to the new `/mnt/ignite/LLM/models/gguf` location from the earlier move.

**OpenArc** (`validation-openarc-qwen3-0.6b.suite.yaml`, real container boot+teardown): PP 156.7, TG 414.0 tok/s. **Real new finding**: OpenArc DOES expose `usage.prompt_tokens` - the original PP/TG session hadn't run a live check against it and had assumed (correctly cautious, but not confirmed) it might not. Closes another real backend, not just qxmx/llama.cpp-family.

**Arcaine** (`validation-arcaine.suite.yaml`, real container boot+teardown, diffusiongemma model): blended-only, confirmed again live (16.24 tok/s, consistent with the prior 16.35 stored result). **Re-confirmed, not a gap**: Arcaine's real HTTP responses genuinely carry no prompt-token count anywhere - checked directly against a live container, not assumed from source alone this time.

**vllm-urak** (new `validation-vllm-urak-pptg.suite.yaml`, `source.mode: external`, read-only, lifecycle never touched): PP 403.6, TG 84.4 tok/s. Needed `request_extra: {stream_options: {include_usage: true}}` - confirmed live that vLLM's stream carries NO `usage` object by default, only when a client explicitly asks via the standard OpenAI `stream_options.include_usage` flag (confirmed working against this exact server: `usage.prompt_tokens: 20` appeared once requested).

**llama-cpp-bonsai** (new `validation-bonsai-pptg.suite.yaml`, `source.mode: external`, read-only): PP 299.5, TG 16.2 tok/s, 9/10 coherence, run through the actual `run_suite()`/flat-file-storage path this time (the earlier ad hoc comparison never went through real storage) - a genuine ~7400-token unique prompt with `cache_prompt: false` via `request_extra`, confirming the earlier one-off finding now as a properly stored, reproducible result.

### Real bug found and fixed live: `fixed-questions` crashed against vLLM's Ornith model

`vllm-urak-pptg`'s first run crashed with `AttributeError: 'NoneType' object has no attribute 'lower'`. Root cause, confirmed via a direct `curl` against the real container: `Ornith-1.0-35B-int4-AutoRound`'s response has `message.content: null` - the entire answer sits in a non-standard `message.reasoning` field instead (`"reasoning": "...12 + 30 = 42."`, `finish_reason: "length"` - the whole token budget went into the thinking trace). Same failure MODE as the already-documented `llama-cpp-sycl` `LLAMA_ARG_REASONING` gotcha, different field name, different backend.

Fixed in `llapdance/plugins/coherence/fixed_questions.py`'s `_ask()`: falls back through `content` → `reasoning_content` → `reasoning` → empty string, so a model that puts its answer somewhere non-standard gets a real, gradeable "wrong/no answer" failure instead of crashing the whole coherence run. 2 new tests (`reasoning`-field fallback, and the empty-string case when nothing is present).

### Results artifact updated

Regenerated with the fresh dataset: 39 real benchmark records (was 34), 5 with a real PP/TG split (was 1) - qxmx, OpenArc, vLLM, and llama.cpp-family now report real PP/TG; Arcaine confirmed genuinely blended-only, not an integration gap. Same URL, republished in place.

162 passing total (2 new coherence tests on top of the prior session's 160).

## Thirtieth session — sweep-value defaults, no more guessing on/off spelling

Direct complaint: the sweep-values field never suggested anything - every sweep meant re-guessing on/off spelling (TRUE? ON? 1?) and re-typing common numeric pairs (2048,4096) from memory every time.

Added `_default_sweep_values(info)` (`llapdance/tui/screens.py`) - derives a real starting pair from the exact same catalog metadata `describe_engine()` already surfaces (`values`/`default`/`type`), never inventing a magnitude that isn't already in that entry:

- Enum-like params (`values` present, e.g. `kv_cache_quant`, `reasoning`, OpenArc's `model_type`) → all real values joined verbatim.
- Anything boolean/binary-shaped (`type` containing "bool"/"presence"/"0/1", or a literal `bool` default) → `"0,1"`. Checked every engine's actual parsing before committing to this - qxmx's `atoi`, ggml-sycl/vulkan's `getenv(...) != nullptr`/`ggml_sycl_get_env` int parsing, vLLM's own truthy `params.get(...)` checks (`vllm.py`'s `build()`) - every one treats a plain `"0"`/`"1"` string correctly. One real, uniform answer across the whole codebase, not a per-engine guess - directly answers "should I use TRUE/FALSE or ON/OFF".
- A numeric `default` with no `values` → bracketed around it (e.g. `context_size` default 4096 → `"2048,4096"`, the exact pair every sweep test this project has wanted anyway). Small-default edge case handled: halving a default of 1 (qxmx's `parallel_slots`) would produce `"1,1"` (not a real second point) - steps up to `"1,2"` instead.
- Nothing declared (no `values`, no `default`) → empty, still just the placeholder - never a fabricated suggestion (e.g. Arcaine's `layer_placement`/`denoising_steps`/`seed`, none of which have a documented sensible default).

Wired into `BuildScreen.on_select_changed`: picking a sweep param now prefills `#sweep-values` with its real default immediately; still a plain editable `Input`, exactly as requested ("the setting can still be a text field... that is fine"). Guarded against a real race found while testing this live: `action_add_sweep_axis()` clears `#sweep-param` back to blank without an intervening event-loop pause in some call sites, so a queued `Select.Changed` for the *previous* selection could still be pending when that happens and re-fill the just-cleared field - fixed by checking the event's value still matches the widget's *current* value before acting on it.

Also added one legitimately-known missing default: `llama-cpp-sycl`/`llama-cpp-vulkan`'s `parallel_slots` had no `"default"` at all despite both translators' own module docstrings already documenting that llama.cpp's real "auto" behavior picks 4 - added `"default": 4` backed by that existing evidence, not new speculation.

Verified live: `context_size` → `"2048,4096"`, `kv_cache_quant` → `"f16,q8_0"` (or `"f16,q8_0,f8"` for qxmx, correctly per-engine), `reasoning` → `"on,off,auto"`, `GGML_SYCL_ENABLE_GRAPH` → `"0,1"`, `GGML_SYCL_MKL_FA_Q_TILE` (default 8192) → `"4096,8192"`, `parallel_slots` (qxmx, default 1) → `"1,2"`. Re-confirmed at a real 100×30 terminal that every button (including `+ axis`) is still visible - no regression on the earlier DataTable-height fix.

9 new tests (helper unit tests covering every branch, plus a live prefill-on-selection test), 168 passing total.

### Real bug found immediately after shipping the above: env sweep values crashed pydantic validation

Direct user report, real error text: `Input should be a valid string [type=string_type, input_value=0, input_type=int]` on `env.GGML_SYCL_ENABLE_GRAPH`/`env.GGML_SYCL_ENABLE_DNN`/`env.GGML_SYCL_FA_ONEDNN` - i.e. exactly the new `"0,1"` defaults this session just added, clicked through as-is.

Root cause: `BackendConfig.env` is `dict[str, str]` (env vars are always strings at the OS/docker level) - `_coerce_sweep_value` coerces `"0"`/`"1"` to real ints for convenience on `params.shared`/`params.backend_specific` (open `Any` dicts, where that's correct and intentional), but was applied uniformly to `env.*` sweep values too. The int survives suite-YAML validation fine (`SweepAxis.values` is `list[Any]`) and only blows up when `llapdance/config/sweep.py`'s `expand_backend_sweep()` writes it into the expanded backend's `env` dict and re-validates - which is why the earlier multi-axis sweep tests (all `params.shared.*`) never caught it.

Fixed with `_coerce_sweep_value_for_param(param, raw)`: skips numeric coercion entirely when `param` starts with `env.`, keeping those values as plain strings; both call sites (`_parse_sweep_axes_text`, `action_generate`'s builder-row fold-in) switched to it. Verified via `expand_backend_sweep()` directly (the actual code path that crashed) - `env.GGML_SYCL_ENABLE_GRAPH` swept across `["0", "1"]` now expands cleanly into two real backends with `env == {"GGML_SYCL_ENABLE_GRAPH": "0"}` / `{"...": "1"}`, no crash.

New regression test goes one step further than the existing sweep tests specifically because they didn't catch this: it calls `expand_backend_sweep()` on the real generated backend, not just `TestSuite.model_validate()` on the compact form. 169 passing total.
