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

## Updated adapter status (see README.md, now reflects reality instead of aspiration)

| Adapter | Status |
|---|---|
| `local-docker` execution | Real, validated against **four** different engines (llama.cpp, qxmx, Arcaine, OpenArc) on real GPU hardware, plus a real build-from-source run. |
| `ssh-docker` execution | **Built and validated.** Real remote host, real stop/test/restore cycle around the host's own production container. `prebuilt` only for now (see above). |
| `generic-http` benchmark | Real, validated against five different real servers/paths (four engines + external/llm-proxy), all producing real TTFT/throughput numbers. Token-counting bug found and fixed (see below) - now records `counted_via` per request. |
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
