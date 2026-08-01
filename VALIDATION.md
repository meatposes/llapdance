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

## Updated adapter status (see README.md, now reflects reality instead of aspiration)

| Adapter | Status |
|---|---|
| `local-docker` execution | Real, validated against **four** different engines (llama.cpp, qxmx, Arcaine, OpenArc) on real GPU hardware, plus a real build-from-source run. |
| `generic-http` benchmark | Real, validated against four different real servers, all producing real TTFT/throughput numbers (though see Arcaine's throughput-undercounting note above). |
| `fixed-questions` coherence | Real, validated — 10/10 (or a genuine 9/10 model error) across four different working backends. (An earlier draft of this doc claimed it also caught a "tokenizer crash bug" — retracted, see above; that crash was my invalid test setup, not a finding.) |
| `flat-file` storage | Real, validated — write + delta-lookup both exercised, across all four backends. |
| `opensearch` storage | **Built and validated.** Real write+query round-trip against a live instance, including catching and fixing a silent timestamp-precision bug (see above). Storage fan-out (flat-file + opensearch simultaneously) confirmed working through a real suite run. |
| Intel VRAM preflight | Real, validated — `xpumcli`-backed, confirmed to actually reject an impossible VRAM requirement with a real free-memory number, not just documented as a placeholder. |
| `source.mode: build` | **Built and validated.** Real git clone (not `prebuilt`), real docker build, real run. Found and fixed a real safety gap (uncommitted-changes check before `git checkout`) and built real build-version tracking (git SHA in image tag). |
| `llama-benchy` benchmark | Still a stub — unrelated to this validation, no new information. |
| SSH execution target | Not built. `RunningBackend`/`ExecutionTargetAdapter` contract exercised only locally across four engines; nothing contradicts it working remotely, but it's unverified. |
| Embedded-DB / Prometheus storage | Not built. |
| MCP integration | Noted in SPEC.md §13 and `cli.py` as explicit future work. Not built. |
| `params.shared` → per-engine `command`/`env`/`devices`/`post_start_requests` translation | **Built and validated against four engines.** `EngineTranslator` plugin kind + `llama-cpp-sycl`/`qxmx`/`arcaine`/`openarc` reference implementations. `post_start_requests` (new this session) covers engines where model-loading is a separate step from container start. Raw passthrough remains available for anything a translator doesn't cover. |
