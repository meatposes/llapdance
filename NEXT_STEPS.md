# Next steps / in-progress work log

Live document — updated as work progresses. See VALIDATION.md for the full
writeup and SPEC_REVIEW.md for the "are we still on track" assessment (now
partly resolved - see its update note at the top).

## Session complete (2026-08-01, continued) — real Qwen3.5 sweep, stale-image bug found+fixed

Direct request: find a real local Qwen3.5/3.6 model Arcaine supports, sweep for optimal flags.

- [x] Found the real model (`unsloth/Qwen3.6-27B-NVFP4`, `config.json` `model_type: "qwen3_5"` matches Arcaine's dispatch exactly) and cataloged all 13 real `ARCAINE_QWEN35_*` env flags by reading every `getenv()` site.
- [x] **Real bug found and fixed**: `arcaine-server:latest` predates Arcaine commit `f6724df` (a KV-cache/recurrent-state reset fix) — every 2nd+ sequential request against any Qwen3.5 model 500s (`"Qwen3.5 KV cache position mismatch"`), and without the crash, the linear-attention layers would silently carry over the previous request's state instead. Rebuilt from current HEAD as `arcaine-server:qwen35fix`, validated live (5/5 sequential requests OK). Recipe saved as `examples/Dockerfile.arcaine-server-rebuild` since the original build file isn't tracked in `~/Arcaine` at all.
- [x] **Ran the real sweep** (`env.ARCAINE_QWEN35_NVFP4_DPAS` = `["0","1"]`) against the fixed image. Result: **refutes the source's own performance comment** — `DPAS=1` (oneDNN BMG f4 path) was both slower (9.90 vs 10.16 tok/s) and measurably less correct (5/10 vs 9/10 fixed-questions, including wrong basic arithmetic) than the `DPAS=0` default. See VALIDATION.md "Tenth session" for full numbers and failure examples.
- [x] Full test suite green (98 passed), committed.

## Prior session (2026-08-01, continued) — sweep automation, image catalog, model catalog

Direct follow-up to SPEC_REVIEW.md's top recommendation:

- [x] **Sweep/parameter-matrix automation built and validated** — `BackendConfig.sweep` + `llapdance/config/sweep.py`, real 2-value sweep produced 2 real automatic runs. Found and fixed a real design bug: sweep axes into `params.shared`/`backend_specific` need to allow introducing a NEW key, not just varying an existing one.
- [x] **Engine sweepable-params catalog built** — `EngineTranslator.sweepable_params` populated for all 4 engines, `llapdance describe-engine <name>` / MCP tool.
- [x] **Image catalog & cleanup built and validated** — `llapdance/core/catalog.py`, `llapdance images list/label/rm`, wraps the `ExecutionTargetAdapter.list_images()` that's existed since session one but was never consumed. Validated against real image sprawl, including the good-label removal safety check (tested against a disposable tag, never the real sprawl).
- [x] **Model catalog + format/backend compatibility built and validated** (new capability, added per direct request) — `llapdance/core/model_catalog.py`, `llapdance models <dir>...`. Scans GGUF/OpenVINO-IR/HF-safetensors, reports quant hint + which engines could plausibly load each (could-run-on, not will-run). Validated against ground truth: all 3 previously-validated cross-format models matched exactly.
- [x] SPEC.md/README.md/VALIDATION.md/SPEC_REVIEW.md updated, full test suite passing (91 tests), committed + pushed.

## Also done this session (in response to a direct follow-up question)

- [x] **Confirmed sweep generalizes beyond `params.shared`** — raw engine env flags (`env.X`) and even build-time cmake flags (`source.build.build_args.X`) sweep via the exact same generic mechanism, no special-casing. Found the real flags by reading `ggml-sycl`'s actual source: `GGML_SYCL_NO_PINNED`, `GGML_OP_OFFLOAD_MIN_BATCH` (runtime env, **validated live** against a real container via `docker exec`), and `GGML_SYCL_DNNL` (a build-time cmake option controlling whether oneDNN gets linked in at all - structurally supported, **not** validated live since a from-source oneDNN rebuild is slow).
- [x] **`describe-engine` extended** — was translator-params-only, now also returns `env_flags` (`EngineTranslator.known_env_flags`), populated for `llama-cpp-sycl` with the three flags found this session.

## Also done this session — cataloged known_env_flags for the other 3 engines

- [x] **qxmx** — read every `getenv()` in its source, cataloged ~16 real flags (perf-tuning + debug-only, labeled which is which). Confirmed qxmx has no oneDNN dependency at all. **Validated live**: swept `env.QXMX_CHUNK` across 2 values, both real runs 10/10 coherence.
- [x] **Arcaine** — cataloged the real oneDNN toggle for the validated `diffusion_gemma` model family: `DIFF_ONEDNN_SDPA`, a **runtime env var** (unlike llama.cpp's build-time `GGML_SYCL_DNNL` - same underlying library, different toggle mechanism). Plus `DIFF_ARENA`/`DISABLE_SCRATCH`, `DIFF_PREFILL_CHUNK`, `DIFF_FORCE_DENOISE_STEPS`, `DIFF_HOST_SAMPLER`. Deliberately left uncataloged (and said so in code): a whole separate Qwen3.5 model family (~15 flags) never validated by this harness, and ~13 NVFP4-specific + MoE-specific flags found but not individually characterized.
- [x] **OpenArc** — real gap found and fixed, not just documented: the translator never forwarded `runtime_config` (OpenArc's actual OpenVINO-tuning surface, confirmed via its own source) even though it looked structurally sweepable. Fixed, **validated live** with a real `PERFORMANCE_HINT` override against the real server.

## Recommended next session

Per the (now mostly-addressed) spec review, remaining real gaps in priority order:

1. **Characterize the remaining Arcaine flags** — ~13 `DIFF_NVFP4_*` flags (directly relevant, the validated model IS NVFP4) and MoE-specific flags (`DIFF_MOE_STATS`, `DIFF_MOE_TAIL_CAP`) were found but not individually read/documented to the same standard as `DIFF_ONEDNN_SDPA`. The Qwen3.5 model family (`ARCAINE_QWEN35_*`) is a bigger undertaking - only worth doing if that model family actually gets validated against this harness.
2. **Validate build-arg sweeping live** — the `source.build.build_args.X` path (e.g. sweeping `GGML_SYCL_DNNL=0` vs `1`) is only unit-tested; a real rebuild-sweep would confirm it end to end, at the cost of a slow oneDNN-from-source build per value.
3. **Device-target sweeping** — sweeping is backend-param-only right now (`BackendConfig.sweep`); there's no way to say "run this same config across every discovered GPU" as a sweep axis.
4. **Multi-GPU expert/layer placement for Arcaine** — still raw-passthrough only; `EngineTranslator.build()` only ever resolves one device per backend.
5. **5th engine, or a web UI** — both reasonable, neither urgent. The image/model catalogs now have real data to show; a simple web view over them would be a natural next UI investment given TUI+CLI+MCP are the only interfaces today.
6. **SSH build-from-source, embedded-DB/Prometheus storage, AMD GPU support** — longer-standing, still open, none blocking.

## Open items carried forward (nothing urgent, unchanged unless noted)

- GPU index spaces: six non-corresponding numbering schemes now confirmed in practice (clinfo/xpumcli/SYCL-level-zero/DRM-render-node/OpenVINO-GPU.N/xmxmon's own numbering). PCI-bus-id/render-node remain the only reconciled pair.
- Image/model catalog labels only live in flat-file, not mirrored into OpenSearch when that adapter is active.
- No embedded-DB or Prometheus storage adapter.
- No web UI (TUI + CLI + MCP now).
- AMD GPU support unimplemented.

## How to pick this up

Read `SPEC_REVIEW.md` first (note its update at the top - two of its flagged
gaps are now resolved), then `VALIDATION.md` for the detailed technical
writeup of everything tested. Eleven `examples/validation*.suite.yaml` files
are now real working references covering four engines, build-from-source,
storage fan-out, SSH remote execution, external/already-loaded backends,
telemetry, and sweep automation - copy the closest one rather than starting
from `example.suite.yaml`.
