# Next steps / in-progress work log

Live document — updated as work progresses. See VALIDATION.md for the full
writeup and SPEC_REVIEW.md for the "are we still on track" assessment (now
partly resolved - see its update note at the top).

## Session complete (2026-08-01, continued) — closed out the MoE-loader question with the official checkpoint

Downloaded `Frosty40/Qwen-AgentWorld-35B-A3B-NVFP4` (Arcaine README's 4th "Supported Model", the exact MoE checkpoint - real answer to "was the AEON-7/urakozz crash a genuine Arcaine bug or bad checkpoints?"). **It loads and runs cleanly** - confirms the earlier crashes were bad third-party quantization recipes, not an Arcaine bug. But found a new, separate issue: only 5/10 fixed-questions passed, and the failures are genuinely degenerate output (repeated empty `<think>` loops or backtick blocks, not a truncation issue) - a real generation-quality/stability finding under Arcaine's default sampling for this checkpoint, not investigated further this session, flagged honestly rather than worked around.

- [ ] Follow-up candidate: investigate the AgentWorld MoE checkpoint's degenerate-repetition failures - likely a sampling-params (temperature/repetition-penalty) or MTP-speculative-decode interaction, not diagnosed this session.

## Prior session (2026-08-01, continued) — model-dir/HF-cache cleanup, 3 more catalog models, a real coherence-adapter fix

Also this session: surveyed model storage sprawl (3 disconnected pools, ~1.15TB), found the "17 duplicate" HF-cache overlaps were actually empty 12K stubs (no real duplication), found and cleaned 2 genuinely broken/partial downloads (root-owned files needed sudo), removed 81 empty stub dirs, and confirmed ignite doesn't have room to absorb the full HF cache (~93G free vs 567G needed) - consolidating onto `/mnt/acheron` (3.0T free) or `/mnt/malebolge` (3.6T free) is the real path if that's still wanted.

- [x] **Fixed a real gap**: `FixedQuestionCoherence` hardcoded `max_tokens: 64`, guaranteeing the same reasoning-model truncation false-negative found last session. Now configurable (`max_tokens` in suite config, default unchanged). 2 new tests, 108 passing.
- [x] Swept 3 more OpenArc models: `Phi-4-mini-instruct-int4-ov` (clean, 10/10), `DeepSeek-R1-Distill-Qwen-7B-int4-ov` with `max_tokens: 512` (clean, 10/10 - **proves the fix above works**), `phi-2-int4-ov` (**real crash, root-caused**: no `chat_template` on the tokenizer, OpenArc's worker dies on first inference and auto-unloads - genuine OpenArc+model incompatibility, not a llapdance bug).
- [x] Clean teardown confirmed for all three, committed.

## Prior session (2026-08-01, continued) — searched beyond the catalog dir, attempted a real fix, found a genuine engine gap

Direct question: any other models anywhere that work on Arcaine? Searched beyond `/mnt/ignite/LLM/models` (the catalog's only scan target so far) - found two real, fully-downloaded Qwen3.5-family models in `~/.cache/huggingface/hub` the catalog never saw.

- [x] Attempted a real fix for the two known-broken `qwen3_5_moe` models: read the loader's actual parser, confirmed AEON-7's checkpoint IS the text-only shape the loader wants (zero vision tensors, `model.language_model.*` keys) just with unflattened config + wrong `model_type` string. Built a non-destructive hard-linked shim with a patched config, tested it live, got further (config now parses, dispatch works) - then hit a genuine engine gap: this checkpoint's `linear_attn` weights are plain/unquantized while the loader hard-requires NVFP4-packed tensors there. Confirmed via reading the safetensors header directly, not guessed. Removed the non-working shim after confirming the dead end; original model directory was never touched.
- [x] Found (by contrast, reading the *dense* qwen3_5 loader) that it already supports 3 weight formats (NVFP4/FP8/dense bf16) unlike the MoE loader's NVFP4-only assumption - a real capability gap between the two loaders, worth a future Arcaine-side patch to bring the MoE loader up to the same standard.
- [x] Found a genuinely new candidate model (`Qwen/Qwen3.5-27B`, dense bf16, exact dispatch match) but it's 52GB and GPU 3 only has ~32GB VRAM (`xpumcli discovery`) - would need multi-GPU layer split, which this harness's engine-translation layer doesn't support yet (only resolves one device per backend). Not run, to avoid a guaranteed OOM.
- [x] Bottom line reported: no additional models work on Arcaine right now beyond the two already validated - the two MoE candidates are real, confirmed-broken by an engine-side gap (not a metadata issue after all), and the dense bf16 candidate needs multi-GPU support that doesn't exist yet.

## Prior session (2026-08-01, continued) — swept more of the catalog using the new tested-status feature

Direct follow-up: "sweep some more models in our catalog." Used the new `tested` field to pick real targets.

- [x] Confirmed (real container test, reading Arcaine's actual `ModelRegistry::create()` dispatch code rather than trusting a comment) that both `qwen3_5_moe`-labeled models (`urakozz/Ornith-1.0-35B-int4-AutoRound`, `AEON-7/Ornith-1.0-35B-...`) fail fast and predictably: exact registry key mismatch, not a fuzzy/partial one. This crash type is permanently invisible to the tested-status feature (crashes before `RunResult` storage) - a real, confirmed limitation, not a hypothetical one.
- [x] **Real finding, not a bug**: OpenArc + `Qwen3-0.6B-int4-ov` got 5/10 fixed-questions - checked the actual failures, all were `<think>` reasoning traces truncated by `max_tokens: 64` before reaching an answer. Config mismatch (reasoning models need bigger budgets or thinking disabled), not model or harness incorrectness.
- [x] Refreshed `diffusion_gemma` on Arcaine (10/10, 16.35 tok/s) - now has a current stored record, confirmed live via `llapdance models --results-dir` flipping from `untested` to `arcaine:pass(10/10)`.
- [x] Full test suite still green, committed.

## Prior session (2026-08-01, continued) — model catalog now shows real test history

Direct follow-up to a question: did the model catalog indicate whether a model actually ran (pass/fail) on a backend before, vs just could-run-on? It didn't. Built it:

- [x] `ModelInfo.tested: dict[engine, TestedStatus]`, built by `annotate_tested_status()` reversing each stored `RunResult`'s volume mount to recover the host path it actually tested, cross-referenced against the catalog scan. Outcome: `pass` / `partial` / `ran` (no coherence adapter configured).
- [x] Documented the real gap honestly: a crashed run never reaches storage, so it's indistinguishable from "never tried" - can't be fixed without also persisting failed attempts, out of scope for now.
- [x] Wired into `llapdance models --results-dir DIR` (CLI) and MCP `list_models(results_dir=...)`.
- [x] **Validated live** against today's own real results - correctly showed `unsloth/Qwen3.6-27B-NVFP4` as the more-recent (worse) of its two real runs, `sakamakismile/Huihui-...` as a clean pass, and honestly reported `RedHatAI/diffusiongemma-...` as untested (its earlier session's result file isn't in the current results dir - no fabrication).
- [x] 8 new tests, 106 passed total, committed.

## Also this session — second Qwen3.5 model validated, real Qwen3.5 sweep, stale-image bug found+fixed

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
