# Next steps / in-progress work log

Live document — updated as work progresses. Read bottom-to-top isn't
necessary anymore; this session's task list is complete. See VALIDATION.md
for the full writeup of everything below.

## Session complete (2026-08-01, overnight) — Arcaine, OpenArc, build-from-source, OpenSearch

All requested items done and validated for real, not just unit-tested:

- [x] Arcaine backend integrated (`llapdance/plugins/engine/arcaine.py`), real run passed (10/10 coherence)
- [x] OpenArc backend integrated (`llapdance/plugins/engine/openarc.py`), real run passed (10/10 coherence) — required a new capability, `EngineInvocation.post_start_requests`, since OpenArc loads models via a separate HTTP call, not at container start
- [x] `source.mode: build` validated for real (fresh git clone of qxmx's real remote + docker build) — found and fixed a safety gap (no dirty-tree check before `git checkout`)
- [x] Build-version tracking validated (git commit SHA baked into image tag, e.g. `llapdance/qxmx-from-source:main-3ae4eff`)
- [x] Real OpenSearch storage adapter built and validated (write+query round-trip against a live instance, plus storage fan-out with flat-file) — found and fixed a real silent-precision bug (timestamp defaulting to 32-bit float)
- [x] MCP integration noted as future work in SPEC.md §13 and `llapdance/cli.py` (not built, as requested)
- [x] README.md / VALIDATION.md updated, full test suite passing, committed + pushed

## Open items for next session (carried forward, not blocking, nothing urgent)

1. **Arcaine's `generic-http` benchmark throughput reads low** (`2.1 tok/s` vs. `~13 tok/s` seen manually) — likely the benchmark adapter's coarse "one SSE line = one token" counting heuristic doesn't fit Arcaine's streaming chunk shape. Coherence content was correct, so this isn't urgent, but don't trust cross-engine `generic-http` throughput comparisons involving Arcaine until this is looked at.
2. **GPU index spaces: now FOUR non-corresponding numbering schemes** confirmed (clinfo, xpumcli, SYCL/level-zero, DRM render-node, OpenArc/OpenVINO's `GPU.N`). `DeviceInfo.pci_bus_id`/`render_node` remain the only two that are actually reconciled with each other. Still an open SPEC.md §7/§15 decision, now with more real evidence behind it.
3. **Arcaine/OpenArc translators only ever resolve ONE GPU device.** Arcaine's model is a 26B MoE that explicitly supports multi-GPU expert placement (`LAYER_PLACEMENT`/`EXPERT_PLACEMENT` env vars, confirmed via a live container's startup logs) - the translator has raw passthrough for these but no generated multi-device support. Not attempted tonight; would need `_apply_engine_translator`/`EngineTranslator.build()` to accept more than one resolved device.
4. **No MCP server built yet** - SPEC.md §13 and `cli.py` both note this is needed; `run_suite`/`run_backend` are the operations to wrap when it's built.
5. Still-standing gaps from prior sessions, unchanged: no SSH remote execution target, `llama-benchy` adapter still a stub (no discoverable API), no embedded-DB or Prometheus storage adapter, no image-catalog/labeling UI, no web UI (TUI + CLI only).

## How to pick this up

Read `VALIDATION.md` in full for the detailed writeup (what was tested, what broke, what got fixed, exact commands used). The six `examples/validation*.suite.yaml` files are real working references for four different engines plus build-from-source plus storage fan-out - copy the closest one rather than starting from `example.suite.yaml` when adding a fifth engine.
