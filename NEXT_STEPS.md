# Next steps / in-progress work log

Live document — updated as work progresses. See VALIDATION.md for the full
writeup and SPEC_REVIEW.md for the "are we still on track" assessment (now
partly resolved - see its update note at the top).

## Session complete (2026-08-01, continued) — sweep automation, image catalog, model catalog

Direct follow-up to SPEC_REVIEW.md's top recommendation:

- [x] **Sweep/parameter-matrix automation built and validated** — `BackendConfig.sweep` + `llapdance/config/sweep.py`, real 2-value sweep produced 2 real automatic runs. Found and fixed a real design bug: sweep axes into `params.shared`/`backend_specific` need to allow introducing a NEW key, not just varying an existing one.
- [x] **Engine sweepable-params catalog built** — `EngineTranslator.sweepable_params` populated for all 4 engines, `llapdance describe-engine <name>` / MCP tool.
- [x] **Image catalog & cleanup built and validated** — `llapdance/core/catalog.py`, `llapdance images list/label/rm`, wraps the `ExecutionTargetAdapter.list_images()` that's existed since session one but was never consumed. Validated against real image sprawl, including the good-label removal safety check (tested against a disposable tag, never the real sprawl).
- [x] **Model catalog + format/backend compatibility built and validated** (new capability, added per direct request) — `llapdance/core/model_catalog.py`, `llapdance models <dir>...`. Scans GGUF/OpenVINO-IR/HF-safetensors, reports quant hint + which engines could plausibly load each (could-run-on, not will-run). Validated against ground truth: all 3 previously-validated cross-format models matched exactly.
- [x] SPEC.md/README.md/VALIDATION.md/SPEC_REVIEW.md updated, full test suite passing (91 tests), committed + pushed.

## Recommended next session

Per the (now mostly-addressed) spec review, remaining real gaps in priority order:

1. **Device-target sweeping** — sweeping is backend-param-only right now (`BackendConfig.sweep`); there's no way to say "run this same config across every discovered GPU" as a sweep axis. A natural follow-on now that param sweeping works.
2. **Multi-GPU expert/layer placement for Arcaine** — still raw-passthrough only; `EngineTranslator.build()` only ever resolves one device per backend.
3. **5th engine, or a web UI** — both reasonable, neither urgent. The image/model catalogs now have real data to show; a simple web view over them would be a natural next UI investment given TUI+CLI+MCP are the only interfaces today.
4. **SSH build-from-source, embedded-DB/Prometheus storage, AMD GPU support** — longer-standing, still open, none blocking.

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
