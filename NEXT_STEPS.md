# Next steps / in-progress work log

Live document — updated as work progresses. See VALIDATION.md for the full
writeup and SPEC_REVIEW.md for the honest "are we still on track" assessment.

## Session complete (2026-08-01, continued) — MCP, telemetry, guidellm attempt, spec review

- [x] **MCP server built and validated** — `llapdance/mcp/server.py`, 5 tools, real stdio client test including a genuine `run_suite` execution pulled back via `get_results`. Real gotcha found: list-returning tools land in `structured_content["result"]`, not `content[0].text`.
- [x] **Telemetry harness built and validated** — new `telemetry` plugin kind (deliberately separate from `benchmark`), `xmxmon` reference adapter. Real run also caught a genuine "watching the wrong physical GPU" mismatch (telemetry device config and backend device targeting aren't reconciled - a live demonstration of the long-standing GPU-index-space problem).
- [x] **guidellm attempted for real** — hit a structural limitation (tokenizer resolution requires a real HF Hub repo id, most of what this harness tests isn't one), shipped as an honest stub like `llama-benchy`.
- [x] **Spec review done** — `SPEC_REVIEW.md`. Portability principle holding, no violations. Spec text fixed in 3 places (stale architecture diagram, missing `source.mode: external`, stale MCP status). Real finding: **sweep automation (§10) and image catalog (§12) are both completely unbuilt**, while engine/execution-target breadth has grown a lot. Recommended pivoting there before adding a 5th engine.
- [x] README.md / VALIDATION.md / SPEC.md updated, full test suite passing (68 tests), committed + pushed.

## Recommended next session (per SPEC_REVIEW.md)

1. **Sweep/parameter-matrix automation** (SPEC.md §10) — take a suite with e.g. `context_size: [2048, 4096, 8192]` and automatically generate + run N backend variants, rather than hand-authoring N suite files. This is the biggest gap between what the spec promised and what exists.
2. **Image catalog & cleanup** (SPEC.md §12) — `RunResult.image_ref` already carries build-version-tracked image tags; nothing consumes the growing pile of `qxmx:*`/`llama-cpp-*`/`llapdance/*` tags into a "which ones are actually good" view yet.
3. Only after those: a 5th engine, multi-GPU expert placement for Arcaine, or an embedded-DB/Prometheus storage adapter — all reasonable, none as urgent as 1-2 above.

## Open items carried forward (nothing urgent, unchanged unless noted)

- SSH execution target still `prebuilt`-only, no remote build-from-source.
- GPU index spaces: six non-corresponding numbering schemes now confirmed in practice (clinfo/xpumcli/SYCL-level-zero/DRM-render-node/OpenVINO-GPU.N/xmxmon's own numbering). PCI-bus-id/render-node remain the only reconciled pair; nothing reconciles a telemetry adapter's device number against a backend's actual target device.
- Multi-GPU expert/layer placement for Arcaine still raw-passthrough only.
- No embedded-DB or Prometheus storage adapter.
- No web UI (TUI + CLI + MCP now).
- AMD GPU support unimplemented.

## How to pick this up

Read `SPEC_REVIEW.md` first for the "are we building the right thing" read, then
`VALIDATION.md` for the detailed technical writeup of everything tested. Ten
`examples/validation*.suite.yaml` files are now real working references
covering four engines, build-from-source, storage fan-out, SSH remote
execution, external/already-loaded backends, and telemetry - copy the closest
one rather than starting from `example.suite.yaml`.
