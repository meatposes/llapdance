# Next steps / in-progress work log

Live document — updated as work progresses. See VALIDATION.md for the full
writeup of everything below.

## Session complete (2026-08-01, continued) — SSH remote target, GPU tracking, Arcaine fix, external mode

Requested items #5, #1, #4 done and validated for real, plus a new
capability (external/already-loaded backend mode) added mid-session:

- [x] **#4 GPU device identity tracking** — `RunResult.device_target` now carries full `DeviceInfo` (vendor/name/pci_bus_id/render_node) plus a `verified` flag, not just a bare index. Local runs record real hostname too.
- [x] **Remote hardware probing** — `core/probe.py` now threads an explicit `CommandRunner` (local or SSH) through every discovery function. Found and handled a real gap live: screamer (the remote host) has neither `xpumcli` nor a working host-level OpenCL runtime, added an `lspci`-based third discovery tier (identification only, structurally excludes non-Intel/non-NVIDIA chips).
- [x] **#5 SSH execution target** — `llapdance/plugins/execution/ssh_docker.py`, built via raw `ssh`+`docker` CLI (not docker-py's `ssh://` transport — needs paramiko, no clean way to pin our identity file). `prebuilt` only, not `build`, over SSH. Also fixed a real bug: the orchestrator had `ExecutionTargetConfig.mode` in its schema for a whole session but never actually read it, hardcoding `"local-docker"` regardless.
- [x] **Real SSH validation against screamer** — stopped its production `bonsai` container, ran our harness against it remotely (smaller context size for the B50's lower VRAM, per the heads-up), confirmed clean teardown, restarted the original container, confirmed it came back healthy and correct.
- [x] **#1 Arcaine benchmark fix** — root cause: Arcaine's diffusion decoding emits the whole completion as ONE SSE chunk (not one-token-per-chunk like autoregressive engines), so the old "count SSE lines" heuristic undercounted ~7x. Fixed generically in `generic_http.py` (prefers `usage.completion_tokens` → `metrics.new_token` → `timings.predicted_n` → line-count fallback, records which was used). Re-validated: Arcaine `2.1 → 14.7 tok/s`, llama.cpp unaffected (`34.6 tok/s`, now via the authoritative field).
- [x] **New: `source.mode: external`** (raised mid-session, not originally on the list) — test an already-loaded backend with zero container lifecycle. Added `api_key`/`headers` support to both adapters (neither sent auth before). Validated against a real already-loaded model through the user's own OpenAI-compatible proxy project.
- [x] README.md / VALIDATION.md updated, full test suite passing (58 tests), committed + pushed.

## Open items for next session (carried forward, nothing urgent)

1. **`source.mode: build` not supported over SSH yet** — `ssh_docker.py` only does `prebuilt`. Would need the build context transferred to the remote host first (rsync, or some other mechanism) - docker-py's local-tar-upload trick doesn't apply since we're not using docker-py's transport for the SSH adapter.
2. **Multi-GPU expert/layer placement for Arcaine** — still raw-passthrough only (`LAYER_PLACEMENT`/`EXPERT_PLACEMENT` env vars), `EngineTranslator.build()` only ever resolves one device per backend. Carried forward from last session, unchanged.
3. **Five (not four) non-corresponding GPU index spaces now confirmed**: clinfo, xpumcli, SYCL/level-zero, DRM render-node, OpenArc/OpenVINO's `GPU.N`, and now `lspci`'s own bare enumeration order too. PCI-bus-id/render-node remain the only reconciled pair.
4. **No MCP server built yet** — SPEC.md §13 and `cli.py` both note this is needed; `run_suite`/`run_backend` are the operations to wrap when it's built.
5. Still-standing gaps, unchanged: `llama-benchy` adapter still a stub, no embedded-DB or Prometheus storage adapter, no image-catalog/labeling UI, no web UI (TUI + CLI only), AMD GPU support unimplemented.

## How to pick this up

Read `VALIDATION.md` in full for the detailed writeup (what was tested, what broke, what got fixed, exact commands used). Eight `examples/validation*.suite.yaml` files are now real working references covering four engines, build-from-source, storage fan-out, SSH remote execution, and external/already-loaded backends - copy the closest one rather than starting from `example.suite.yaml`.
