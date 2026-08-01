# Spec review — are we proceeding correctly? (2026-08-01)

Requested after building and validating the MCP integration: step back, compare
`SPEC.md` and the original intent against what's actually been built across five
sessions, and give an honest read on whether this is still the right shape.

## Short answer

Mostly yes, with one real strategic imbalance worth correcting before adding a
fifth engine or a fifth execution target: **breadth has significantly outpaced
depth**. Every engine/execution-target/storage-adapter addition has been real,
validated, and individually justified — this isn't speculative scope creep, the
project's own discipline (test everything for real, document every bug) has
held throughout. But two capabilities the spec calls out as directly motivating
this whole project — automated parameter sweeps (§10) and image catalog/cleanup
(§12) — are **still completely unbuilt**, while four inference engines, two
execution targets, a telemetry adapter, and an MCP server all exist. The things
that make this actually pleasant to use day-to-day (not manually hand-writing a
new suite YAML per parameter variant, not drowning in `qxmx:*`/`llama-cpp-*`
image sprawl) are the least-developed part of the system.

## Portability principle (§0) — holding, no violations found

Checked against everything built: no hardcoded paths, GPU counts, or vendor
assumptions in core code. Every new capability this session (SSH remote target,
external/already-loaded mode, telemetry) went through the same discipline —
config-driven, probed at runtime, documented gotchas rather than assumptions.
`examples/*.suite.yaml` files reference this box's real paths/hostnames, but
that's the spec's own explicitly-sanctioned pattern (§16 "reference deployment,
not part of the spec") — the *harness* doesn't hardcode them, the *example
configs* do, on purpose, same as `SPEC.md` itself does.

## Where the spec text has gone stale (needs updating, not rebuilding)

1. **§5's architecture diagram is out of date.** It draws four boxes (Execution
   Target / Telemetry-Benchmark / Coherence-Quality / Storage) and folds
   telemetry and benchmark into one combined slot. What actually got built and
   validated has **six** plugin kinds: `execution`, `benchmark`, `coherence`,
   `storage`, `engine` (the per-engine translator — real, validated against
   four engines, but never drawn as its own box in the original diagram), and
   `telemetry` (split out from benchmark this session, because xmxmon doesn't
   behave like a benchmark adapter at all — it brackets around a run watching
   hardware counters rather than making requests itself). This split was the
   right call once xmxmon was actually integrated, but the spec text should
   reflect it rather than describe an architecture that's now inaccurate.

2. **`source.mode: external` doesn't exist in the spec at all.** It was added
   this session in direct response to a real ask ("test a backend that's
   already loaded") and is fully real/validated — but §4 only documents
   `build`/`prebuilt`, and §7's "VRAM preflight is mandatory before any
   bench/coherence run" is written as an unconditional rule that the actual
   implementation correctly does NOT apply to external backends (there's no
   container of the harness's own to preflight). The spec should say so
   explicitly rather than leave a real, working code path unmentioned and
   apparently contradicting §7's wording.

3. **§13's MCP line still reads "future, not built."** It's built and
   validated (real stdio client, all 5 tools, a genuine `run_suite` execution
   pulled back via `get_results`) — should flip to done, with the real gotcha
   found (list-returning tools wrap output as `{"result": [...]}` in
   `structured_content`, not JSON text) noted for whoever builds the next tool.

None of these are design mistakes — they're the spec not having caught up to
validated reality. Fixing the text is cheap; I've done it as part of this
review (see the updated `SPEC.md`).

## Where the actual build has gone quiet (real gaps, not stale text)

1. **§10 Sweep/parameter matrix — not built.** This is the most significant
   gap. The spec's own motivating language — "the same trigger path drives
   sweeps and truth tables... without necessarily touching source," "device
   selection is itself sweepable" — describes an *automated* matrix generator.
   What exists today is a `backends: list[BackendConfig]` field and separate
   suite YAML files per variant, run one at a time. Every comparison done so
   far (llama.cpp vs. qxmx vs. Arcaine vs. OpenArc, local vs. remote, different
   context sizes) was a **hand-authored separate suite file**, not a sweep.
   There is no code path that takes `context_size: [2048, 4096, 8192]` and
   produces three runs automatically. This was fine while the priority was
   proving each individual capability works at all — it stops being fine the
   moment someone wants to actually compare 5 quant levels × 3 context sizes
   without hand-writing 15 files.

2. **§12 Image catalog & cleanup — not built at all.** Zero code exists for
   this. Notably, this is the section the spec itself says was *directly
   motivated* by real image sprawl already documented in §16 (`llama-cpp-*`,
   `qxmx:*` — dozens of tags). That sprawl has only grown since (this session
   added `llapdance/qxmx-from-source:main-3ae4eff`-style build-tracked tags on
   top of it). Every stored `RunResult` already carries `image_ref` and full
   backend config, which is exactly the metadata §12 says makes cataloging
   possible — the data is there, nothing consumes it into a "which builds are
   good, which are dead weight" view yet.

3. **§8 Storage — 1 of 3 optional adapters built.** OpenSearch is real and
   validated (including a genuine bug fix). Embedded-DB and Prometheus/Grafana
   remain open decisions (§15.1), unchanged since they were first flagged.

4. **§13 Web UI — not built.** TUI + CLI only, unchanged since v0.1.

## Recommendation

Before adding a fifth engine or building out multi-GPU expert placement (both
reasonable, both currently on the open-items list): consider pivoting toward
sweep automation and image cataloging instead. The engine-integration muscle
is proven four times over now — the risk isn't "can we wrap another backend,"
it's "does anyone actually want to use this by hand-writing YAML files
forever." Sweeps and image catalog are also lower-risk to build than they were
earlier: `RunResult` now carries build-version-tracked image refs, full GPU
identity, and telemetry, so a sweep runner and a catalog view both have
real, validated data to consume rather than needing new plumbing first.

Not a hard blocker — the breadth built so far is genuinely useful and none of
it needs to be undone. This is a sequencing observation, not a correction.
