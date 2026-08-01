"""MCP server for LLAPDANCE (SPEC.md §13, previously noted as future work -
now built). Lets an agent push test suites/runs and pull back results
programmatically, not just a human operator via CLI/TUI.

Deliberately thin: every tool here calls straight into the same
orchestrator functions (`run_suite`, adapter registry, storage adapters)
the CLI uses - no separate business logic lives here, per the note left
in `llapdance/cli.py` when this was still unbuilt. If this server and the
CLI ever disagree about what a suite run does, that's a bug in this file,
not a second implementation to keep in sync by hand.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from llapdance.config.loader import load_suite
from llapdance.core.orchestrator import run_suite as orchestrator_run_suite
from llapdance.core.result import RunResult
from llapdance.plugins.registry import available, load_builtin_adapters
from llapdance.plugins.storage.flat_file import FlatFileStorage

server = MCPServer(
    name="llapdance",
    description="LLM Automated Pipeline for Deployment, Analysis aNd Coherence Evaluation "
    "- orchestrates LLM inference engine test suites and their results.",
)


@server.tool()
def list_adapters() -> dict[str, list[str]]:
    """List every registered plugin adapter, grouped by kind (benchmark,
    coherence, storage, execution, engine, telemetry). Use this to see
    what's available before writing/editing a suite."""
    return {kind: available(kind) for kind in ("benchmark", "coherence", "storage", "execution", "engine", "telemetry")}


@server.tool()
def list_suites(directory: str = ".") -> list[str]:
    """Find *.suite.yaml files under `directory` (non-recursive-safe glob,
    matches the TUI's own discovery pattern)."""
    return sorted(str(p) for p in Path(directory).glob("**/*.suite.yaml"))


@server.tool()
def get_suite(suite_path: str) -> dict[str, Any]:
    """Load and validate a suite file, returning its fully-resolved config
    (defaults filled in) as JSON - inspect before running, without
    executing anything."""
    return load_suite(suite_path).model_dump(mode="json")


@server.tool()
def run_suite(suite_path: str, overrides: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Run every backend in a suite (build/start/benchmark/coherence/stop
    per SPEC.md §9-11, or the external/no-lifecycle path for
    source.mode='external') and return each outcome: the full result plus
    whether a prior run was found to diff against.

    `overrides` is the same nested-dict shape as the suite YAML itself
    (e.g. {"backends": {"0": {"model": "other-model"}}}), merged on top -
    matches what `llapdance.config.loader.load_suite` already accepts, so
    an agent overriding a config value here behaves identically to the
    CLI's `--set` flag.
    """
    suite = load_suite(suite_path, overrides)
    outcomes = orchestrator_run_suite(suite)
    return [
        {
            "result": outcome.result.model_dump(mode="json"),
            "delta_against_run_id": outcome.delta_against.run_id if outcome.delta_against else None,
        }
        for outcome in outcomes
    ]


@server.tool()
def get_results(flat_file_dir: str, backend_name: str, limit: int = 5) -> list[dict[str, Any]]:
    """Pull the most recent stored results for a backend from flat-file
    storage (the always-on default, SPEC.md §8) - most recent first.
    Point `flat_file_dir` at whatever a suite's `storage.flat_file_dir`
    was for the run(s) you want to pull back."""
    storage = FlatFileStorage({"flat_file_dir": flat_file_dir})
    results: list[RunResult] = storage.previous_for(backend_name, limit=limit)
    return [r.model_dump(mode="json") for r in results]


def main() -> None:
    load_builtin_adapters()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
