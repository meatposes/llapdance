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
from llapdance.core.catalog import LabeledImageRemovalError
from llapdance.core.catalog import label_image as catalog_label_image
from llapdance.core.catalog import list_images as catalog_list_images
from llapdance.core.catalog import remove_image as catalog_remove_image
from llapdance.core.model_catalog import annotate_tested_status, load_run_history, scan_models
from llapdance.core.orchestrator import run_suite as orchestrator_run_suite
from llapdance.core.result import RunResult
from llapdance.plugins import registry
from llapdance.plugins.registry import available, describe_engine as registry_describe_engine, load_builtin_adapters
from llapdance.plugins.storage.flat_file import FlatFileStorage


def _execution_adapter(host: str | None, user: str | None, ssh_key_path: str | None):
    if host:
        return registry.get("execution", "ssh-docker")({"host": host, "user": user, "ssh_key_path": ssh_key_path})
    return registry.get("execution", "local-docker")({})

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
def describe_engine(engine_name: str) -> dict[str, Any]:
    """What's known to be sweepable for a registered EngineTranslator
    (SPEC.md §10's 'catalog of build switches to sweep') - `params`
    (translator-consumed, swept via params.shared/backend_specific) and
    `env_flags` (raw engine/library env vars the translator never
    touches, swept directly via env.<NAME> - same mechanism, different
    config section). Use this to find valid dotted param paths for a
    suite's `sweep` axes before writing one."""
    return registry_describe_engine(engine_name)


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


@server.tool()
def list_images(
    catalog_dir: str | None = None,
    name_filter: str | None = None,
    host: str | None = None,
    user: str | None = None,
    ssh_key_path: str | None = None,
) -> list[dict[str, Any]]:
    """Enumerate docker images (local by default, or a remote host if
    host/user/ssh_key_path are given), enriched with any label and stored
    run history if `catalog_dir` (a suite's flat_file_dir) is given
    (SPEC.md §12)."""
    execution = _execution_adapter(host, user, ssh_key_path)
    return catalog_list_images(execution, catalog_dir=catalog_dir, name_filter=name_filter)


@server.tool()
def label_image(catalog_dir: str, image_ref: str, label: str, note: str = "") -> dict[str, str]:
    """Label an image good/bad/unknown with an optional note - labels live
    in `catalog_dir` (a suite's flat_file_dir), SPEC.md §12."""
    catalog_label_image(catalog_dir, image_ref, label, note)
    return {"image_ref": image_ref, "label": label}


@server.tool()
def remove_image(
    image_ref: str,
    catalog_dir: str | None = None,
    force: bool = False,
    host: str | None = None,
    user: str | None = None,
    ssh_key_path: str | None = None,
) -> dict[str, str]:
    """Remove an image - refuses if labeled 'good' unless force=True."""
    execution = _execution_adapter(host, user, ssh_key_path)
    try:
        catalog_remove_image(execution, image_ref, catalog_dir=catalog_dir, force=force)
    except LabeledImageRemovalError as exc:
        return {"error": str(exc)}
    return {"removed": image_ref}


@server.tool()
def list_models(directories: list[str], results_dir: str = "./results") -> list[dict[str, Any]]:
    """Scan directories for models, reporting format + quant hint + which
    registered engines could plausibly load each (format-compatible, not
    a guarantee it will run), plus real prior test outcomes cross-referenced
    from results_dir (a `tested` dict keyed by engine name - empty means no
    stored run was found, which includes both genuinely-untested models AND
    runs that crashed before a result could be written, see
    TestedStatus's docstring in model_catalog.py)."""
    from dataclasses import asdict

    models = scan_models(directories)
    annotate_tested_status(models, load_run_history(results_dir))
    return [asdict(m) for m in models]


def main() -> None:
    load_builtin_adapters()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
