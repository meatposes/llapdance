"""CLI entrypoint. See `llapdance/mcp/server.py` for the MCP integration
(SPEC.md §13) - it wraps the same orchestrator functions used here
(`run_suite`), not this CLI, so the two stay in sync by construction.
"""
from __future__ import annotations

import click

from llapdance.config.loader import load_suite, parse_kv_overrides
from llapdance.core.catalog import LabeledImageRemovalError
from llapdance.core.catalog import label_image as catalog_label_image
from llapdance.core.catalog import list_images as catalog_list_images
from llapdance.core.catalog import remove_image as catalog_remove_image
from llapdance.core.model_catalog import annotate_tested_status, load_run_history, scan_models
from llapdance.core.orchestrator import best_outcome, run_suite
from llapdance.plugins.registry import available, describe_engine, load_builtin_adapters


def _execution_adapter(host: str | None, user: str | None, ssh_key_path: str | None):
    from llapdance.plugins import registry

    if host:
        return registry.get("execution", "ssh-docker")({"host": host, "user": user, "ssh_key_path": ssh_key_path})
    return registry.get("execution", "local-docker")({})


@click.group()
def main() -> None:
    """LLAPDANCE - LLM Automated Pipeline for Deployment, Analysis aNd Coherence Evaluation."""
    load_builtin_adapters()


@main.command()
@click.argument("suite_path")
@click.option("--set", "overrides", multiple=True, help="Override a config value, e.g. --set backends.0.model=foo")
def run(suite_path: str, overrides: tuple[str, ...]) -> None:
    """Run a test suite defined in SUITE_PATH."""
    suite = load_suite(suite_path, parse_kv_overrides(list(overrides)))
    # Real gap: run_suite() used to abort the ENTIRE run (silently
    # discarding every already-succeeded backend) the moment any one
    # backend raised - a sweep with N combinations and one bad one
    # produced zero output, not N-1. It now skips a failing backend and
    # reports it via on_event instead - echo those events live so a
    # multi-backend/sweep run shows which combination failed, not just a
    # final list of the ones that happened to succeed.
    outcomes = run_suite(suite, on_event=click.echo)
    for outcome in outcomes:
        click.echo(f"\n=== {outcome.result.backend_name} ({outcome.result.run_id}) ===")
        for bench in outcome.result.benchmarks:
            click.echo(f"  [{bench.adapter}] {bench.metrics}")
        for coh in outcome.result.coherence:
            click.echo(f"  [{coh.adapter}] {coh.passed}/{coh.total} passed")
        for tel in outcome.result.telemetry:
            click.echo(f"  [{tel.adapter}] {tel.metrics}")
        if outcome.delta_against:
            click.echo(f"  delta against run {outcome.delta_against.run_id} available")

    # Direct user question: for a multi-backend/sweep run, how do you tell
    # which combination had the best result? Ranks by real per-backend
    # throughput (decode-only tok/s where available, else blended),
    # restricted to backends whose coherence check (if any) fully passed.
    best = best_outcome(outcomes)
    if best is not None:
        best_throughput, best_name, comparable_count = best
        click.echo(f"\nBest: {best_name} ({best_throughput:.2f} tok/s) among {comparable_count} comparable result(s)")


@main.command()
def adapters() -> None:
    """List available plugin adapters by kind."""
    for kind in ("benchmark", "coherence", "storage", "execution", "engine", "telemetry"):
        click.echo(f"{kind}: {', '.join(available(kind))}")


@main.command("describe-engine")
@click.argument("engine_name")
def describe_engine_cmd(engine_name: str) -> None:
    """Show what's known to be sweepable for a registered EngineTranslator
    (SPEC.md §10's 'catalog of build switches to sweep') - translator-
    consumed params (swept via params.shared/backend_specific) and raw
    engine/library env flags the translator never touches (swept directly
    via env.<NAME> - same mechanism, different config section)."""
    catalog = describe_engine(engine_name)
    if not catalog["params"] and not catalog["env_flags"] and not catalog["image_hints"]:
        click.echo(f"{engine_name}: nothing cataloged")
        return
    if catalog["image_hints"]:
        click.echo(f"image_hints (patterns of docker tags known to fit this engine): {', '.join(catalog['image_hints'])}")
    if catalog["params"]:
        click.echo("params (swept via params.shared / params.backend_specific):")
        for param, info in catalog["params"].items():
            click.echo(f"  {param}:")
            for key, value in info.items():
                click.echo(f"    {key}: {value}")
    if catalog["env_flags"]:
        click.echo("env_flags (swept directly via env.<NAME>):")
        for flag, info in catalog["env_flags"].items():
            click.echo(f"  {flag}:")
            for key, value in info.items():
                click.echo(f"    {key}: {value}")


@main.group()
def images() -> None:
    """Docker image catalog & cleanup (SPEC.md §12)."""


_host_option = click.option("--host", default=None, help="Remote host (omit for local docker socket)")
_user_option = click.option("--user", default=None, help="Remote SSH user")
_key_option = click.option("--ssh-key-path", default=None, help="Remote SSH identity file")
_catalog_dir_option = click.option(
    "--catalog-dir", default=None, help="A suite's flat_file_dir - where labels/run history live"
)


@images.command("list")
@click.option("--filter", "name_filter", default=None, help="Only images whose tag contains this substring")
@_catalog_dir_option
@_host_option
@_user_option
@_key_option
def images_list(name_filter, catalog_dir, host, user, ssh_key_path) -> None:
    """List images, enriched with any label and stored run history."""
    execution = _execution_adapter(host, user, ssh_key_path)
    for image in catalog_list_images(execution, catalog_dir=catalog_dir, name_filter=name_filter):
        tags = ", ".join(image["tags"]) or "<untagged>"
        label = image["label"]["label"] if image["label"] else "unlabeled"
        click.echo(f"{tags}  [{label}]  runs={len(image['runs'])}  size={image.get('size')}")


@images.command("label")
@click.argument("image_ref")
@click.argument("label", type=click.Choice(["good", "bad", "unknown"]))
@click.option("--note", default="")
@_catalog_dir_option
def images_label(image_ref, label, note, catalog_dir) -> None:
    """Label an image good/bad/unknown, with an optional note."""
    if not catalog_dir:
        raise click.UsageError("--catalog-dir is required (labels live alongside a suite's flat_file_dir)")
    catalog_label_image(catalog_dir, image_ref, label, note)
    click.echo(f"{image_ref} labeled {label!r}")


@images.command("rm")
@click.argument("image_ref")
@click.option("--force", is_flag=True, help="Remove even if labeled 'good'")
@_catalog_dir_option
@_host_option
@_user_option
@_key_option
def images_rm(image_ref, force, catalog_dir, host, user, ssh_key_path) -> None:
    """Remove an image - refuses if labeled 'good' unless --force is given."""
    execution = _execution_adapter(host, user, ssh_key_path)
    try:
        catalog_remove_image(execution, image_ref, catalog_dir=catalog_dir, force=force)
    except LabeledImageRemovalError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"removed {image_ref}")


@main.command("models")
@click.argument("directories", nargs=-1, required=True)
@click.option(
    "--results-dir",
    default="./results",
    show_default=True,
    help="Flat-file results directory to cross-reference for real prior run "
    "outcomes per model+engine (see 'tested' column). A crashed run never "
    "reaches storage, so it's indistinguishable from untested here.",
)
def models_cmd(directories: tuple[str, ...], results_dir: str) -> None:
    """Scan directories for models, reporting format + quant hint + which
    registered engines could plausibly load each (format-compatible, not
    a guarantee it will run - see llapdance/core/model_catalog.py), plus
    real prior test outcomes cross-referenced from --results-dir."""
    models = scan_models(list(directories))
    annotate_tested_status(models, load_run_history(results_dir))
    for model in models:
        tested = (
            ", ".join(f"{engine}:{status.outcome}({status.coherence_summary or 'n/a'})" for engine, status in model.tested.items())
            or "untested"
        )
        click.echo(f"{model.format:12} {model.quant_hint:24} {model.compatible_engines}  tested=[{tested}]  {model.path}")


@main.command()
def tui() -> None:
    """Launch the terminal UI."""
    from llapdance.tui.app import LLAPDanceApp

    LLAPDanceApp().run()


@main.command()
def mcp() -> None:
    """Start the MCP server (stdio transport) so agents can push suites/runs
    and pull back results programmatically (SPEC.md §13)."""
    from llapdance.mcp.server import main as mcp_main

    mcp_main()


if __name__ == "__main__":
    main()
