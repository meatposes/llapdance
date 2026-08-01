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
from llapdance.core.model_catalog import scan_models
from llapdance.core.orchestrator import run_suite
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
    outcomes = run_suite(suite)
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


@main.command()
def adapters() -> None:
    """List available plugin adapters by kind."""
    for kind in ("benchmark", "coherence", "storage", "execution", "engine", "telemetry"):
        click.echo(f"{kind}: {', '.join(available(kind))}")


@main.command("describe-engine")
@click.argument("engine_name")
def describe_engine_cmd(engine_name: str) -> None:
    """Show the sweepable params a registered EngineTranslator declares
    (SPEC.md §10's 'catalog of build switches to sweep')."""
    params = describe_engine(engine_name)
    if not params:
        click.echo(f"{engine_name}: no sweepable params declared")
        return
    for param, info in params.items():
        click.echo(f"{param}:")
        for key, value in info.items():
            click.echo(f"  {key}: {value}")


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
def models_cmd(directories: tuple[str, ...]) -> None:
    """Scan directories for models, reporting format + quant hint + which
    registered engines could plausibly load each (format-compatible, not
    a guarantee it will run - see llapdance/core/model_catalog.py)."""
    for model in scan_models(list(directories)):
        click.echo(f"{model.format:12} {model.quant_hint:24} {model.compatible_engines}  {model.path}")


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
