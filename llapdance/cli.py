"""CLI entrypoint.

NOTE (future work, not built - see SPEC.md §13): this suite will need an
MCP integration so agents can push test suites/runs and pull back results
programmatically, not just human operators via this CLI/the TUI. `run()`
below is the operation an MCP tool would wrap almost directly - keep that
in mind if this CLI ever grows business logic that isn't also reachable
by calling `run_suite()`/`run_backend()` directly, since an MCP layer will
want to call the orchestrator, not shell out to this CLI.
"""
from __future__ import annotations

import click

from llapdance.config.loader import load_suite, parse_kv_overrides
from llapdance.core.orchestrator import run_suite
from llapdance.plugins.registry import available, load_builtin_adapters


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
        if outcome.delta_against:
            click.echo(f"  delta against run {outcome.delta_against.run_id} available")


@main.command()
def adapters() -> None:
    """List available plugin adapters by kind."""
    for kind in ("benchmark", "coherence", "storage", "execution"):
        click.echo(f"{kind}: {', '.join(available(kind))}")


@main.command()
def tui() -> None:
    """Launch the terminal UI."""
    from llapdance.tui.app import LLAPDanceApp

    LLAPDanceApp().run()


if __name__ == "__main__":
    main()
