"""Minimal Textual TUI: browse suite YAML files in the current directory,
trigger a run, watch results stream in. Same orchestrator core as the CLI -
this is a thin view on top, not a second code path (SPEC.md §13)."""
from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import DataTable, Footer, Header, Log

from llapdance.config.loader import load_suite
from llapdance.core.orchestrator import run_suite
from llapdance.plugins.registry import load_builtin_adapters


class LLAPDanceApp(App):
    CSS = """
    DataTable { width: 40%; }
    Log { width: 60%; }
    """
    BINDINGS = [("r", "run_selected", "Run selected suite"), ("q", "quit", "Quit")]

    def __init__(self, search_dir: str = ".") -> None:
        super().__init__()
        self._search_dir = Path(search_dir)
        self._suite_paths: list[Path] = []
        load_builtin_adapters()

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield DataTable(id="suites")
            yield Log(id="output")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#suites", DataTable)
        table.add_columns("Suite file")
        self._suite_paths = sorted(self._search_dir.glob("**/*.suite.yaml"))
        for path in self._suite_paths:
            table.add_row(str(path))
        table.cursor_type = "row"
        table.focus()

    def action_run_selected(self) -> None:
        table = self.query_one("#suites", DataTable)
        log = self.query_one("#output", Log)
        if table.cursor_row is None or not self._suite_paths:
            return
        path = self._suite_paths[table.cursor_row]
        log.write_line(f"loading {path} ...")
        self.run_worker(self._run(path), exclusive=True, thread=True)

    async def _run(self, path: Path) -> None:
        log = self.query_one("#output", Log)
        try:
            suite = load_suite(path)
            outcomes = run_suite(suite)
        except Exception as exc:  # surfaced to the user, not swallowed
            log.write_line(f"ERROR: {exc}")
            return
        for outcome in outcomes:
            log.write_line(f"=== {outcome.result.backend_name} ({outcome.result.run_id}) ===")
            for bench in outcome.result.benchmarks:
                log.write_line(f"  [{bench.adapter}] {bench.metrics}")
            for coh in outcome.result.coherence:
                log.write_line(f"  [{coh.adapter}] {coh.passed}/{coh.total} passed")
            if outcome.delta_against:
                log.write_line(f"  delta against run {outcome.delta_against.run_id} available")


if __name__ == "__main__":
    LLAPDanceApp().run()
