"""Real interactive TUI screens (replaces the old placeholder - see
VALIDATION.md "TUI rebuild" section for what was wrong with it: a bare list
of file paths, no model/engine discovery, no live progress, a raw Python
dict dumped at the end).

Flow: browse real scanned models (reusing `model_catalog.scan_models` +
tested-status, the same code the CLI's `llapdance models` uses) -> pick a
real registered engine + device + image -> review/edit a generated suite
as YAML (this harness has too many real per-engine gotchas - HF cache
symlinks, vLLM's /dev/dri/by-path mount, health_path conventions - to
pretend one-size-fits-all defaults are always correct; showing the exact
YAML and letting a human fix it before running is the honest choice) ->
run it with live per-stage progress (via `orchestrator.run_backend`'s
`on_event` callback, added alongside this rebuild - the orchestrator had
NO progress visibility at all before) and a clear pass/fail summary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Input, RichLog, Select, Static, TextArea

from llapdance.config.models import TestSuite
from llapdance.core import orchestrator
from llapdance.core.catalog import list_images
from llapdance.core.model_catalog import ModelInfo, annotate_tested_status, load_run_history, scan_models
from llapdance.core.probe import discover_devices
from llapdance.plugins.registry import available

# Known real health-check conventions per engine, confirmed against real
# containers this session (see VALIDATION.md) - not guessed. Still just a
# starting point in the generated YAML, not silently trusted: the preview
# step shows it and lets a human correct it before running.
_ENGINE_HEALTH_PATH_HINTS: dict[str, str] = {
    "llama-cpp-sycl": "/health",
    "qxmx": "/health",
    "arcaine": "/v1/models",  # Arcaine has no /health endpoint at all - confirmed via a live container
    "openarc": "/v1/models",  # returns 200 with an empty model list before any model is loaded
    "vllm": "/health",
}


def _default_model_path_and_volumes(model: ModelInfo) -> tuple[dict[str, str], str]:
    """A real host model path, mounted the same way every validated example
    suite this session mounts one - not a guess, but also not infallible:
    engines with extra mount requirements (vLLM's /dev/dri/by-path, HF
    cache's relative-symlink gotcha - see VALIDATION.md) still need a human
    to add those in the preview step; this only gets the model volume right."""
    if model.format == "gguf":
        host_dir = str(Path(model.path).parent)
        return {host_dir: "/models"}, f"/models/{Path(model.path).name}"
    return {model.path: "/models"}, "/models"


def _tested_summary(model: ModelInfo) -> str:
    if not model.tested:
        return "untested"
    return ", ".join(f"{engine}:{status.outcome}({status.coherence_summary or 'n/a'})" for engine, status in model.tested.items())


class ModelBrowserScreen(Screen):
    """Step 1 of 3: real model discovery - the same `scan_models` +
    tested-status the CLI's `llapdance models` command uses, not a
    hardcoded example list.

    Real gap found from direct user feedback after the first rebuild: this
    file `import`ed Button and never placed one anywhere - the entire
    interaction relied on keybindings visible only as small text in the
    Footer widget (`s`/`enter`/`g`/`r`), with no on-screen instructions and
    no clickable path through the app at all. Fixed: a numbered step
    banner plus real, visible, clickable buttons for every action -
    keybindings still work too, but nothing requires knowing them."""

    BINDINGS = [("s", "scan", "Scan directories"), ("enter", "configure", "Configure a run for this model"), ("q", "quit", "Quit")]

    def __init__(self) -> None:
        super().__init__()
        self._models: list[ModelInfo] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Static("[b]Step 1 of 3:[/b] pick a model to test. Real models found on this machine, cross-referenced against real prior test results.")
            with Horizontal():
                yield Input(value="/mnt/ignite/LLM/models", id="scan-dirs")
                yield Button("Scan", id="scan-btn", variant="primary")
            yield DataTable(id="models")
            yield Static("", id="status")
            yield Button("Configure a run for the selected model →", id="configure-btn", variant="success")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#models", DataTable)
        table.add_columns("Format", "Quant", "Compatible engines", "Tested", "Path")
        table.cursor_type = "row"
        self.action_scan()

    def action_scan(self) -> None:
        dirs = self.query_one("#scan-dirs", Input).value.split()
        table = self.query_one("#models", DataTable)
        table.clear()
        self._models = scan_models(dirs)
        annotate_tested_status(self._models, load_run_history("./results"))
        for m in self._models:
            table.add_row(m.format, m.quant_hint, ", ".join(m.compatible_engines) or "(none)", _tested_summary(m), m.path)
        status = self.query_one("#status", Static)
        if self._models:
            status.update(f"{len(self._models)} models found across {len(dirs)} director(ies). Pick a row, then click/press Enter on 'Configure'.")
        else:
            status.update(f"No models found under {' '.join(dirs) or '(no directories given)'} - check the path above and press Scan again.")

    def action_configure(self) -> None:
        table = self.query_one("#models", DataTable)
        if not self._models:
            self.query_one("#status", Static).update("[red]no models to configure - scan a directory with real models in it first[/red]")
            return
        if table.cursor_row is None:
            self.query_one("#status", Static).update("[red]select a row in the table first (arrow keys, then Enter)[/red]")
            return
        model = self._models[table.cursor_row]
        self.app.push_screen(BuildScreen(model))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.action_configure()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "scan-btn":
            self.action_scan()
        elif event.button.id == "configure-btn":
            self.action_configure()


def _local_image_options() -> list[tuple[str, str]]:
    """Real local docker images (the same `LocalDockerExecutionTarget.list_images()`
    the CLI's `llapdance images` command uses) - so a user picks from what
    actually exists on this machine instead of typing a tag from memory."""
    from llapdance.plugins.execution.local_docker import LocalDockerExecutionTarget

    try:
        images = list_images(LocalDockerExecutionTarget({}), catalog_dir="./results")
    except Exception:
        return []
    options = []
    for image in images:
        for tag in image["tags"]:
            options.append((tag, tag))
    return options


class BuildScreen(Screen):
    """Step 2 of 3: pick a real registered engine + real discovered device
    + a real local image, generate a real suite (same Pydantic models the
    CLI validates against), and review/edit it as YAML before running -
    see module docstring for why this isn't fully automatic."""

    BINDINGS = [("escape", "back", "Back"), ("g", "generate", "Generate/refresh YAML"), ("r", "launch", "Run this suite")]

    def __init__(self, model: ModelInfo) -> None:
        super().__init__()
        self._model = model

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Static(
                f"[b]Step 2 of 3:[/b] configure a run for [b]{self._model.path}[/b] "
                f"({self._model.format}, {self._model.quant_hint})."
            )
            with Horizontal():
                engines = self._model.compatible_engines or available("engine")
                yield Select([(e, e) for e in engines], id="engine", value=engines[0] if engines else Select.BLANK)
                devices = discover_devices()
                yield Select(
                    [(f"{d.index}: {d.name}", d.index) for d in devices],
                    id="device",
                    value=devices[0].index if devices else Select.BLANK,
                )
            image_options = _local_image_options()
            yield Select(
                image_options,
                id="image-select",
                value=image_options[0][1] if image_options else Select.BLANK,
                allow_blank=True,
                prompt="pick a local image, or type one below",
            )
            yield Input(
                value=image_options[0][1] if image_options else "",
                placeholder="...or type an image ref directly (e.g. openarc:dev)",
                id="image",
            )
            with Horizontal():
                yield Input(value="8000", placeholder="port", id="port")
                yield Input(value="4096", placeholder="context_size", id="context-size")
            with Horizontal():
                yield Button("Generate config ↓", id="generate-btn", variant="primary")
                yield Button("Run this suite ▶", id="launch-btn", variant="success")
                yield Button("← Back", id="back-btn")
            yield TextArea.code_editor("", language="yaml", id="yaml-preview")
            yield Static("", id="build-status")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "generate-btn":
            self.action_generate()
        elif event.button.id == "launch-btn":
            self.action_launch()
        elif event.button.id == "back-btn":
            self.action_back()

    def on_select_changed(self, event: Select.Changed) -> None:
        # picking a real local image fills the free-text field too, so
        # action_generate() only ever needs to read one source of truth
        if event.select.id == "image-select" and event.value != Select.BLANK:
            self.query_one("#image", Input).value = str(event.value)

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_generate(self) -> None:
        engine = self.query_one("#engine", Select).value
        device_index = self.query_one("#device", Select).value
        image = self.query_one("#image", Input).value.strip()
        port = int(self.query_one("#port", Input).value or "8000")
        context_size = int(self.query_one("#context-size", Input).value or "4096")

        if not image:
            self.query_one("#build-status", Static).update("[red]image ref is required[/red]")
            return

        volumes, model_path = _default_model_path_and_volumes(self._model)
        name = Path(self._model.path).stem.lower().replace(" ", "-").replace("_", "-")
        # REAL BUG found validating this live (see VALIDATION.md): the
        # benchmark/coherence adapters' `model` config defaults to the
        # literal string "default" if unset, which almost never matches
        # what the engine actually serves - e.g. OpenArc's real served name
        # comes from `params.backend_specific.model_name` (defaults to the
        # container model_path's basename if unset, NOT the string
        # "default"). A mismatched model name doesn't fail cleanly here -
        # confirmed via a real container: OpenArc returns HTTP 200 then
        # crashes the SSE stream mid-response
        # (`ValueError: Model 'default' is not loaded or no worker is
        # available`), which httpx surfaces as a raw disconnect. Setting
        # the SAME served name everywhere (backend_specific AND both
        # adapters' `model` config) is required, not optional - putting it
        # in multiple engines' backend_specific keys is harmless, each
        # translator only reads the keys it recognizes.
        served_name = name
        suite_dict: dict[str, Any] = {
            "name": f"tui-{name}",
            "backends": [
                {
                    "name": name,
                    "source": {"mode": "prebuilt", "image": image},
                    "model": Path(self._model.path).name,
                    "model_path": model_path,
                    "engine": engine,
                    "port": port,
                    "volumes": volumes,
                    "params": {
                        "shared": {"context_size": context_size},
                        "backend_specific": {"model_name": served_name, "served_model_name": served_name},
                    },
                    "health_path": _ENGINE_HEALTH_PATH_HINTS.get(engine, "/health"),
                    "startup_timeout_s": 180,
                    "network": {"mode": "disabled"},
                }
            ],
            "device_target": {"mode": "indices", "indices": [device_index] if device_index != Select.BLANK else []},
            "execution_target": {"mode": "local"},
            "benchmark_adapters": [
                {
                    "adapter": "generic-http",
                    "config": {"model": served_name, "prompt": "In one sentence, what is a hash table?", "max_tokens": 64, "num_requests": 3},
                }
            ],
            "coherence_adapters": [{"adapter": "fixed-questions", "config": {"model": served_name}}],
            "storage": {"flat_file_dir": "./results", "extra_adapters": []},
        }
        self.query_one("#yaml-preview", TextArea).text = yaml.safe_dump(suite_dict, sort_keys=False)
        self.query_one("#build-status", Static).update(
            "Generated - review/edit above if needed (params.shared, health_path, volumes for engine-specific mounts), "
            "then click [b]Run this suite[/b]."
        )

    def action_launch(self) -> None:
        raw = self.query_one("#yaml-preview", TextArea).text
        if not raw.strip():
            self.query_one("#build-status", Static).update("[red]click 'Generate config' first[/red]")
            return
        try:
            suite = TestSuite.model_validate(yaml.safe_load(raw))
        except Exception as exc:
            self.query_one("#build-status", Static).update(f"[red]invalid config: {exc}[/red]")
            return
        self.app.push_screen(RunScreen(suite))


class RunScreen(Screen):
    """Live progress via `orchestrator.run_backend`'s `on_event` callback -
    the orchestrator was completely silent internally before this rebuild;
    a run used to look frozen for however long build+start+health-check+
    benchmark+coherence took, then dump a raw Python dict at the very end."""

    BINDINGS = [("escape", "back", "Back to model browser")]

    def __init__(self, suite: TestSuite) -> None:
        super().__init__()
        self._suite = suite

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Static(f"[b]Step 3 of 3:[/b] running [b]{self._suite.name}[/b]. Live progress below.", id="run-title")
            yield RichLog(id="log", wrap=True, highlight=True)
            with VerticalScroll(id="summary-container"):
                yield Static("", id="summary")
            yield Button("← Back to model browser", id="back-btn")
        yield Footer()

    def on_mount(self) -> None:
        self.run_worker(self._run, thread=True, exclusive=True)

    def action_back(self) -> None:
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.action_back()

    def _log(self, message: str) -> None:
        self.app.call_from_thread(self.query_one("#log", RichLog).write, message)

    def _run(self) -> None:
        log = self.query_one("#log", RichLog)
        try:
            outcomes = orchestrator.run_suite(self._suite, on_event=self._log)
        except Exception as exc:
            self._log(f"[bold red]ERROR[/bold red]: {exc}")
            return

        lines = []
        for outcome in outcomes:
            result = outcome.result
            all_passed = all(c.passed == c.total for c in result.coherence) if result.coherence else None
            if all_passed is True:
                verdict = "[bold green]PASS[/bold green]"
            elif all_passed is False:
                verdict = "[bold red]FAIL[/bold red]"
            else:
                verdict = "[yellow]NO COHERENCE CHECK[/yellow]"
            lines.append(f"{verdict}  {result.backend_name}  (run_id={result.run_id})")
            for b in result.benchmarks:
                metrics = ", ".join(f"{k}={v:.2f}" for k, v in b.metrics.items())
                lines.append(f"    [{b.adapter}] {metrics}")
            for c in result.coherence:
                lines.append(f"    [{c.adapter}] {c.passed}/{c.total} passed")
            if outcome.delta_against:
                lines.append(f"    (delta available against prior run {outcome.delta_against.run_id})")
        summary_text = "\n".join(lines) if lines else "(no backends ran)"
        self.app.call_from_thread(self.query_one("#summary", Static).update, summary_text)
