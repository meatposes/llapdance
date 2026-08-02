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
from llapdance.plugins.registry import available, describe_engine

# Known real health-check conventions per engine, confirmed against real
# containers this session (see VALIDATION.md) - not guessed. Still just a
# starting point in the generated YAML, not silently trusted: the preview
# step shows it and lets a human correct it before running.
_ENGINE_HEALTH_PATH_HINTS: dict[str, str] = {
    "llama-cpp-sycl": "/health",
    "llama-cpp-vulkan": "/health",  # same llama-server binary/health endpoint as llama-cpp-sycl
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


def _coerce_sweep_value(raw: str) -> Any:
    """A sweep value typed into the TUI is always a string - coerce it to
    int/float where it genuinely parses as one, so e.g. sweeping
    params.shared.context_size across "2048,4096" produces real integers
    in the generated YAML, not strings that only look numeric."""
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _coerce_sweep_value_for_param(param: str, raw: str) -> Any:
    """Real bug found live: `BackendConfig.env` is strictly `dict[str, str]`
    (env vars are always strings at the OS/docker level, never real ints)
    - coercing an `env.<NAME>` sweep value like "0"/"1" to an actual int
    (as `_coerce_sweep_value` does for params.shared/backend_specific,
    which are open `Any` dicts) fails pydantic validation the moment
    `sweep.py` writes it into the expanded backend's `env` dict
    ("Input should be a valid string [type=string_type, input_value=0,
    input_type=int]"). `env.*` sweep values are left as plain strings;
    only `params.*` paths get numeric coercion."""
    if param.startswith("env."):
        return raw
    return _coerce_sweep_value(raw)


def _parse_sweep_axes_text(text: str) -> list[dict[str, Any]]:
    """Parses the `#sweep-axes` box's `param=values` lines (one per axis,
    blank lines ignored) into real `{"param", "values"}` dicts - the same
    shape `SweepAxis` expects. Direct user request: sweeping several
    params "at a time" - each line becomes one more axis in the cartesian
    product `llapdance/config/sweep.py` already expands at run time; this
    is purely a TUI-side parser, the multi-axis mechanism itself already
    existed and was already validated (see VALIDATION.md)."""
    axes = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"sweep axis line {lineno} ({line!r}) must be 'param=values', e.g. 'params.shared.context_size=2048,4096'")
        param, raw_values = line.split("=", 1)
        param = param.strip()
        values = [_coerce_sweep_value_for_param(param, v.strip()) for v in raw_values.split(",") if v.strip()]
        if not param or not values:
            raise ValueError(f"sweep axis line {lineno} ({line!r}) needs both a param and at least one value")
        axes.append({"param": param, "values": values})
    return axes


def _default_sweep_values(info: dict[str, Any]) -> str:
    """Direct user complaint: the sweep-values field never suggested
    anything, so every sweep meant re-guessing correct on/off spelling
    (TRUE? ON? 1?) and re-typing common numeric pairs from memory every
    time. Derives a real starting pair from the same catalog metadata
    already shown in the engine-info line (`describe_engine()`'s
    `values`/`default`/`type`) - never invents a magnitude or unit that
    isn't already in that entry; a param with no `default` and no
    `values` gets no suggestion (still just the placeholder), rather than
    a fabricated one.

    On/off spelling: checked every engine's actual parsing this project
    has (qxmx's `atoi`, ggml-sycl/vulkan's `getenv(...) != nullptr` or
    `ggml_sycl_get_env` int parsing, vLLM's own truthy `params.get(...)`
    checks in vllm.py's build()) - every single one treats a plain "0"/"1"
    string correctly (falsy/truthy, or exact match for the "0 disables"
    ones), so "0,1" is a real, uniform answer across this whole codebase,
    not a per-engine guess."""
    values = info.get("values")
    if values:
        return ",".join(str(v) for v in values)

    type_str = str(info.get("type", "")).lower()
    default = info.get("default")

    if isinstance(default, bool) or "bool" in type_str or "presence" in type_str or "0/1" in type_str:
        return "0,1"

    if isinstance(default, (int, float)) and not isinstance(default, bool):
        if isinstance(default, int) and default < 2:
            # halving would duplicate a small default (e.g. qxmx's
            # parallel_slots default of 1 -> "1,1", not a real second
            # point) - step up by one instead, still two genuinely
            # distinct values either side of the default.
            return f"{default},{default + 1}"
        low = default // 2 if isinstance(default, int) else default
        return f"{low},{default}"

    return ""


def _short_model_name(path: str) -> str:
    """Real complaint fixed: the model table showed the FULL absolute host
    path (e.g. `/mnt/ignite/LLM/models/AEON-7/Ornith-1.0-...`), both as a
    column and reused in on-screen text. Real model layouts on this
    project's actual scan directories are consistently `<root>/<org>/
    <model>` or `<root>/<model>` (confirmed against every path seen this
    session) - the last two path segments are what a human actually reads
    to identify a model ("AEON-7/Ornith-1.0-...", "phi-2-int4-ov"), not the
    scan root prefix repeated on every row."""
    parts = Path(path).parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else path


def _model_name_cell(m: ModelInfo) -> str:
    """Table-cell text for a model's name column: the short relative name
    plus a compact `(m)` indicator when a sibling mmproj (multimodal
    projector) file exists alongside it (see `ModelInfo.has_mmproj`) -
    direct user feedback: hide the mmproj files themselves as their own
    rows (they aren't standalone models), but don't hide the fact that one
    exists, since it matters for e.g. vision-capable serving."""
    name = _short_model_name(m.path)
    return f"{name} (m)" if m.has_mmproj else name


def _tested_summary(model: ModelInfo) -> str:
    if not model.tested:
        return "untested"
    return ", ".join(f"{engine}:{status.outcome}({status.coherence_summary or 'n/a'})" for engine, status in model.tested.items())


class HomeScreen(Screen):
    """Entry point: two real ways to start a test, per direct user request.
    'Test by model' starts from what you have on disk and narrows to
    compatible engines (`ModelBrowserScreen`, the original flow). 'Test by
    backend' starts from a real registered engine (its real sweepable
    params/env flags shown up front) and narrows to models known to be
    compatible with it (`BackendBrowserScreen`) - useful when the question
    is "what can I throw at OpenArc" rather than "what can I do with this
    model.\""""

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Button("Test by model →", id="by-model-btn", variant="primary")
            yield Button("Test by backend →", id="by-backend-btn", variant="primary")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "by-model-btn":
            self.app.push_screen(ModelBrowserScreen())
        elif event.button.id == "by-backend-btn":
            self.app.push_screen(BackendBrowserScreen())


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

    BINDINGS = [
        ("s", "scan", "Scan directories"),
        ("enter", "configure", "Configure a run for this model"),
        ("escape", "back", "Back"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._models: list[ModelInfo] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            with Horizontal():
                yield Input(value="/mnt/ignite/LLM/models", id="scan-dirs")
                yield Button("Scan", id="scan-btn", variant="primary")
            yield DataTable(id="models")
            yield Static("", id="status")
            with Horizontal():
                yield Button("← Back", id="back-btn")
                yield Button("Configure →", id="configure-btn", variant="success")
        yield Footer()

    def action_back(self) -> None:
        self.app.pop_screen()

    def on_mount(self) -> None:
        table = self.query_one("#models", DataTable)
        table.add_columns("Model", "Format", "Quant", "Engines", "Tested")
        table.cursor_type = "row"
        self.action_scan()

    def action_scan(self) -> None:
        dirs = self.query_one("#scan-dirs", Input).value.split()
        table = self.query_one("#models", DataTable)
        table.clear()
        self._models = scan_models(dirs)
        annotate_tested_status(self._models, load_run_history("./results"))
        for m in self._models:
            table.add_row(_model_name_cell(m), m.format, m.quant_hint, ", ".join(m.compatible_engines) or "-", _tested_summary(m))
        status = self.query_one("#status", Static)
        if self._models:
            status.update(f"{len(self._models)} found. Pick a row, click Configure.")
        else:
            status.update(f"No models under {' '.join(dirs) or '(none given)'}.")

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
        elif event.button.id == "back-btn":
            self.action_back()


def _short_list(items: list[str], limit: int = 4) -> str:
    """Cap how many names get printed on one line - real engines can have
    a dozen+ known env flags (e.g. Arcaine's ARCAINE_QWEN35_* family), and
    printing all of them wraps a single Static across many lines, which is
    exactly the "too many lines" complaint this pass is fixing."""
    if not items:
        return "-"
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f", +{len(items) - limit} more"


def _engine_info_text(engine: str) -> str:
    info = describe_engine(engine)
    params = _short_list(list(info["params"].keys()))
    env_flags = _short_list(list(info["env_flags"].keys()))
    images = _short_list(info["image_hints"])
    return f"[b]{engine}[/b] params: {params} | env: {env_flags} | images: {images}"


class BackendBrowserScreen(Screen):
    """Test-by-backend: the other real entry point, per direct request -
    pick a real registered engine first (its real sweepable params/env
    flags shown immediately, from `describe_engine` - the same info
    `llapdance describe-engine` prints), THEN narrow to models known to be
    compatible with it. Converges on the same `BuildScreen` as
    test-by-model, just with the engine already chosen."""

    BINDINGS = [("escape", "back", "Back")]

    def __init__(self) -> None:
        super().__init__()
        self._models: list[ModelInfo] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            engines = available("engine")
            yield Select([(e, e) for e in engines], id="engine", value=engines[0] if engines else Select.NULL)
            yield Static(_engine_info_text(engines[0]) if engines else "(no engines registered)", id="engine-info")
            with Horizontal():
                yield Input(value="/mnt/ignite/LLM/models", id="scan-dirs")
                yield Button("Scan", id="scan-btn", variant="primary")
            yield DataTable(id="models")
            yield Static("", id="status")
            with Horizontal():
                yield Button("← Back", id="back-btn")
                yield Button("Configure →", id="configure-btn", variant="success")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#models", DataTable)
        table.add_columns("Model", "Format", "Quant", "OK?", "Tested")
        table.cursor_type = "row"

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "engine" and event.value != Select.NULL:
            self.query_one("#engine-info", Static).update(_engine_info_text(event.value))
            self._refresh_table()

    def action_scan(self) -> None:
        self._refresh_table()

    def _refresh_table(self) -> None:
        engine = self.query_one("#engine", Select).value
        dirs = self.query_one("#scan-dirs", Input).value.split()
        table = self.query_one("#models", DataTable)
        table.clear()
        self._models = scan_models(dirs)
        annotate_tested_status(self._models, load_run_history("./results"))
        # compatible-first, so the models this backend can actually load
        # aren't buried under ones it can't
        self._models.sort(key=lambda m: engine not in m.compatible_engines)
        for m in self._models:
            compatible = "yes" if engine in m.compatible_engines else "no"
            table.add_row(_model_name_cell(m), m.format, m.quant_hint, compatible, _tested_summary(m))
        status = self.query_one("#status", Static)
        n_compatible = sum(1 for m in self._models if engine in m.compatible_engines)
        status.update(f"{n_compatible}/{len(self._models)} compatible with {engine}.")

    def action_configure(self) -> None:
        table = self.query_one("#models", DataTable)
        if not self._models:
            self.query_one("#status", Static).update("[red]scan a directory with real models in it first[/red]")
            return
        if table.cursor_row is None:
            self.query_one("#status", Static).update("[red]select a row in the table first[/red]")
            return
        model = self._models[table.cursor_row]
        engine = self.query_one("#engine", Select).value
        self.app.push_screen(BuildScreen(model, preselected_engine=engine))

    def action_back(self) -> None:
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "scan-btn":
            self.action_scan()
        elif event.button.id == "configure-btn":
            self.action_configure()
        elif event.button.id == "back-btn":
            self.action_back()


def _local_image_options(engine: str | None = None) -> list[tuple[str, str]]:
    """Real local docker images (the same `LocalDockerExecutionTarget.list_images()`
    the CLI's `llapdance images` command uses) - so a user picks from what
    actually exists on this machine instead of typing a tag from memory.

    Real bug found from direct user feedback (screenshot): with no filter,
    this listed every local image regardless of engine, and the picker
    defaulted to whatever docker returned first - which was an unrelated
    `arcaine-server:qwen35fix` image while the engine was `qxmx`, silently
    filling both the Select AND the Input with a mismatched image.

    Filters by the engine's own `image_hints` (real glob patterns of tags
    it's actually been validated/run against - see EngineTranslator's
    docstring, llapdance/plugins/base.py) when the registered engine
    declares any. Falls back to a plain substring match on the engine
    name (the previous, cruder behavior) for an engine that hasn't
    declared hints yet, rather than showing nothing - e.g. this caught
    that `llama-cpp-sycl`'s real validated image is tagged
    `llama-cpp-bonsai:*`, which the old substring-on-engine-name approach
    would never have matched at all."""
    import fnmatch

    from llapdance.plugins.execution.local_docker import LocalDockerExecutionTarget
    from llapdance.plugins.registry import get as get_adapter

    hints: list[str] = []
    if engine:
        try:
            hints = list(get_adapter("engine", engine).image_hints)
        except KeyError:
            hints = []

    try:
        # hints (if any) do their own matching below; only use the crude
        # substring name_filter when there's nothing better to go on
        images = list_images(
            LocalDockerExecutionTarget({}), catalog_dir="./results", name_filter=None if hints else engine
        )
    except Exception:
        return []
    options = []
    for image in images:
        for tag in image["tags"]:
            if hints and not any(fnmatch.fnmatch(tag, pattern) for pattern in hints):
                continue
            options.append((tag, tag))
    return options


class BuildScreen(Screen):
    """Step 2 of 3: pick a real registered engine + real discovered device
    + a real local image, generate a real suite (same Pydantic models the
    CLI validates against), and review/edit it as YAML before running -
    see module docstring for why this isn't fully automatic."""

    BINDINGS = [("escape", "back", "Back"), ("g", "generate", "Generate/refresh YAML"), ("r", "launch", "Run this suite")]

    def __init__(self, model: ModelInfo, preselected_engine: str | None = None) -> None:
        super().__init__()
        self._model = model
        self._preselected_engine = preselected_engine
        self._sweep_param_info: dict[str, dict[str, Any]] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Static(f"[b]{_short_model_name(self._model.path)}[/b] ({self._model.format}, {self._model.quant_hint})")
            with Horizontal():
                engines = self._model.compatible_engines or available("engine")
                default_engine = self._preselected_engine if self._preselected_engine in engines else (engines[0] if engines else Select.NULL)
                yield Select([(e, e) for e in engines], id="engine", value=default_engine)
                devices = discover_devices()
                yield Select(
                    [(f"{d.index}: {d.name}", d.index) for d in devices],
                    id="device",
                    value=devices[0].index if devices else Select.NULL,
                )
            image_options = _local_image_options(default_engine if default_engine != Select.NULL else None)
            with Horizontal():
                yield Select(
                    image_options,
                    id="image-select",
                    value=image_options[0][1] if image_options else Select.NULL,
                    allow_blank=True,
                    prompt="local image...",
                )
                yield Input(
                    value=image_options[0][1] if image_options else "",
                    placeholder="...or type image ref",
                    id="image",
                )
            with Horizontal():
                yield Input(value="8000", placeholder="port", id="port")
                yield Input(value="4096", placeholder="context_size", id="context-size")
            with Horizontal():
                yield Select([], id="sweep-param", allow_blank=True, prompt="sweep param (optional)")
                yield Input(placeholder="sweep values, e.g. 2048,4096", id="sweep-values")
                yield Button("+ axis", id="add-sweep-btn")
            # Direct user feedback: sweeping was "one at a time" - this box
            # holds one `param=values` line per axis (editable directly, or
            # built up via the Select/Input/+axis row above), so multiple
            # axes can be swept together as a real cartesian product (the
            # underlying mechanism, llapdance/config/sweep.py, already
            # supported this - only the TUI was single-axis).
            yield TextArea("", id="sweep-axes")
            with Horizontal():
                yield Button("Generate", id="generate-btn", variant="primary")
                yield Button("Run ▶", id="launch-btn", variant="success")
                yield Button("← Back", id="back-btn")
            yield TextArea.code_editor("", language="yaml", id="yaml-preview")
            yield Static("", id="build-status")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_sweep_options()

    def _refresh_sweep_options(self) -> None:
        engine = self.query_one("#engine", Select).value
        if engine == Select.NULL:
            return
        info = describe_engine(engine)
        options = [(f"params.shared.{k}", f"params.shared.{k}") for k in info["params"]]
        options += [(f"env.{k}", f"env.{k}") for k in info["env_flags"]]
        self._sweep_param_info = {f"params.shared.{k}": v for k, v in info["params"].items()}
        self._sweep_param_info.update({f"env.{k}": v for k, v in info["env_flags"].items()})
        sweep_select = self.query_one("#sweep-param", Select)
        sweep_select.set_options(options)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "generate-btn":
            self.action_generate()
        elif event.button.id == "launch-btn":
            self.action_launch()
        elif event.button.id == "back-btn":
            self.action_back()
        elif event.button.id == "add-sweep-btn":
            self.action_add_sweep_axis()

    def action_add_sweep_axis(self) -> None:
        """Appends the current sweep-param/sweep-values builder row as one
        more line in `#sweep-axes`, then clears the builder row so the same
        two widgets can be reused to add another axis - direct user
        request: the sweep needs "the ability to select several at a
        time", not just one param per run."""
        sweep_param = self.query_one("#sweep-param", Select).value
        sweep_values_raw = self.query_one("#sweep-values", Input).value.strip()
        if sweep_param == Select.NULL or not sweep_values_raw:
            self.query_one("#build-status", Static).update("[red]pick a sweep param and enter values first[/red]")
            return
        axes_box = self.query_one("#sweep-axes", TextArea)
        existing = axes_box.text
        prefix = existing if (not existing or existing.endswith("\n")) else existing + "\n"
        axes_box.text = f"{prefix}{sweep_param}={sweep_values_raw}\n"
        self.query_one("#sweep-values", Input).value = ""
        self.query_one("#sweep-param", Select).value = Select.NULL

    def on_select_changed(self, event: Select.Changed) -> None:
        # picking a real local image fills the free-text field too, so
        # action_generate() only ever needs to read one source of truth
        if event.select.id == "image-select" and event.value != Select.NULL:
            self.query_one("#image", Input).value = str(event.value)
        elif event.select.id == "engine":
            self._refresh_sweep_options()
            self._refresh_image_options()
        elif event.select.id == "sweep-param" and event.value != Select.NULL:
            # Direct user complaint: picking a param never suggested real
            # values to sweep - always prefill a real default (from the
            # same catalog metadata the engine-info line already shows),
            # still just a plain Input the user can overwrite.
            #
            # Guard against a stale event: action_add_sweep_axis() sets
            # sweep-param back to Select.NULL synchronously, without an
            # intervening pilot.pause() in some callers, so a queued
            # Changed message for the PREVIOUS selection can still be
            # pending when that happens - only act if this event still
            # matches the widget's current value.
            sweep_param_select = self.query_one("#sweep-param", Select)
            if event.value != sweep_param_select.value:
                return
            info = self._sweep_param_info.get(event.value, {})
            default = _default_sweep_values(info)
            if default:
                self.query_one("#sweep-values", Input).value = default

    def _refresh_image_options(self) -> None:
        engine = self.query_one("#engine", Select).value
        image_options = _local_image_options(engine if engine != Select.NULL else None)
        image_select = self.query_one("#image-select", Select)
        image_select.set_options(image_options)
        new_value = image_options[0][1] if image_options else Select.NULL
        image_select.value = new_value
        self.query_one("#image", Input).value = image_options[0][1] if image_options else ""

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
            "device_target": {"mode": "indices", "indices": [device_index] if device_index != Select.NULL else []},
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

        # Real sweep mechanism (SPEC.md §10, llapdance/config/sweep.py) -
        # this ONE backend config expands into the cartesian product of
        # every axis's values at `run_suite` time, same as any hand-written
        # sweep suite. Direct user request: sweeping needs to handle
        # SEVERAL params at once, not just one - axes come from the
        # `#sweep-axes` box (one `param=values` line each), with whatever's
        # still sitting unclicked in the param/values builder row folded in
        # too, so a user doesn't have to click "+ axis" for a single-axis
        # sweep.
        try:
            axes = _parse_sweep_axes_text(self.query_one("#sweep-axes", TextArea).text)
        except ValueError as exc:
            self.query_one("#build-status", Static).update(f"[red]{exc}[/red]")
            return

        sweep_param = self.query_one("#sweep-param", Select).value
        sweep_values_raw = self.query_one("#sweep-values", Input).value.strip()
        if sweep_param != Select.NULL and sweep_values_raw:
            values = [_coerce_sweep_value_for_param(sweep_param, v.strip()) for v in sweep_values_raw.split(",") if v.strip()]
            axes.append({"param": sweep_param, "values": values})

        if axes:
            suite_dict["backends"][0]["sweep"] = axes

        self.query_one("#yaml-preview", TextArea).text = yaml.safe_dump(suite_dict, sort_keys=False)
        if axes:
            total_runs = 1
            for axis in axes:
                total_runs *= len(axis["values"])
            params_list = ", ".join(axis["param"] for axis in axes)
            sweep_note = f" ({total_runs} runs, sweeping {params_list})"
        else:
            sweep_note = ""
        self.query_one("#build-status", Static).update(f"Generated{sweep_note} - edit above if needed, then Run.")

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

    BINDINGS = [("escape", "back", "Back")]

    def __init__(self, suite: TestSuite) -> None:
        super().__init__()
        self._suite = suite

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Static(f"Running [b]{self._suite.name}[/b]...", id="run-title")
            yield RichLog(id="log", wrap=True, highlight=True)
            with VerticalScroll(id="summary-container"):
                yield Static("", id="summary")
            yield Button("← Back", id="back-btn")
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
