import asyncio
from unittest.mock import patch

import yaml
from textual.widgets import DataTable, Input, Select, Static, TextArea

from llapdance.config.models import TestSuite
from llapdance.core.model_catalog import ModelInfo
from llapdance.plugins.registry import load_builtin_adapters
from llapdance.tui.app import LLAPDanceApp
from llapdance.tui.screens import (
    BackendBrowserScreen,
    BuildScreen,
    HomeScreen,
    ModelBrowserScreen,
    _coerce_sweep_value,
    _default_model_path_and_volumes,
    _local_image_options,
    _short_model_name,
    _tested_summary,
)

_FAKE_IMAGES = [
    {"id": "1", "tags": ["arcaine-server:qwen35fix"], "size": 0},
    {"id": "2", "tags": ["llapdance/qxmx-from-source:main-abc123"], "size": 0},
]


def test_short_model_name_strips_scan_root_prefix():
    # real complaint fixed: the table showed the full absolute host path
    # (e.g. /mnt/ignite/LLM/models/AEON-7/Ornith-1.0-...) - only the last
    # two segments should be shown
    assert _short_model_name("/mnt/ignite/LLM/models/AEON-7/Ornith-1.0-abc") == "AEON-7/Ornith-1.0-abc"
    assert _short_model_name("/mnt/ignite/LLM/models/OpenVINO/phi-2-int4-ov") == "OpenVINO/phi-2-int4-ov"


def test_gguf_model_path_and_volumes_mount_parent_dir(tmp_path):
    gguf = tmp_path / "some-model.gguf"
    gguf.write_bytes(b"x")
    model = ModelInfo(path=str(gguf), format="gguf", compatible_engines=["qxmx"])
    volumes, model_path = _default_model_path_and_volumes(model)
    assert volumes == {str(tmp_path): "/models"}
    assert model_path == "/models/some-model.gguf"


def test_directory_format_model_mounts_itself(tmp_path):
    model = ModelInfo(path=str(tmp_path), format="safetensors", compatible_engines=["arcaine"])
    volumes, model_path = _default_model_path_and_volumes(model)
    assert volumes == {str(tmp_path): "/models"}
    assert model_path == "/models"


def test_tested_summary_reports_untested_when_empty():
    model = ModelInfo(path="/x", format="gguf")
    assert _tested_summary(model) == "untested"


def test_model_browser_scans_real_directory_and_populates_table(tmp_path):
    load_builtin_adapters()
    (tmp_path / "test-model.gguf").write_bytes(b"x")

    async def scenario():
        app = LLAPDanceApp()
        async with app.run_test() as pilot:
            await app.push_screen(ModelBrowserScreen())  # HomeScreen -> "Test by model"
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ModelBrowserScreen)
            screen.query_one("#scan-dirs", Input).value = str(tmp_path)
            screen.action_scan()
            await pilot.pause()

            table = screen.query_one("#models", DataTable)
            assert table.row_count == 1
            # real complaint fixed: "Model" (short relative name) is the
            # first column, no full absolute path anywhere in the row
            first_row = table.get_row_at(0)
            assert str(first_row[0]) == _short_model_name(str(tmp_path / "test-model.gguf"))
            assert str(tmp_path) not in str(first_row[0])
            status = screen.query_one("#status", Static).render()
            assert "1 found" in str(status)

    asyncio.run(scenario())


def test_build_screen_generates_valid_suite_yaml(tmp_path):
    load_builtin_adapters()
    gguf = tmp_path / "test-model.gguf"
    gguf.write_bytes(b"x")
    model = ModelInfo(path=str(gguf), format="gguf", compatible_engines=["qxmx"])

    async def scenario():
        app = LLAPDanceApp()
        async with app.run_test() as pilot:
            await app.push_screen(BuildScreen(model))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, BuildScreen)

            screen.query_one("#image", Input).value = "qxmx:latest"
            screen.action_generate()
            await pilot.pause()

            raw = screen.query_one("#yaml-preview", TextArea).text
            parsed = yaml.safe_load(raw)
            suite = TestSuite.model_validate(parsed)  # must be a real, valid suite

            backend = suite.backends[0]
            assert backend.engine == "qxmx"
            assert backend.source.image == "qxmx:latest"
            assert backend.model_path == f"/models/{gguf.name}"
            assert backend.health_path == "/health"  # real qxmx convention, see _ENGINE_HEALTH_PATH_HINTS
            assert suite.benchmark_adapters[0].adapter == "generic-http"
            assert suite.coherence_adapters[0].adapter == "fixed-questions"

            # real bug found validating this live (see VALIDATION.md): the
            # served name must be consistent everywhere, or a real backend
            # (e.g. OpenArc) crashes mid-stream on a mismatched model name
            served_name = backend.params.backend_specific["model_name"]
            assert backend.params.backend_specific["served_model_name"] == served_name
            assert suite.benchmark_adapters[0].config["model"] == served_name
            assert suite.coherence_adapters[0].config["model"] == served_name

    asyncio.run(scenario())


def test_model_browser_configure_button_navigates_to_build_screen(tmp_path):
    # real bug found from direct user feedback: the app `import`ed Button
    # but never placed one anywhere - every action required knowing a
    # keybinding from the Footer's small text. This clicks the actual
    # button widget, not the underlying action method, to prove the fix.
    load_builtin_adapters()
    (tmp_path / "test-model.gguf").write_bytes(b"x")

    async def scenario():
        app = LLAPDanceApp()
        async with app.run_test(size=(120, 60)) as pilot:
            await pilot.pause()
            await pilot.click("#by-model-btn")  # real HomeScreen click, not a direct push
            await pilot.pause()
            screen = app.screen
            screen.query_one("#scan-dirs", Input).value = str(tmp_path)
            await pilot.click("#scan-btn")
            await pilot.pause()

            table = screen.query_one("#models", DataTable)
            table.move_cursor(row=0)
            await pilot.click("#configure-btn")
            await pilot.pause()

            assert isinstance(app.screen, BuildScreen)

    asyncio.run(scenario())


def test_build_screen_generate_and_run_buttons_are_clickable(tmp_path):
    load_builtin_adapters()
    gguf = tmp_path / "test-model.gguf"
    gguf.write_bytes(b"x")
    model = ModelInfo(path=str(gguf), format="gguf", compatible_engines=["qxmx"])

    async def scenario():
        app = LLAPDanceApp()
        async with app.run_test() as pilot:
            await app.push_screen(BuildScreen(model))
            await pilot.pause()
            screen = app.screen

            screen.query_one("#image", Input).value = "qxmx:latest"
            await pilot.click("#generate-btn")
            await pilot.pause()

            assert screen.query_one("#yaml-preview", TextArea).text != ""

    asyncio.run(scenario())


def test_build_screen_requires_image_before_generating(tmp_path):
    load_builtin_adapters()
    model = ModelInfo(path=str(tmp_path), format="safetensors", compatible_engines=["arcaine"])

    async def scenario():
        app = LLAPDanceApp()
        async with app.run_test() as pilot:
            await app.push_screen(BuildScreen(model))
            await pilot.pause()
            screen = app.screen
            # the image field auto-fills from real local docker images on
            # this machine (see _local_image_options) - clear it to
            # exercise the "no image" path deliberately, not by accident
            screen.query_one("#image", Input).value = ""

            screen.action_generate()
            await pilot.pause()

            assert screen.query_one("#yaml-preview", TextArea).text == ""
            assert "required" in str(screen.query_one("#build-status", Static).render())

    asyncio.run(scenario())


def test_coerce_sweep_value_parses_numbers_but_leaves_strings_alone():
    assert _coerce_sweep_value("2048") == 2048
    assert isinstance(_coerce_sweep_value("2048"), int)
    assert _coerce_sweep_value("0.5") == 0.5
    assert _coerce_sweep_value("q8_0") == "q8_0"


def test_home_screen_by_model_button_pushes_model_browser():
    load_builtin_adapters()

    async def scenario():
        app = LLAPDanceApp()
        async with app.run_test() as pilot:
            assert isinstance(app.screen, HomeScreen)
            await pilot.click("#by-model-btn")
            await pilot.pause()
            assert isinstance(app.screen, ModelBrowserScreen)

    asyncio.run(scenario())


def test_home_screen_by_backend_button_pushes_backend_browser():
    load_builtin_adapters()

    async def scenario():
        app = LLAPDanceApp()
        async with app.run_test() as pilot:
            await pilot.click("#by-backend-btn")
            await pilot.pause()
            assert isinstance(app.screen, BackendBrowserScreen)

    asyncio.run(scenario())


def test_backend_browser_marks_compatible_models_and_configures_with_preselected_engine(tmp_path):
    load_builtin_adapters()
    (tmp_path / "test-model.gguf").write_bytes(b"x")

    async def scenario():
        app = LLAPDanceApp()
        async with app.run_test(size=(140, 60)) as pilot:
            await app.push_screen(BackendBrowserScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, BackendBrowserScreen)

            engine_select = screen.query_one("#engine", Select)
            engine_select.value = "qxmx"
            await pilot.pause()

            screen.query_one("#scan-dirs", Input).value = str(tmp_path)
            screen.action_scan()
            await pilot.pause()

            table = screen.query_one("#models", DataTable)
            assert table.row_count == 1
            table.move_cursor(row=0)
            screen.action_configure()
            await pilot.pause()

            build_screen = app.screen
            assert isinstance(build_screen, BuildScreen)
            assert build_screen.query_one("#engine", Select).value == "qxmx"

    asyncio.run(scenario())


def test_build_screen_generate_applies_real_sweep_axis(tmp_path):
    load_builtin_adapters()
    gguf = tmp_path / "test-model.gguf"
    gguf.write_bytes(b"x")
    model = ModelInfo(path=str(gguf), format="gguf", compatible_engines=["qxmx"])

    async def scenario():
        app = LLAPDanceApp()
        async with app.run_test() as pilot:
            await app.push_screen(BuildScreen(model))
            await pilot.pause()
            screen = app.screen

            screen.query_one("#image", Input).value = "qxmx:latest"
            screen.query_one("#sweep-param", Select).value = "params.shared.context_size"
            screen.query_one("#sweep-values", Input).value = "2048,4096"
            screen.action_generate()
            await pilot.pause()

            raw = screen.query_one("#yaml-preview", TextArea).text
            suite = TestSuite.model_validate(yaml.safe_load(raw))
            sweep = suite.backends[0].sweep
            assert len(sweep) == 1
            assert sweep[0].param == "params.shared.context_size"
            assert sweep[0].values == [2048, 4096]  # real ints, not strings

    asyncio.run(scenario())


def test_local_image_options_filters_by_engine_name():
    # real bug found from direct user feedback (screenshot): the image
    # picker listed every local image unfiltered, and defaulted to
    # whichever docker returned first - an unrelated arcaine image while
    # the engine was qxmx. Filtering by engine-name substring (the same
    # name_filter list_images already supports) fixes it.
    with patch("llapdance.tui.screens.list_images", return_value=[
        {**img, "label": None, "runs": []} for img in _FAKE_IMAGES if "qxmx" in img["tags"][0]
    ]):
        options = _local_image_options("qxmx")
    assert options == [("llapdance/qxmx-from-source:main-abc123", "llapdance/qxmx-from-source:main-abc123")]


def test_build_screen_image_picker_refreshes_on_engine_change(tmp_path):
    load_builtin_adapters()
    gguf = tmp_path / "test-model.gguf"
    gguf.write_bytes(b"x")
    model = ModelInfo(path=str(gguf), format="gguf", compatible_engines=["llama-cpp-sycl", "qxmx"])

    def fake_list_images(execution, catalog_dir=None, name_filter=None):
        matches = [img for img in _FAKE_IMAGES if not name_filter or name_filter in img["tags"][0]]
        return [{**img, "label": None, "runs": []} for img in matches]

    async def scenario():
        with patch("llapdance.tui.screens.list_images", side_effect=fake_list_images):
            app = LLAPDanceApp()
            async with app.run_test() as pilot:
                await app.push_screen(BuildScreen(model))
                await pilot.pause()
                screen = app.screen

                # default engine is llama-cpp-sycl (first compatible) - no
                # local image matches that name in the fake set, so the
                # picker must NOT fall back to the arcaine image
                assert screen.query_one("#image", Input).value == ""
                assert "arcaine" not in str(screen.query_one("#image-select", Select).value)

                screen.query_one("#engine", Select).value = "qxmx"
                await pilot.pause()
                assert screen.query_one("#image", Input).value == "llapdance/qxmx-from-source:main-abc123"

    asyncio.run(scenario())


def test_configure_button_visible_at_a_normal_terminal_size():
    # real complaint fixed: DataTable defaulted to filling ALL remaining
    # vertical space, pushing the Configure button below the visible
    # screen entirely at any normal (non-huge) terminal height. Confirmed
    # via a real pilot check at 100x30 before the fix: the button's
    # region.y (31) landed one row past the screen height (30).
    load_builtin_adapters()

    async def scenario():
        app = LLAPDanceApp()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.click("#by-model-btn")
            await pilot.pause()
            screen = app.screen
            btn = screen.query_one("#configure-btn")
            assert btn.region.offset in screen.size.region

    asyncio.run(scenario())
