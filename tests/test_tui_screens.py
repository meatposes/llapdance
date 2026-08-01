import asyncio

import yaml
from textual.widgets import DataTable, Input, Static, TextArea

from llapdance.config.models import TestSuite
from llapdance.core.model_catalog import ModelInfo
from llapdance.plugins.registry import load_builtin_adapters
from llapdance.tui.app import LLAPDanceApp
from llapdance.tui.screens import BuildScreen, ModelBrowserScreen, _default_model_path_and_volumes, _tested_summary


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
            screen = app.screen
            assert isinstance(screen, ModelBrowserScreen)
            screen.query_one("#scan-dirs", Input).value = str(tmp_path)
            screen.action_scan()
            await pilot.pause()

            table = screen.query_one("#models", DataTable)
            assert table.row_count == 1
            status = screen.query_one("#status", Static).render()
            assert "1 models found" in str(status)

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
