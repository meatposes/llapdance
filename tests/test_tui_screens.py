import asyncio
from unittest.mock import patch

import pytest
import yaml
from textual.widgets import DataTable, Input, Select, Static, TextArea

from llapdance.config.models import TestSuite
from llapdance.core.model_catalog import ModelInfo
from llapdance.plugins.registry import available, describe_engine, load_builtin_adapters
from llapdance.tui.app import LLAPDanceApp
from llapdance.tui.screens import (
    BackendBrowserScreen,
    BuildScreen,
    HomeScreen,
    ModelBrowserScreen,
    _coerce_sweep_value,
    _default_model_path_and_volumes,
    _local_image_options,
    _default_sweep_values,
    _parse_sweep_axes_text,
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


def test_parse_sweep_axes_text_handles_multiple_lines():
    text = "params.shared.context_size=2048,4096\nenv.GGML_SYCL_ENABLE_GRAPH=0,1\n"
    axes = _parse_sweep_axes_text(text)
    assert axes == [
        {"param": "params.shared.context_size", "values": [2048, 4096]},
        # Real bug found live: BackendConfig.env is dict[str, str] - an
        # env.* sweep value must stay a string ("0"/"1"), never coerce to
        # a real int like params.shared.* does, or sweep expansion fails
        # pydantic validation the moment it writes into the env dict.
        {"param": "env.GGML_SYCL_ENABLE_GRAPH", "values": ["0", "1"]},
    ]


def test_parse_sweep_axes_text_ignores_blank_lines():
    assert _parse_sweep_axes_text("\n\n  \n") == []


def test_parse_sweep_axes_text_rejects_malformed_line():
    with pytest.raises(ValueError, match="line 1"):
        _parse_sweep_axes_text("not-a-valid-line")


def test_default_sweep_values_uses_enum_values_verbatim():
    assert _default_sweep_values({"type": "str", "values": ["f16", "q8_0"]}) == "f16,q8_0"
    assert _default_sweep_values({"type": "str", "values": ["on", "off", "auto"]}) == "on,off,auto"


def test_default_sweep_values_is_0_1_for_every_boolean_shape():
    # direct user complaint: shouldn't have to guess TRUE/FALSE vs ON/OFF -
    # every real engine in this codebase parses a plain "0"/"1" correctly
    # (checked each translator's own atoi/getenv/truthy-check parsing)
    assert _default_sweep_values({"type": "bool", "default": True}) == "0,1"
    assert _default_sweep_values({"type": "int (0/1)", "default": 1}) == "0,1"
    assert _default_sweep_values({"type": "bool (atoi != 0)"}) == "0,1"
    # explicit "value '0' disables" (qxmx's QXMX_FD/FOLD/VEC) is a real,
    # distinct 0-vs-1 behavior, confirmed via qxmx's own source
    assert _default_sweep_values({"type": "presence (value '0' disables)"}) == "0,1"


def test_default_sweep_values_empty_for_bare_presence_flags():
    # Real bug found auditing this after a live report: a bare/generic
    # "presence" flag (ggml-sycl's GGML_SYCL_NO_PINNED, every GGML_VK_*
    # flag in ggml-vulkan) triggers on ANY set value INCLUDING "0" -
    # confirmed via their real `getenv(...) != nullptr` call sites, no
    # atoi/strcmp involved. Suggesting "0,1" for one of these would
    # silently produce two runs with IDENTICAL real behavior (both "flag
    # present"), not an actual A/B - worse than no suggestion, since it
    # looks like a valid sweep. There is no way to express "leave this
    # env var unset" as one value of a sweep axis, so no default pair can
    # honestly represent that comparison.
    assert _default_sweep_values({"type": "presence"}) == ""
    assert _default_sweep_values({"type": "presence (any non-empty value disables pinned host memory)"}) == ""


def test_default_sweep_values_brackets_a_numeric_default():
    assert _default_sweep_values({"type": "int", "default": 4096}) == "2048,4096"
    assert _default_sweep_values({"type": "int", "default": 8192}) == "4096,8192"


def test_default_sweep_values_steps_up_instead_of_duplicating_a_small_default():
    # halving a default of 1 would produce "1,1" - not a real second point
    assert _default_sweep_values({"type": "int", "default": 1}) == "1,2"


def test_default_sweep_values_empty_when_nothing_to_go_on():
    # no default, no values - no suggestion, never a fabricated one
    assert _default_sweep_values({"type": "str", "maps_to": "LAYER_PLACEMENT env"}) == ""


def test_default_sweep_values_never_suggests_env_string_for_a_bare_presence_flag_across_every_real_engine():
    # Real regression guard for the bare-presence bug: walks every actual
    # registered engine's real catalog (not a hand-picked sample) and
    # confirms any entry whose type is a bare "presence" (no explicit
    # "'0' disables" note) gets no suggestion at all - so this class of
    # bug (a degenerate "0,1" sweep that's actually the same value twice
    # in real behavior) can't silently come back for a newly added flag.
    load_builtin_adapters()
    for engine in available("engine"):
        info = describe_engine(engine)
        for name, entry in {**info["params"], **info["env_flags"]}.items():
            type_str = str(entry.get("type", ""))
            if type_str.strip() == "presence" or (
                "presence" in type_str and "'0' disables" not in type_str and "atoi" not in type_str
            ):
                assert _default_sweep_values(entry) == "", f"{engine}.{name}: {entry}"


def test_default_sweep_values_covers_vllms_presence_flags_now_that_they_have_a_verified_default():
    # Direct follow-up audit ("check other engines' sweep params too"):
    # vllm.py's own build() reads these via a plain params.get(name)
    # truthy check (verified in that file), so unset (None) really is
    # equivalent to False - a real, not fabricated, default.
    load_builtin_adapters()
    info = describe_engine("vllm")["params"]
    for name in ("enable_auto_tool_choice", "trust_remote_code", "language_model_only"):
        assert _default_sweep_values(info[name]) == "0,1", name


def test_build_screen_sweep_values_prefills_a_real_default_on_selection(tmp_path):
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

            screen.query_one("#sweep-param", Select).value = "params.shared.context_size"
            await pilot.pause()
            assert screen.query_one("#sweep-values", Input).value == "2048,4096"

            # switching to an enum param overwrites it with that param's
            # own real values, not a leftover numeric guess
            screen.query_one("#sweep-param", Select).value = "params.shared.kv_cache_quant"
            await pilot.pause()
            assert screen.query_one("#sweep-values", Input).value == "f16,q8_0,f8"

    asyncio.run(scenario())


def test_build_screen_add_sweep_axis_button_appends_line_and_clears_builder(tmp_path):
    # Direct user request: sweeping needs "the ability to select several
    # at a time" - clicking + axis should stack another line rather than
    # overwrite/replace the previous one.
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

            screen.query_one("#sweep-param", Select).value = "params.shared.context_size"
            screen.query_one("#sweep-values", Input).value = "2048,4096"
            screen.action_add_sweep_axis()
            await pilot.pause()

            assert screen.query_one("#sweep-values", Input).value == ""  # builder row cleared
            assert screen.query_one("#sweep-param", Select).value == Select.NULL

            screen.query_one("#sweep-param", Select).value = "params.shared.parallel_slots"
            screen.query_one("#sweep-values", Input).value = "1,2"
            screen.action_add_sweep_axis()
            await pilot.pause()

            axes_text = screen.query_one("#sweep-axes", TextArea).text
            assert "params.shared.context_size=2048,4096" in axes_text
            assert "params.shared.parallel_slots=1,2" in axes_text

    asyncio.run(scenario())


def test_build_screen_generate_applies_multiple_sweep_axes(tmp_path):
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
            screen.query_one("#sweep-axes", TextArea).text = "params.shared.context_size=2048,4096\n"
            # plus one more axis still sitting in the unclicked builder row
            screen.query_one("#sweep-param", Select).value = "params.shared.parallel_slots"
            screen.query_one("#sweep-values", Input).value = "1,2"
            screen.action_generate()
            await pilot.pause()

            raw = screen.query_one("#yaml-preview", TextArea).text
            suite = TestSuite.model_validate(yaml.safe_load(raw))
            sweep = suite.backends[0].sweep
            assert len(sweep) == 2
            assert sweep[0].param == "params.shared.context_size"
            assert sweep[0].values == [2048, 4096]
            assert sweep[1].param == "params.shared.parallel_slots"
            assert sweep[1].values == [1, 2]

            status = str(screen.query_one("#build-status", Static).render())
            assert "4 runs" in status  # 2 * 2 combinations

    asyncio.run(scenario())


def test_build_screen_env_sweep_survives_real_expansion(tmp_path):
    # Real bug, found live: "Input should be a valid string [type=
    # string_type, input_value=0, input_type=int]" - BackendConfig.env is
    # dict[str, str], but sweeping env.GGML_SYCL_ENABLE_GRAPH with the
    # TUI's own suggested default ("0,1") coerced those to real ints,
    # which crashed the moment sweep.py's expand_backend_sweep() wrote
    # them into the expanded backend's env dict and re-validated it. The
    # compact TestSuite.model_validate() in the other sweep tests doesn't
    # catch this - SweepAxis.values is `list[Any]` - only real expansion
    # does, which is why this test goes one step further than the others.
    load_builtin_adapters()
    gguf = tmp_path / "test-model.gguf"
    gguf.write_bytes(b"x")
    model = ModelInfo(path=str(gguf), format="gguf", compatible_engines=["llama-cpp-sycl"])

    async def scenario():
        app = LLAPDanceApp()
        async with app.run_test() as pilot:
            await app.push_screen(BuildScreen(model))
            await pilot.pause()
            screen = app.screen

            screen.query_one("#image", Input).value = "llama-cpp-bonsai:meat6-hardened"
            screen.query_one("#sweep-param", Select).value = "env.GGML_SYCL_ENABLE_GRAPH"
            await pilot.pause()
            # the TUI's own suggested default for this param - exactly
            # what a user clicking through, not hand-typing, would sweep
            assert screen.query_one("#sweep-values", Input).value == "0,1"
            screen.action_generate()
            await pilot.pause()

            raw = screen.query_one("#yaml-preview", TextArea).text
            suite = TestSuite.model_validate(yaml.safe_load(raw))
            assert suite.backends[0].sweep[0].values == ["0", "1"]

            from llapdance.config.sweep import expand_backend_sweep

            expanded = expand_backend_sweep(suite.backends[0])
            assert len(expanded) == 2
            assert expanded[0].env["GGML_SYCL_ENABLE_GRAPH"] == "0"
            assert expanded[1].env["GGML_SYCL_ENABLE_GRAPH"] == "1"

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
