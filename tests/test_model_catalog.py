import json

from llapdance.core.model_catalog import annotate_tested_status, load_run_history, scan_models
from llapdance.core.result import CoherenceResult, RunResult


def test_detects_standalone_gguf_file_and_quant_hint(tmp_path):
    gguf_dir = tmp_path / "gguf"
    gguf_dir.mkdir()
    (gguf_dir / "Ternary-Bonsai-27B-Q2_0.gguf").write_bytes(b"x" * 100)

    models = scan_models([str(tmp_path)])
    assert len(models) == 1
    assert models[0].format == "gguf"
    assert models[0].quant_hint == "Q2_0"
    assert models[0].compatible_engines == ["llama-cpp-sycl", "qxmx"]
    assert models[0].size_bytes == 100


def test_detects_openvino_ir_model_and_reads_dtype(tmp_path):
    model_dir = tmp_path / "OpenVINO" / "Phi-4-mini-instruct-int4-ov"
    model_dir.mkdir(parents=True)
    (model_dir / "openvino_model.xml").write_text("<xml/>")
    (model_dir / "openvino_model.bin").write_bytes(b"x" * 50)
    (model_dir / "openvino_config.json").write_text('{"dtype": "int4"}')

    models = scan_models([str(tmp_path)])
    assert len(models) == 1
    assert models[0].format == "openvino_ir"
    assert models[0].quant_hint == "int4"
    assert models[0].compatible_engines == ["openarc"]


def test_detects_safetensors_model_and_reads_quant_format(tmp_path):
    model_dir = tmp_path / "RedHatAI" / "diffusiongemma-26B-A4B-it-NVFP4"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"x" * 50)
    (model_dir / "config.json").write_text('{"quantization_config": {"format": "nvfp4-pack-quantized"}}')

    models = scan_models([str(tmp_path)])
    assert len(models) == 1
    assert models[0].format == "safetensors"
    assert models[0].quant_hint == "nvfp4-pack-quantized"
    assert models[0].compatible_engines == ["arcaine"]


def test_finds_models_nested_under_an_org_namespace_directory(tmp_path):
    # real layout found on this box: OpenVINO/droans/qwen3.5-9B-int4-ov/...
    model_dir = tmp_path / "OpenVINO" / "droans" / "qwen3.5-9B-int4-ov"
    model_dir.mkdir(parents=True)
    (model_dir / "openvino_model.xml").write_text("<xml/>")

    models = scan_models([str(tmp_path)])
    assert len(models) == 1
    assert "droans" in models[0].path


def test_does_not_descend_into_a_detected_model_roots_own_subdirectories(tmp_path):
    model_dir = tmp_path / "some-model"
    model_dir.mkdir()
    (model_dir / "openvino_model.xml").write_text("<xml/>")
    nested_junk = model_dir / "checkpoints" / "openvino_model.xml"
    nested_junk.parent.mkdir()
    nested_junk.write_text("<xml/>")

    models = scan_models([str(tmp_path)])
    assert len(models) == 1  # the nested one under checkpoints/ is not a second model


def test_unknown_quant_hint_when_no_marker_found(tmp_path):
    gguf_dir = tmp_path / "gguf"
    gguf_dir.mkdir()
    (gguf_dir / "some-model-file.gguf").write_bytes(b"x")
    models = scan_models([str(tmp_path)])
    assert models[0].quant_hint == "unknown"


def test_nonexistent_directory_is_skipped_not_an_error():
    assert scan_models(["/nonexistent/path/xyz"]) == []


def _make_run_result(*, engine: str, model_path: str, volumes: dict[str, str], passed: int, total: int) -> RunResult:
    return RunResult(
        backend_name="test-backend",
        backend_config={"engine": engine, "model_path": model_path, "volumes": volumes},
        execution_target={},
        device_target={},
        coherence=[CoherenceResult(adapter="fixed-questions", total=total, passed=passed, graded_by_match=passed, graded_by_llm_judge=0)],
    )


def test_annotate_tested_status_matches_via_reverse_volume_mount(tmp_path):
    model_dir = tmp_path / "unsloth" / "Qwen3.6-27B-NVFP4"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"x")
    (model_dir / "config.json").write_text('{"model_type": "qwen3_5"}')

    models = scan_models([str(tmp_path)])
    result = _make_run_result(
        engine="arcaine",
        model_path="/models/qwen35",
        volumes={str(model_dir): "/models/qwen35"},
        passed=9,
        total=10,
    )
    annotate_tested_status(models, [result])

    assert "arcaine" in models[0].tested
    status = models[0].tested["arcaine"]
    assert status.outcome == "partial"
    assert status.coherence_summary == "9/10"


def test_annotate_tested_status_matches_gguf_file_nested_under_mount(tmp_path):
    gguf_dir = tmp_path / "gguf" / "Ternary-Bonsai-27B-gguf"
    gguf_dir.mkdir(parents=True)
    (gguf_dir / "Ternary-Bonsai-27B-Q2_0.gguf").write_bytes(b"x")

    models = scan_models([str(tmp_path)])
    result = _make_run_result(
        engine="qxmx",
        model_path="/models/Ternary-Bonsai-27B-Q2_0.gguf",
        volumes={str(gguf_dir): "/models"},
        passed=10,
        total=10,
    )
    annotate_tested_status(models, [result])

    assert models[0].tested["qxmx"].outcome == "pass"


def test_annotate_tested_status_reports_ran_when_no_coherence_adapter_configured(tmp_path):
    model_dir = tmp_path / "some-model"
    model_dir.mkdir()
    (model_dir / "openvino_model.xml").write_text("<xml/>")

    models = scan_models([str(tmp_path)])
    result = RunResult(
        backend_name="test-backend",
        backend_config={"engine": "openarc", "model_path": "/models/foo", "volumes": {str(model_dir): "/models/foo"}},
        execution_target={},
        device_target={},
    )
    annotate_tested_status(models, [result])

    assert models[0].tested["openarc"].outcome == "ran"
    assert models[0].tested["openarc"].coherence_summary is None


def test_annotate_tested_status_leaves_unmatched_models_untested(tmp_path):
    model_dir = tmp_path / "some-model"
    model_dir.mkdir()
    (model_dir / "openvino_model.xml").write_text("<xml/>")

    models = scan_models([str(tmp_path)])
    annotate_tested_status(models, [])

    assert models[0].tested == {}


def test_annotate_tested_status_keeps_most_recent_run_per_engine(tmp_path):
    model_dir = tmp_path / "some-model"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"x")
    (model_dir / "config.json").write_text("{}")

    models = scan_models([str(tmp_path)])
    volumes = {str(model_dir): "/models/foo"}
    older = _make_run_result(engine="arcaine", model_path="/models/foo", volumes=volumes, passed=5, total=10)
    older.timestamp = 100.0
    newer = _make_run_result(engine="arcaine", model_path="/models/foo", volumes=volumes, passed=10, total=10)
    newer.timestamp = 200.0
    annotate_tested_status(models, [newer, older])

    assert models[0].tested["arcaine"].outcome == "pass"


def test_load_run_history_skips_non_run_result_json(tmp_path):
    (tmp_path / "_image_labels.json").write_text(json.dumps({"some-image:latest": "good"}))
    history = load_run_history(str(tmp_path))
    assert history == []


def test_load_run_history_reads_real_stored_results(tmp_path):
    result = _make_run_result(engine="arcaine", model_path="/models/foo", volumes={}, passed=10, total=10)
    (tmp_path / f"{result.run_id}.json").write_text(result.model_dump_json())
    history = load_run_history(str(tmp_path))
    assert len(history) == 1
    assert history[0].run_id == result.run_id


def test_load_run_history_on_nonexistent_directory_returns_empty():
    assert load_run_history("/nonexistent/path/xyz") == []
