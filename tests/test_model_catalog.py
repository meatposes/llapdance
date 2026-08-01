from llapdance.core.model_catalog import scan_models


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
