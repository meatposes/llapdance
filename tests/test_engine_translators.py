import pytest

from llapdance.core.probe import DeviceInfo
from llapdance.plugins.engine.arcaine import ArcaineEngine
from llapdance.plugins.engine.llama_cpp_sycl import LlamaCppSyclEngine
from llapdance.plugins.engine.openarc import OpenArcEngine
from llapdance.plugins.engine.qxmx import QxmxEngine

DEVICE = DeviceInfo(index=3, vendor="intel", name="Arc B70", integrated=False, pci_bus_id="0000:8a:00.0", render_node="/dev/dri/renderD131")
DEVICE_NO_RENDER_NODE = DeviceInfo(index=0, vendor="intel", name="Arc B70 (clinfo)", integrated=False)


class TestLlamaCppSycl:
    def test_minimal_params_matches_validated_defaults(self):
        inv = LlamaCppSyclEngine().build(model_path="/models/x.gguf", params={}, port=8080, device=DEVICE)
        assert inv.command == ["-m", "/models/x.gguf", "-c", "4096", "--port", "8080", "-ngl", "99"]
        assert inv.env == {"LLAMA_ARG_HOST": "0.0.0.0"}
        assert inv.devices == ["/dev/dri/renderD131:/dev/dri/renderD131"]

    def test_optional_params_added_when_present(self):
        inv = LlamaCppSyclEngine().build(
            model_path="/models/x.gguf",
            params={"context_size": 8192, "batch_size": 512, "kv_cache_quant": "q8_0", "parallel_slots": 2},
            port=8080,
            device=DEVICE,
        )
        assert "-b" in inv.command and inv.command[inv.command.index("-b") + 1] == "512"
        assert "-ub" in inv.command and inv.command[inv.command.index("-ub") + 1] == "512"
        assert "--cache-type-k" in inv.command
        assert "--parallel" in inv.command and inv.command[inv.command.index("--parallel") + 1] == "2"

    def test_reasoning_off_sets_env_var(self):
        # GOTCHA regression test: an earlier version of this translator
        # omitted --reasoning entirely, assuming it was fork-specific fluff.
        # It's a real upstream llama.cpp flag (LLAMA_ARG_REASONING) whose
        # default ("auto") silently broke coherence checking on models
        # whose chat template auto-enables thinking - see VALIDATION.md.
        inv = LlamaCppSyclEngine().build(model_path="/m.gguf", params={"reasoning": "off"}, port=8080, device=DEVICE)
        assert inv.env["LLAMA_ARG_REASONING"] == "off"

    def test_reasoning_omitted_when_not_set(self):
        inv = LlamaCppSyclEngine().build(model_path="/m.gguf", params={}, port=8080, device=DEVICE)
        assert "LLAMA_ARG_REASONING" not in inv.env

    def test_invalid_reasoning_value_rejected(self):
        with pytest.raises(ValueError, match="reasoning"):
            LlamaCppSyclEngine().build(model_path="/m.gguf", params={"reasoning": "sometimes"}, port=8080, device=DEVICE)

    def test_f8_kv_quant_rejected(self):
        with pytest.raises(ValueError, match="fp8"):
            LlamaCppSyclEngine().build(model_path="/m.gguf", params={"kv_cache_quant": "f8"}, port=8080, device=DEVICE)

    def test_no_device_means_cpu_only(self):
        inv = LlamaCppSyclEngine().build(model_path="/m.gguf", params={}, port=8080, device=None)
        assert inv.devices == []
        assert "-ngl" in inv.command and inv.command[inv.command.index("-ngl") + 1] == "0"

    def test_device_without_render_node_raises(self):
        with pytest.raises(ValueError, match="render_node"):
            LlamaCppSyclEngine().build(model_path="/m.gguf", params={}, port=8080, device=DEVICE_NO_RENDER_NODE)


class TestQxmx:
    def test_minimal_params_matches_validated_defaults(self):
        inv = QxmxEngine().build(model_path="/models/x.gguf", params={}, port=8080, device=DEVICE)
        assert inv.command == [
            "./build/qxmx_serve", "--host", "0.0.0.0", "--port", "8080",
            "--slots", "1", "--ctx-per-slot", "4096", "/models/x.gguf",
        ]
        assert inv.devices == ["/dev/dri/renderD131:/dev/dri/renderD131"]

    def test_kv_quant_value_translated_to_qxmx_spelling(self):
        inv = QxmxEngine().build(model_path="/m.gguf", params={"kv_cache_quant": "f16"}, port=8080, device=DEVICE)
        assert "--ctk" in inv.command
        assert inv.command[inv.command.index("--ctk") + 1] == "fp16"  # not "f16" - qxmx's own spelling

    def test_no_device_raises(self):
        with pytest.raises(ValueError, match="GPU"):
            QxmxEngine().build(model_path="/m.gguf", params={}, port=8080, device=None)

    def test_device_without_render_node_raises(self):
        with pytest.raises(ValueError, match="render_node"):
            QxmxEngine().build(model_path="/m.gguf", params={}, port=8080, device=DEVICE_NO_RENDER_NODE)


class TestArcaine:
    def test_minimal_params_matches_validated_defaults(self):
        inv = ArcaineEngine().build(model_path="/models/diffusiongemma", params={}, port=7461, device=DEVICE)
        assert inv.command == []  # fully env-driven, confirmed via a real container
        assert inv.env["MODEL_PATH"] == "/models/diffusiongemma"
        assert inv.env["MAX_SEQ"] == "4096"
        assert inv.env["DEFAULT_MAX_TOKENS"] == "2048"
        assert inv.devices == ["/dev/dri/renderD131:/dev/dri/renderD131"]

    def test_diffusion_specific_params_added_when_present(self):
        inv = ArcaineEngine().build(
            model_path="/m", params={"denoising_steps": 20, "seed": 42, "served_model_name": "my-model"}, port=7461, device=DEVICE
        )
        assert inv.env["DENOISING_STEPS"] == "20"
        assert inv.env["DEFAULT_SEED"] == "42"
        assert inv.env["SERVED_MODEL_NAME"] == "my-model"

    def test_no_device_means_no_devices_list(self):
        inv = ArcaineEngine().build(model_path="/m", params={}, port=7461, device=None)
        assert inv.devices == []

    def test_device_without_render_node_raises(self):
        with pytest.raises(ValueError, match="render_node"):
            ArcaineEngine().build(model_path="/m", params={}, port=7461, device=DEVICE_NO_RENDER_NODE)


class TestOpenArc:
    def test_minimal_params_generates_load_request(self):
        inv = OpenArcEngine().build(model_path="/models/phi4-mini", params={}, port=8000, device=DEVICE)
        assert inv.command == []
        assert inv.devices == ["/dev/dri/renderD131:/dev/dri/renderD131"]
        assert len(inv.post_start_requests) == 1
        req = inv.post_start_requests[0]
        assert req["method"] == "POST"
        assert req["path"] == "/openarc/load"
        assert req["json"] == {
            "model_path": "/models/phi4-mini",
            "model_name": "phi4-mini",  # derived from model_path's final component
            "model_type": "llm",
            "engine": "ovgenai",
            "device": "GPU",
            "runtime_config": {},
        }

    def test_runtime_config_passed_through_to_openvino(self):
        # real gap found cataloging this engine's sweepable params: an
        # earlier version silently dropped runtime_config even though
        # OpenArc's own source (ov_genai/llm.py) merges it straight into
        # the LLMPipeline call - fixed, see module docstring
        inv = OpenArcEngine().build(
            model_path="/models/phi4-mini",
            params={"runtime_config": {"NUM_STREAMS": "2", "PERFORMANCE_HINT": "THROUGHPUT"}},
            port=8000,
            device=DEVICE,
        )
        assert inv.post_start_requests[0]["json"]["runtime_config"] == {
            "NUM_STREAMS": "2",
            "PERFORMANCE_HINT": "THROUGHPUT",
        }

    def test_explicit_model_name_and_type_used(self):
        inv = OpenArcEngine().build(
            model_path="/models/phi4-mini",
            params={"model_name": "my-name", "model_type": "vlm", "openarc_engine": "optimum"},
            port=8000,
            device=DEVICE,
        )
        req = inv.post_start_requests[0]
        assert req["json"]["model_name"] == "my-name"
        assert req["json"]["model_type"] == "vlm"
        assert req["json"]["engine"] == "optimum"

    def test_no_device_raises(self):
        with pytest.raises(ValueError, match="GPU device"):
            OpenArcEngine().build(model_path="/m", params={}, port=8000, device=None)

    def test_device_without_render_node_raises(self):
        with pytest.raises(ValueError, match="render_node"):
            OpenArcEngine().build(model_path="/m", params={}, port=8000, device=DEVICE_NO_RENDER_NODE)
