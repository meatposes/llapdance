import pytest

from llapdance.core.probe import DeviceInfo
from llapdance.plugins.engine.llama_cpp_sycl import LlamaCppSyclEngine
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
