from llapdance.plugins.registry import describe_engine, load_builtin_adapters


def test_describe_engine_returns_declared_params_for_each_reference_engine():
    load_builtin_adapters()
    llama = describe_engine("llama-cpp-sycl")
    assert "context_size" in llama
    assert llama["context_size"]["default"] == 4096
    assert llama["reasoning"]["values"] == ["on", "off", "auto"]

    qxmx = describe_engine("qxmx")
    assert "batch_size" not in qxmx  # genuinely not applicable, see module docstring
    assert "context_size" in qxmx

    arcaine = describe_engine("arcaine")
    assert "layer_placement" in arcaine
    assert "expert_placement" in arcaine

    openarc = describe_engine("openarc")
    assert "context_size" not in openarc  # not applicable, different config surface
    assert "openarc_engine" in openarc


def test_describe_engine_returns_empty_dict_for_engine_with_no_declarations():
    from llapdance.plugins.base import EngineInvocation, EngineTranslator
    from llapdance.plugins.registry import register

    class NoDeclarationsEngine(EngineTranslator):
        name = "no-declarations-engine"

        def build(self, model_path, params, port, device):
            return EngineInvocation()

    register("engine", NoDeclarationsEngine.name, NoDeclarationsEngine)
    assert describe_engine("no-declarations-engine") == {}
