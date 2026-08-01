from llapdance.plugins.registry import describe_engine, load_builtin_adapters


def test_describe_engine_returns_declared_params_for_each_reference_engine():
    load_builtin_adapters()
    llama = describe_engine("llama-cpp-sycl")["params"]
    assert "context_size" in llama
    assert llama["context_size"]["default"] == 4096
    assert llama["reasoning"]["values"] == ["on", "off", "auto"]

    qxmx = describe_engine("qxmx")["params"]
    assert "batch_size" not in qxmx  # genuinely not applicable, see module docstring
    assert "context_size" in qxmx

    arcaine = describe_engine("arcaine")["params"]
    assert "layer_placement" in arcaine
    assert "expert_placement" in arcaine

    openarc = describe_engine("openarc")["params"]
    assert "context_size" not in openarc  # not applicable, different config surface
    assert "openarc_engine" in openarc


def test_describe_engine_returns_known_env_flags_for_llama_cpp_sycl():
    load_builtin_adapters()
    env_flags = describe_engine("llama-cpp-sycl")["env_flags"]
    # real flags confirmed by reading ggml-sycl's actual getenv() call sites,
    # not guessed at - see llapdance/plugins/engine/llama_cpp_sycl.py
    assert "GGML_SYCL_NO_PINNED" in env_flags
    assert env_flags["GGML_OP_OFFLOAD_MIN_BATCH"]["default"] == 32


def test_describe_engine_returns_known_env_flags_for_qxmx():
    load_builtin_adapters()
    env_flags = describe_engine("qxmx")["env_flags"]
    # real flags confirmed by reading qxmx's own getenv() call sites - qxmx
    # is from-scratch, no oneDNN dependency at all (see module docstring)
    assert env_flags["QXMX_CHUNK"]["default"] == 256
    assert "QXMX_GEMM_WGM8" in env_flags
    assert "QXMX_PROFILE" in env_flags  # debug flag, still cataloged for completeness


def test_describe_engine_returns_known_env_flags_for_arcaine():
    load_builtin_adapters()
    env_flags = describe_engine("arcaine")["env_flags"]
    # real flags for the validated diffusion_gemma model family, confirmed
    # by reading Arcaine's own source - DIFF_ONEDNN_SDPA is Arcaine's real
    # oneDNN toggle, a RUNTIME env var (unlike llama.cpp's GGML_SYCL_DNNL,
    # which is build-time)
    assert "DIFF_ONEDNN_SDPA" in env_flags
    assert env_flags["DIFF_PREFILL_CHUNK"]["default"] == 2048


def test_openarc_sweepable_params_includes_runtime_config():
    load_builtin_adapters()
    params = describe_engine("openarc")["params"]
    # real gap found and fixed cataloging this engine: runtime_config is
    # OpenArc's actual tuning surface (raw OpenVINO plugin properties),
    # confirmed by reading its source - the translator didn't wire it
    # through before this was caught
    assert "runtime_config" in params


def test_describe_engine_returns_empty_catalogs_for_engine_with_no_declarations():
    from llapdance.plugins.base import EngineInvocation, EngineTranslator
    from llapdance.plugins.registry import register

    class NoDeclarationsEngine(EngineTranslator):
        name = "no-declarations-engine"

        def build(self, model_path, params, port, device):
            return EngineInvocation()

    register("engine", NoDeclarationsEngine.name, NoDeclarationsEngine)
    assert describe_engine("no-declarations-engine") == {"params": {}, "env_flags": {}}
