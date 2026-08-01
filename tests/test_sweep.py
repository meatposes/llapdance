import pytest

from llapdance.config.models import BackendConfig, BackendSource, SourceMode, SweepAxis, TestSuite, StorageConfig
from llapdance.config.sweep import expand_backend_sweep, expand_suite_sweep


def _backend(**overrides) -> BackendConfig:
    base = dict(
        name="engine-a",
        source=BackendSource(mode=SourceMode.prebuilt, image="x:y"),
        model="m",
        params={"shared": {"context_size": 4096}},
    )
    base.update(overrides)
    return BackendConfig(**base)


def test_no_sweep_returns_backend_unchanged():
    backend = _backend()
    assert expand_backend_sweep(backend) == [backend]


def test_single_axis_expands_to_one_backend_per_value():
    backend = _backend(sweep=[SweepAxis(param="params.shared.context_size", values=[2048, 4096, 8192])])
    expanded = expand_backend_sweep(backend)
    assert len(expanded) == 3
    assert [b.params.shared["context_size"] for b in expanded] == [2048, 4096, 8192]
    assert [b.name for b in expanded] == [
        "engine-a--context_size_2048",
        "engine-a--context_size_4096",
        "engine-a--context_size_8192",
    ]
    # expanded configs are concrete - not themselves further sweepable
    assert all(b.sweep == [] for b in expanded)


def test_multiple_axes_produce_cartesian_product():
    backend = _backend(
        sweep=[
            SweepAxis(param="params.shared.context_size", values=[2048, 4096]),
            SweepAxis(param="params.shared.kv_cache_quant", values=["f16", "q8_0"]),
        ]
    )
    expanded = expand_backend_sweep(backend)
    assert len(expanded) == 4  # 2 x 2
    combos = {(b.params.shared["context_size"], b.params.shared["kv_cache_quant"]) for b in expanded}
    assert combos == {(2048, "f16"), (2048, "q8_0"), (4096, "f16"), (4096, "q8_0")}
    names = {b.name for b in expanded}
    assert names == {
        "engine-a--context_size_2048--kv_cache_quant_f16",
        "engine-a--context_size_2048--kv_cache_quant_q8_0",
        "engine-a--context_size_4096--kv_cache_quant_f16",
        "engine-a--context_size_4096--kv_cache_quant_q8_0",
    }


def test_invalid_param_path_raises():
    backend = _backend(sweep=[SweepAxis(param="params.shared.does_not_exist.deeper", values=[1])])
    with pytest.raises(ValueError, match="does not resolve"):
        expand_backend_sweep(backend)


def test_empty_values_rejected_at_config_validation():
    with pytest.raises(Exception, match="at least one value"):
        SweepAxis(param="params.shared.context_size", values=[])


def test_expand_suite_sweep_expands_all_backends_leaves_rest_untouched(tmp_path):
    swept = _backend(name="swept", sweep=[SweepAxis(param="params.shared.context_size", values=[1, 2])])
    plain = _backend(name="plain")
    suite = TestSuite(
        name="s",
        backends=[swept, plain],
        benchmark_adapters=[],
        storage=StorageConfig(flat_file_dir=str(tmp_path)),
    )
    expanded = expand_suite_sweep(suite)
    assert len(expanded.backends) == 3  # 2 from swept + 1 unchanged
    assert expanded.backends[0].name == "swept--context_size_1"
    assert expanded.backends[1].name == "swept--context_size_2"
    assert expanded.backends[2] is plain
    # original suite object is untouched
    assert len(suite.backends) == 2


def test_sweep_works_on_env_vars_not_just_params_shared():
    # confirmed live against a real container (see VALIDATION.md): raw
    # engine env flags (e.g. GGML_OP_OFFLOAD_MIN_BATCH) sweep via the
    # exact same generic mechanism, no special-casing per config section
    backend = _backend(env={"GGML_OP_OFFLOAD_MIN_BATCH": "32"}, sweep=[SweepAxis(param="env.GGML_OP_OFFLOAD_MIN_BATCH", values=["16", "64"])])
    expanded = expand_backend_sweep(backend)
    assert [b.env["GGML_OP_OFFLOAD_MIN_BATCH"] for b in expanded] == ["16", "64"]
    assert [b.name for b in expanded] == ["engine-a--GGML_OP_OFFLOAD_MIN_BATCH_16", "engine-a--GGML_OP_OFFLOAD_MIN_BATCH_64"]


def test_sweep_works_on_build_args_for_build_time_flags():
    # e.g. GGML_SYCL_DNNL - a cmake build-time flag, not a runtime env var
    # (see llama_cpp_sycl.py's known_env_flags docstring) - structurally
    # supported by the same mechanism since build_args is just another
    # dict on the backend config; NOT validated live (a from-source oneDNN
    # rebuild is slow), unlike the env-var case above.
    backend = BackendConfig(
        name="engine-a",
        source=BackendSource(mode=SourceMode.build, build={"repo": "https://example/x", "path": "/tmp/x"}),
        model="m",
        sweep=[SweepAxis(param="source.build.build_args.GGML_SYCL_DNNL", values=["0", "1"])],
    )
    expanded = expand_backend_sweep(backend)
    assert [b.source.build.build_args["GGML_SYCL_DNNL"] for b in expanded] == ["0", "1"]
