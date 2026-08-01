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
