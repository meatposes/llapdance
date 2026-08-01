import pytest
from pydantic import ValidationError

from llapdance.config.loader import _deep_merge, parse_kv_overrides
from llapdance.config.models import (
    BackendConfig,
    BackendSource,
    DeviceTarget,
    ExecutionTargetConfig,
    NetworkConfig,
    SourceMode,
)


def test_backend_source_requires_build_for_build_mode():
    with pytest.raises(ValidationError):
        BackendSource(mode=SourceMode.build)


def test_backend_source_requires_image_for_prebuilt_mode():
    with pytest.raises(ValidationError):
        BackendSource(mode=SourceMode.prebuilt)


def test_backend_source_prebuilt_ok():
    src = BackendSource(mode=SourceMode.prebuilt, image="foo:latest")
    assert src.image == "foo:latest"


def test_network_enabled_requires_name():
    with pytest.raises(ValidationError):
        NetworkConfig(mode="enabled")
    cfg = NetworkConfig(mode="enabled", network="ai-net")
    assert cfg.network == "ai-net"


def test_device_target_indices_mode_requires_indices():
    with pytest.raises(ValidationError):
        DeviceTarget(mode="indices")
    dt = DeviceTarget(mode="indices", indices=[0, 1])
    assert dt.indices == [0, 1]


def test_execution_target_ssh_requires_fields():
    with pytest.raises(ValidationError):
        ExecutionTargetConfig(mode="ssh")
    cfg = ExecutionTargetConfig(mode="ssh", host="h", user="u", ssh_key_path="/k")
    assert cfg.host == "h"


def test_backend_config_no_hardcoded_defaults():
    cfg = BackendConfig(
        name="test",
        source=BackendSource(mode=SourceMode.prebuilt, image="x:y"),
        model="m",
    )
    # port has a documented default, but network/build path must never be
    # silently filled in with a real filesystem path or account-specific value
    assert cfg.network.mode.value == "disabled"


def test_deep_merge_overrides_nested():
    base = {"a": {"b": 1, "c": 2}}
    override = {"a": {"b": 9}}
    assert _deep_merge(base, override) == {"a": {"b": 9, "c": 2}}


def test_deep_merge_indexes_into_lists():
    base = {"backends": [{"name": "a", "model": "m1"}, {"name": "b", "model": "m2"}]}
    override = {"backends": {"0": {"model": "m1-override"}}}
    merged = _deep_merge(base, override)
    assert merged["backends"][0]["model"] == "m1-override"
    assert merged["backends"][0]["name"] == "a"
    assert merged["backends"][1]["model"] == "m2"


def test_parse_kv_overrides():
    result = parse_kv_overrides(["backends.0.model=foo", "min_free_vram_mb=4096"])
    assert result == {"backends": {"0": {"model": "foo"}}, "min_free_vram_mb": 4096}
