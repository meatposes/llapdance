"""Config schema for LLAPDANCE, matching SPEC.md sections 4, 6, 7, 8, 9.

Nothing here should default to a value that only makes sense on one
machine (no hardcoded paths, GPU ids, or network names) - see SPEC.md 0.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class SourceMode(str, Enum):
    build = "build"
    prebuilt = "prebuilt"


class BuildSpec(BaseModel):
    repo: str
    ref: str = "main"
    path: str = Field(
        description="Clone target directory. Caller-supplied; never defaulted to a "
        "particular user's home directory."
    )
    dockerfile: str = "Dockerfile"
    build_args: dict[str, str] = Field(default_factory=dict)


class BackendSource(BaseModel):
    mode: SourceMode
    build: BuildSpec | None = None
    image: str | None = None

    @model_validator(mode="after")
    def _check_mode_fields(self) -> "BackendSource":
        if self.mode is SourceMode.build and self.build is None:
            raise ValueError("source.build is required when mode='build'")
        if self.mode is SourceMode.prebuilt and not self.image:
            raise ValueError("source.image is required when mode='prebuilt'")
        return self


class BackendParams(BaseModel):
    """Normalized cross-backend knobs plus an open per-backend escape hatch."""

    shared: dict[str, Any] = Field(default_factory=dict)
    backend_specific: dict[str, Any] = Field(default_factory=dict)


class NetworkMode(str, Enum):
    disabled = "disabled"
    enabled = "enabled"
    isolated = "isolated"


class NetworkConfig(BaseModel):
    mode: NetworkMode = NetworkMode.disabled
    network: str | None = None

    @model_validator(mode="after")
    def _check_network_name(self) -> "NetworkConfig":
        if self.mode is NetworkMode.enabled and not self.network:
            raise ValueError("network.network is required when mode='enabled'")
        return self


class BackendConfig(BaseModel):
    name: str
    source: BackendSource
    model: str
    params: BackendParams = Field(default_factory=BackendParams)
    endpoint_contract: str = "openai-compatible"
    port: int = 8000
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    env: dict[str, str] = Field(default_factory=dict)


class DeviceTargetMode(str, Enum):
    all_discrete = "all_discrete"
    indices = "indices"
    none = "none"


class DeviceTarget(BaseModel):
    """GPU target as a test parameter - resolved against whatever the hardware
    prober discovers at run time, never a hardcoded device list (SPEC.md §7)."""

    mode: DeviceTargetMode = DeviceTargetMode.all_discrete
    indices: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_indices(self) -> "DeviceTarget":
        if self.mode is DeviceTargetMode.indices and not self.indices:
            raise ValueError("device_target.indices is required when mode='indices'")
        return self


class ExecutionTargetMode(str, Enum):
    local = "local"
    ssh = "ssh"


class ExecutionTargetConfig(BaseModel):
    mode: ExecutionTargetMode = ExecutionTargetMode.local
    host: str | None = None
    user: str | None = None
    ssh_key_path: str | None = None

    @model_validator(mode="after")
    def _check_ssh_fields(self) -> "ExecutionTargetConfig":
        if self.mode is ExecutionTargetMode.ssh:
            missing = [f for f in ("host", "user", "ssh_key_path") if not getattr(self, f)]
            if missing:
                raise ValueError(f"execution_target.ssh requires: {', '.join(missing)}")
        return self


class AdapterRef(BaseModel):
    """A named plugin plus whatever config that plugin needs."""

    adapter: str
    config: dict[str, Any] = Field(default_factory=dict)


class StorageConfig(BaseModel):
    """Flat file is the only always-on adapter (SPEC.md §8). Everything else
    is explicit opt-in, and any number of them can be active simultaneously."""

    flat_file_dir: str
    extra_adapters: list[AdapterRef] = Field(default_factory=list)


class TestSuite(BaseModel):
    name: str
    backends: list[BackendConfig]
    device_target: DeviceTarget = Field(default_factory=DeviceTarget)
    execution_target: ExecutionTargetConfig = Field(default_factory=ExecutionTargetConfig)
    benchmark_adapters: list[AdapterRef]
    coherence_adapters: list[AdapterRef] = Field(default_factory=list)
    storage: StorageConfig
    min_free_vram_mb: float = 2048
    allow_unknown_vram: bool = False
