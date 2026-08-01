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
    external = "external"  # already running elsewhere - no build/start/stop lifecycle at all


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
    endpoint: str | None = Field(
        default=None,
        description="Full base URL of an already-running backend (e.g. through llm-proxy or "
        "directly). Only used when mode='external' - no container is built, started, or "
        "stopped; benchmark/coherence adapters are pointed at this endpoint as-is.",
    )

    @model_validator(mode="after")
    def _check_mode_fields(self) -> "BackendSource":
        if self.mode is SourceMode.build and self.build is None:
            raise ValueError("source.build is required when mode='build'")
        if self.mode is SourceMode.prebuilt and not self.image:
            raise ValueError("source.image is required when mode='prebuilt'")
        if self.mode is SourceMode.external and not self.endpoint:
            raise ValueError("source.endpoint is required when mode='external'")
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
    engine: str | None = Field(
        default=None,
        description="Name of a registered EngineTranslator (e.g. 'llama-cpp-sycl', 'qxmx') "
        "that generates command/env/devices from params.shared + model_path + the resolved "
        "GPU device. Any of command/env/devices set explicitly below still wins over what "
        "the translator generates for that field - this is additive convenience, not a "
        "replacement for the raw passthrough escape hatch.",
    )
    model_path: str | None = Field(
        default=None,
        description="In-container path to the model file (e.g. '/models/foo.gguf'), used by "
        "an EngineTranslator. `model` above remains a human-readable label for records/"
        "storage; this is the actual path a translator needs to build a command.",
    )
    command: list[str] = Field(
        default_factory=list,
        description="Raw CLI args appended after (or replacing, if the image has no "
        "ENTRYPOINT - check `docker inspect --format '{{.Config.Entrypoint}}'` first, see "
        "VALIDATION.md) the image's default command. If `engine` is set, this is only needed "
        "to override what the translator would otherwise generate.",
    )
    volumes: dict[str, str] = Field(
        default_factory=dict,
        description="host_path -> container_path, always mounted read-only. "
        "No rw/tmpfs support yet - add if/when an adapter needs to write into "
        "its own container (unlikely for inference-serving backends).",
    )
    health_path: str = "/health"
    startup_timeout_s: float = 120
    devices: list[str] = Field(
        default_factory=list,
        description="Host device nodes to pass through, docker CLI '--device' syntax "
        "('/dev/dri:/dev/dri:rwm'). Required for GPU access on every backend tested so "
        "far (Intel /dev/dri render nodes) - there is no vendor-aware auto-detection of "
        "which device nodes a GPU needs, this is raw passthrough, deployer's responsibility.",
    )
    post_start_requests: list[dict[str, Any]] = Field(
        default_factory=list,
        description="HTTP requests ({'method','path','json'}) fired against the running "
        "backend after health check passes, before benchmark/coherence adapters run. For "
        "engines where 'container started' and 'model loaded' are separate steps (e.g. "
        "OpenArc: POST /openarc/load after the server is already up). Non-2xx aborts the run.",
    )
    device_note: str | None = Field(
        default=None,
        description="Free-text description of which GPU this backend is known to run on "
        "(e.g. 'GPU1, B70'). Purely informational, stored as-is in RunResult - for "
        "source.mode='external' backends there is no container of ours to probe, so this is "
        "the only device identity captured for them; NEVER treated as verified the way a "
        "probed DeviceInfo is (see RunResult.device_target's 'verified' flag).",
    )


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
