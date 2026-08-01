"""EngineTranslator for vLLM (real images: `intel/vllm:latest`,
`intel/llm-scaler-vllm`, `urakozz/vllm-xpu-env` - the last is an already-running
production container, `vllm-urak`, serving a real model - see VALIDATION.md
"vLLM engine translator" section and the fork-provenance note below).

CLI/env shape confirmed by inspecting the real running `vllm-urak`
container (`docker inspect`), not guessed from vLLM's public docs alone -
this harness targets Intel XPU builds specifically, which carry extra
XPU-only env vars on top of upstream vLLM's normal CLI.

ENTRYPOINT is `vllm serve` (confirmed via `docker inspect intel/vllm:latest`)
- CMD args are `<model_path> --served-model-name <name> --host 0.0.0.0
--port <port> [tuning flags...]`, i.e. this translator only ever emits
`command`, never `env` for the params below (mirrors llama-cpp-sycl/qxmx's
"CMD, not baked-in" style, unlike Arcaine's all-env style).

Normalized params this translator understands (all confirmed present on
the real running container's actual CMD args, not invented):
  - context_size (int): -> `--max-model-len`.
  - tensor_parallel_size (int): -> `--tensor-parallel-size`. Default 1.
  - kv_cache_dtype (str): -> `--kv-cache-dtype` (e.g. "fp8", "auto").
  - max_num_seqs (int): -> `--max-num-seqs`.
  - max_num_batched_tokens (int): -> `--max-num-batched-tokens`.
  - block_size (int): -> `--block-size`.
  - quantization (str): -> `--quantization` (e.g. "awq", "compressed-tensors") -
    NOT observed on the real container (that run relies on the checkpoint's
    own `quantization_config`, auto-detected), but this is standard upstream
    vLLM CLI - included since raw HF checkpoints needing an explicit
    override are a real, not hypothetical, case.
  - reasoning_parser (str): -> `--reasoning-parser` (e.g. "qwen3").
  - tool_call_parser (str): -> `--tool-call-parser` (e.g. "qwen3_xml").
  - enable_auto_tool_choice (bool): -> `--enable-auto-tool-choice` (presence flag).
  - trust_remote_code (bool): -> `--trust-remote-code` (presence flag) - needed
    for checkpoints shipping custom modeling_*.py (e.g. DeepSeek-V2-family).
  - language_model_only (bool): -> `--language-model-only` (presence flag) -
    real flag seen on the running container, skips loading a multimodal
    checkpoint's vision tower.

Health check: real vLLM (Intel XPU build) serves a standard `/health`
(confirmed 200 via `docker exec vllm-urak curl .../health`) and
`/v1/models` - suites can use the default `health_path` unmodified.

GPU: unlike every other engine here, a single render node is NOT enough -
confirmed the hard way, a live validation attempt passing just the
render node crashed with `oneCCL: ze_fd_manager.cpp:144 init_device_fds:
EXCEPTION: opendir failed: could not open device directory` (oneCCL
enumerates `/dev/dri` as a directory even at `tensor_parallel_size=1`, and
a lone device node leaves no directory to open). This translator passes
through the WHOLE `/dev/dri` directory instead, matching the real running
`vllm-urak` container's actual `HostConfig.Devices`.

The real container ALSO sets `ONEAPI_DEVICE_SELECTOR=level_zero:0` to pin
the specific GPU within that shared `/dev/dri`. The level_zero index is
NOT confirmed to correspond to any existing DeviceInfo field (xpumcli/
clinfo/render-node/OpenVINO GPU.N/SYCL/xmxmon are SIX already-confirmed
non-corresponding numbering schemes in this project - see VALIDATION.md -
a level_zero-index mapping would be a SEVENTH, unverified). This
translator therefore does NOT auto-derive `ONEAPI_DEVICE_SELECTOR` from
`device` - with `/dev/dri` passed through whole, single-GPU boxes need
nothing further, but a multi-GPU host needs `env.ONEAPI_DEVICE_SELECTOR`
set explicitly once a suite author has independently confirmed the right
index, rather than this translator guessing at a mapping.

FORK PROVENANCE NOTE (see VALIDATION.md): `urakozz/vllm-xpu-env` is a fork
of vLLM's XPU support (real, currently running in production), believed to
be unpublished/deleted from wherever it lived - meaning its exact source
diff vs upstream `intel/vllm` is NOT currently reproducible from this
image alone. This translator only depends on the CLI/env CONTRACT (which
is the same across all three vLLM images inspected), not on the fork's
internals, so it should work unmodified against `intel/vllm` too - but the
running container itself is a real, currently-irreplaceable artifact if
its source truly is gone. Flagged here rather than silently treated as
equivalent to a reproducible build.
"""
from __future__ import annotations

from typing import Any

from llapdance.core.probe import DeviceInfo
from llapdance.plugins.base import EngineInvocation, EngineTranslator
from llapdance.plugins.registry import register


class VLLMEngine(EngineTranslator):
    name = "vllm"

    sweepable_params = {
        "context_size": {"type": "int", "maps_to": "--max-model-len"},
        "tensor_parallel_size": {"type": "int", "default": 1, "maps_to": "--tensor-parallel-size"},
        "kv_cache_dtype": {"type": "str", "maps_to": "--kv-cache-dtype", "note": "e.g. 'fp8', 'auto'"},
        "max_num_seqs": {"type": "int", "maps_to": "--max-num-seqs"},
        "max_num_batched_tokens": {"type": "int", "maps_to": "--max-num-batched-tokens"},
        "block_size": {"type": "int", "maps_to": "--block-size"},
        "quantization": {"type": "str", "maps_to": "--quantization", "note": "e.g. 'awq', 'compressed-tensors' - usually unnecessary, auto-detected from the checkpoint's own quantization_config"},
        "reasoning_parser": {"type": "str", "maps_to": "--reasoning-parser", "note": "e.g. 'qwen3'"},
        "tool_call_parser": {"type": "str", "maps_to": "--tool-call-parser", "note": "e.g. 'qwen3_xml'"},
        "enable_auto_tool_choice": {"type": "bool", "maps_to": "--enable-auto-tool-choice (presence flag)"},
        "trust_remote_code": {"type": "bool", "maps_to": "--trust-remote-code (presence flag)", "note": "needed for checkpoints shipping custom modeling_*.py, e.g. DeepSeek-V2-family"},
        "language_model_only": {"type": "bool", "maps_to": "--language-model-only (presence flag)", "note": "skip loading a multimodal checkpoint's vision tower"},
    }

    # Real env vars seen on the actual running vllm-urak container
    # (`docker inspect`), not guessed from generic vLLM docs - the XPU
    # build carries these on top of upstream vLLM's normal env surface.
    known_env_flags = {
        "VLLM_TARGET_DEVICE": {"type": "str", "note": "build-time-flavored but observed set at runtime too on the real container; value seen: 'xpu'"},
        "ONEAPI_DEVICE_SELECTOR": {
            "type": "str",
            "note": "real device-pinning mechanism (value seen: 'level_zero:0') - NOT auto-derived from "
            "the resolved DeviceInfo here, see module docstring: mapping level_zero index to any other "
            "numbering scheme in this project is unconfirmed, a suite author must supply it directly if needed.",
        },
        "VLLM_WORKER_MULTIPROC_METHOD": {"type": "str", "note": "value seen: 'spawn'"},
        "VLLM_XPU_ENABLE_XPU_GRAPH": {"type": "bool", "note": "value seen: '1'"},
        "HF_HUB_OFFLINE": {"type": "bool", "note": "value seen: '1' - set when the model is a local mount, no HF Hub access needed/wanted"},
        "HF_HUB_ENABLE_HF_TRANSFER": {"type": "bool", "note": "value seen: '0' on the real container"},
    }

    def build(self, model_path: str, params: dict[str, Any], port: int, device: DeviceInfo | None) -> EngineInvocation:
        command: list[str] = [
            model_path,
            "--host", "0.0.0.0",
            "--port", str(port),
        ]
        if "served_model_name" in params:
            command += ["--served-model-name", str(params["served_model_name"])]
        if "context_size" in params:
            command += ["--max-model-len", str(params["context_size"])]
        command += ["--tensor-parallel-size", str(params.get("tensor_parallel_size", 1))]
        if "kv_cache_dtype" in params:
            command += ["--kv-cache-dtype", str(params["kv_cache_dtype"])]
        if "max_num_seqs" in params:
            command += ["--max-num-seqs", str(params["max_num_seqs"])]
        if "max_num_batched_tokens" in params:
            command += ["--max-num-batched-tokens", str(params["max_num_batched_tokens"])]
        if "block_size" in params:
            command += ["--block-size", str(params["block_size"])]
        if "quantization" in params:
            command += ["--quantization", str(params["quantization"])]
        if "reasoning_parser" in params:
            command += ["--reasoning-parser", str(params["reasoning_parser"])]
        if "tool_call_parser" in params:
            command += ["--tool-call-parser", str(params["tool_call_parser"])]
        if params.get("enable_auto_tool_choice"):
            command.append("--enable-auto-tool-choice")
        if params.get("trust_remote_code"):
            command.append("--trust-remote-code")
        if params.get("language_model_only"):
            command.append("--language-model-only")

        devices: list[str] = []
        if device is not None:
            if device.render_node is None:
                raise ValueError(
                    f"device {device.index} ({device.name}) has no known render_node - "
                    "discover devices via xpumcli (populates render_node), not the clinfo fallback."
                )
            # REAL FINDING (see VALIDATION.md "vLLM engine translator"
            # section): unlike every other engine here, a single render
            # node is NOT enough. First live validation attempt passed
            # just the render node (the pattern every other engine uses)
            # and crashed: `oneCCL: ze_fd_manager.cpp:144 init_device_fds:
            # EXCEPTION: opendir failed: could not open device directory`
            # - oneCCL enumerates /dev/dri as a directory even at
            # tensor_parallel_size=1, and a single passed-through device
            # node leaves no /dev/dri directory to opendir() at all.
            # Confirmed against the real running vllm-urak container's
            # actual HostConfig.Devices, which passes through the WHOLE
            # /dev/dri directory, not one render node - matched here.
            devices = ["/dev/dri:/dev/dri"]

        return EngineInvocation(command=command, env={}, devices=devices)


register("engine", VLLMEngine.name, VLLMEngine)
