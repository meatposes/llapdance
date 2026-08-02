"""EngineTranslator for llama.cpp's Vulkan server.

Direct user question surfaced this gap: is `llama-cpp-vulkan` secretly the
same thing as `llama-cpp-sycl`/`llama-cpp-intel`/`llama-cpp-bonsai`? Checked
for real, not assumed - `docker run --entrypoint sh <image> -c "ls /app |
grep ggml"` shows `libggml-vulkan.so` in the Vulkan-tagged images and
`libggml-sycl.so` in all three SYCL-tagged ones, never both in the same
image. Genuinely a different GGML backend, confirmed via `docker inspect`
too: no oneAPI env baked in at all (SYCL images all carry
CCL_CONFIGURATION/ONEAPI_ROOT/etc.). Before this file existed there was NO
EngineTranslator at all for these images - `llama-cpp-sycl` was the only
registered llama.cpp engine, so a Vulkan image had no translator-driven
path through the TUI/CLI (raw command/env passthrough on BackendConfig
remained available, just not this convenience layer).

Same llama.cpp server binary/CLI as the SYCL build (`llama-server`, backed
by ggml's shared, backend-agnostic frontend) - the params.shared mapping
below is deliberately identical to llama_cpp_sycl.py's for exactly that
reason. What differs is the GGML runtime env surface (Vulkan's own
`GGML_VK_*` flags, confirmed by reading this project's real local Vulkan
checkout, `~/llama.cpp.git/llama.cpp.prism/ggml/src/ggml-vulkan/
ggml-vulkan.cpp`'s own `getenv()` call sites - NOT the SYCL flags, which
don't exist in a Vulkan build at all) and, unvalidated so far, whether
Vulkan's device enumeration behaves identically to SYCL's under a single
passed-through render node (see GPU pinning note below).

GOTCHA found investigating this (real, not hypothetical): of the two
`llama-cpp-vulkan` tags found locally, only one is actually servable this
way. `llama-cpp-vulkan:prism-bonsai` sets `ENTRYPOINT ["/app/llama-server"]`
(confirmed via `docker inspect`) - `command` below becomes its CMD args,
exactly like every other engine here. `llama-cpp-vulkan:newmeat2` sets
`ENTRYPOINT ["/app/tools.sh"]` instead (still contains a working
llama-server binary + libggml-vulkan.so internally, but this harness's
`local_docker` execution adapter has no per-backend entrypoint override -
see llapdance/plugins/execution/local_docker.py's `start()` - so `command`
would be appended to `tools.sh`, not run llama-server directly). Deliberately
NOT included in `image_hints` below for this reason; **any newly built
`llama-cpp-vulkan` tag should have its `ENTRYPOINT` checked with `docker
inspect --format '{{.Config.Entrypoint}}'` before assuming it fits this
translator** - same breadcrumb as the qxmx/llama-cpp-bonsai CMD-vs-ENTRYPOINT
gotcha documented elsewhere in this project.

GPU pinning: NOT validated live against real Vulkan hardware this session
(unlike the SYCL translator, which was). Passing a single render node
through is assumed sufficient by analogy to the SYCL translator's confirmed
behavior, but Vulkan's device enumeration is a genuinely different code
path (see `ggml-vulkan.cpp`'s `GGML_VK_VISIBLE_DEVICES` handling) - this
assumption needs a real multi-GPU-host check before being trusted the way
the SYCL translator's is. `GGML_VK_VISIBLE_DEVICES` is listed in
known_env_flags as the raw fallback selector if render-node scoping proves
insufficient in practice.
"""
from __future__ import annotations

from typing import Any

from llapdance.core.probe import DeviceInfo
from llapdance.plugins.base import EngineInvocation, EngineTranslator
from llapdance.plugins.registry import register

_KV_QUANT_MAP = {"f16": "f16", "q8_0": "q8_0"}  # "f8" deliberately absent - same as llama-cpp-sycl, llama.cpp-wide limit
_REASONING_VALUES = {"on", "off", "auto"}


class LlamaCppVulkanEngine(EngineTranslator):
    name = "llama-cpp-vulkan"

    # Deliberately excludes llama-cpp-vulkan:newmeat2 - see module
    # docstring's ENTRYPOINT gotcha. Only tags confirmed (via `docker
    # inspect`) to set ENTRYPOINT ["/app/llama-server"] belong here.
    image_hints = ["llama-cpp-vulkan:prism*"]

    # Identical to llama-cpp-sycl's - same llama.cpp server CLI, this is a
    # backend-agnostic part of llama.cpp itself, not SYCL-specific.
    sweepable_params = {
        "context_size": {"type": "int", "default": 4096, "maps_to": "-c"},
        "batch_size": {"type": "int", "maps_to": "-b / -ub (same value both)"},
        "kv_cache_quant": {"type": "str", "values": ["f16", "q8_0"], "maps_to": "--cache-type-k / --cache-type-v"},
        "parallel_slots": {"type": "int", "maps_to": "--parallel"},
        "reasoning": {"type": "str", "values": ["on", "off", "auto"], "default": "auto", "maps_to": "LLAMA_ARG_REASONING env"},
    }

    # Real GGML_VK_* runtime env flags, found by reading this project's own
    # local Vulkan checkout's ggml-vulkan.cpp getenv() call sites (not
    # guessed, not the SYCL flag list) - swept via env.<NAME>, same generic
    # mechanism as every other engine's known_env_flags.
    known_env_flags = {
        "GGML_VK_VISIBLE_DEVICES": {
            "type": "str (comma-separated device indices)",
            "note": "Vulkan's analogue of CUDA_VISIBLE_DEVICES. NOT set by this translator (render-"
            "node passthrough is used instead, unvalidated for Vulkan - see module docstring) - "
            "listed here as the raw fallback selector if that assumption doesn't hold up.",
        },
        "GGML_VK_DISABLE_F16": {"type": "presence", "note": "forces fp32 fallback, disables fp16 arithmetic path"},
        "GGML_VK_FORCE_MAX_ALLOCATION_SIZE": {"type": "int (bytes)", "note": "override device max single-allocation size"},
        "GGML_VK_FORCE_MAX_BUFFER_SIZE": {"type": "int (bytes)", "note": "override device max buffer size"},
        "GGML_VK_SUBALLOCATION_BLOCK_SIZE": {"type": "int (bytes)", "note": "suballocator block size tuning"},
        "GGML_VK_DISABLE_COOPMAT": {"type": "presence", "note": "disables cooperative-matrix (coopmat) matmul path"},
        "GGML_VK_DISABLE_COOPMAT2": {"type": "presence", "note": "disables coopmat2 matmul path"},
        "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT": {"type": "presence", "note": "disables integer dot-product extension usage"},
        "GGML_VK_DISABLE_BFLOAT16": {"type": "presence", "note": "disables bfloat16 arithmetic path"},
        "GGML_VK_ALLOW_SYSMEM_FALLBACK": {"type": "presence", "note": "allows falling back to system memory when device memory is exhausted"},
        "GGML_VK_PREFER_HOST_MEMORY": {"type": "presence", "note": "prefers host-visible memory over device-local"},
        "GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM": {"type": "presence", "note": "disables host-visible VRAM usage"},
        "GGML_VK_DISABLE_ASYNC": {"type": "presence", "note": "disables async transfer queue usage"},
        "GGML_VK_ALLOW_GRAPHICS_QUEUE": {"type": "presence", "note": "allows using a graphics-capable queue for compute if no dedicated compute queue exists"},
        "GGML_VK_DISABLE_FUSION": {"type": "presence", "note": "disables ggml op-fusion optimizations"},
        "GGML_VK_DISABLE_MMVQ": {"type": "presence", "note": "disables mat-vec-quant kernel path"},
        "GGML_VK_FORCE_MMVQ": {"type": "presence", "note": "forces mat-vec-quant kernel path on"},
        "GGML_VK_DISABLE_GRAPH_OPTIMIZE": {"type": "presence", "note": "disables ggml graph-level optimization pass"},
        "GGML_VK_ENABLE_MEMORY_PRIORITY": {"type": "presence", "note": "enables VK_EXT_memory_priority hinting"},
        "GGML_VK_PERF_LOGGER": {"type": "presence", "note": "debug/profiling output, not a perf switch"},
        "GGML_VK_DEBUG_MARKERS": {"type": "presence", "note": "debug: enables Vulkan debug-utils markers"},
    }

    def build(self, model_path: str, params: dict[str, Any], port: int, device: DeviceInfo | None) -> EngineInvocation:
        context_size = params.get("context_size", 4096)
        batch_size = params.get("batch_size")
        kv_cache_quant = params.get("kv_cache_quant")
        parallel_slots = params.get("parallel_slots")
        reasoning = params.get("reasoning")

        command = ["-m", model_path, "-c", str(context_size), "--port", str(port)]
        env = {"LLAMA_ARG_HOST": "0.0.0.0"}

        if reasoning is not None:
            if reasoning not in _REASONING_VALUES:
                raise ValueError(f"llama-cpp-vulkan reasoning must be one of {sorted(_REASONING_VALUES)}, got {reasoning!r}")
            env["LLAMA_ARG_REASONING"] = reasoning

        if batch_size is not None:
            command += ["-b", str(batch_size), "-ub", str(batch_size)]

        if kv_cache_quant is not None:
            if kv_cache_quant not in _KV_QUANT_MAP:
                raise ValueError(
                    f"llama-cpp-vulkan does not support kv_cache_quant={kv_cache_quant!r} "
                    f"(supported: {sorted(_KV_QUANT_MAP)}); llama.cpp has no fp8 KV cache type"
                )
            mapped = _KV_QUANT_MAP[kv_cache_quant]
            command += ["--cache-type-k", mapped, "--cache-type-v", mapped]

        if parallel_slots is not None:
            command += ["--parallel", str(parallel_slots)]

        devices: list[str] = []
        if device is not None:
            if device.render_node is None:
                raise ValueError(
                    f"device {device.index} ({device.name}) has no known render_node - cannot "
                    "safely pin llama.cpp to it without passing through the whole /dev/dri "
                    "(which risks giving it a different card than intended). Discover devices "
                    "via xpumcli (populates render_node) rather than the clinfo fallback."
                )
            devices = [f"{device.render_node}:{device.render_node}"]
            command += ["-ngl", "99"]
        else:
            command += ["-ngl", "0"]

        return EngineInvocation(command=command, env=env, devices=devices)


register("engine", LlamaCppVulkanEngine.name, LlamaCppVulkanEngine)
