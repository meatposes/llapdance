"""EngineTranslator for llama.cpp's SYCL server (validated against
llama-cpp-bonsai:meat6-hardened - see VALIDATION.md).

Normalized params.shared this translator understands (all optional except
context_size, which defaults to the value used in the validated run):
  - context_size (int): -> `-c`. Default 4096.
  - batch_size (int): -> `-b` and `-ub`, set identically (matches the
    validated command, which used the same value for both).
  - kv_cache_quant (str, one of "f16"/"q8_0"/"f8"): -> `--cache-type-k`/
    `--cache-type-v`. "f8" is NOT supported by llama.cpp - raises rather
    than silently passing a bad value. Omitted entirely if not set (the
    validated run never set this and relied on llama.cpp's own default).
  - parallel_slots (int): -> `--parallel`. Omitted if not set (the
    validated run relied on llama.cpp's "n_parallel is set to auto"
    default, which picked 4).
  - reasoning (str, one of "on"/"off"/"auto"): -> `LLAMA_ARG_REASONING` env
    var. GOTCHA, found the hard way (see VALIDATION.md): this is a real,
    general upstream llama.cpp flag, NOT something specific to the
    "-hardened" fork - an earlier version of this file guessed it was
    fork-specific fluff and omitted it, which silently broke every
    coherence-check answer (model's chat template defaults to
    reasoning='auto', which turned "on" for this model, so the entire
    token budget went into a hidden `reasoning_content` field and
    `message.content` came back empty). llama.cpp's own default is "auto"
    - this translator does NOT override that default when unset, so a
    suite testing a reasoning-capable model MUST set reasoning: "off"
    explicitly if it wants direct-answer coherence checking.

GPU pinning: confirmed on real hardware (VALIDATION.md "params translation
layer" section) that passing through a SINGLE render node is sufficient on
its own - llama.cpp only ever sees one device and uses it as SYCL0, no
ONEAPI_DEVICE_SELECTOR / GGML_SYCL_VISIBLE_DEVICES env needed. This
translator deliberately does NOT set those env vars, because this session
could not verify a reliable mapping between llama.cpp's SYCL/level-zero
device index and any other tool's device numbering (see core/probe.py's
docstring) - render-node-scoped passthrough sidesteps needing that mapping
at all.

"""
from __future__ import annotations

from typing import Any

from llapdance.core.probe import DeviceInfo
from llapdance.plugins.base import EngineInvocation, EngineTranslator
from llapdance.plugins.registry import register

_KV_QUANT_MAP = {"f16": "f16", "q8_0": "q8_0"}  # "f8" deliberately absent - see module docstring
_REASONING_VALUES = {"on", "off", "auto"}


class LlamaCppSyclEngine(EngineTranslator):
    name = "llama-cpp-sycl"

    # Real tags confirmed to be this same SYCL backend build (direct user
    # question: are llama-cpp-sycl/llama-cpp-intel/llama-cpp-bonsai
    # secretly different things? Checked for real, not assumed - `docker
    # run --entrypoint sh <image> -c "ls /app | grep ggml"` against all
    # three shows `libggml-sycl.so` present in every one of them, and
    # identical baked-in oneAPI env (CCL_CONFIGURATION, ONEAPI_ROOT, etc.)
    # via `docker inspect`. Same backend, just different build tags/dates -
    # `llama-cpp-vulkan:*` is the genuinely different one, see
    # llama_cpp_vulkan.py: it links libggml-vulkan.so instead, no oneAPI
    # env at all, needs its own translator).
    image_hints = ["llama-cpp-bonsai:*", "llama-cpp-sycl:*", "llama-cpp-intel:*"]

    sweepable_params = {
        "context_size": {"type": "int", "default": 4096, "maps_to": "-c"},
        "batch_size": {"type": "int", "maps_to": "-b / -ub (same value both)"},
        "kv_cache_quant": {"type": "str", "values": ["f16", "q8_0"], "maps_to": "--cache-type-k / --cache-type-v"},
        # "default": 4 here is llama.cpp's own real "auto" behavior when
        # unset (confirmed in the validated run - see module docstring),
        # not this translator's own default - --parallel is omitted
        # entirely unless a suite sets this explicitly.
        "parallel_slots": {"type": "int", "default": 4, "maps_to": "--parallel"},
        "reasoning": {"type": "str", "values": ["on", "off", "auto"], "default": "auto", "maps_to": "LLAMA_ARG_REASONING env"},
    }

    # Raw GGML/SYCL runtime env flags, NOT read by this translator - swept
    # directly via `env.<NAME>` on the backend config (validated live: a
    # real 2-value sweep of GGML_OP_OFFLOAD_MIN_BATCH landed correctly
    # inside a real running container, confirmed via `docker exec`).
    #
    # Direct user feedback: the sweep options "do not actually contain some
    # of the flags that can tune performance" - correct, this catalog was
    # previously built from a naive `grep getenv(` (3 flags), which MISSED
    # every flag read through ggml-sycl's own `ggml_sycl_get_env()` helper
    # (common.cpp) instead of a bare `getenv()` call. Re-derived from this
    # project's own local checkout (~/llama.cpp.git/llama.cpp,
    # ggml/src/ggml-sycl/{ggml-sycl.cpp,fattn.cpp,fattn-mkl.cpp}) - every
    # `ggml_sycl_get_env(...)` AND bare `getenv(...)` call site, not
    # guessed. This is what "turn graph on/off for intel" (the user's own
    # example) turned out to be: `GGML_SYCL_ENABLE_GRAPH`, below.
    known_env_flags = {
        "GGML_SYCL_NO_PINNED": {
            "type": "presence (any non-empty value disables pinned host memory)",
            "source": "ggml/src/ggml-sycl/common.cpp, ggml-sycl.cpp: getenv(...) != nullptr",
        },
        "GGML_OP_OFFLOAD_MIN_BATCH": {
            "type": "int", "default": 32,
            "source": "ggml/src/ggml-sycl/ggml-sycl.cpp: atoi(getenv(...))",
        },
        "GGML_SYCL_VISIBLE_DEVICES": {
            "type": "str (device index)",
            "note": "NOT set by this translator (render-node passthrough is used for GPU pinning "
            "instead, see module docstring) - listed here only because it's a real flag this "
            "backend's binary reads, in case a suite needs to set it directly via raw env passthrough.",
        },
        "GGML_SYCL_ENABLE_GRAPH": {
            "type": "int (0/1)", "default": 0,
            "note": "direct user ask ('turn graph on/off for intel') - this is it: enables SYCL "
            "command-graph capture/replay for the compute graph. GOTCHA: gated behind a BUILD-TIME "
            "cmake option (GGML_SYCL_GRAPH, ggml/src/ggml-sycl/CMakeLists.txt) - if the image wasn't "
            "compiled with it, this env var is a silent no-op (the binary logs 'graph disabled by "
            "compile flag' at startup). NOT confirmed which locally-built images (if any) were "
            "compiled with GGML_SYCL_GRAPH - check the container's own startup log before trusting a "
            "sweep of this to do anything.",
            "source": "ggml/src/ggml-sycl/ggml-sycl.cpp:290,332-377",
        },
        "GGML_SYCL_ENABLE_OPT": {"type": "int (0/1)", "default": 1, "note": "master toggle for ggml-sycl's graph-level op optimization pass"},
        "GGML_SYCL_ENABLE_DNN": {"type": "int (0/1)", "default": 1, "note": "master toggle for oneDNN kernel usage - a real, previously-confirmed-impactful lever (see Arcaine's DIFF_ONEDNN_SDPA A/B, same underlying library)"},
        "GGML_SYCL_FA_ONEDNN": {"type": "int (0/1)", "default": 1, "note": "flash-attention specifically via the fused-XMX oneDNN SDPA path (fattn-onednn.cpp) vs the non-oneDNN flash-attn path"},
        "GGML_SYCL_FA_ONEDNN_MAX_KV": {"type": "int", "default": 0, "note": "KV-length cap above which the oneDNN flash-attn path is skipped (0 = no cap)"},
        "GGML_SYCL_ENABLE_MKL_FA": {"type": "int (0/1)", "default": 1, "note": "flash-attention via the oneMKL GEMM path toggle"},
        "GGML_SYCL_MKL_FA_Q_TILE": {"type": "int", "default": 8192, "note": "query-tile size for the oneMKL flash-attn path"},
        "GGML_SYCL_MKL_FA_DEBUG": {"type": "int (0/1)", "default": 0, "note": "debug output, not a perf switch"},
        "GGML_SYCL_MKL_FA_DIAG": {"type": "int (0/1)", "default": 0, "note": "debug diagnostics, not a perf switch"},
        "GGML_SYCL_ENABLE_VMM": {"type": "int (0/1)", "default": 1, "note": "SYCL virtual-memory-management allocator path toggle"},
        "GGML_SYCL_ENABLE_FUSION": {"type": "int (0/1)", "default": 1, "note": "op-fusion optimization toggle (see fusion.hpp)"},
        "GGML_SYCL_PRIORITIZE_DMMV": {"type": "int (0/1)", "default": 0, "note": "prioritizes the dequantize-matmul-vec kernel path over alternatives"},
        "GGML_SYCL_DEV2DEV_MEMCPY": {"type": "int (0=SYCL, 1=level-zero, 2=forward)", "default": 0, "note": "device-to-device copy implementation selector (ggml_sycl_dev2dev_memcpy_mode enum)"},
        "GGML_SYCL_ENABLE_FLASH_ATTN": {"type": "int (0/1)", "default": 1, "note": "master flash-attention toggle across all SYCL FA paths"},
        "GGML_SYCL_USM_SYSTEM": {"type": "int (0/1)", "default": 0, "note": "use USM system-memory allocations instead of device allocations"},
        "GGML_SYCL_USE_ASYNC_MEM_OP": {"type": "int (0/1)", "default": 1, "note": "async USM alloc/free path toggle - independent of graph capture, useful outside it too"},
        "GGML_SYCL_USE_LEVEL_ZERO_API": {"type": "int (0/1)", "default": 1, "note": "use the level-zero backend API directly vs the generic SYCL API, when available"},
        "GGML_SYCL_DEBUG": {"type": "int (0/1)", "default": 0, "note": "debug logging, not a perf switch"},
    }
    # GGML_SYCL_DNNL (whether oneDNN kernels are linked in at all, e.g. for
    # flash-attention) is a BUILD-TIME cmake option (`-DGGML_SYCL_DNNL=0|1`,
    # ggml/src/ggml-sycl/CMakeLists.txt), not a runtime env var - sweeping
    # it means sweeping `source.build.build_args.GGML_SYCL_DNNL` with
    # `source.mode: build`, which triggers a real rebuild per value rather
    # than just a container restart. Structurally supported by the same
    # generic dotted-path sweep mechanism (build_args is just another dict
    # on the backend config) - NOT validated live this session (a from-
    # source oneDNN build is a real, slow rebuild, not a quick check) - do
    # that validation before relying on it for something that matters.

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
                raise ValueError(f"llama-cpp-sycl reasoning must be one of {sorted(_REASONING_VALUES)}, got {reasoning!r}")
            env["LLAMA_ARG_REASONING"] = reasoning

        if batch_size is not None:
            command += ["-b", str(batch_size), "-ub", str(batch_size)]

        if kv_cache_quant is not None:
            if kv_cache_quant not in _KV_QUANT_MAP:
                raise ValueError(
                    f"llama-cpp-sycl does not support kv_cache_quant={kv_cache_quant!r} "
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


register("engine", LlamaCppSyclEngine.name, LlamaCppSyclEngine)
