"""EngineTranslator for Arcaine (~/Arcaine, remotes: origin=meatposes/Arcaine,
upstream=SearchSavior/Arcaine). Validated against the local `arcaine-server:latest`
image - see VALIDATION.md.

Unlike llama.cpp/qxmx, Arcaine's server-entrypoint image is ENTIRELY env-var
driven (no CLI args at all) - `server-entrypoint.sh` builds its own argv from
env vars. This translator therefore only ever emits `env`, never `command`.

Normalized params.shared this translator understands:
  - context_size (int): -> MAX_SEQ. Default 4096.
  - max_tokens (int): -> DEFAULT_MAX_TOKENS. Default 2048 (Arcaine's own default).

Arcaine-specific params (genuinely not cross-backend concepts, read from the
same merged params dict, see orchestrator._apply_engine_translator):
  - denoising_steps (int, diffusion-model-specific): -> DENOISING_STEPS
  - seed (int): -> DEFAULT_SEED
  - layer_placement / expert_placement (str, MoE/multi-GPU expert-sharding
    config): -> LAYER_PLACEMENT / EXPERT_PLACEMENT. NOT translated from the
    resolved `device` the way GPU pinning is for other engines - this model
    (26B-A4B MoE) supports placing experts across MULTIPLE GPUs, which is
    out of scope for this translator (see VALIDATION.md - the translation
    layer only ever resolves ONE device per backend). Pass these through
    raw if a suite wants multi-GPU expert placement; single-GPU (the
    validated case) needs neither set.

Health check: Arcaine's image has NO /health endpoint (confirmed via a live
container - 404). Suites using this engine MUST set `health_path: "/v1/models"`
themselves (a normal 200-with-model-list response once the server is up) -
this translator does not set health_path because BackendConfig owns that
field directly, not EngineInvocation.
"""
from __future__ import annotations

from typing import Any

from llapdance.core.probe import DeviceInfo
from llapdance.plugins.base import EngineInvocation, EngineTranslator
from llapdance.plugins.registry import register


class ArcaineEngine(EngineTranslator):
    name = "arcaine"

    sweepable_params = {
        "context_size": {"type": "int", "default": 4096, "maps_to": "MAX_SEQ env"},
        "max_tokens": {"type": "int", "default": 2048, "maps_to": "DEFAULT_MAX_TOKENS env"},
        "denoising_steps": {"type": "int", "maps_to": "DENOISING_STEPS env", "note": "diffusion-model-specific"},
        "seed": {"type": "int", "maps_to": "DEFAULT_SEED env"},
        "layer_placement": {"type": "str", "maps_to": "LAYER_PLACEMENT env", "note": "multi-GPU MoE expert sharding - not resolved from device, raw passthrough only, see module docstring"},
        "expert_placement": {"type": "str", "maps_to": "EXPERT_PLACEMENT env", "note": "see layer_placement"},
    }

    # Real runtime env flags for the validated model family (diffusion_gemma
    # - src/modeling/diffusion_gemma/), found by reading Arcaine's own
    # source. Arcaine IS a from-scratch engine (own modeling/gpu code, not
    # vendored ggml-sycl) but DOES link oneDNN (CMakeLists.txt:
    # find_package(dnnl CONFIG REQUIRED)) - unlike llama.cpp's GGML_SYCL_DNNL
    # (a build-time cmake flag), Arcaine's oneDNN attention path is a
    # RUNTIME toggle: DIFF_ONEDNN_SDPA unset/"off"/"0"/"false"/"no" (any
    # case) disables it; any other value enables oneDNN-backed
    # scaled-dot-product-attention and is passed through as an
    # implementation-variant string (exact valid non-empty values not
    # enumerated here - not fully characterized this session).
    #
    # Also found: a separate Qwen3.5 model family (src/modeling/qwen3_5*/,
    # ARCAINE_QWEN35_* / ARCAINE_QWEN_MAX_LAYERS env vars, ~15 flags) that
    # this harness has NOT validated against (only diffusion_gemma has been
    # tested here) - deliberately not cataloged below; read
    # src/modeling/qwen3_5/ and src/modeling/qwen3_5_moe/ directly before
    # relying on any of those if that model family gets validated later.
    known_env_flags = {
        "DIFF_ONEDNN_SDPA": {
            "type": "str",
            "note": "unset or 'off'/'0'/'false'/'no' (any case) = oneDNN SDPA disabled (the default); "
            "any other value = enabled, value passed through as an implementation-variant selector "
            "(exact valid values not enumerated this session) - src/modeling/diffusion_gemma/attention.cpp",
        },
        "DIFF_ARENA": {
            "type": "str", "default": "on (pooled)",
            "note": "'off'/'0'/'false'/'no' drops to non-pooled allocation (fresh sycl::malloc_device "
            "per alloc, freed on scope exit) - the pre-pool A/B baseline. See DISABLE_SCRATCH.",
        },
        "DISABLE_SCRATCH": {
            "type": "str", "note": "'1'/'true'/'TRUE'/'yes' has the same effect as DIFF_ARENA=off - two "
            "env vars control the same allocator toggle",
        },
        "DIFF_PREFILL_CHUNK": {"type": "int", "default": 2048, "note": "<=0 disables chunking - bounds activation storage for long prompts"},
        "DIFF_FORCE_DENOISE_STEPS": {"type": "presence", "note": "skips the normal early-stop loop-break logic, forces the full step count"},
        "DIFF_HOST_SAMPLER": {"type": "presence", "note": "use the original fully host-side sampler instead of the device sampler path"},
    }
    # NVFP4-quant-specific flags exist too (DIFF_NVFP4_* - ~13 flags,
    # directly relevant since the validated model IS NVFP4-quantized) and
    # MoE-specific flags (DIFF_MOE_STATS, DIFF_MOE_TAIL_CAP) - found but not
    # individually characterized this session (each needs its own context
    # read, like DIFF_ONEDNN_SDPA above, to document honestly rather than
    # guess at valid values/defaults). Read src/modeling/diffusion_gemma/
    # (attention_kernels.hpp, moe.cpp, fusions/int4_awq.hpp) directly before
    # sweeping any of them.

    def build(self, model_path: str, params: dict[str, Any], port: int, device: DeviceInfo | None) -> EngineInvocation:
        context_size = params.get("context_size", 4096)
        max_tokens = params.get("max_tokens", 2048)

        env = {
            "MODEL_PATH": model_path,
            "SERVER_HOST": "0.0.0.0",
            "SERVER_PORT": str(port),
            "MAX_SEQ": str(context_size),
            "DEFAULT_MAX_TOKENS": str(max_tokens),
        }
        if "served_model_name" in params:
            env["SERVED_MODEL_NAME"] = params["served_model_name"]
        if "denoising_steps" in params:
            env["DENOISING_STEPS"] = str(params["denoising_steps"])
        if "seed" in params:
            env["DEFAULT_SEED"] = str(params["seed"])
        if "layer_placement" in params:
            env["LAYER_PLACEMENT"] = str(params["layer_placement"])
        if "expert_placement" in params:
            env["EXPERT_PLACEMENT"] = str(params["expert_placement"])

        devices: list[str] = []
        if device is not None:
            if device.render_node is None:
                raise ValueError(
                    f"device {device.index} ({device.name}) has no known render_node - "
                    "discover devices via xpumcli (populates render_node), not the clinfo fallback."
                )
            devices = [f"{device.render_node}:{device.render_node}"]

        return EngineInvocation(command=[], env=env, devices=devices)


register("engine", ArcaineEngine.name, ArcaineEngine)
