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

    sweepable_params = {
        "context_size": {"type": "int", "default": 4096, "maps_to": "-c"},
        "batch_size": {"type": "int", "maps_to": "-b / -ub (same value both)"},
        "kv_cache_quant": {"type": "str", "values": ["f16", "q8_0"], "maps_to": "--cache-type-k / --cache-type-v"},
        "parallel_slots": {"type": "int", "maps_to": "--parallel"},
        "reasoning": {"type": "str", "values": ["on", "off", "auto"], "default": "auto", "maps_to": "LLAMA_ARG_REASONING env"},
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
