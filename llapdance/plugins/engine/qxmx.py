"""EngineTranslator for qxmx (~/qxmx, validated against qxmx:latest - see
VALIDATION.md).

Normalized params.shared this translator understands:
  - context_size (int): -> `--ctx-per-slot`. Default 4096 (matches the
    validated run).
  - kv_cache_quant (str, one of "f16"/"q8_0"/"f8"): -> `--ctk`, using
    qxmx's own value spelling ("f16" -> "fp16", others unchanged). Omitted
    if not set - qxmx's own usage banner shows this as optional
    (`[--ctk fp16|q8_0|fp8]`), so its own default applies.
  - parallel_slots (int): -> `--slots`. Default 1 (matches the validated
    run) - unlike llama.cpp, qxmx's usage requires this flag, there is no
    "auto" to fall back to.
  - batch_size: NOT APPLICABLE. qxmx's CLI has no batching flag at all
    (confirmed via its own `--help`/usage banner); if set, this translator
    ignores it rather than inventing a flag that doesn't exist.

GOTCHA (see VALIDATION.md): qxmx:latest sets no ENTRYPOINT, only a default
CMD - overriding `command` REPLACES it entirely, so the binary path itself
must be `command[0]`. This is baked into the generated command below; it's
exactly the kind of image-specific detail an EngineTranslator exists to
absorb so a suite author doesn't have to know it.

GPU: qxmx has NO device-selector flag or env var at all - which render
node(s) are passed through to the container IS its entire GPU-pinning
mechanism (confirmed via its own README: "only one GPU is supported").
This translator requires `device` to be given; qxmx cannot meaningfully
run without exactly one GPU.
"""
from __future__ import annotations

from typing import Any

from llapdance.core.probe import DeviceInfo
from llapdance.plugins.base import EngineInvocation, EngineTranslator
from llapdance.plugins.registry import register

_KV_QUANT_MAP = {"f16": "fp16", "q8_0": "q8_0", "f8": "fp8"}


class QxmxEngine(EngineTranslator):
    name = "qxmx"

    sweepable_params = {
        "context_size": {"type": "int", "default": 4096, "maps_to": "--ctx-per-slot"},
        "kv_cache_quant": {"type": "str", "values": ["f16", "q8_0", "f8"], "maps_to": "--ctk (spelling translated per value)"},
        "parallel_slots": {"type": "int", "default": 1, "maps_to": "--slots"},
        # batch_size deliberately absent - qxmx has no batching flag at all (see module docstring)
    }

    def build(self, model_path: str, params: dict[str, Any], port: int, device: DeviceInfo | None) -> EngineInvocation:
        if device is None:
            raise ValueError("qxmx requires a GPU device - it has no CPU fallback (see its own README)")
        if device.render_node is None:
            raise ValueError(
                f"device {device.index} ({device.name}) has no known render_node - qxmx's GPU "
                "pinning IS render-node passthrough, there is no other selector to fall back to. "
                "Discover devices via xpumcli (populates render_node) rather than the clinfo fallback."
            )

        context_size = params.get("context_size", 4096)
        parallel_slots = params.get("parallel_slots", 1)
        kv_cache_quant = params.get("kv_cache_quant")

        command = [
            "./build/qxmx_serve",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--slots", str(parallel_slots),
            "--ctx-per-slot", str(context_size),
        ]
        if kv_cache_quant is not None:
            if kv_cache_quant not in _KV_QUANT_MAP:
                raise ValueError(f"qxmx does not support kv_cache_quant={kv_cache_quant!r} (supported: {sorted(_KV_QUANT_MAP)})")
            command += ["--ctk", _KV_QUANT_MAP[kv_cache_quant]]
        command.append(model_path)

        devices = [f"{device.render_node}:{device.render_node}"]
        return EngineInvocation(command=command, env={}, devices=devices)


register("engine", QxmxEngine.name, QxmxEngine)
