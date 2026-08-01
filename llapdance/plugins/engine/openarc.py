"""EngineTranslator for OpenArc (~/OpenArc/OpenArc, remote: origin=SearchSavior/OpenArc).
Validated against the local `openarc:dev` image - see VALIDATION.md.

GOTCHA that shaped this translator's whole design: OpenArc is NOT a
"start container with a model baked into the command/env" engine like the
other three. It starts with NO model loaded (`openarc serve start`, no
model args at all), and a model only becomes servable after a separate
`POST /openarc/load` call. This is why `EngineInvocation.post_start_requests`
exists (see llapdance/plugins/base.py) - it didn't before this backend was
integrated, because none of the other three engines needed it.

Discovered live, not guessed: `engine` must be `ovgenai` for `model_type:
llm` - `optimum` is only valid for `emb`/`rerank` (the server told me the
exact valid combinations directly when I got it wrong, no need to guess a
second time).

Model path here is an OpenVINO IR model DIRECTORY (openvino_model.xml/.bin
+ tokenizer/detokenizer IR + configs) - genuinely different format from the
GGUF single-file (llama.cpp/qxmx) or HF-safetensors-dir (Arcaine) models
the other three translators expect. `model_path` still just means
"whatever this engine's own convention wants at that path" - the config
schema doesn't encode format, the translator does.

Normalized params.shared this translator understands:
  - none. OpenArc's load config (model_type, engine, device) doesn't map
    onto context_size/batch_size/kv_cache_quant/parallel_slots/reasoning
    the way the other three do - it's a fundamentally different "which
    engine backend + device to load onto" surface, not a runtime-tuning
    surface. Read from the merged params dict as OpenArc-specific keys
    instead (see below) - there was nothing to normalize across backends
    here, and inventing a fake shared concept just to force uniformity
    would be worse than admitting this one doesn't share much with the
    other three.

OpenArc-specific params (read from the merged params dict):
  - model_type (str, default "llm"): passed straight to /openarc/load.
  - openarc_engine (str, default "ovgenai"): passed as the load config's
    `engine` field. Named openarc_engine (not `engine`) to avoid colliding
    with BackendConfig.engine (which names the *translator*, a different
    concept entirely).
  - model_name (str, default: derived from model_path's final path
    component): the public-facing name this model is loaded/served under.
  - runtime_config (dict, default {}): raw passthrough to OpenVINO/ov_genai
    plugin properties (e.g. NUM_STREAMS, PERFORMANCE_HINT,
    INFERENCE_PRECISION_HINT, CACHE_DIR) - confirmed by reading OpenArc's
    own source (src/engine/ov_genai/llm.py: `pipeline_kwargs =
    {**(loader.runtime_config or {})}`, merged straight into the
    LLMPipeline call) rather than guessed at. GOTCHA found cataloging
    this: an earlier version of this translator didn't read or forward
    `runtime_config` at all - OpenArc's real tuning surface existed and
    was reachable through its own API, but this translator silently
    dropped it, making it unreachable through this harness even though it
    looked structurally sweepable. Fixed - now merged into the
    `/openarc/load` JSON body.

GPU: only one render node ever passed through (same as the other
translators); OpenArc's own device naming ("GPU", "GPU.0", "GPU.1" - yet a
FOURTH GPU index space, on top of clinfo/xpumcli/SYCL-level-zero/DRM
render-node, none reconciled with each other - see VALIDATION.md) is
sidestepped entirely by just using the literal string "GPU": with only one
render node visible in the container, OpenVINO only ever enumerates one
GPU regardless of what number it would otherwise get.

Health check: OpenArc DOES have a real health-ish endpoint - `/v1/models`
returns 200 with an empty list before any model is loaded. Suites using
this engine MUST set `health_path: "/v1/models"` themselves (same
requirement as the arcaine translator, for the same reason: no image here
provides a real /health).
"""
from __future__ import annotations

from typing import Any

from llapdance.core.probe import DeviceInfo
from llapdance.plugins.base import EngineInvocation, EngineTranslator
from llapdance.plugins.registry import register


class OpenArcEngine(EngineTranslator):
    name = "openarc"

    sweepable_params = {
        "model_type": {"type": "str", "default": "llm", "values": ["llm", "vlm", "whisper", "qwen3_asr", "kokoro", "emb", "rerank"], "maps_to": "/openarc/load model_type"},
        "openarc_engine": {"type": "str", "default": "ovgenai", "values": ["ovgenai", "openvino", "optimum"], "maps_to": "/openarc/load engine", "note": "not all model_type/engine combinations are valid, see module docstring"},
        "model_name": {"type": "str", "maps_to": "/openarc/load model_name", "note": "defaults to model_path's final path component if unset"},
        "runtime_config": {
            "type": "dict", "default": {},
            "maps_to": "/openarc/load runtime_config -> raw OpenVINO/ov_genai plugin properties",
            "note": "e.g. {'NUM_STREAMS': '2'} or {'PERFORMANCE_HINT': 'THROUGHPUT'} - sweep individual "
            "keys via params.shared.runtime_config.<KEY> (confirmed real by reading OpenArc's source, "
            "see module docstring; the specific property names/values are OpenVINO's own, not OpenArc's)",
        },
        # no context_size/batch_size/kv_cache_quant/parallel_slots/reasoning
        # - genuinely not applicable, see module docstring
    }

    def build(self, model_path: str, params: dict[str, Any], port: int, device: DeviceInfo | None) -> EngineInvocation:
        if device is None:
            raise ValueError("openarc requires a GPU device to be resolved (device: 'GPU' is hardcoded downstream)")
        if device.render_node is None:
            raise ValueError(
                f"device {device.index} ({device.name}) has no known render_node - "
                "discover devices via xpumcli (populates render_node), not the clinfo fallback."
            )

        model_name = params.get("model_name") or model_path.rstrip("/").rsplit("/", 1)[-1]
        model_type = params.get("model_type", "llm")
        openarc_engine = params.get("openarc_engine", "ovgenai")
        runtime_config = params.get("runtime_config", {})

        devices = [f"{device.render_node}:{device.render_node}"]
        post_start_requests = [
            {
                "method": "POST",
                "path": "/openarc/load",
                "json": {
                    "model_path": model_path,
                    "model_name": model_name,
                    "model_type": model_type,
                    "engine": openarc_engine,
                    "device": "GPU",
                    "runtime_config": runtime_config,
                },
            }
        ]
        return EngineInvocation(command=[], env={}, devices=devices, post_start_requests=post_start_requests)


register("engine", OpenArcEngine.name, OpenArcEngine)
