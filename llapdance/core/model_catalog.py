"""Model catalog: format detection + backend compatibility (new capability,
not originally in SPEC.md - added per request alongside the image catalog).

Distinct from the image catalog (SPEC.md §12, docker images): this scans
model FILES/directories on disk and answers "which of this harness's
engine translators could plausibly load this, based on format alone" -
explicitly a could-run-on signal, never a will-run guarantee. A GGUF file
that's corrupt, a safetensors model too large for available VRAM, or a
quant an engine doesn't actually support (see llama-cpp-sycl's f8 KV-cache
rejection) all pass this check and still fail at runtime - format
compatibility is necessary, not sufficient.

Real directory layouts found scanning this box's actual model folders
(not assumed): OpenVINO IR and HF-safetensors model roots are frequently
nested under an org/contributor namespace directory (e.g.
`OpenVINO/droans/qwen3.5-9B-int4-ov/`), same convention as the HF Hub's
own `org/model` structure - detection walks recursively and stops
descending once a directory is identified as a model root (its own
files), rather than assuming models sit directly under the configured
scan directory.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# Engine translator names (llapdance/plugins/engine/) that can load each
# format - informational, "could run on", see module docstring. Kept here
# rather than on each EngineTranslator because format compatibility is a
# property of the FORMAT, not of any one engine's params (unlike
# sweepable_params, which is genuinely per-engine).
COMPATIBLE_ENGINES: dict[str, list[str]] = {
    "gguf": ["llama-cpp-sycl", "qxmx"],
    "openvino_ir": ["openarc"],
    "safetensors": ["arcaine"],
}

_GGUF_QUANT_RE = re.compile(
    r"(NVFP4|dspark|IQ[0-9]_[A-Z0-9]+|Q[0-9]_[A-Z0-9]+(?:_[A-Z0-9]+)?|[BF]F?16|F32)", re.IGNORECASE
)


@dataclass
class ModelInfo:
    path: str
    format: str  # "gguf" | "openvino_ir" | "safetensors"
    compatible_engines: list[str] = field(default_factory=list)
    quant_hint: str = "unknown"
    size_bytes: int = 0


def _dir_size(path: Path) -> int:
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def _gguf_quant_hint(filename: str) -> str:
    match = _GGUF_QUANT_RE.search(filename)
    return match.group(1) if match else "unknown"


def _openvino_ir_quant_hint(model_dir: Path) -> str:
    config_path = model_dir / "openvino_config.json"
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text())
            if data.get("dtype"):
                return data["dtype"]
            quant_method = data.get("quantization_config", {}).get("quant_method")
            if quant_method:
                return quant_method
        except (json.JSONDecodeError, OSError):
            pass
    return "unknown"


def _safetensors_quant_hint(model_dir: Path) -> str:
    config_path = model_dir / "config.json"
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text())
            quant_config = data.get("quantization_config", {})
            if quant_config.get("format"):
                return quant_config["format"]
            if quant_config.get("quant_method"):
                return quant_config["quant_method"]
        except (json.JSONDecodeError, OSError):
            pass
    # fall back to a filename/dirname regex - the same heuristic used for
    # GGUF, since some safetensors dirs encode quant in the dir name only
    match = _GGUF_QUANT_RE.search(model_dir.name)
    return match.group(1) if match else "unknown"


def scan_models(directories: list[str]) -> list[ModelInfo]:
    """Recursively scans each directory for GGUF files, OpenVINO IR model
    roots, and HF-safetensors model roots. Never assumes models sit
    directly under the given directory - see module docstring."""
    results: list[ModelInfo] = []

    for base in directories:
        base_path = Path(base)
        if not base_path.is_dir():
            continue

        # GGUF: standalone files, found anywhere under the tree.
        for gguf_file in base_path.rglob("*.gguf"):
            results.append(
                ModelInfo(
                    path=str(gguf_file),
                    format="gguf",
                    compatible_engines=COMPATIBLE_ENGINES["gguf"],
                    quant_hint=_gguf_quant_hint(gguf_file.name),
                    size_bytes=gguf_file.stat().st_size,
                )
            )

        # OpenVINO IR / safetensors: directory roots, identified by their
        # own marker files - stop descending once found (a model root's
        # own subdirectories, if any, aren't separate models).
        for dirpath, dirnames, filenames in os.walk(base_path):
            current = Path(dirpath)
            if (current / "openvino_model.xml").exists():
                results.append(
                    ModelInfo(
                        path=str(current),
                        format="openvino_ir",
                        compatible_engines=COMPATIBLE_ENGINES["openvino_ir"],
                        quant_hint=_openvino_ir_quant_hint(current),
                        size_bytes=_dir_size(current),
                    )
                )
                dirnames[:] = []
            elif any(f.endswith(".safetensors") for f in filenames) and "config.json" in filenames:
                results.append(
                    ModelInfo(
                        path=str(current),
                        format="safetensors",
                        compatible_engines=COMPATIBLE_ENGINES["safetensors"],
                        quant_hint=_safetensors_quant_hint(current),
                        size_bytes=_dir_size(current),
                    )
                )
                dirnames[:] = []

    return results
