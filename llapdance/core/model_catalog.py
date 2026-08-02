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

from llapdance.core.result import RunResult

# Engine translator names (llapdance/plugins/engine/) that can load each
# format - informational, "could run on", see module docstring. Kept here
# rather than on each EngineTranslator because format compatibility is a
# property of the FORMAT, not of any one engine's params (unlike
# sweepable_params, which is genuinely per-engine).
COMPATIBLE_ENGINES: dict[str, list[str]] = {
    "gguf": ["llama-cpp-sycl", "llama-cpp-vulkan", "qxmx"],
    "openvino_ir": ["openarc"],
    # arcaine only dispatches a narrow model_type allowlist (see
    # llapdance/plugins/engine/arcaine.py); vllm has no such restriction -
    # any HF-transformers-format safetensors checkpoint is a plausible fit
    # (confirmed real: gemma3/qwen3/deepseek_v2 all load under vLLM natively,
    # see VALIDATION.md "vLLM engine translator" section).
    "safetensors": ["arcaine", "vllm"],
}

_GGUF_QUANT_RE = re.compile(
    r"(NVFP4|dspark|IQ[0-9]_[A-Z0-9]+|Q[0-9]_[A-Z0-9]+(?:_[A-Z0-9]+)?|[BF]F?16|F32)", re.IGNORECASE
)


@dataclass
class TestedStatus:
    """A real prior run found in stored results (SPEC.md §8 flat-file
    records) for this model on this engine - never inferred, only ever
    built from an actual RunResult.

    outcome is one of:
      "pass"    - every coherence adapter that ran got a clean 100% pass rate
      "partial" - at least one coherence adapter ran and had failures
      "ran"     - the run completed and got recorded, but no coherence
                  adapter was configured, so there's no correctness signal,
                  only "it started and didn't crash"

    NOTE the real gap this can't cover: `run_backend` only writes a
    RunResult after the run finishes (see orchestrator.py's `finally:
    execution.stop(running)` wrapping `_run_adapters_with_telemetry` before
    storage). A run that crashed mid-request (like the real Arcaine
    KV-cache 500 found this session, see VALIDATION.md) never reaches
    storage at all - there is no stored record to find, so a genuinely
    broken model/engine/flag combination looks identical to "never tried"
    here, not "tried and failed". Cross-check VALIDATION.md/session notes
    for crash history this can't see.
    """

    engine: str
    run_id: str
    timestamp: float
    outcome: str
    coherence_summary: str | None = None


@dataclass
class ModelInfo:
    path: str
    format: str  # "gguf" | "openvino_ir" | "safetensors"
    compatible_engines: list[str] = field(default_factory=list)
    quant_hint: str = "unknown"
    size_bytes: int = 0
    tested: dict[str, TestedStatus] = field(default_factory=dict)
    has_mmproj: bool = False
    """True if a sibling `*mmproj*.gguf` file (a multimodal-projector
    companion artifact, not a standalone servable model - see
    `_MMPROJ_RE`) exists in the same directory as this model. Direct user
    feedback: these companion files were being scanned and listed as their
    own independent "models", which is real clutter/confusion (they load
    as valid GGUF files, so nothing structural rejected them, even though
    running one alone as the main model is never a meaningful test) - now
    excluded from scan results entirely, with this flag surfacing the same
    information as a small indicator on the model(s) they belong to
    instead of a hidden fact."""


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

        # GGUF: standalone files, found anywhere under the tree. mmproj
        # companion files (multimodal-projector artifacts, not standalone
        # servable models - see ModelInfo.has_mmproj) are excluded from
        # results entirely, but their directory is noted so sibling real
        # models in the same directory can be flagged as having one.
        gguf_files = list(base_path.rglob("*.gguf"))
        mmproj_dirs = {f.parent for f in gguf_files if "mmproj" in f.name.lower()}
        for gguf_file in gguf_files:
            if "mmproj" in gguf_file.name.lower():
                continue
            results.append(
                ModelInfo(
                    path=str(gguf_file),
                    format="gguf",
                    compatible_engines=COMPATIBLE_ENGINES["gguf"],
                    quant_hint=_gguf_quant_hint(gguf_file.name),
                    size_bytes=gguf_file.stat().st_size,
                    has_mmproj=gguf_file.parent in mmproj_dirs,
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


def load_run_history(results_dir: str) -> list[RunResult]:
    """Reads every stored RunResult from a flat-file results directory
    (SPEC.md §8). Skips files that don't parse as a RunResult (e.g. the
    catalog's own `_image_labels.json` living alongside it - see
    llapdance/core/catalog.py) rather than raising, since a directory
    scan shouldn't hard-fail on an unrelated file."""
    results: list[RunResult] = []
    base = Path(results_dir)
    if not base.is_dir():
        return results
    for f in base.glob("*.json"):
        try:
            results.append(RunResult.model_validate_json(f.read_text()))
        except (json.JSONDecodeError, ValueError, OSError):
            continue
    return results


def _resolve_host_path(model_path: str, volumes: dict[str, str]) -> str | None:
    """Reverses a BackendConfig's host->container volume mount to recover
    the host path a run's in-container `model_path` actually pointed at -
    the only way to line a stored RunResult back up against a ModelInfo
    (which is always a host path), since RunResult only ever records the
    container-side path directly."""
    for host_vol, container_vol in volumes.items():
        if model_path == container_vol:
            return host_vol
        prefix = container_vol.rstrip("/") + "/"
        if model_path.startswith(prefix):
            return str(Path(host_vol) / model_path[len(prefix):])
    return None


def _run_outcome(result: RunResult) -> tuple[str, str | None]:
    if not result.coherence:
        return "ran", None
    total = sum(c.total for c in result.coherence)
    passed = sum(c.passed for c in result.coherence)
    summary = f"{passed}/{total}"
    return ("pass" if total and passed == total else "partial"), summary


def annotate_tested_status(models: list[ModelInfo], history: list[RunResult]) -> None:
    """Cross-references real stored run history against the (purely
    static, format-based) model catalog - mutates each ModelInfo.tested
    in place. Only ever reports what's actually in a stored RunResult;
    see TestedStatus's docstring for the real gap (crashed runs never
    reach storage, so they're indistinguishable from untested here)."""
    by_path = {str(Path(m.path)): m for m in models}

    for result in sorted(history, key=lambda r: r.timestamp):
        engine = result.backend_config.get("engine")
        model_path = result.backend_config.get("model_path")
        volumes = result.backend_config.get("volumes", {})
        if not engine or not model_path:
            continue
        host_path = _resolve_host_path(model_path, volumes)
        if host_path is None:
            continue
        model = by_path.get(str(Path(host_path)))
        if model is None:
            continue
        outcome, summary = _run_outcome(result)
        # sorted ascending by timestamp above, so the last write for a
        # given engine is always the most recent - no explicit max needed.
        model.tested[engine] = TestedStatus(
            engine=engine,
            run_id=result.run_id,
            timestamp=result.timestamp,
            outcome=outcome,
            coherence_summary=summary,
        )
