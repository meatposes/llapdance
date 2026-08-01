"""Docker image catalog & cleanup (SPEC.md §12) - the first thing this
harness has built to consume the sprawl of experimental image tags it was
originally motivated by, rather than just leaving `docker images | grep`
as the only way to see what's there.

Labels follow flat-file storage (the always-on default, SPEC.md §8/§12):
a small JSON file (`_image_labels.json`) alongside a suite's flat-file
results, not a separate database. If a suite also has OpenSearch or
another storage adapter enabled, the flat-file label file remains the
source of truth for this v1 - cross-referencing labels into other storage
adapters is not built (SPEC.md §12 explicitly allows a DB to be the
source of truth instead when one's active; that's future work here, not
done this pass).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from llapdance.plugins.base import ExecutionTargetAdapter

_LABELS_FILENAME = "_image_labels.json"


def _labels_path(catalog_dir: str) -> Path:
    return Path(catalog_dir) / _LABELS_FILENAME


def get_labels(catalog_dir: str) -> dict[str, dict[str, Any]]:
    path = _labels_path(catalog_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def label_image(catalog_dir: str, image_ref: str, label: str, note: str = "") -> None:
    path = _labels_path(catalog_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = get_labels(catalog_dir)
    data[image_ref] = {"label": label, "note": note, "labeled_at": time.time()}
    path.write_text(json.dumps(data, indent=2))


def _index_runs_by_image(catalog_dir: str) -> dict[str, list[dict[str, Any]]]:
    """Cross-references stored RunResults (flat-file JSON files, not the
    labels file itself) by image_ref - this is the "trace a result back
    to the exact image" half of SPEC.md §12, using metadata every run
    already carries (image_ref, per RunResult) rather than needing new
    plumbing."""
    index: dict[str, list[dict[str, Any]]] = {}
    catalog_path = Path(catalog_dir)
    if not catalog_path.is_dir():
        return index
    for path in catalog_path.glob("*.json"):
        if path.name == _LABELS_FILENAME:
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        image_ref = data.get("image_ref")
        if not image_ref:
            continue
        index.setdefault(image_ref, []).append(
            {"run_id": data.get("run_id"), "backend_name": data.get("backend_name"), "timestamp": data.get("timestamp")}
        )
    return index


def list_images(
    execution: ExecutionTargetAdapter, catalog_dir: str | None = None, name_filter: str | None = None
) -> list[dict[str, Any]]:
    """Enumerate images via whichever ExecutionTargetAdapter is active
    (local or SSH - both already implement list_images(), just never
    consumed anywhere until now), enriched with any label and any stored
    run results that reference each tag."""
    labels = get_labels(catalog_dir) if catalog_dir else {}
    runs_by_image = _index_runs_by_image(catalog_dir) if catalog_dir else {}

    enriched = []
    for image in execution.list_images(name_filter):
        label_info = None
        runs: list[dict[str, Any]] = []
        for tag in image["tags"]:
            if tag in labels:
                label_info = labels[tag]
            runs.extend(runs_by_image.get(tag, []))
        enriched.append({**image, "label": label_info, "runs": runs})
    return enriched


class LabeledImageRemovalError(RuntimeError):
    """Raised when removing an image labeled 'good' without force=True -
    a labeled image is exactly the thing SPEC.md §12 cataloging exists to
    protect from accidental cleanup."""


def remove_image(
    execution: ExecutionTargetAdapter, image_ref: str, catalog_dir: str | None = None, force: bool = False
) -> None:
    if catalog_dir and not force:
        label = get_labels(catalog_dir).get(image_ref, {}).get("label")
        if label == "good":
            raise LabeledImageRemovalError(
                f"{image_ref} is labeled 'good' - refusing to remove without force=True"
            )
    execution.remove_image(image_ref)
