"""Default storage adapter (SPEC.md §8) - always available, no external
service required. One JSON file per run, named for sortable recency."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llapdance.core.result import RunResult
from llapdance.plugins.base import StorageAdapter
from llapdance.plugins.registry import register


class FlatFileStorage(StorageAdapter):
    name = "flat-file"

    def __init__(self, config: dict[str, Any]) -> None:
        self._dir = Path(config["flat_file_dir"])
        self._dir.mkdir(parents=True, exist_ok=True)

    def write(self, result: RunResult) -> None:
        fname = f"{int(result.timestamp)}_{result.backend_name}_{result.run_id}.json"
        (self._dir / fname).write_text(result.model_dump_json(indent=2))

    def previous_for(self, backend_name: str, limit: int = 1) -> list[RunResult]:
        matches = sorted(
            (p for p in self._dir.glob(f"*_{backend_name}_*.json")),
            key=lambda p: p.name,
            reverse=True,
        )
        results = []
        for path in matches[:limit]:
            results.append(RunResult.model_validate(json.loads(path.read_text())))
        return results


register("storage", FlatFileStorage.name, FlatFileStorage)
