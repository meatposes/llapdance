"""YAML config loading with CLI-flag override merging (SPEC.md §10)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import TestSuite


def _deep_merge(base: Any, override: Any) -> Any:
    """Recursive merge. `override` dicts with all-digit keys against a list
    `base` are treated as index assignments (this is how `--set a.0.b=x`
    reaches into a YAML list) - GOTCHA: plain dict/dict merging alone
    silently produced `{"0": {...}}` clobbering the list's type instead of
    indexing into it; caught by trying to override a real suite's
    benchmark_adapters list end-to-end, not by unit tests against dicts only."""
    if isinstance(base, list) and isinstance(override, dict) and override and all(k.isdigit() for k in override):
        result = list(base)
        for key, value in override.items():
            idx = int(key)
            if idx < len(result):
                result[idx] = _deep_merge(result[idx], value)
            else:
                result.append(value)
        return result
    if isinstance(base, dict) and isinstance(override, dict):
        result = dict(base)
        for key, value in override.items():
            result[key] = _deep_merge(result.get(key), value) if key in result else value
        return result
    return override


def load_suite(path: str | Path, overrides: dict[str, Any] | None = None) -> TestSuite:
    """Load a TestSuite from a YAML file, applying optional CLI-flag overrides
    on top. `overrides` uses the same nested-dict shape as the YAML file."""
    raw = yaml.safe_load(Path(path).read_text())
    if overrides:
        raw = _deep_merge(raw, overrides)
    return TestSuite.model_validate(raw)


def parse_kv_overrides(pairs: list[str]) -> dict[str, Any]:
    """Turn `--set a.b.c=value` style CLI flags into a nested override dict."""
    overrides: dict[str, Any] = {}
    for pair in pairs:
        key_path, _, value = pair.partition("=")
        if not key_path or not value:
            raise ValueError(f"invalid override, expected key.path=value: {pair!r}")
        node = overrides
        parts = key_path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return overrides
