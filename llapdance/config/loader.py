"""YAML config loading with CLI-flag override merging (SPEC.md §10)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import TestSuite


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


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
