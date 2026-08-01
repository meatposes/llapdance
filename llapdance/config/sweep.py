"""Parameter-sweep expansion (SPEC.md §10). Turns a backend config with one
or more `sweep` axes into the cartesian product of concrete backend
configs - real automation, not the "hand-author one suite file per
variant" pattern every prior validation run in this project used.

Expansion happens at `run_suite()` time (llapdance/core/orchestrator.py),
not at `load_suite()` time - `get_suite`/`list_suites` (CLI and MCP) show
the compact sweep-spec form a suite author actually wrote; only an actual
run sees the expanded set. Nothing about a swept backend's raw config
changes: this module only ever produces NEW BackendConfig objects, it
never mutates the one a suite author wrote.
"""
from __future__ import annotations

import itertools
import re
from typing import Any

from llapdance.config.models import BackendConfig, TestSuite

_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _set_dotted(d: dict[str, Any], path: str, value: Any) -> None:
    """Intermediate path components must already exist as dicts (catches a
    typo'd path before it silently no-ops) - but the FINAL leaf key is
    allowed to be new: params.shared/backend_specific are open dicts, and a
    sweep should be able to introduce a param the base config never set a
    default for, not just vary an existing one."""
    parts = path.split(".")
    node = d
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise ValueError(f"sweep param path {path!r} does not resolve to an existing dict at {part!r}")
        node = node[part]
    node[parts[-1]] = value


def _slug(value: Any) -> str:
    return _SLUG_RE.sub("_", str(value))


def expand_backend_sweep(backend: BackendConfig) -> list[BackendConfig]:
    """One backend, no sweep axes -> itself, unchanged (list of 1). One
    backend with N axes -> the cartesian product across all of them, each
    combination a distinct concrete BackendConfig with a name suffixed by
    every axis's value (e.g. `qxmx--context_size_2048--context_size_4096`
    would be wrong - it's `qxmx--context_size_2048`, `qxmx--context_size_4096`,
    one per combination, not one name per axis)."""
    if not backend.sweep:
        return [backend]

    axes = backend.sweep
    combinations = itertools.product(*(axis.values for axis in axes))
    expanded: list[BackendConfig] = []
    for combo in combinations:
        raw = backend.model_dump()
        name_parts = []
        for axis, value in zip(axes, combo):
            _set_dotted(raw, axis.param, value)
            leaf = axis.param.rsplit(".", 1)[-1]
            name_parts.append(f"{leaf}_{_slug(value)}")
        raw["sweep"] = []  # expanded configs are concrete, not themselves sweepable
        raw["name"] = f"{backend.name}--{'--'.join(name_parts)}"
        expanded.append(BackendConfig.model_validate(raw))
    return expanded


def expand_suite_sweep(suite: TestSuite) -> TestSuite:
    """Expand every backend in a suite, in place order, returning a NEW
    TestSuite (the original is never mutated) - everything else about the
    suite (device_target, storage, adapters) is shared unchanged across
    every expanded backend."""
    expanded_backends: list[BackendConfig] = []
    for backend in suite.backends:
        expanded_backends.extend(expand_backend_sweep(backend))
    return suite.model_copy(update={"backends": expanded_backends})
