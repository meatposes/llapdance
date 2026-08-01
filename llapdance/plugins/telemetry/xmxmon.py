"""Reference TelemetryAdapter for xmxmon (a real, already-running local GPU
telemetry daemon - hardware counter sampling, not an OpenAI-compatible
endpoint). Validated against a live instance - see VALIDATION.md.

Real API confirmed by reading the daemon's own source directly (no docs to
guess from): `GET /now` returns a per-device rolling-window snapshot
(`gauges`: percent-like averages, `rates`: per-second counter rates,
`derived`: human-labeled derived metrics) computed over `window_s` seconds
of recent samples. `POST /capture` / `POST /capture/stop` also exist, but
deliberately NOT used here: `stop_capture()` writes a tagged ndjson file
INSIDE the daemon's own container and returns only file metadata (row
count, duration) over the API - the actual samples are never exposed via
HTTP, only to a filesystem this harness has no access-checked reason to
assume it can read. The `/now` window snapshot IS a real, complete summary
already computed by the daemon itself, so that's what this adapter uses:
`stop()` takes one snapshot after the run completes, covering the recent
window average (accurate as long as the run's duration comfortably exceeds
`window_s`, which is xmxmon's own config, not this adapter's).

Device numbering here is xmxmon's own (confirmed: on the box tested, only
device "0" is sampled at all, itself the daemon's own config choice, not
auto-discovered) - yet another GPU index space not reconciled with
`DeviceInfo`, same as OpenArc's `GPU.N` and everyone else's - `device` in
config is a bare informational int/str, not resolved from a probed device.
"""
from __future__ import annotations

from typing import Any

import httpx

from llapdance.core.result import TelemetryResult
from llapdance.plugins.base import TelemetryAdapter
from llapdance.plugins.registry import register


class XmxmonTelemetry(TelemetryAdapter):
    name = "xmxmon"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._default_config = config or {}

    def start(self, config: dict[str, Any]) -> Any:
        # No-op by design (see module docstring) - xmxmon's rolling window
        # already covers "recent activity" without needing an explicit
        # start marker; returning the device here just threads it to stop().
        cfg = {**self._default_config, **config}
        return {"base_url": cfg.get("base_url", "http://localhost:9143"), "device": cfg.get("device", 0)}

    def stop(self, handle: Any) -> TelemetryResult:
        base_url, device = handle["base_url"], handle["device"]
        resp = httpx.get(f"{base_url}/now", timeout=10)
        resp.raise_for_status()
        snapshot = resp.json().get(str(device))
        if snapshot is None:
            raise RuntimeError(f"xmxmon has no device {device!r} (base_url={base_url})")

        metrics: dict[str, Any] = dict(snapshot.get("gauges", {}))
        metrics.update(snapshot.get("rates", {}))
        for d in snapshot.get("derived", []):
            metrics[d["label"]] = d["value"]

        return TelemetryResult(adapter=self.name, metrics=metrics, raw=snapshot)


register("telemetry", XmxmonTelemetry.name, XmxmonTelemetry)
