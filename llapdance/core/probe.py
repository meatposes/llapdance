"""Vendor-agnostic hardware discovery (SPEC.md §7).

Nothing here hardcodes a device count/topology - it shells out to whatever
vendor tooling is present and classifies what it finds. If a device can't be
classified as integrated-vs-discrete, or free memory can't be determined,
callers must fail closed rather than guess (SPEC.md §7 hard requirement).
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class DeviceInfo:
    index: int
    vendor: str
    name: str
    integrated: bool
    pci_bus_id: str | None = None


def _run(cmd: list[str]) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except (subprocess.SubprocessError, OSError):
        return None


_INTEGRATED_NAME_RE = re.compile(r"^Intel\(R\) Graphics$")


def discover_devices() -> list[DeviceInfo]:
    """Best-effort enumeration across vendors. Extend per-vendor as new
    hardware is validated (SPEC.md §15.2) - this must never be the place a
    specific machine's GPU count gets hardcoded."""
    devices: list[DeviceInfo] = []
    devices.extend(_discover_intel_opencl())
    devices.extend(_discover_nvidia())
    return devices


def _discover_intel_opencl() -> list[DeviceInfo]:
    out = _run(["clinfo", "-l"])
    if not out:
        return []
    devices: list[DeviceInfo] = []
    index = 0
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith(("+-- Device", "`-- Device")):
            continue
        name = line.split("Device #", 1)[1].split(":", 1)[1].strip()
        devices.append(
            DeviceInfo(
                index=index,
                vendor="intel",
                name=name,
                integrated=bool(_INTEGRATED_NAME_RE.match(name)),
            )
        )
        index += 1
    return devices


def _discover_nvidia() -> list[DeviceInfo]:
    out = _run(["nvidia-smi", "-L"])
    if not out:
        return []
    devices: list[DeviceInfo] = []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("GPU "):
            continue
        idx_str, rest = line.split(":", 1)
        index = int(idx_str.replace("GPU", "").strip())
        # NVIDIA data-center/workstation cards are discrete by construction;
        # there is no NVIDIA "integrated" case to special-case here.
        devices.append(DeviceInfo(index=index, vendor="nvidia", name=rest.strip(), integrated=False))
    return devices


def discrete_devices() -> list[DeviceInfo]:
    return [d for d in discover_devices() if not d.integrated]


def free_vram_mb(device: DeviceInfo) -> float | None:
    """Returns None when free VRAM can't be determined for this vendor/device
    - callers MUST treat None as "unknown, refuse to run" per SPEC.md §7,
    not as "assume it's fine"."""
    if device.vendor == "nvidia":
        out = _run(["nvidia-smi", f"--id={device.index}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        if out and out.strip():
            try:
                return float(out.strip().splitlines()[0])
            except ValueError:
                return None
        return None
    # Intel free-VRAM reporting needs xpu-smi or a sysfs path resolved per
    # card; not resolved yet (SPEC.md §15.2) - fail closed rather than guess.
    return None
