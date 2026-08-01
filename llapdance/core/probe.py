"""Vendor-agnostic hardware discovery (SPEC.md §7).

Nothing here hardcodes a device count/topology - it shells out to whatever
vendor tooling is present and classifies what it finds. If a device can't be
classified as integrated-vs-discrete, or free memory can't be determined,
callers must fail closed rather than guess (SPEC.md §7 hard requirement).

GOTCHA, confirmed on real hardware (see VALIDATION.md): the same 4 physical
GPUs are enumerated in at least FOUR different, non-corresponding index
spaces depending on which tool you ask - `clinfo` (OpenCL) order, `xpumcli`'s
own device_id, llama.cpp's SYCL/level-zero index, and the kernel's DRM
card/render node number. The only thing that reliably ties them together
across tools is the PCI bus address (`pci_bus_id` below) - that is why it's
treated as the canonical identity here, with `index` being whichever
enumeration produced the DeviceInfo (documented per-source, not assumed
comparable across sources).
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DeviceInfo:
    index: int
    vendor: str
    name: str
    integrated: bool
    pci_bus_id: str | None = None
    render_node: str | None = None


def _run(cmd: list[str]) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except (subprocess.SubprocessError, OSError):
        return None


def _run_json(cmd: list[str]):
    import json

    out = _run(cmd)
    if not out:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


_INTEGRATED_NAME_RE = re.compile(r"^Intel\(R\) Graphics$")


def _render_node_for_pci(pci_bus_id: str) -> str | None:
    """Resolve a PCI bus address to its DRM render node via sysfs - no
    vendor tool needed, works for any DRM-backed GPU. `pci_bus_id` is
    expected in `0000:04:00.0` form (as reported by xpumcli/lspci)."""
    drm_dir = Path("/sys/class/drm")
    if not drm_dir.is_dir():
        return None
    for entry in drm_dir.glob("renderD*"):
        device_link = entry / "device"
        try:
            resolved = device_link.resolve()
        except OSError:
            continue
        if resolved.name == pci_bus_id:
            return f"/dev/dri/{entry.name}"
    return None


def discover_devices() -> list[DeviceInfo]:
    """Best-effort enumeration across vendors. Extend per-vendor as new
    hardware is validated (SPEC.md §15.2) - this must never be the place a
    specific machine's GPU count gets hardcoded.

    Intel: prefers `xpumcli` (gives PCI bus id + free/total VRAM in one call,
    see free_vram_mb below) and falls back to `clinfo` (enumeration only, no
    VRAM reporting) when xpumcli isn't installed.
    """
    devices: list[DeviceInfo] = []
    intel_devices = _discover_intel_xpumcli()
    devices.extend(intel_devices if intel_devices else _discover_intel_opencl())
    devices.extend(_discover_nvidia())
    return devices


def _discover_intel_xpumcli() -> list[DeviceInfo]:
    data = _run_json(["xpumcli", "discovery", "-j"])
    if not data:
        return []
    devices: list[DeviceInfo] = []
    for entry in data.get("device_list", []):
        name = entry.get("device_name", "")
        pci_bus_id = entry.get("pci_bdf_address")
        devices.append(
            DeviceInfo(
                index=int(entry["device_id"]),
                vendor="intel",
                name=name,
                integrated=bool(_INTEGRATED_NAME_RE.match(name)),
                pci_bus_id=pci_bus_id,
                render_node=_render_node_for_pci(pci_bus_id) if pci_bus_id else None,
            )
        )
    return devices


def _discover_intel_opencl() -> list[DeviceInfo]:
    """Fallback when xpumcli isn't installed: enumeration only, via clinfo.
    No PCI bus id / render node / VRAM reporting - callers relying on those
    will fail closed, which is correct (see free_vram_mb)."""
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
    if device.vendor == "intel":
        # device.index is the xpumcli device_id only when discovery went
        # through _discover_intel_xpumcli - if the clinfo fallback populated
        # this DeviceInfo instead, index means something else and querying
        # xpumcli with it would silently ask about the wrong physical card.
        # pci_bus_id is only ever set by the xpumcli path, so its presence
        # is exactly the signal that index is safe to use here.
        if device.pci_bus_id is None:
            return None
        data = _run_json(["xpumcli", "discovery", "-d", str(device.index), "-j"])
        if not data or "memory_free_size_byte" not in data:
            return None
        try:
            return float(data["memory_free_size_byte"]) / (1024 * 1024)
        except (TypeError, ValueError):
            return None
    return None
