"""Vendor-agnostic hardware discovery (SPEC.md §7).

Nothing here hardcodes a device count/topology - it shells out to whatever
vendor tooling is present and classifies what it finds. If a device can't be
classified as integrated-vs-discrete, or free memory can't be determined,
callers must fail closed rather than guess (SPEC.md §7 hard requirement).

GOTCHA, confirmed on real hardware (see VALIDATION.md): the same physical
GPUs can be enumerated in different, non-corresponding index spaces
depending on which tool you ask - `clinfo` (OpenCL) order, `xpumcli`'s own
device_id, an engine's own SYCL/level-zero index, the kernel's DRM
card/render node number, and (found integrating OpenArc) OpenVINO's own
`GPU.N` naming. The only thing that reliably ties them together across
tools is the PCI bus address (`pci_bus_id` below) - that is why it's
treated as the canonical identity here, with `index` being whichever
enumeration produced the DeviceInfo (documented per-source, not assumed
comparable across sources).

Every probing function here takes an optional `runner` (default: local
subprocess) - this is what makes hardware discovery work identically for
a local execution target and a remote one over SSH (SPEC.md §5, §7: "probing
happens against whichever execution target is active"). A host with no
compute-runtime tooling installed at all (found on a real remote box -
`clinfo` reports 0 platforms, no `xpumcli`) falls back to `lspci`, which
gives vendor/model/PCI-bus-id identification (enough for GPU tracking) but
never free-VRAM (fail-closed applies exactly as it does locally).
"""
from __future__ import annotations

import json as _json
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class DeviceInfo:
    index: int
    vendor: str
    name: str
    integrated: bool
    pci_bus_id: str | None = None
    render_node: str | None = None


class CommandRunner(Protocol):
    """Where a probing command actually executes - local subprocess, or a
    remote host over SSH. Both raise nothing; a command that can't run
    (tool missing, host unreachable) just returns None, same contract as
    the old bare `_run()` had locally."""

    def run(self, cmd: list[str]) -> str | None: ...
    def which(self, tool: str) -> bool: ...


class LocalRunner:
    def which(self, tool: str) -> bool:
        return shutil.which(tool) is not None

    def run(self, cmd: list[str]) -> str | None:
        if not self.which(cmd[0]):
            return None
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
        except (subprocess.SubprocessError, OSError):
            return None


class SSHRunner:
    """Runs probing commands on a remote host over SSH, using an explicit
    identity file rather than relying on ssh-agent/~/.ssh/config state,
    which this process has no guarantee persists between invocations."""

    def __init__(self, host: str, user: str | None = None, ssh_key_path: str | None = None) -> None:
        self._target = f"{user}@{host}" if user else host
        self._ssh_key_path = ssh_key_path
        self._which_cache: dict[str, bool] = {}

    def _ssh_base(self) -> list[str]:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
        if self._ssh_key_path:
            cmd += ["-i", self._ssh_key_path]
        return cmd + [self._target]

    def which(self, tool: str) -> bool:
        if tool not in self._which_cache:
            out = self.run(["sh", "-c", f"command -v {shlex.quote(tool)}"])
            self._which_cache[tool] = bool(out and out.strip())
        return self._which_cache[tool]

    def run(self, cmd: list[str]) -> str | None:
        try:
            resp = subprocess.run(
                self._ssh_base() + [shlex.join(cmd)], capture_output=True, text=True, timeout=20
            )
            return resp.stdout
        except (subprocess.SubprocessError, OSError):
            return None


_DEFAULT_RUNNER = LocalRunner()


def _run(cmd: list[str], runner: CommandRunner) -> str | None:
    if not runner.which(cmd[0]):
        return None
    return runner.run(cmd)


def _run_json(cmd: list[str], runner: CommandRunner):
    out = _run(cmd, runner)
    if not out:
        return None
    try:
        return _json.loads(out)
    except _json.JSONDecodeError:
        return None


_INTEGRATED_NAME_RE = re.compile(r"^Intel\(R\) Graphics$")
_DISCRETE_MARKETING_NAME_RE = re.compile(r"\bArc\b|\bData Center GPU\b|\bFlex\b", re.IGNORECASE)


def _render_node_for_pci(pci_bus_id: str, runner: CommandRunner) -> str | None:
    """Resolve a PCI bus address to its DRM render node via sysfs - no
    vendor tool needed, works for any DRM-backed GPU, local or remote
    (the resolution itself runs through `runner`, so this works
    identically over SSH)."""
    out = runner.run(
        ["sh", "-c", "for f in /sys/class/drm/renderD*; do echo \"$(basename $f) $(basename $(readlink -f $f/device))\"; done"]
    )
    if not out:
        return None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == pci_bus_id:
            return f"/dev/dri/{parts[0]}"
    return None


def discover_devices(runner: CommandRunner | None = None) -> list[DeviceInfo]:
    """Best-effort enumeration across vendors. Extend per-vendor as new
    hardware is validated (SPEC.md §15.2) - this must never be the place a
    specific machine's GPU count gets hardcoded.

    Tries, in order, for Intel: `xpumcli` (PCI bus id + free/total VRAM in
    one call) -> `clinfo` (enumeration only) -> `lspci` (identification
    only, no VRAM, for hosts with no compute-runtime tooling installed at
    all - confirmed necessary on a real remote box, see VALIDATION.md).
    """
    runner = runner or _DEFAULT_RUNNER
    devices: list[DeviceInfo] = []
    intel_devices = _discover_intel_xpumcli(runner)
    if not intel_devices:
        intel_devices = _discover_intel_opencl(runner)
    if not intel_devices:
        intel_devices = _discover_lspci(runner)
    devices.extend(intel_devices)
    devices.extend(_discover_nvidia(runner))
    return devices


def _discover_intel_xpumcli(runner: CommandRunner) -> list[DeviceInfo]:
    data = _run_json(["xpumcli", "discovery", "-j"], runner)
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
                render_node=_render_node_for_pci(pci_bus_id, runner) if pci_bus_id else None,
            )
        )
    return devices


def _discover_intel_opencl(runner: CommandRunner) -> list[DeviceInfo]:
    """Fallback when xpumcli isn't installed: enumeration only, via clinfo.
    No PCI bus id / render node / VRAM reporting - callers relying on those
    will fail closed, which is correct (see free_vram_mb)."""
    out = _run(["clinfo", "-l"], runner)
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


# PCI vendor IDs, from the PCI-SIG registry - structural filter (any device
# from any OTHER vendor, e.g. server BMC graphics like ASPEED/Matrox, is
# never a usable compute GPU and is excluded by this filter, not by name
# matching a specific chip).
_PCI_VENDOR_INTEL = "8086"
_PCI_VENDOR_NVIDIA = "10de"


def _discover_lspci(runner: CommandRunner) -> list[DeviceInfo]:
    """Last-resort fallback for a host with no compute-runtime tooling
    installed at all (confirmed necessary on a real remote box - clinfo
    reports 0 platforms, no xpumcli binary). Identification only - no
    free-VRAM reporting is possible this way, so free_vram_mb() will
    correctly return None for anything discovered only via this path."""
    out = _run(["lspci", "-nn"], runner)
    if not out:
        return []
    devices: list[DeviceInfo] = []
    index = 0
    for line in out.splitlines():
        if "VGA compatible controller" not in line and "3D controller" not in line:
            continue
        match = re.search(r"^(\S+) .*?\[([0-9a-f]{4}):([0-9a-f]{4})\]", line)
        if not match:
            continue
        pci_slot, vendor_id, _device_id = match.groups()
        if vendor_id not in (_PCI_VENDOR_INTEL, _PCI_VENDOR_NVIDIA):
            continue
        vendor = "intel" if vendor_id == _PCI_VENDOR_INTEL else "nvidia"
        name = line.split(": ", 1)[-1] if ": " in line else line
        pci_bus_id = pci_slot if re.match(r"^[0-9a-f]{4}:", pci_slot) else f"0000:{pci_slot}"
        # Best-effort only (see module docstring): NVIDIA has no integrated
        # case; for Intel, lack of a discrete marketing name (Arc/Data
        # Center GPU/Flex) is the only signal available without xpumcli/
        # clinfo - recommend installing one of those for reliable
        # classification rather than trusting this heuristic long-term.
        integrated = vendor == "intel" and not _DISCRETE_MARKETING_NAME_RE.search(name)
        devices.append(
            DeviceInfo(
                index=index,
                vendor=vendor,
                name=name,
                integrated=integrated,
                pci_bus_id=pci_bus_id,
                render_node=_render_node_for_pci(pci_bus_id, runner),
            )
        )
        index += 1
    return devices


def _discover_nvidia(runner: CommandRunner) -> list[DeviceInfo]:
    out = _run(["nvidia-smi", "-L"], runner)
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


def discrete_devices(runner: CommandRunner | None = None) -> list[DeviceInfo]:
    return [d for d in discover_devices(runner) if not d.integrated]


def free_vram_mb(device: DeviceInfo, runner: CommandRunner | None = None) -> float | None:
    """Returns None when free VRAM can't be determined for this vendor/device
    - callers MUST treat None as "unknown, refuse to run" per SPEC.md §7,
    not as "assume it's fine"."""
    runner = runner or _DEFAULT_RUNNER
    if device.vendor == "nvidia":
        out = _run(
            ["nvidia-smi", f"--id={device.index}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], runner
        )
        if out and out.strip():
            try:
                return float(out.strip().splitlines()[0])
            except ValueError:
                return None
        return None
    if device.vendor == "intel":
        # device.index is the xpumcli device_id only when discovery went
        # through _discover_intel_xpumcli - if the clinfo/lspci fallback
        # populated this DeviceInfo instead, index means something else and
        # querying xpumcli with it would silently ask about the wrong
        # physical card. pci_bus_id combined with a *successful* xpumcli
        # call is the actual signal that index is safe to use here (lspci
        # also sets pci_bus_id, but never has xpumcli to query in the first
        # place, so this still correctly falls through to None below).
        if device.pci_bus_id is None:
            return None
        data = _run_json(["xpumcli", "discovery", "-d", str(device.index), "-j"], runner)
        if not data or "memory_free_size_byte" not in data:
            return None
        try:
            return float(data["memory_free_size_byte"]) / (1024 * 1024)
        except (TypeError, ValueError):
            return None
    return None
