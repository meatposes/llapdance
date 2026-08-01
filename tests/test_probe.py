from llapdance.core import probe


CLINFO_OUTPUT = """\
Platform #0: Intel(R) OpenCL Graphics
 +-- Device #0: Intel(R) Arc(TM) Pro B70 Graphics
 +-- Device #1: Intel(R) Arc(TM) Pro B50 Graphics
 `-- Device #2: Intel(R) Arc(TM) Pro B70 Graphics
Platform #1: Intel(R) OpenCL Graphics
 `-- Device #0: Intel(R) Graphics
"""

XPUMCLI_DISCOVERY = {
    "device_list": [
        {"device_id": 1, "device_name": "Intel(R) Arc(TM) Pro B70 Graphics", "pci_bdf_address": "0000:04:00.0"},
        {"device_id": 2, "device_name": "Intel(R) Arc(TM) Pro B50 Graphics", "pci_bdf_address": "0000:84:00.0"},
        {"device_id": 3, "device_name": "Intel(R) Arc(TM) Pro B70 Graphics", "pci_bdf_address": "0000:8a:00.0"},
        {"device_id": 4, "device_name": "Intel(R) Graphics", "pci_bdf_address": "0000:00:02.0"},
    ]
}


def test_discover_intel_opencl_classifies_integrated(monkeypatch):
    def fake_run(cmd):
        if cmd[0] == "clinfo":
            return CLINFO_OUTPUT
        return None

    monkeypatch.setattr(probe, "_run", fake_run)
    devices = probe._discover_intel_opencl()
    assert len(devices) == 4
    integrated = [d for d in devices if d.integrated]
    discrete = [d for d in devices if not d.integrated]
    assert len(integrated) == 1
    assert integrated[0].name == "Intel(R) Graphics"
    assert len(discrete) == 3


def test_discover_devices_empty_when_no_tools(monkeypatch):
    monkeypatch.setattr(probe, "_run", lambda cmd: None)
    monkeypatch.setattr(probe, "_run_json", lambda cmd: None)
    assert probe.discover_devices() == []


def test_discover_intel_xpumcli_prefers_over_clinfo(monkeypatch):
    # if xpumcli succeeds, clinfo should never even be tried
    monkeypatch.setattr(probe, "_run_json", lambda cmd: XPUMCLI_DISCOVERY if cmd[0] == "xpumcli" else None)
    monkeypatch.setattr(probe, "_render_node_for_pci", lambda pci: f"/dev/dri/renderD_for_{pci}")
    def fail_on_clinfo(cmd):
        if cmd[0] == "clinfo":
            raise AssertionError("clinfo should not run when xpumcli already succeeded")
        return None  # e.g. nvidia-smi, legitimately probed too, just absent here

    monkeypatch.setattr(probe, "_run", fail_on_clinfo)

    devices = probe.discover_devices()
    assert len(devices) == 4
    by_pci = {d.pci_bus_id: d for d in devices}
    assert by_pci["0000:00:02.0"].integrated is True
    assert by_pci["0000:04:00.0"].integrated is False
    assert by_pci["0000:04:00.0"].render_node == "/dev/dri/renderD_for_0000:04:00.0"


def test_free_vram_mb_unknown_for_clinfo_only_device():
    # no pci_bus_id => came from the clinfo fallback, not xpumcli - must
    # fail closed rather than query xpumcli with a mismatched index
    device = probe.DeviceInfo(index=0, vendor="intel", name="x", integrated=False)
    assert probe.free_vram_mb(device) is None


def test_free_vram_mb_uses_xpumcli_when_pci_known(monkeypatch):
    monkeypatch.setattr(
        probe,
        "_run_json",
        lambda cmd: {"memory_free_size_byte": 34206572544} if "discovery" in cmd else None,
    )
    device = probe.DeviceInfo(index=3, vendor="intel", name="B70", integrated=False, pci_bus_id="0000:8a:00.0")
    free = probe.free_vram_mb(device)
    assert free == 34206572544 / (1024 * 1024)


def test_render_node_for_pci_resolves_via_sysfs(tmp_path, monkeypatch):
    from pathlib import Path as RealPath

    drm_dir = tmp_path / "drm"
    render_dir = drm_dir / "renderD131"
    render_dir.mkdir(parents=True)
    pci_dir = tmp_path / "0000:8a:00.0"
    pci_dir.mkdir()
    (render_dir / "device").symlink_to(pci_dir)

    monkeypatch.setattr(probe, "Path", lambda p: drm_dir if p == "/sys/class/drm" else RealPath(p))
    result = probe._render_node_for_pci("0000:8a:00.0")
    assert result == "/dev/dri/renderD131"
