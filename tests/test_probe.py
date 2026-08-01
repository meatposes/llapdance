from llapdance.core import probe


CLINFO_OUTPUT = """\
Platform #0: Intel(R) OpenCL Graphics
 +-- Device #0: Intel(R) Arc(TM) Pro B70 Graphics
 +-- Device #1: Intel(R) Arc(TM) Pro B50 Graphics
 `-- Device #2: Intel(R) Arc(TM) Pro B70 Graphics
Platform #1: Intel(R) OpenCL Graphics
 `-- Device #0: Intel(R) Graphics
"""


def test_discover_intel_opencl_classifies_integrated(monkeypatch):
    def fake_run(cmd):
        if cmd[0] == "clinfo":
            return CLINFO_OUTPUT
        return None

    monkeypatch.setattr(probe, "_run", fake_run)
    devices = probe.discover_devices()
    assert len(devices) == 4
    integrated = [d for d in devices if d.integrated]
    discrete = [d for d in devices if not d.integrated]
    assert len(integrated) == 1
    assert integrated[0].name == "Intel(R) Graphics"
    assert len(discrete) == 3


def test_discover_devices_empty_when_no_tools(monkeypatch):
    monkeypatch.setattr(probe, "_run", lambda cmd: None)
    assert probe.discover_devices() == []


def test_free_vram_mb_unknown_for_intel_fails_closed(monkeypatch):
    monkeypatch.setattr(probe, "_run", lambda cmd: None)
    device = probe.DeviceInfo(index=0, vendor="intel", name="x", integrated=False)
    assert probe.free_vram_mb(device) is None
