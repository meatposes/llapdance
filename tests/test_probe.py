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

# Real lspci -nn line shapes (Intel Arc Pro B50, and an ASPEED BMC chip that
# must be excluded structurally, not by name - see VALIDATION.md/screamer).
LSPCI_OUTPUT = (
    "09:00.0 VGA compatible controller [0300]: ASPEED Technology, Inc. ASPEED Graphics Family [1a03:2000] (rev 30)\n"
    "84:00.0 VGA compatible controller [0300]: Intel Corporation Battlemage G21 [Arc Pro B50] [8086:e212]\n"
)


class FakeRunner:
    """Implements the CommandRunner protocol with canned responses, keyed
    by the command's first token - stands in for both LocalRunner and
    SSHRunner in tests, since discovery logic must not care which one it's
    talking to."""

    def __init__(self, responses: dict[str, str]):
        self._responses = responses

    def which(self, tool: str) -> bool:
        return tool in self._responses or tool == "sh"

    def run(self, cmd: list[str]) -> str | None:
        return self._responses.get(cmd[0])


def _no_render_node_runner(**responses) -> FakeRunner:
    # sh (render-node resolution) intentionally returns nothing => render_node None
    return FakeRunner({**responses, "sh": ""})


def test_discover_intel_opencl_classifies_integrated():
    runner = _no_render_node_runner(clinfo=CLINFO_OUTPUT)
    devices = probe._discover_intel_opencl(runner)
    assert len(devices) == 4
    integrated = [d for d in devices if d.integrated]
    discrete = [d for d in devices if not d.integrated]
    assert len(integrated) == 1
    assert integrated[0].name == "Intel(R) Graphics"
    assert len(discrete) == 3


def test_discover_devices_empty_when_no_tools():
    runner = FakeRunner({})
    assert probe.discover_devices(runner) == []


def test_discover_intel_xpumcli_prefers_over_clinfo():
    # if xpumcli succeeds, clinfo should never even be consulted
    runner = _no_render_node_runner(xpumcli=_as_json(XPUMCLI_DISCOVERY), clinfo="SHOULD NOT BE USED")
    devices = probe.discover_devices(runner)
    assert len(devices) == 4
    by_pci = {d.pci_bus_id: d for d in devices}
    assert by_pci["0000:00:02.0"].integrated is True
    assert by_pci["0000:04:00.0"].integrated is False


def test_discover_lspci_fallback_excludes_non_intel_nvidia_vendor():
    runner = _no_render_node_runner(lspci=LSPCI_OUTPUT)
    devices = probe._discover_lspci(runner)
    assert len(devices) == 1  # ASPEED (vendor 1a03) excluded structurally
    assert devices[0].vendor == "intel"
    assert devices[0].pci_bus_id == "0000:84:00.0"
    assert devices[0].integrated is False  # "Arc Pro B50" matches the discrete marketing-name heuristic


def test_free_vram_mb_unknown_for_clinfo_only_device():
    # no pci_bus_id => came from the clinfo fallback, not xpumcli - must
    # fail closed rather than query xpumcli with a mismatched index
    device = probe.DeviceInfo(index=0, vendor="intel", name="x", integrated=False)
    assert probe.free_vram_mb(device, FakeRunner({})) is None


def test_free_vram_mb_uses_xpumcli_when_pci_known():
    runner = FakeRunner({"xpumcli": _as_json({"memory_free_size_byte": 34206572544})})
    device = probe.DeviceInfo(index=3, vendor="intel", name="B70", integrated=False, pci_bus_id="0000:8a:00.0")
    assert probe.free_vram_mb(device, runner) == 34206572544 / (1024 * 1024)


def test_render_node_for_pci_resolves_via_runner():
    runner = FakeRunner({"sh": "renderD131 0000:8a:00.0\nrenderD129 0000:04:00.0\n"})
    assert probe._render_node_for_pci("0000:8a:00.0", runner) == "/dev/dri/renderD131"
    assert probe._render_node_for_pci("0000:99:99.9", runner) is None


def test_ssh_runner_builds_expected_command(monkeypatch):
    captured = {}

    class FakeCompleted:
        stdout = "ok\n"

    def fake_run(args, **kwargs):
        captured["args"] = args
        return FakeCompleted()

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    runner = probe.SSHRunner(host="screamer", user="nullraptor", ssh_key_path="/home/x/.ssh/id_nullraptor")
    result = runner.run(["clinfo", "-l"])

    assert result == "ok\n"
    assert captured["args"][:2] == ["ssh", "-o"]
    assert "-i" in captured["args"] and "/home/x/.ssh/id_nullraptor" in captured["args"]
    assert captured["args"][-2] == "nullraptor@screamer"
    assert captured["args"][-1] == "clinfo -l"


def _as_json(obj) -> str:
    import json

    return json.dumps(obj)
