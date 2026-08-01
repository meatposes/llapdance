import httpx
import pytest

from llapdance.plugins.telemetry.xmxmon import XmxmonTelemetry

NOW_RESPONSE = {
    "0": {
        "device": 0,
        "n": 3,
        "gauges": {"GPU_BUSY": 74.2},
        "rates": {"GPU_MEMORY_BYTE_READ": 123456.0},
        "derived": [{"label": "XMX peak", "value": 5.1, "unit": "T/s", "note": ""}],
    }
}


def test_stop_flattens_gauges_rates_and_derived(monkeypatch):
    def fake_get(url, timeout=None):
        assert url == "http://localhost:9143/now"
        return httpx.Response(200, json=NOW_RESPONSE, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)

    adapter = XmxmonTelemetry()
    handle = adapter.start({"device": 0})
    result = adapter.stop(handle)

    assert result.adapter == "xmxmon"
    assert result.metrics["GPU_BUSY"] == 74.2
    assert result.metrics["GPU_MEMORY_BYTE_READ"] == 123456.0
    assert result.metrics["XMX peak"] == 5.1
    assert result.raw == NOW_RESPONSE["0"]


def test_start_threads_custom_base_url_and_device(monkeypatch):
    captured = {}

    def fake_get(url, timeout=None):
        captured["url"] = url
        return httpx.Response(200, json={"1": NOW_RESPONSE["0"]}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)

    adapter = XmxmonTelemetry()
    handle = adapter.start({"base_url": "http://other-host:9143", "device": 1})
    adapter.stop(handle)

    assert captured["url"] == "http://other-host:9143/now"


def test_stop_raises_for_unknown_device(monkeypatch):
    monkeypatch.setattr(
        httpx, "get", lambda url, timeout=None: httpx.Response(200, json={}, request=httpx.Request("GET", url))
    )
    adapter = XmxmonTelemetry()
    handle = adapter.start({"device": 5})
    with pytest.raises(RuntimeError, match="device 5"):
        adapter.stop(handle)
