import httpx
import pytest

from llapdance.core.orchestrator import StartupTimeoutError, _wait_until_ready


def test_wait_until_ready_returns_on_200(monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        return httpx.Response(200, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    _wait_until_ready("http://example:8080/", "/health", timeout_s=5)
    assert calls == ["http://example:8080/health"]


def test_wait_until_ready_times_out_when_never_healthy(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda url, timeout: httpx.Response(503, request=httpx.Request("GET", url)))
    monkeypatch.setattr("time.sleep", lambda s: None)  # don't actually wait in the test
    with pytest.raises(StartupTimeoutError):
        _wait_until_ready("http://example:8080", "/health", timeout_s=0.01)


def test_wait_until_ready_survives_connection_errors_until_deadline(monkeypatch):
    def raise_conn_error(url, timeout):
        raise httpx.ConnectError("refused", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", raise_conn_error)
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(StartupTimeoutError, match="refused"):
        _wait_until_ready("http://example:8080", "/health", timeout_s=0.01)
