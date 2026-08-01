"""Unit tests mock opensearchpy entirely - the real write+query round-trip
(including the float32-precision timestamp bug this caught) was validated
by hand against a live OpenSearch instance, see VALIDATION.md. These tests
guard the query/mapping shape against regressions without needing a live
OpenSearch for CI.
"""
import sys
import types

import pytest


@pytest.fixture
def fake_opensearchpy(monkeypatch):
    calls = {"index_creates": [], "indexed_docs": [], "searches": []}

    class FakeIndicesClient:
        def __init__(self):
            self._existing = set()

        def exists(self, index):
            return index in self._existing

        def create(self, index, body=None):
            calls["index_creates"].append({"index": index, "body": body})
            self._existing.add(index)

    class FakeOpenSearch:
        def __init__(self, **kwargs):
            self.indices = FakeIndicesClient()
            self._docs = []

        def index(self, index, id, body, refresh=None):
            calls["indexed_docs"].append({"index": index, "id": id, "body": body})
            self._docs.append(body)

        def search(self, index, body):
            calls["searches"].append({"index": index, "body": body})
            backend_name = body["query"]["term"]["backend_name"]
            matches = [d for d in self._docs if d["backend_name"] == backend_name]
            matches.sort(key=lambda d: d["timestamp"], reverse=True)
            size = body.get("size", 10)
            return {"hits": {"hits": [{"_source": d} for d in matches[:size]]}}

    fake_module = types.ModuleType("opensearchpy")
    fake_module.OpenSearch = FakeOpenSearch
    monkeypatch.setitem(sys.modules, "opensearchpy", fake_module)
    return calls


def test_init_creates_index_with_explicit_mapping_when_missing(fake_opensearchpy):
    from llapdance.plugins.storage.opensearch import OpenSearchStorage

    OpenSearchStorage({"index": "my-index"})

    assert fake_opensearchpy["index_creates"] == [
        {
            "index": "my-index",
            "body": {
                "mappings": {
                    "properties": {
                        "run_id": {"type": "keyword"},
                        "backend_name": {"type": "keyword"},
                        "timestamp": {"type": "double"},  # NOT "float" - see module docstring
                    }
                }
            },
        }
    ]


def test_write_and_previous_for_round_trip(fake_opensearchpy):
    from llapdance.core.result import RunResult
    from llapdance.plugins.storage.opensearch import OpenSearchStorage

    storage = OpenSearchStorage({"index": "my-index"})

    r1 = RunResult(backend_name="engine-a", backend_config={}, execution_target={}, device_target={})
    storage.write(r1)
    r2 = RunResult(backend_name="engine-a", backend_config={}, execution_target={}, device_target={})
    r2.timestamp = r1.timestamp + 10
    storage.write(r2)

    prev = storage.previous_for("engine-a", limit=5)
    assert [p.run_id for p in prev] == [r2.run_id, r1.run_id]  # most recent first
    assert storage.previous_for("nonexistent") == []


def test_missing_opensearch_py_raises_clear_error(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "opensearchpy", None)
    from llapdance.plugins.storage.opensearch import OpenSearchStorage

    with pytest.raises(ImportError, match="opensearch-py"):
        OpenSearchStorage({})
