"""Optional OpenSearch storage adapter (SPEC.md §8) - opt-in, not default.
`opensearch-py` is imported lazily inside __init__ (not at module level) so
importing this module during `load_builtin_adapters()` never requires the
dependency for suites that don't configure this adapter - only actually
selecting it does.

Config (AdapterRef.config):
    host, port: OpenSearch endpoint (default localhost:9200)
    username, password: basic auth
    use_ssl (bool, default True), verify_certs (bool, default False - most
        local/dev OpenSearch deployments run self-signed certs; set true for
        anything internet-facing)
    index (str, default "llapdance-results")

Validated against a real local OpenSearch 3.7.0 instance - see VALIDATION.md.
"""
from __future__ import annotations

from typing import Any

from llapdance.core.result import RunResult
from llapdance.plugins.base import StorageAdapter
from llapdance.plugins.registry import register


class OpenSearchStorage(StorageAdapter):
    name = "opensearch"

    def __init__(self, config: dict[str, Any]) -> None:
        try:
            from opensearchpy import OpenSearch
        except ImportError as exc:
            raise ImportError(
                "the opensearch storage adapter requires opensearch-py - install with "
                "`pip install llapdance[opensearch]`"
            ) from exc

        self._index = config.get("index", "llapdance-results")
        self._client = OpenSearch(
            hosts=[{"host": config.get("host", "localhost"), "port": config.get("port", 9200)}],
            http_auth=(config["username"], config["password"]) if "username" in config else None,
            use_ssl=config.get("use_ssl", True),
            verify_certs=config.get("verify_certs", False),
        )
        # opensearch-py 3.x: indices.exists/create require `index=` as a
        # keyword, not positional - confirmed via a real TypeError, not
        # assumed from docs (which show it either way in older examples).
        #
        # GOTCHA, found the hard way: without an explicit mapping,
        # OpenSearch's dynamic mapping guesses `timestamp` as 32-bit
        # "float" (Lucene default for JSON floats), which cannot represent
        # a Unix epoch timestamp (~1.7e9 + fractional seconds) to better
        # than ~2-minute precision - two results written seconds apart
        # rounded to the SAME sort key, silently breaking delta ordering
        # (confirmed via a real write+query round-trip, not assumed from
        # docs). `run_id`/`backend_name` are explicitly `keyword` too,
        # rather than relying on a dynamic `.keyword` sub-field existing.
        if not self._client.indices.exists(index=self._index):
            self._client.indices.create(
                index=self._index,
                body={
                    "mappings": {
                        "properties": {
                            "run_id": {"type": "keyword"},
                            "backend_name": {"type": "keyword"},
                            "timestamp": {"type": "double"},
                        }
                    }
                },
            )

    def write(self, result: RunResult) -> None:
        self._client.index(index=self._index, id=result.run_id, body=result.model_dump(mode="json"), refresh=True)

    def previous_for(self, backend_name: str, limit: int = 1) -> list[RunResult]:
        resp = self._client.search(
            index=self._index,
            body={
                "query": {"term": {"backend_name": backend_name}},  # explicit keyword mapping, see __init__
                "sort": [{"timestamp": {"order": "desc"}}],
                "size": limit,
            },
        )
        return [RunResult.model_validate(hit["_source"]) for hit in resp["hits"]["hits"]]


register("storage", OpenSearchStorage.name, OpenSearchStorage)
