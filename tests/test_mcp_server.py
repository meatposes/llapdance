"""Unit-level coverage for the MCP tool functions themselves (calling them
directly, not through the wire protocol - the `@server.tool()` decorator
preserves plain callability). The full stdio-protocol round trip (list
tools, call a tool, get structured_content back) was validated by hand
against a live client using the real `mcp` SDK - see VALIDATION.md,
including a real gotcha: list-returning tools wrap their result as
`{"result": [...]}` in `structured_content`, not as a JSON string in
`content[0].text`.
"""
from llapdance.config.models import AdapterRef, BackendConfig, BackendSource, SourceMode, StorageConfig, TestSuite
from llapdance.mcp.server import get_results, get_suite, list_adapters, list_suites, run_suite
from llapdance.plugins import registry
from llapdance.plugins.base import BenchmarkAdapter, CoherenceAdapter
from llapdance.plugins.registry import load_builtin_adapters
from llapdance.core.result import BenchmarkResult, CoherenceResult


class FakeBenchmark(BenchmarkAdapter):
    name = "fake-mcp-benchmark"

    def __init__(self, config=None):
        pass

    def run(self, endpoint, config):
        return BenchmarkResult(adapter=self.name, metrics={"avg_tokens_per_sec": 10.0})


class FakeCoherence(CoherenceAdapter):
    name = "fake-mcp-coherence"

    def __init__(self, config=None):
        pass

    def run(self, endpoint, config):
        return CoherenceResult(adapter=self.name, total=1, passed=1, graded_by_match=1, graded_by_llm_judge=0)


def test_list_adapters_returns_all_registered_kinds():
    load_builtin_adapters()
    result = list_adapters()
    assert set(result.keys()) == {"benchmark", "coherence", "storage", "execution", "engine", "telemetry"}
    assert "generic-http" in result["benchmark"]
    assert "xmxmon" in result["telemetry"]


def test_list_suites_finds_yaml_files(tmp_path):
    (tmp_path / "a.suite.yaml").write_text("name: a\n")
    (tmp_path / "b.suite.yaml").write_text("name: b\n")
    (tmp_path / "not-a-suite.yaml").write_text("name: c\n")

    result = list_suites(str(tmp_path))
    assert result == sorted([str(tmp_path / "a.suite.yaml"), str(tmp_path / "b.suite.yaml")])


def test_get_suite_returns_resolved_config(tmp_path):
    suite_path = tmp_path / "x.suite.yaml"
    suite_path.write_text(
        """
name: test-suite
backends:
  - name: b
    source:
      mode: external
      endpoint: "http://fake:8000"
    model: m
benchmark_adapters: []
storage:
  flat_file_dir: "./results"
"""
    )
    result = get_suite(str(suite_path))
    assert result["name"] == "test-suite"
    assert result["backends"][0]["source"]["mode"] == "external"


def test_run_suite_returns_result_and_delta_info(tmp_path):
    registry.register("benchmark", FakeBenchmark.name, FakeBenchmark)
    registry.register("coherence", FakeCoherence.name, FakeCoherence)

    suite_path = tmp_path / "x.suite.yaml"
    suite_path.write_text(
        f"""
name: test-suite
backends:
  - name: ext-backend
    source:
      mode: external
      endpoint: "http://fake:8000"
    model: m
benchmark_adapters:
  - adapter: {FakeBenchmark.name}
coherence_adapters:
  - adapter: {FakeCoherence.name}
storage:
  flat_file_dir: "{tmp_path / 'results'}"
"""
    )

    outcomes = run_suite(str(suite_path))
    assert len(outcomes) == 1
    assert outcomes[0]["result"]["backend_name"] == "ext-backend"
    assert outcomes[0]["result"]["benchmarks"][0]["metrics"]["avg_tokens_per_sec"] == 10.0
    assert outcomes[0]["delta_against_run_id"] is None  # first run

    second = run_suite(str(suite_path))
    assert second[0]["delta_against_run_id"] == outcomes[0]["result"]["run_id"]


def test_get_results_reads_flat_file_storage(tmp_path):
    registry.register("benchmark", FakeBenchmark.name, FakeBenchmark)
    results_dir = tmp_path / "results"

    suite_path = tmp_path / "x.suite.yaml"
    suite_path.write_text(
        f"""
name: test-suite
backends:
  - name: ext-backend
    source:
      mode: external
      endpoint: "http://fake:8000"
    model: m
benchmark_adapters:
  - adapter: {FakeBenchmark.name}
storage:
  flat_file_dir: "{results_dir}"
"""
    )
    run_suite(str(suite_path))

    history = get_results(str(results_dir), "ext-backend", limit=5)
    assert len(history) == 1
    assert history[0]["backend_name"] == "ext-backend"
