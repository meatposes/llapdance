from typing import Any

from llapdance.config.models import (
    BackendConfig,
    BackendSource,
    DeviceTarget,
    ExecutionTargetConfig,
    SourceMode,
    StorageConfig,
    TestSuite,
    AdapterRef,
)
from llapdance.core import orchestrator
from llapdance.core.result import BenchmarkResult, CoherenceResult, RunResult
from llapdance.plugins.base import BenchmarkAdapter, CoherenceAdapter, ExecutionTargetAdapter, RunningBackend
from llapdance.plugins import registry


class FakeRunningBackend(RunningBackend):
    @property
    def endpoint(self) -> str:
        return "http://fake:8000"

    def logs(self, tail: int = 200) -> str:
        return ""


class FakeExecutionTarget(ExecutionTargetAdapter):
    name = "fake-execution"
    started_with: list[Any] = []

    def __init__(self, config=None):
        pass

    def build(self, backend_config):
        return "fake-image:latest"

    def start(self, backend_config, image_ref, device_indices):
        FakeExecutionTarget.started_with.append(device_indices)
        return FakeRunningBackend()

    def stop(self, backend):
        pass

    def list_images(self, name_filter=None):
        return []

    def remove_image(self, image_ref):
        pass


class FakeBenchmark(BenchmarkAdapter):
    name = "fake-benchmark"

    def __init__(self, config=None):
        pass

    def run(self, endpoint, config):
        return BenchmarkResult(adapter=self.name, metrics={"avg_tokens_per_sec": 42.0})


class FakeCoherence(CoherenceAdapter):
    name = "fake-coherence"

    def __init__(self, config=None):
        pass

    def run(self, endpoint, config):
        return CoherenceResult(adapter=self.name, total=10, passed=9, graded_by_match=9, graded_by_llm_judge=0)


def _register_fakes():
    registry.register("execution", FakeExecutionTarget.name, FakeExecutionTarget)
    registry.register("benchmark", FakeBenchmark.name, FakeBenchmark)
    registry.register("coherence", FakeCoherence.name, FakeCoherence)


def _suite(tmp_path, **overrides) -> TestSuite:
    base = dict(
        name="test-suite",
        backends=[
            BackendConfig(
                name="engine-a",
                source=BackendSource(mode=SourceMode.prebuilt, image="x:y"),
                model="m",
            )
        ],
        device_target=DeviceTarget(mode="none"),
        execution_target=ExecutionTargetConfig(mode="local"),
        benchmark_adapters=[AdapterRef(adapter="fake-benchmark", config={})],
        coherence_adapters=[AdapterRef(adapter="fake-coherence", config={})],
        storage=StorageConfig(flat_file_dir=str(tmp_path)),
    )
    base.update(overrides)
    return TestSuite(**base)


def test_run_backend_end_to_end(tmp_path, monkeypatch):
    # _wait_until_ready is a real network poll (proven necessary against a
    # real llama.cpp container - see VALIDATION.md); the fake backend below
    # has no real endpoint to poll, so bypass it here rather than let it
    # burn through startup_timeout_s trying to resolve a fake hostname.
    monkeypatch.setattr(orchestrator, "_wait_until_ready", lambda *a, **k: None)
    _register_fakes()
    suite = _suite(tmp_path)
    orig_get = registry.get

    def get_with_fake_execution(kind, name):
        if kind == "execution":
            return FakeExecutionTarget
        return orig_get(kind, name)

    monkeypatch.setattr(orchestrator.registry, "get", get_with_fake_execution)

    outcome = orchestrator.run_backend(suite, suite.backends[0])

    assert outcome.result.backend_name == "engine-a"
    assert outcome.result.benchmarks[0].metrics["avg_tokens_per_sec"] == 42.0
    assert outcome.result.coherence[0].passed == 9
    assert outcome.delta_against is None  # first run, nothing to diff against

    # second run should see the first as its delta target
    second = orchestrator.run_backend(suite, suite.backends[0])
    assert second.delta_against is not None
    assert second.delta_against.run_id == outcome.result.run_id


def test_run_backend_skips_device_probe_when_mode_none(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "_wait_until_ready", lambda *a, **k: None)
    _register_fakes()
    orig_get = registry.get

    def get_with_fake_execution(kind, name):
        if kind == "execution":
            return FakeExecutionTarget
        return orig_get(kind, name)

    monkeypatch.setattr(orchestrator.registry, "get", get_with_fake_execution)
    FakeExecutionTarget.started_with.clear()

    suite = _suite(tmp_path)
    orchestrator.run_backend(suite, suite.backends[0])

    assert FakeExecutionTarget.started_with == [[]]
