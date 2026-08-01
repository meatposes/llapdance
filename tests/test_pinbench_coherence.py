import json
import subprocess
from pathlib import Path

import pytest

from llapdance.plugins.coherence.pinbench import PinBenchCoherence


def _fake_run_benchmark(summary: dict, results: list[dict]):
    """Returns a subprocess.run stand-in that writes the same real output
    shape PinBench's run_benchmark.py produces (<output_dir>/<run_id>/...
    - run_id is a runtime timestamp, so the adapter must discover it, not
    assume a fixed name - this fixture uses an arbitrary name on purpose)."""

    def fake_run(cmd, cwd, capture_output, text, timeout):
        config_path = Path(cmd[cmd.index("-c") + 1])
        run_config = config_path.read_text()
        assert "base_url:" in run_config  # real config was actually written

        import yaml

        parsed = yaml.safe_load(run_config)
        output_dir = Path(parsed["output_dir"])
        run_dir = output_dir / "20260101_000000"
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(json.dumps({"summary": summary}))
        (run_dir / "results.json").write_text(json.dumps({"results": results}))
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    return fake_run


def test_requires_pinbench_dir():
    adapter = PinBenchCoherence({})
    with pytest.raises(ValueError, match="pinbench_dir"):
        adapter.run("http://fake", {})


def test_base_url_gets_v1_appended(monkeypatch):
    seen = {}

    def fake_run(cmd, cwd, capture_output, text, timeout):
        import yaml

        config_path = Path(cmd[cmd.index("-c") + 1])
        parsed = yaml.safe_load(config_path.read_text())
        seen["base_url"] = parsed["providers"][0]["base_url"]
        output_dir = Path(parsed["output_dir"])
        run_dir = output_dir / "run1"
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(json.dumps({"summary": {"count": 0, "passed": 0}}))
        (run_dir / "results.json").write_text(json.dumps({"results": []}))
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    adapter = PinBenchCoherence({"pinbench_dir": "/fake/pinbench", "model": "m"})
    adapter.run("http://127.0.0.1:8001", {})
    assert seen["base_url"] == "http://127.0.0.1:8001/v1"


def test_real_result_shape_and_failures_extracted(monkeypatch):
    summary = {"count": 3, "passed": 2, "pass_rate": 0.667}
    results = [
        {"test_id": 1, "test_title": "a", "category": "basic", "passed": True, "score": 1.0, "details": {}, "errors": []},
        {"test_id": 2, "test_title": "b", "category": "basic", "passed": True, "score": 1.0, "details": {}, "errors": []},
        {
            "test_id": 3,
            "test_title": "c",
            "category": "english_names",
            "passed": False,
            "score": 0.3,
            "details": {"table_content": False},
            "errors": ["mismatch"],
        },
    ]
    monkeypatch.setattr(subprocess, "run", _fake_run_benchmark(summary, results))
    adapter = PinBenchCoherence({"pinbench_dir": "/fake/pinbench", "model": "m"})
    result = adapter.run("http://127.0.0.1:8001", {})

    assert result.adapter == "pinbench"
    assert result.total == 3
    assert result.passed == 2
    assert result.graded_by_match == 2
    assert result.graded_by_llm_judge == 0
    assert len(result.failures) == 1
    assert result.failures[0]["test_id"] == 3
    assert result.failures[0]["score"] == 0.3


def test_raises_with_stdout_stderr_on_nonzero_exit(monkeypatch):
    def fake_run(cmd, cwd, capture_output, text, timeout):
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="out", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    adapter = PinBenchCoherence({"pinbench_dir": "/fake/pinbench", "model": "m"})
    with pytest.raises(RuntimeError, match="boom"):
        adapter.run("http://127.0.0.1:8001", {})


def test_filter_test_ids_passed_through_to_config(monkeypatch):
    seen = {}

    def fake_run(cmd, cwd, capture_output, text, timeout):
        import yaml

        config_path = Path(cmd[cmd.index("-c") + 1])
        parsed = yaml.safe_load(config_path.read_text())
        seen["filter_test_ids"] = parsed.get("filter_test_ids")
        output_dir = Path(parsed["output_dir"])
        run_dir = output_dir / "run1"
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(json.dumps({"summary": {"count": 0, "passed": 0}}))
        (run_dir / "results.json").write_text(json.dumps({"results": []}))
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    adapter = PinBenchCoherence({"pinbench_dir": "/fake/pinbench", "model": "m"})
    adapter.run("http://127.0.0.1:8001", {"filter_test_ids": [5, 6]})
    assert seen["filter_test_ids"] == [5, 6]
