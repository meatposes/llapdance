"""PinBench coherence adapter (see SPEC.md's "prior art" table) - a real
external tool (https://github.com/ShadyHippo/PinBench), NOT vendored into
this repo. Requires a local checkout, given via `pinbench_dir`. Shells out
to its own `run_benchmark.py` - its provider/grading code (pinyin<->hanzi
matching, weighted structured-output criteria) is real, non-trivial domain
logic; reimplementing it natively here would just be a worse copy.

Real interface, confirmed by reading providers.py/runner.py/grader.py in a
real clone and validating live against a real running backend
(llama-cpp-bonsai's Ternary-Bonsai-27B-Q2_0.gguf, see VALIDATION.md) - not
guessed from the README alone:
  - `RunConfig.providers` entries are plain dicts; `type: "vllm"` (despite
    the name) is `OpenAICompatibleProvider` underneath - a bare
    `openai.OpenAI(api_key=..., base_url=...)` client, so it works against
    ANY OpenAI-compatible endpoint, which is exactly what every engine
    translator in this harness exposes.
  - `base_url` must include the trailing `/v1` (PinBench's own example
    configs do this) - this harness's `endpoint` values never do, so it's
    appended here.
  - Real output lands at `<output_dir>/<run_id>/summary.json` (aggregate
    counts/pass-rate) and `results.json` (per-test `GradingResult`, used
    here for the `failures` list) - `run_id` is a timestamp generated at
    run time, not knowable in advance, so this reads back whatever
    subdirectory actually got created.

Needs the `pinbench` extra (`pip install llapdance[pinbench]`) - PinBench's
own providers.py imports `openai` directly. `python_bin` defaults to this
same interpreter (`sys.executable`), since installing the extra puts the
dependency there; point it elsewhere if PinBench has its own venv.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from llapdance.core.result import CoherenceResult
from llapdance.plugins.base import CoherenceAdapter
from llapdance.plugins.registry import register


class PinBenchCoherence(CoherenceAdapter):
    name = "pinbench"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def run(self, endpoint: str, config: dict[str, Any]) -> CoherenceResult:
        cfg = {**self._config, **config}
        pinbench_dir = cfg.get("pinbench_dir")
        if not pinbench_dir:
            raise ValueError(
                "pinbench coherence adapter requires 'pinbench_dir' - a local checkout of "
                "https://github.com/ShadyHippo/PinBench (external tool, not vendored here)."
            )
        model = cfg.get("model", "default")
        python_bin = cfg.get("python_bin", sys.executable)

        run_config: dict[str, Any] = {
            "name": "llapdance-pinbench",
            "test_file": "test_cases.json",
            "providers": [
                {
                    "type": "vllm",
                    "model": model,
                    "base_url": endpoint.rstrip("/") + "/v1",
                    "api_key": cfg.get("api_key", "EMPTY"),
                    "temperature": cfg.get("temperature", 0.0),
                    "max_tokens": cfg.get("max_tokens", 4096),
                    "timeout": cfg.get("timeout", 300),
                }
            ],
            "runs_per_model": cfg.get("runs_per_model", 1),
            "max_workers": cfg.get("max_workers", 1),
            "save_raw_responses": False,
            "save_results_csv": False,
            "print_progress": False,
        }
        if "filter_test_ids" in cfg:
            run_config["filter_test_ids"] = cfg["filter_test_ids"]
        if "filter_categories" in cfg:
            run_config["filter_categories"] = cfg["filter_categories"]

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = str(Path(tmp) / "results")
            run_config["output_dir"] = output_dir
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(yaml.safe_dump(run_config))

            proc = subprocess.run(
                [python_bin, "run_benchmark.py", "-c", str(config_path), "--no-progress"],
                cwd=pinbench_dir,
                capture_output=True,
                text=True,
                timeout=cfg.get("subprocess_timeout", 1800),
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"PinBench run_benchmark.py failed (exit {proc.returncode}):\n"
                    f"{proc.stdout}\n{proc.stderr}"
                )

            run_dirs = sorted(Path(output_dir).iterdir())
            if not run_dirs:
                raise RuntimeError(f"PinBench produced no output under {output_dir}")
            run_dir = run_dirs[-1]

            summary = json.loads((run_dir / "summary.json").read_text())["summary"]
            detailed = json.loads((run_dir / "results.json").read_text())["results"]

        failures = [
            {
                "test_id": r["test_id"],
                "test_title": r["test_title"],
                "category": r["category"],
                "score": r["score"],
                "criteria": r["details"],
                "errors": r["errors"],
            }
            for r in detailed
            if not r["passed"]
        ]

        return CoherenceResult(
            adapter=self.name,
            total=summary["count"],
            passed=summary["passed"],
            # PinBench's grading is deterministic weighted-criteria matching
            # (pinyin/hanzi/table-format checks), not an LLM judge - maps to
            # graded_by_match, mirroring fixed-questions' own convention.
            graded_by_match=summary["passed"],
            graded_by_llm_judge=0,
            failures=failures,
        )


register("coherence", PinBenchCoherence.name, PinBenchCoherence)
