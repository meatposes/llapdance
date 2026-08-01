"""guidellm (vLLM project) benchmark adapter - INTENTIONAL STUB, same
pattern as llama_benchy.py.

Tried this for real against a live endpoint before stubbing it:

    guidellm run --backend "kind=openai_http,target=...,model=<name>,api_key=..." \\
      --tokenizer "kind=huggingface_auto,pretrained_model_name_or_path=gpt2" \\
      --data "kind=synthetic_text,prompt_tokens=32,output_tokens=32" \\
      --profile "kind=synchronous" --constraint "kind=max_requests,count=3" \\
      --output "kind=json,path=/tmp/out.json"

Root cause, confirmed by reading guidellm's own source
(`guidellm/data/tokenizers/huggingface.py`): its `synthetic_text` data
source ALWAYS resolves a HuggingFace tokenizer via
`AutoTokenizer.from_pretrained(model_name)`, using the backend's `model=`
value directly as the HF repo id - `HuggingFaceTokenizerArgs` has no field
to point it at a different repo. This works fine when the served model
name IS a real HF Hub repo id, but fails hard (`OSError: Repo id must use
alphanumeric chars...`) against:
  - llm-proxy's `<file>@<backend>` naming convention (this session's
    external-mode validation target)
  - any locally-built custom-quant model name in general (which is most of
    what this harness tests) - there is no guarantee a llama.cpp/qxmx/
    Arcaine/OpenArc model name is ever a resolvable HF repo id.

Registered anyway so a suite config referencing `adapter: guidellm` fails
with this explanation instead of a bare error deep in guidellm's own
tokenizer resolution code. Replace `run()` if guidellm adds a tokenizer
override field, or if this harness starts tracking a model's real
upstream HF repo id (e.g. via BackendConfig) separately from its served
name specifically for this purpose.
"""
from __future__ import annotations

from typing import Any

from llapdance.core.result import BenchmarkResult
from llapdance.plugins.base import BenchmarkAdapter
from llapdance.plugins.registry import register


class GuidellmBenchmark(BenchmarkAdapter):
    name = "guidellm"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def run(self, endpoint: str, config: dict[str, Any]) -> BenchmarkResult:
        raise NotImplementedError(
            "guidellm adapter is not implemented: its synthetic_text data source resolves a "
            "tokenizer via AutoTokenizer.from_pretrained(model_name) with no override, which "
            "fails against any model name that isn't a real HF Hub repo id (confirmed live - "
            "see module docstring). Use 'generic-http' instead, or a --data source/tokenizer "
            "combination that doesn't need HF tokenizer resolution."
        )


register("benchmark", GuidellmBenchmark.name, GuidellmBenchmark)
