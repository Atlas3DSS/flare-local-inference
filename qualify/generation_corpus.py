"""Freeze exact checkpoint token contexts for 64-token greedy agreement.

The corpus reuses all standard tool-benchmark initial contexts, seven frontend
fixtures, and seeded context-pressure conversations. It contains model inputs,
not benchmark scores or generated outputs. Loading requires no tensor library.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import struct
import subprocess
from typing import Any

from ignition.preparation.qwen38 import (
    CONFIG_SHA256, FRONTEND_SHA256, GENERATION_SHA256, MODEL_ID, REVISION, TRANSFORMERS_REVISION,
)

BENCHMARK_REVISION = "6be685f0e6b9e0df05ed024848cf7fe1eca48752"
REFERENCE_DATE = "2026-09-16"
CHAT_CASES = (
    "generation_prompt", "thinking_disabled", "thinking_enabled", "reasoning_low",
    "reasoning_medium", "reasoning_xhigh", "assistant_continuation",
)
LONG_BUDGETS = (8192, 32768, 65536)
LONG_SCENARIOS = ("TC-01", "TC-07")
GENERATION = {
    "model": MODEL_ID, "temperature": 0, "max_tokens": 64, "top_p": 1,
    "top_k": 0, "min_p": 0, "repetition_penalty": 1,
    "presence_penalty": 0, "frequency_penalty": 0,
}
_CONTEXT_FIELDS = ("messages", "tools", "chat_template_kwargs")
_FORBIDDEN = ("stop", "response_format", "grammar", "guided_json", "logit_bias")


def _json_bytes(value: Any) -> bytes:
    # Mapping order is significant to tool-schema rendering.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def corpus_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def token_sha256(tokens: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read(path: Path, expected: str | None = None) -> dict:
    data = path.read_bytes()
    if expected is not None and hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"corpus SHA256 mismatch: {path}")
    result = json.loads(data, object_pairs_hook=_object)
    if not isinstance(result, dict):
        raise ValueError("corpus must be a JSON object")
    return result


def expected_ids() -> list[str]:
    return [f"tool-TC-{i:02}" for i in range(1, 70)] + [
        f"chat-{name}" for name in CHAT_CASES
    ] + [f"long-{budget}-{scenario}" for budget in LONG_BUDGETS for scenario in LONG_SCENARIOS]


def validate_corpus(corpus: dict, *, verify_sources: bool = True) -> None:
    """Reject incomplete, modified, or differently sampled corpus records."""
    for key, value in {"schema_version": 1, "model_id": MODEL_ID, "revision": REVISION,
                       "max_new_tokens": 64, "ignore_eos": True}.items():
        if corpus.get(key) != value or type(corpus.get(key)) is not type(value):
            raise ValueError(f"invalid corpus {key}")
    prompts = corpus.get("prompts")
    if not isinstance(prompts, list) or [p.get("id") for p in prompts] != expected_ids():
        raise ValueError("corpus must contain all 82 ordered fixed prompt IDs")
    if Counter(p.get("category") for p in prompts) != {"tool": 69, "chat": 7, "long_context": 6}:
        raise ValueError("invalid corpus category coverage")
    if corpus.get("eos_policy", {}).get("production_eos_token_ids") != [248046, 248044]:
        raise ValueError("invalid production EOS provenance")
    for prompt in prompts:
        name = prompt["id"]
        request = prompt.get("request")
        if not isinstance(request, dict) or any(request.get(k) != v for k, v in GENERATION.items()):
            raise ValueError(f"{name}: explicit greedy generation contract mismatch")
        if any(isinstance(request[k], bool) for k in GENERATION):
            raise ValueError(f"{name}: boolean sampling parameters are invalid")
        allowed = set(GENERATION) | set(_CONTEXT_FIELDS)
        if request.get("tool_choice") == "none":
            allowed.add("tool_choice")
        if set(request) != allowed:
            raise ValueError(f"{name}: unsupported request fields")
        if any(key in request for key in _FORBIDDEN):
            raise ValueError(f"{name}: constrained decoding is not part of the corpus")
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages or not isinstance(request.get("tools"), list):
            raise ValueError(f"{name}: invalid messages or tools")
        if not isinstance(request.get("chat_template_kwargs"), dict):
            raise ValueError(f"{name}: invalid template options")
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise ValueError(f"{name}: only text messages are supported")
        tokens = prompt.get("prompt_tokens")
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int or not 0 <= t < 248320 for t in tokens):
            raise ValueError(f"{name}: invalid prompt token IDs")
        if len(tokens) + 64 > 262144 or prompt.get("prompt_token_count") != len(tokens):
            raise ValueError(f"{name}: invalid context length")
        rendered = prompt.get("rendered_prompt")
        if not isinstance(rendered, str) or not rendered:
            raise ValueError(f"{name}: missing rendered prompt")
        checks = {
            "prompt_tokens_sha256": token_sha256(tokens),
            "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "request_sha256": _digest(request),
        }
        if any(prompt.get(k) != v for k, v in checks.items()):
            raise ValueError(f"{name}: prompt/request fingerprint mismatch")
        source = prompt.get("source", {})
        if not isinstance(source.get("source_request"), dict) or source.get("source_request_sha256") != _digest(source["source_request"]):
            raise ValueError(f"{name}: source request fingerprint mismatch")
        original = source["source_request"]
        if original.get("messages") != messages or (original.get("tools") or []) != request["tools"]:
            raise ValueError(f"{name}: original messages or tools changed")
        if (original.get("tool_choice") == "none") != (request.get("tool_choice") == "none"):
            raise ValueError(f"{name}: disabled tool semantics changed")
        if prompt.get("excluded_decode_constraints") != _excluded_constraints(original):
            raise ValueError(f"{name}: excluded decoding constraints are not declared exactly")
    sources = corpus.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("corpus source fingerprints are required")
    if verify_sources:
        for path, expected in sources.items():
            if corpus_sha256(path) != expected:
                raise ValueError(f"corpus source SHA256 mismatch: {path}")


def load_corpus(
    path: str | Path, *, expected_sha256: str | None = None, verify_sources: bool = True,
) -> dict:
    """Read a raw corpus or manifest, verifying bytes before returning its data."""
    path = Path(path).resolve()
    corpus = _read(path, expected_sha256)
    if corpus.get("artifact_type") == "greedy_agreement_corpus_manifest":
        raw = Path(corpus["fixture"]["path"])
        if not raw.is_absolute():
            raw = path.parent / raw
        corpus = _read(raw, corpus["fixture"]["sha256"])
    validate_corpus(corpus, verify_sources=verify_sources)
    return corpus


_EXTRACT = r'''
import asyncio, copy, inspect, json
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS
from tool_eval_bench.runner.orchestrator import run_scenario
from tool_eval_bench.runner.context_pressure import ContextPressureConfig, build_pressure_messages

class Captured(BaseException):
    pass

class Adapter:
    async def chat_completion(self, **kwargs):
        self.request = kwargs
        raise Captured

async def initial(scenario, pressure=None):
    adapter = Adapter()
    try:
        await run_scenario(adapter, model=MODEL, base_url='http://unused.invalid', api_key=None,
                           scenario=scenario, seed=42, reference_date='2026-09-16',
                           reference_day='Wednesday', context_pressure_messages=pressure)
    except Captured:
        k = adapter.request
        body = dict(model=k['model'], messages=k['messages'], max_tokens=k['max_tokens'],
                    temperature=k['temperature'])
        if k['tools']:
            body.update(tools=k['tools'], tool_choice=k['tool_choice'], parallel_tool_calls=True)
        if k['response_format']:
            body['response_format'] = k['response_format']
        body.update(k['extra_params'])
        body.update(stream=True, stream_options={'include_usage':True})
        return dict(scenario_id=scenario.id, title=scenario.title, category=scenario.category.value,
                    source_file=inspect.getsourcefile(scenario.evaluate), source_request=body)
    raise RuntimeError('scenario did not reach its initial model request')

async def main():
    scenarios = {s.id:s for s in ALL_SCENARIOS}
    if list(scenarios) != [f'TC-{i:02}' for i in range(1,70)]:
        raise ValueError('standard scenario registry differs')
    short = [await initial(s) for s in scenarios.values()]
    long = []
    for budget in (8192,32768,65536):
        pressure = build_pressure_messages(ContextPressureConfig(fill_tokens=budget), seed=42)
        for name in ('TC-01','TC-07'):
            item = await initial(scenarios[name], pressure)
            item['pressure_fill_tokens'] = budget
            long.append(item)
    print(json.dumps(dict(tool=short,long_context=long),ensure_ascii=False))

asyncio.run(main())
'''


def _benchmark_inputs(root: Path, python: Path) -> dict:
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
    if revision != BENCHMARK_REVISION or status:
        raise ValueError("benchmark must be a clean checkout of the pinned revision")
    result = subprocess.run(
        [str(python), "-c", f"MODEL={MODEL_ID!r}\n" + _EXTRACT], check=True,
        env={**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONDONTWRITEBYTECODE": "1",
             "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def _excluded_constraints(original: dict) -> dict:
    result = {}
    if original.get("tool_choice") not in (None, "auto", "none"):
        result["tool_choice"] = original["tool_choice"]
    if original.get("response_format"):
        result["response_format"] = original["response_format"]
    return result


def _prompt(identifier: str, category: str, source: dict, tokenizer, options: dict) -> dict:
    original = deepcopy(source["source_request"])
    source = {**source, "source_request": original, "source_request_sha256": _digest(original)}
    request = {**GENERATION, "messages": deepcopy(original["messages"]),
               "tools": deepcopy(original.get("tools") or []), "chat_template_kwargs": options}
    if original.get("tool_choice") == "none":
        request["tool_choice"] = "none"
    rendered_tools = [] if request.get("tool_choice") == "none" else request["tools"]
    rendered = tokenizer.apply_chat_template(request["messages"], tools=rendered_tools, tokenize=False, **options)
    tokens = tokenizer.apply_chat_template(request["messages"], tools=rendered_tools, tokenize=True,
                                           return_dict=False, **options)
    if tokenizer.encode(rendered, add_special_tokens=False) != tokens:
        raise ValueError(f"{identifier}: rendered text and template tokenization differ")
    return {"id": identifier, "category": category, "request": request, "rendered_prompt": rendered,
            "prompt_tokens": tokens, "prompt_token_count": len(tokens),
            "prompt_tokens_sha256": token_sha256(tokens),
            "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "request_sha256": _digest(request), "source": source,
            "excluded_decode_constraints": _excluded_constraints(original)}


def generate(snapshot: Path, benchmark_root: Path, benchmark_python: Path, frontend_fixture: Path) -> dict:
    """Render the fixed corpus using only verified local source and tokenizer assets."""
    from transformers import AutoTokenizer

    distribution = importlib.metadata.distribution("transformers")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != TRANSFORMERS_REVISION:
        raise ValueError("corpus generation requires pinned Transformers")
    assets = {**FRONTEND_SHA256, "generation_config.json": GENERATION_SHA256,
              "config.json": CONFIG_SHA256}
    for name, expected in assets.items():
        if corpus_sha256(snapshot / name) != expected:
            raise ValueError(f"checkpoint asset mismatch: {name}")
    frontend = _read(frontend_fixture)
    if frontend["model_id"] != MODEL_ID or frontend["revision"] != REVISION or frontend["assets"] != FRONTEND_SHA256:
        raise ValueError("frontend fixture checkpoint identity differs")
    benchmark = _benchmark_inputs(benchmark_root, benchmark_python)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    prompts = [_prompt(f"tool-{source['scenario_id']}", "tool", source, tokenizer,
                       {"add_generation_prompt": True}) for source in benchmark["tool"]]
    cases = {case["name"]: case for case in frontend["cases"]}
    for name in CHAT_CASES:
        case = cases[name]
        options = deepcopy(case["options"])
        options.setdefault("add_generation_prompt", not bool(options.get("continue_final_message")))
        source = {"fixture_name": name, "source_file": str(frontend_fixture),
                  "source_request": {"messages": case["messages"], "chat_template_kwargs": case["options"]}}
        prompt = _prompt(f"chat-{name}", "chat", source, tokenizer, options)
        if prompt["rendered_prompt"] != case["rendered"] or prompt["prompt_tokens"] != case["token_ids"]:
            raise ValueError(f"{name}: frontend fixture rendering changed")
        prompts.append(prompt)
    for source in benchmark["long_context"]:
        prompts.append(_prompt(f"long-{source['pressure_fill_tokens']}-{source['scenario_id']}",
                               "long_context", source, tokenizer, {"add_generation_prompt": True}))
    paths = set(benchmark_root.glob("src/**/*.py")) | {
        Path(__file__).resolve(), frontend_fixture, frontend_fixture.parent / "generate_fixtures.py",
        Path(__file__).resolve().parents[1] / "ignition/preparation/qwen38.py",
        benchmark_root / "pyproject.toml", benchmark_root / "uv.lock",
    } | {snapshot / name for name in assets}
    result = {
        "schema_version": 1, "model_id": MODEL_ID, "revision": REVISION,
        "max_new_tokens": 64, "ignore_eos": True,
        "tokenizer_assets": {name: {"path": str(snapshot / name), "sha256": expected}
                             for name, expected in assets.items()},
        "eos_policy": {"production_eos_token_ids": [248046, 248044],
                       "generation": "Record exactly 64 raw greedy token IDs, including tokens after EOS.",
                       "production": "Ordinary requests stop at configured EOS or token budget."},
        "construction": {"benchmark_revision": BENCHMARK_REVISION, "seed": 42,
                         "reference_date": REFERENCE_DATE, "reference_day": "Wednesday",
                         "transformers_revision": TRANSFORMERS_REVISION,
                         "tokenizers_version": importlib.metadata.version("tokenizers"),
                         "long_context": "Pinned build_pressure_messages fill budgets; actual rendered token counts are recorded without calibration.",
                         "scope": "Initial model contexts; no tool execution, scenario scoring, grammar, or forced-tool constraints.",
                         "token_hash_encoding": "little-endian uint32"},
        "sources": {str(path): corpus_sha256(path) for path in sorted(paths)}, "prompts": prompts,
    }
    validate_corpus(result)
    return result


def write_corpus(corpus: dict, output: Path, manifest: Path) -> None:
    validate_corpus(corpus)
    if output.exists() or manifest.exists():
        raise FileExistsError("corpus and manifest destinations must be unused")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_json_bytes(corpus) + b"\n")
    summary = {"artifact_type": "greedy_agreement_corpus_manifest", "schema_version": 1,
               "model_id": MODEL_ID, "revision": REVISION,
               "fixture": {"path": str(output.resolve()), "sha256": corpus_sha256(output),
                           "bytes": output.stat().st_size},
               "max_new_tokens": 64, "ignore_eos": True, "prompt_count": len(corpus["prompts"]),
               "categories": dict(Counter(p["category"] for p in corpus["prompts"])),
               "construction": corpus["construction"], "eos_policy": corpus["eos_policy"],
               "prompts": [{k: p[k] for k in ("id", "category", "prompt_token_count",
                           "prompt_tokens_sha256", "rendered_prompt_sha256", "request_sha256",
                           "excluded_decode_constraints")}
                           for p in corpus["prompts"]]}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


def frontend_validation_fixture(corpus: dict, baseline: dict) -> dict:
    """Adapt inputs to the existing native frontend qualifier without changing it."""
    validate_corpus(corpus)
    result = deepcopy(baseline)
    result["cases"].extend({"name": p["id"], "messages": p["request"]["messages"],
                            "options": {**p["request"]["chat_template_kwargs"],
                                        "tools": [] if p["request"].get("tool_choice") == "none" else p["request"]["tools"]},
                            "rendered": p["rendered_prompt"], "token_ids": p["prompt_tokens"]}
                           for p in corpus["prompts"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--benchmark-python", type=Path, required=True)
    parser.add_argument("--frontend-fixture", type=Path, default=Path("tests/frontend/qwen38.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--native-frontend-fixture", type=Path)
    args = parser.parse_args()
    corpus = generate(args.snapshot.resolve(), args.benchmark_root.resolve(),
                      args.benchmark_python.absolute(), args.frontend_fixture.resolve())
    write_corpus(corpus, args.output, args.manifest)
    if args.native_frontend_fixture:
        with args.native_frontend_fixture.open("x") as stream:
            json.dump(frontend_validation_fixture(corpus, _read(args.frontend_fixture)), stream, ensure_ascii=False)
    print(json.dumps({"manifest": str(args.manifest), "fixture": str(args.output),
                      "sha256": corpus_sha256(args.output), "prompts": len(corpus["prompts"]),
                      "created_at": datetime.now(timezone.utc).isoformat()}))


if __name__ == "__main__":
    main()
