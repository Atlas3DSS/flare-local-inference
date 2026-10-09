"""CPU integrity and exact-input contracts for the greedy-generation corpus."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from qualify import generation_corpus as corpus


class Tokenizer:
    def apply_chat_template(self, messages, *, tools, tokenize, return_dict=None, **options):
        rendered = json.dumps([messages, tools, options], ensure_ascii=False)
        return self.encode(rendered, add_special_tokens=False) if tokenize else rendered

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode())


@pytest.fixture
def fixture(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("fixed source\n")
    prompts = []
    for index, name in enumerate(corpus.expected_ids()):
        category = "tool" if index < 69 else "chat" if index < 76 else "long_context"
        original = {"messages": [{"role": "user", "content": f"Unicode café 世界 {name}"}],
                    "tools": [], "temperature": 0.7, "max_tokens": 4096}
        prompts.append(corpus._prompt(name, category, {"source_request": original},
                                      Tokenizer(), {"add_generation_prompt": True}))
    return {"schema_version": 1, "model_id": corpus.MODEL_ID, "revision": corpus.REVISION,
            "max_new_tokens": 64, "ignore_eos": True,
            "eos_policy": {"production_eos_token_ids": [248046, 248044]},
            "construction": {}, "sources": {str(source): corpus.corpus_sha256(source)},
            "prompts": prompts}


def save(path, value):
    path.write_bytes(corpus._json_bytes(value) + b"\n")
    return path


def test_complete_corpus_preserves_sources_and_explicit_sampling(fixture, tmp_path):
    path = save(tmp_path / "corpus.json", fixture)
    loaded = corpus.load_corpus(path, expected_sha256=corpus.corpus_sha256(path))
    assert loaded == fixture
    assert len(loaded["prompts"]) == 82
    for prompt in loaded["prompts"]:
        assert prompt["request"]["temperature"] == 0
        assert prompt["request"]["max_tokens"] == 64
        assert prompt["source"]["source_request"]["temperature"] == 0.7
        assert prompt["source"]["source_request"]["max_tokens"] == 4096
        assert prompt["request"]["messages"] == prompt["source"]["source_request"]["messages"]


def test_manifest_resolution_and_expected_input_digest(fixture, tmp_path):
    raw, manifest = tmp_path / "data" / "corpus.json", tmp_path / "manifest.json"
    corpus.write_corpus(fixture, raw, manifest)
    assert corpus.load_corpus(manifest, expected_sha256=corpus.corpus_sha256(manifest)) == fixture
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        corpus.load_corpus(manifest, expected_sha256="0" * 64)
    raw.write_bytes(raw.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        corpus.load_corpus(manifest)


def test_fixture_destinations_cannot_be_overwritten(fixture, tmp_path):
    path = save(tmp_path / "corpus.json", fixture)
    with pytest.raises(FileExistsError):
        corpus.write_corpus(fixture, path, tmp_path / "manifest.json")


@pytest.mark.parametrize("change", [
    lambda c: c["prompts"].pop(44),
    lambda c: c["prompts"].reverse(),
    lambda c: c["prompts"][1].update(id=c["prompts"][0]["id"]),
])
def test_missing_excluded_duplicated_or_reordered_cases_rejected(fixture, change):
    change(fixture)
    with pytest.raises(ValueError, match="all 82 ordered"):
        corpus.validate_corpus(fixture)


@pytest.mark.parametrize("key,value", [
    ("temperature", 0.1), ("top_k", 1), ("top_p", 0.9), ("min_p", .01),
    ("max_tokens", 63), ("repetition_penalty", 1.1),
    ("presence_penalty", 0.1), ("frequency_penalty", .1), ("temperature", False),
])
def test_sampling_contract_cannot_silently_inherit_or_change(fixture, key, value):
    fixture["prompts"][0]["request"][key] = value
    with pytest.raises(ValueError, match="generation contract|boolean sampling"):
        corpus.validate_corpus(fixture)


@pytest.mark.parametrize("change", [
    lambda p: p["prompt_tokens"].__setitem__(0, 1),
    lambda p: p.update(rendered_prompt=p["rendered_prompt"] + " "),
    lambda p: p["request"]["chat_template_kwargs"].update(enable_thinking=False),
    lambda p: p["source"]["source_request"].update(max_tokens=32),
])
def test_input_content_tampering_is_detected(fixture, change):
    change(fixture["prompts"][0])
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        corpus.validate_corpus(fixture)


@pytest.mark.parametrize("token", [-1, 248320, True, 1.0])
def test_invalid_token_ids_rejected_before_hashing(fixture, token):
    fixture["prompts"][0]["prompt_tokens"][0] = token
    with pytest.raises(ValueError, match="invalid prompt token"):
        corpus.validate_corpus(fixture)


def test_source_modification_fails_default_loading(fixture, tmp_path):
    path = save(tmp_path / "corpus.json", fixture)
    Path(next(iter(fixture["sources"]))).write_text("modified source\n")
    with pytest.raises(ValueError, match="source SHA256 mismatch"):
        corpus.load_corpus(path)
    assert corpus.load_corpus(path, verify_sources=False) == fixture


def test_no_empty_source_provenance_or_eos_policy_substitution(fixture):
    modified = deepcopy(fixture)
    modified["sources"] = {}
    with pytest.raises(ValueError, match="source fingerprints"):
        corpus.validate_corpus(modified)
    fixture["ignore_eos"] = False
    with pytest.raises(ValueError, match="ignore_eos"):
        corpus.validate_corpus(fixture)


def test_duplicate_json_members_are_not_silently_overwritten(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":0,"schema_version":1}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        corpus.load_corpus(path)


def test_order_of_tool_properties_is_part_of_recorded_digest(fixture):
    original = {"messages": [{"role": "user", "content": "Use the tool."}],
                "tools": [{"type": "function", "function": {"name": "lookup", "description": "Find"}}]}
    prompt = corpus._prompt("tool-TC-01", "tool", {"source_request": original},
                            Tokenizer(), {"add_generation_prompt": True})
    function = prompt["request"]["tools"][0]["function"]
    prompt["request"]["tools"][0]["function"] = dict(reversed(list(function.items())))
    fixture["prompts"][0] = prompt
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        corpus.validate_corpus(fixture)


def test_native_fixture_exports_every_exact_render_and_id(fixture):
    base = {"assets": {}, "cases": [{"name": "default"}], "encode_cases": [], "decode_cases": []}
    exported = corpus.frontend_validation_fixture(fixture, base)
    assert len(exported["cases"]) == 83
    assert base["cases"] == [{"name": "default"}]
    for actual, expected in zip(exported["cases"][1:], fixture["prompts"], strict=True):
        assert actual["token_ids"] == expected["prompt_tokens"]
        assert actual["rendered"] == expected["rendered_prompt"]
        assert actual["messages"] == expected["request"]["messages"]


def test_generator_rejects_different_render_and_encode_laws():
    class InconsistentTokenizer(Tokenizer):
        def apply_chat_template(self, *args, tokenize, **kwargs):
            return [1] if tokenize else "different"
    source = {"source_request": {"messages": [{"role": "user", "content": "x"}]}}
    with pytest.raises(ValueError, match="template tokenization differ"):
        corpus._prompt("case", "chat", source, InconsistentTokenizer(), {})


def test_tool_choice_none_preserves_schema_but_disables_its_rendering(fixture):
    original = {"messages": [{"role": "user", "content": "Do not use tools."}],
                "tools": [{"type": "function", "function": {"name": "search"}}],
                "tool_choice": "none"}
    prompt = corpus._prompt("tool-TC-44", "tool", {"source_request": original},
                            Tokenizer(), {"add_generation_prompt": True})
    assert prompt["request"]["tools"] == original["tools"]
    assert prompt["request"]["tool_choice"] == "none"
    assert json.loads(prompt["rendered_prompt"])[1] == []
    fixture["prompts"][43] = prompt
    corpus.validate_corpus(fixture)
    exported = corpus.frontend_validation_fixture(fixture, {"cases": []})
    assert exported["cases"][43]["options"]["tools"] == []
    del prompt["request"]["tool_choice"]
    prompt["request_sha256"] = corpus._digest(prompt["request"])
    with pytest.raises(ValueError, match="disabled tool semantics"):
        corpus.validate_corpus(fixture)


@pytest.mark.parametrize("constraint", [
    {"tool_choice": "required"},
    {"response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "string"}}}},
])
def test_unconstrained_cases_keep_original_constraints_explicit(fixture, constraint):
    original = {"messages": [{"role": "user", "content": "Return required output."}], **constraint}
    prompt = corpus._prompt("tool-TC-01", "tool", {"source_request": original}, Tokenizer(), {})
    assert prompt["excluded_decode_constraints"] == constraint
    assert all(key not in prompt["request"] for key in constraint)
    fixture["prompts"][0] = prompt
    corpus.validate_corpus(fixture)
    prompt["excluded_decode_constraints"] = {}
    with pytest.raises(ValueError, match="excluded decoding constraints"):
        corpus.validate_corpus(fixture)
