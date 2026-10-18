"""Prompt rendering against DeepSeek V4.1 Flash 1017's canonical encoder."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import jinja2
import pytest

from ignition.recipes import resolve

CALL = {"id": "call1", "type": "function", "function": {"name": "lookup", "arguments": {"key": "x", "limit": 2}}}
SECOND = {"id": "call2", "type": "function", "function": {"name": "lookup", "arguments": {"key": "y"}}}
TOOLS = [{"type": "function", "function": {"name": "lookup", "description": "Find a key.",
          "parameters": {"type": "object", "properties": {"key": {"type": "string"}}}}}]
AGENT = [
    {"role": "user", "content": "A"},
    {"role": "assistant", "content": "", "reasoning_content": "R1", "tool_calls": [CALL, SECOND]},
    {"role": "tool", "tool_call_id": "call2", "content": "second"},
    {"role": "tool", "tool_call_id": "call1", "content": [{"type": "text", "text": "first"}]},
    {"role": "assistant", "content": "B", "reasoning_content": "R2", "tool_calls": [CALL]},
    {"role": "tool", "tool_call_id": "call1", "content": "again"},
]


@pytest.fixture(scope="module")
def reference():
    model = resolve("deepseek-v4.1-flash")
    path = model.snapshot(Path.home() / ".cache/huggingface/hub") / "encoding/encoding.py"
    if not path.is_file():
        pytest.skip("canonical DeepSeek V4.1 Flash 1017 encoder is not cached")
    spec = importlib.util.spec_from_file_location("deepseek_v41_encoding_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", ("chat", "thinking"))
@pytest.mark.parametrize("effort", ("low", "high", "max", 7))
@pytest.mark.parametrize(
    "messages",
    (
        [{"role": "user", "content": "Hello."}],
        [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "Hello."}],
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B", "reasoning_content": "C"},
            {"role": "user", "content": "D"},
        ],
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B", "reasoning_content": "C"},
            {"role": "system", "content": "Mid-conversation instruction."},
        ],
        [{"role": "user", "content": [{"type": "text", "text": "One."}, {"type": "text", "text": "Two."}]}],
        AGENT,
        [*AGENT, {"role": "assistant", "content": "C", "reasoning_content": "R3"}, {"role": "user", "content": "D"}],
        [*AGENT[:3], {"role": "user", "content": "Also this."}],
        [{"role": "system", "content": "", "tools": TOOLS}, *AGENT],
    ),
)
def test_prompt_matches_checkpoint(reference, mode, effort, messages):
    env = jinja2.Environment()
    env.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)

    def fail(message):
        raise ValueError(message)

    env.globals["raise_exception"] = fail
    template = env.from_string(Path("ignition/recipes/deepseek41/assets/chat_template.jinja").read_text())
    actual = template.render(
        messages=messages, thinking_mode=mode, reasoning_effort=effort, add_generation_prompt=True
    )
    wire_messages = deepcopy(messages)
    for message in wire_messages:
        for call in message.get("tool_calls", []):
            call["function"]["arguments"] = json.dumps(call["function"]["arguments"], ensure_ascii=False)
    expected = reference.encode_messages(wire_messages, thinking_mode=mode, reasoning_effort=effort)
    assert actual == expected
