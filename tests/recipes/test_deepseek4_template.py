"""Prompt rendering against the checkpoint's canonical text/vision encoder."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import jinja2
import pytest

from ignition.recipes.deepseek4.config import REVISION


@pytest.fixture(scope="module")
def reference():
    path = (
        Path.home()
        / f".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp/snapshots/{REVISION}/encoding/encoding_dsv4.py"
    )
    if not path.is_file():
        pytest.skip("canonical DeepSeek V4 encoder is not cached")
    spec = importlib.util.spec_from_file_location("deepseek_v4_encoding_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", ("chat", "thinking"))
@pytest.mark.parametrize("effort", ("low", "high", "max"))
@pytest.mark.parametrize(
    "messages",
    (
        [{"role": "user", "content": "Hello."}],
        [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "Hello."}],
        [{"role": "developer", "content": "Be concise."}, {"role": "user", "content": "Hello."}],
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B", "reasoning_content": "C"},
            {"role": "user", "content": "D"},
        ],
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe."},
                    {"type": "image_url", "image_url": {"url": "image.png"}},
                ],
            }
        ],
        [
            {"role": "user", "content": "A"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": {"key": "x", "limit": 2}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call1", "content": "result"},
        ],
    ),
)
def test_prompt_matches_checkpoint(reference, mode, effort, messages):
    env = jinja2.Environment()
    env.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)

    def fail(message):
        raise ValueError(message)

    env.globals["raise_exception"] = fail
    template = env.from_string(Path("ignition/recipes/deepseek4/assets/chat_template.jinja").read_text())
    actual = template.render(
        messages=messages, thinking_mode=mode, reasoning_effort=effort, add_generation_prompt=True
    )
    wire_messages = deepcopy(messages)
    # Helios validates and decodes OpenAI argument strings before rendering Jinja.
    for message in wire_messages:
        for call in message.get("tool_calls", []):
            call["function"]["arguments"] = json.dumps(call["function"]["arguments"], ensure_ascii=False)
    expected = reference.encode_messages(wire_messages, thinking_mode=mode, reasoning_effort=effort)
    assert actual == expected
