"""Prepare Step-5's ChatML template for inference without training-span annotations."""

import json
from pathlib import Path

from tokenizers import Tokenizer

from ignition.preparation.frontend import prepare_frontend as _prepare_frontend
from ignition.preparation.frontend import tokenizer_settings


def prepare_frontend(snapshot, *, eos_token_ids):
    snapshot = Path(snapshot)
    settings = tokenizer_settings(json.loads((snapshot / "tokenizer_config.json").read_text()))
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    turn_end = tokenizer.token_to_id(settings["eos_token"])
    if turn_end is None:
        raise ValueError("Step-5 tokenizer EOS marker is missing from its vocabulary")
    # The chat tokenizer's end-of-turn token differs from the base-model EOS tokens.
    effective = list(eos_token_ids)
    if turn_end not in effective:
        effective.append(turn_end)
    return _prepare_frontend(snapshot, eos_token_ids=effective)


def inference_template(snapshot):
    source = (Path(snapshot) / "chat_template.jinja").read_text()
    opening, closing = "{%- generation %}", "{%- endgeneration %}"
    if source.count(opening) != 1 or source.count(closing) != 1:
        raise ValueError("Step-5 chat template must contain one assistant generation span")
    return source.replace(opening, "{%- if true %}").replace(closing, "{%- endif %}")
