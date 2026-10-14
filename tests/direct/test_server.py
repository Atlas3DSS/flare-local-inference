"""Direct execution's request parsing and reply decoding, without a model."""

import pytest

from ignition.direct.scheduler import AdmissionError
from ignition.direct.server import _Frontend, _sampling

DEFAULTS = {"temperature": 1., "top_p": .95, "top_k": -1, "seed": 0, "min_p": 0., "repetition_penalty": 1.,
            "presence_penalty": 0., "frequency_penalty": 0.}


def test_request_sampling_overrides_the_recipe_defaults():
    sampling = _sampling({"seed": 1101, "top_p": 1}, DEFAULTS)
    assert (sampling.seed, sampling.temperature, sampling.top_p, sampling.top_k) == (1101, 1., 1., -1)
    assert _sampling({}, DEFAULTS).values() == (1., .95, 0., 1., 0., 0.)
    for body in ({"temperature": 0}, {"seed": -1}, {"top_p": "high"}, {"seed": 1.5}):
        with pytest.raises(AdmissionError):
            _sampling(body, DEFAULTS)


class Tokenizer:
    def decode(self, tokens, skip_special_tokens):
        assert not skip_special_tokens
        return "".join(chr(token) for token in tokens)


def frontend():
    result = object.__new__(_Frontend)
    result.tokenizer, result.eos = Tokenizer(), {0}
    return result


def test_replies_split_reasoning_when_the_prompt_opens_a_think_block():
    text = [ord(c) for c in "plan</think>\nThe code is 2U2GYCZF, 198."] + [0]
    assert frontend().reply(text, "<|user|>q<|assistant|><think>") == ("The code is 2U2GYCZF, 198.", "plan")
    assert frontend().reply([ord(c) for c in "still thinking"], "<|assistant|><think>") == ("", "still thinking")
    assert frontend().reply([ord(c) for c in "plain"], "<|assistant|>") == ("plain", None)
