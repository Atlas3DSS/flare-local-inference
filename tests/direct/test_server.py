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


def test_graph_replay_saves_graph_times_that_eager_execution_reads(tmp_path):
    import threading

    from ignition.direct.scheduler import Layout, Scheduler
    from ignition.direct.server import DirectServer, load_graph_times

    graphs = {("speculative", 1, 1): "speculative1x1", ("speculative", 1, 2): "speculative1x2",
              ("prefill", 1, 128): "prefill128"}
    layout = Layout(bindings={}, buffer_specs={}, page_views={}, capacity=1, max_active=1, context=1024, page_tokens=64, table_width=16, pages=16, snapshots=0,
                    prefill_chunk=128, vocabulary=16, eos=(1,), graphs=graphs, widths=(1, 2), record_fields=24)
    server = object.__new__(DirectServer)
    server.layout, server.scheduler, server.lock = layout, Scheduler(layout), threading.Lock()
    server.graph_times = tmp_path / "times.json"
    server.scheduler.policy.costs.observe(("speculative", 1, 2), 9_000_000)
    server.scheduler.policy.costs.observe(("prefill", 1, 128), 30_000_000)
    server._save_graph_times()
    assert load_graph_times(server.graph_times, layout) == {"speculative1x2": 9_000_000, "prefill128": 30_000_000}
    # Graphs another preparation named are ignored.
    server.graph_times.write_text('{"graph_ns": {"speculative1x1": 4000000, "decode64": 5}}')
    assert load_graph_times(server.graph_times, layout) == {"speculative1x1": 4_000_000}
    server.graph_times.write_text("{}")
    with pytest.raises(AdmissionError, match="cannot read graph times"):
        load_graph_times(server.graph_times, layout)
