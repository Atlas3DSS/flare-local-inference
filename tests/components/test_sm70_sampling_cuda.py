"""Production whole-forward sampling compilation and capture on real SM70."""

import math
import os

import pytest
import torch

from ignition.components.completion import CompletionOutputs
from ignition.components.sampling import (
    ERROR, FINISHED, INVALID_DISTRIBUTION, RUNNING, DeviceRequestState,
    PreparedSampling, RequestAdmission,
)
from ignition.preparation.compilation import compile_forward


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned V100 device")


class _CapturedSampling:
    def __init__(self, rows, vocab):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
            pytest.skip("requires SM70")
        self.rows = rows
        self.state = DeviceRequestState.allocate("cuda", vocab_size=vocab)
        self.logits = torch.zeros((rows, vocab), dtype=torch.float32, device="cuda")
        self.slots = torch.arange(rows, dtype=torch.int32, device="cuda")
        self.epochs = torch.ones(rows, dtype=torch.int64, device="cuda")
        self.live = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.frame = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.sequence = torch.ones(1, dtype=torch.int64, device="cuda")
        self.completion = CompletionOutputs.allocate((2, 4, 8), device="cuda")
        self.sampler = PreparedSampling(self.logits, self.state, self.slots, self.epochs,
            self.live, self.completion, self.frame, self.sequence)
        self.reset()
        self.sampler.prepare()
        self.compiled = compile_forward(self.sampler)
        self.compiled()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.compiled()
        self.pointer = self.sampler.probabilities.data_ptr()

    def admit(self, slot, *, counter=0, seed=2**64 - 97, prompt=(), **parameters):
        admission = dict(generation=1, seed=seed, max_new_tokens=100,
                         temperature=1., top_k=0, top_p=1., eos_token_ids=())
        admission.update(parameters)
        self.state.admit(slot, RequestAdmission(**admission), prompt)
        self.state.rng_counter[slot] = counter

    def reset(self, **parameters):
        for slot in range(4):
            self.admit(slot, **parameters)

    def replay(self):
        torch.cuda.synchronize()
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        self.graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        assert self.sampler.probabilities.data_ptr() == self.pointer
        return self.completion.records[self.frame.item()].cpu()


@pytest.fixture(params=[257, 248320], ids=["small", "qwen"])
def sampler(request):
    return _CapturedSampling(4, request.param)


@torch.no_grad()
def test_sampling_filter_ties_parameters_and_completion_survive_compiled_replay(sampler):
    cases = (
        ([4., 3., 3., 1.], 2, 1., 0., [0, 1, 2]),
        ([math.log(2.), 0., 0.], 0, .6, 0., [0, 1, 2]),
        ([math.log(2.), 0., 0.], -1, 1., .6, [0]),
        ([0., 0., 0.], 0, 1., 1., [0, 1, 2]),
    )
    sampler.slots.copy_(torch.tensor([2, 0, 3, 1], dtype=torch.int32))
    sampler.live.fill_(1)
    for iteration, (values, top_k, top_p, min_p, support) in enumerate(cases):
        sampler.reset()
        sampler.admit(2, counter=2**32 + iteration, seed=2**64 - 97 - iteration,
                      top_k=top_k, top_p=top_p, min_p=min_p)
        sampler.logits.fill_(-torch.inf)
        sampler.logits[0, :len(values)].copy_(torch.tensor(values))
        sampler.frame.fill_(iteration % 2)
        sampler.sequence.fill_(101 + iteration)
        records = sampler.replay()
        probabilities = sampler.sampler.probabilities[0].cpu()
        assert probabilities.isfinite().all()
        assert torch.nonzero(probabilities > 0).flatten().tolist() == support
        torch.testing.assert_close(probabilities.sum(), torch.tensor(1.), rtol=0, atol=2e-6)
        token = records[2, 4].item()
        assert token in support
        assert records[2].tolist() == [1, 101 + iteration, 0, 1, token, RUNNING, 0, 0]
        assert sampler.state.rng_counter[2].item() == 2**32 + iteration + 1
        counts = sampler.state.histogram.cpu()
        assert counts.sum() == 1 and counts[2, token] == 1
        assert (records[[0, 1, 3], 4] == -1).all()
        assert records[[0, 1, 3], 5].sum() == 0


@torch.no_grad()
def test_sampling_request_philox_is_invariant_to_packing_and_capacity(sampler):
    single = _CapturedSampling(1, sampler.state.vocab_size)
    for seed, counter in ((17, 0), (2**64 - 79, 2**32 - 1), (2**63 + 19, 2**32 + 7)):
        tokens = []
        for owner in (single, sampler):
            for rotation in range(4):
                order = [(rotation + row) % 4 for row in range(owner.rows)]
                owner.reset(seed=seed, counter=counter)
                owner.slots.copy_(torch.tensor(order, dtype=torch.int32))
                owner.live.fill_(owner.rows)
                owner.logits.fill_(-torch.inf)
                owner.logits[:, :16].zero_()
                records = owner.replay()
                tokens.extend(records[order, 4].tolist())
                assert owner.state.rng_counter.cpu().tolist() == [
                    counter + int(slot in order) for slot in range(4)]
                assert owner.state.generated_tokens.cpu().tolist() == [int(slot in order) for slot in range(4)]
        assert len(set(tokens)) == 1 and 0 <= tokens[0] < 16


@torch.no_grad()
def test_sampling_penalties_grammar_errors_and_eos_preserve_request_state(sampler):
    sampler.reset()
    sampler.admit(3, prompt=(3, 3, 4), repetition_penalty=2., presence_penalty=1.,
                  frequency_penalty=.5, temperature=.5, eos_token_ids=(7,))
    sampler.live.fill_(4)
    sampler.logits.zero_()
    sampler.logits[0, 5] = torch.nan
    sampler.logits[1, 5] = torch.inf
    sampler.logits[3, 5] = torch.nan
    sampler.state.grammar_enabled[2:] = 1
    sampler.state.grammar_mask[2:].zero_()
    sampler.state.grammar_mask[3, 0] = 1 << 7
    records = sampler.replay()
    for slot in (0, 1, 2):
        assert records[slot].tolist() == [1, 1, 0, 0, -1, ERROR, INVALID_DISTRIBUTION, 0]
    assert records[3].tolist() == [1, 1, 0, 1, 7, FINISHED, 0, 0]
    assert sampler.state.rng_counter.cpu().tolist() == [0, 0, 0, 1]
    assert sampler.state.histogram[3, [3, 4, 7]].cpu().tolist() == [2, 1, 1]
    before = {name: getattr(sampler.state, name).cpu().clone() for name in sampler.state.__dataclass_fields__}
    sampler.live.zero_()
    sampler.replay()
    for name, value in before.items():
        assert torch.equal(getattr(sampler.state, name).cpu(), value), name

    sampler.reset(prompt=(3, 3, 4), repetition_penalty=2., presence_penalty=1.,
                  frequency_penalty=.5, temperature=.5)
    sampler.live.fill_(1)
    sampler.logits.fill_(-torch.inf)
    sampler.logits[0, :5].copy_(torch.tensor([0., 1., -2., 4., -4.]))
    sampler.replay()
    # Repetition divides positive values and multiplies negative ones; presence
    # and occurrence penalties apply before temperature and vocabulary filters.
    expected = torch.tensor([0., 2., -4., 0., -19.]).softmax(dim=0)
    torch.testing.assert_close(sampler.sampler.probabilities[0, :5].cpu(), expected, rtol=2e-6, atol=2e-7)
