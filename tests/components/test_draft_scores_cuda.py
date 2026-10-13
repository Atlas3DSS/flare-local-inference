"""Fused one-position draft scores match the unfused speculative score preparation."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not (os.environ.get("FLARE_RUN_CUDA_TESTS") and torch.cuda.is_available()),
                                reason="set FLARE_RUN_CUDA_TESTS=1 on a CUDA host")

VOCAB = 10000


def _state(generator):
    from ignition.components.sampling import RUNNING, DeviceRequestState

    state = DeviceRequestState.allocate("cuda", vocab_size=VOCAB, capacity=4)
    state.generation.copy_(torch.tensor([7, 8, 9, 10]))
    state.status.fill_(RUNNING)
    state.status[3] = 0
    state.histogram.copy_(torch.randint(0, 3, (4, VOCAB), generator=generator, dtype=torch.int32))
    state.parameters.copy_(torch.tensor([[0.7, 0.9, 0.0, 1.3, 0.2, 0.05],
                                         [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                                         [1.2, 0.8, 0.1, 0.8, -0.1, 0.1],
                                         [1.0, 1.0, 0.0, 1.0, 0.0, 0.0]]))
    state.top_k.copy_(torch.tensor([0, 40, 0, 5], dtype=torch.int32))
    masks = torch.randint(-2**31, 2**31 - 1, state.grammar_mask.shape, generator=generator, dtype=torch.int64)
    state.grammar_mask.copy_(masks.to(torch.int32))
    state.grammar_enabled.copy_(torch.tensor([0, 1, 0, 1], dtype=torch.int32))
    return state


@pytest.mark.parametrize("prefix", [0, 1, 3])
@pytest.mark.parametrize("live", [4, 3])
@pytest.mark.parametrize("layout", ["duplicate", "stopped"])
def test_fused_draft_scores_match_reference(prefix, live, layout):
    from ignition.components.draft_sampling import PreparedDraftDistribution
    from ignition.components.speculation import PreparedSpeculativeDistributions

    generator = torch.Generator().manual_seed(prefix * 10 + live)
    state = _state(generator)
    batch = 4
    # Rows: ordinary, grammar-constrained, stale generation, then a duplicate of row 0 or a stopped slot.
    last = 0 if layout == "duplicate" else 3
    slots = torch.tensor([0, 1, 2, last], dtype=torch.int32, device="cuda")
    generations = torch.tensor([7, 8, 99, 7 if layout == "duplicate" else 10], dtype=torch.int64, device="cuda")
    live_rows = torch.tensor([live], dtype=torch.int32, device="cuda")
    logits = (torch.randn((batch, 1, VOCAB), generator=generator) * 4).to(torch.bfloat16).cuda()
    candidates = torch.randint(0, VOCAB, (batch, 4), generator=generator).cuda()
    candidates[1, 0] = -1
    reference = PreparedSpeculativeDistributions(logits, state, slots, generations, live_rows, candidates,
                                                 prefix_offset=prefix)
    reference.prepare_scores()
    seed, offset = torch.zeros(batch, dtype=torch.int64, device="cuda"), torch.zeros(batch, dtype=torch.int64,
                                                                                       device="cuda")
    sampled = torch.zeros(batch, dtype=torch.int64, device="cuda")
    fused = PreparedDraftDistribution(logits, state, slots, generations, live_rows, candidates,
                                      seed=seed, offset=offset, sampled=sampled, prefix_offset=prefix)
    fused.prepare()
    fused.prepare_scores()
    torch.cuda.synchronize()
    assert torch.equal(fused.valid, reference.valid), (fused.valid, reference.valid)
    rows = reference.valid[:, 0]
    assert torch.equal(fused.scores[rows], reference.scores[rows])
    assert torch.equal(fused.maximum[rows], reference.maximum[rows])
    assert torch.equal(fused.point_mass_ids[rows], reference.point_mass_ids[rows])
    assert torch.equal(fused.packed_top_k, reference.packed_top_k)
    assert torch.equal(fused.packed_top_p, reference.packed_top_p)
