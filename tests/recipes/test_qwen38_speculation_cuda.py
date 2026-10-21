from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import replace

import pytest
import torch

from tests.components.completion_support import completion_outputs

from ignition.components.sampling import FINISHED
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.recipes.qwen38.sampling import DeviceRequestState, RequestAdmission
from ignition.components.speculation import (
    TARGET_RNG_DOMAIN,
    PreparedCopyProposals,
    PreparedSpeculativeCommit,
    PreparedSpeculativeDistributions,
    PreparedStochasticVerification,
    SpeculativeRngState,
)

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires explicitly assigned CUDA device and FLARE_RUN_CUDA_TESTS=1")
VOCAB = 248320


@torch.no_grad()
def test_compiled_commit_preserves_pooled_state_without_pool_copies():
    from ignition.preparation.compilation import compile_forward
    from ignition.components.speculation import SpeculativeAcceptance
    from ignition.recipes.qwen38.state import HybridStatePool

    state, slots, epochs, live, inputs, records, frame, sequence = tensors(4, 2)
    pool = HybridStatePool.allocate(34, "cuda", speculative_tokens=7)
    consumed = pool.consumed_tokens[1:5, 0]
    history = pool.ple_history[1:5]
    consumed.copy_(state.consumed_tokens)
    history.copy_(state.ple_history)
    state = replace(state, consumed_tokens=consumed, ple_history=history)
    state.status[1] = 0
    acceptance = SpeculativeAcceptance.allocate(4, 2, "cuda")
    acceptance.token_ids[:, 0] = 4
    acceptance.token_ids[:, 1] = 100
    acceptance.proposed_count.fill_(2)
    acceptance.accepted_drafts.fill_(1)
    acceptance.offered_drafts.fill_(1)
    acceptance.valid.fill_(True)
    logits = torch.zeros((*inputs.shape, state.vocab_size), device=inputs.device)
    commit = PreparedSpeculativeCommit(state, acceptance, inputs, slots, epochs, live,
        completion_outputs(records), frame, sequence, logits=logits)
    commit.prepare()
    restore = snapshot(state)
    def run():
        commit()
    compiled = compile_forward(run)
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    compiled()
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - allocated < pool.backing.nbytes // 4

    with captured(compiled, (commit, pool, state), {"records": records}, restore) as replay:
        replay()
        assert consumed.tolist() == [4, 2, 4, 4]
        assert history.tolist() == [[3, 4], [1, 2], [3, 4], [3, 4]]
        assert records[0, :, 2].tolist() == [4, 2, 4, 4]


def tensors(batch, width):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    state = DeviceRequestState.allocate("cuda", vocab_size=VOCAB)
    slots = torch.tensor([3, 1, 0, 2][:batch], device="cuda", dtype=torch.int32)
    epochs = torch.full((batch,), 7, device="cuda", dtype=torch.int64)
    live = torch.tensor([batch], device="cuda", dtype=torch.int32)
    inputs = torch.arange(3, 3 + width, device="cuda").expand(batch, -1).contiguous()
    records = torch.full((2, 4, 24), -99, device="cuda", dtype=torch.int64)
    frame = torch.tensor([0], device="cuda", dtype=torch.int32)
    sequence = torch.tensor([1], device="cuda", dtype=torch.int64)
    for slot in range(4):
        state.admit(slot, RequestAdmission(generation=7, seed=2**64 - 19 + slot,
            max_new_tokens=1000, top_k=0, top_p=1, eos_token_ids=()), [1, 2], consumed_tokens=2, ple_history=(1, 2))
        state.pending_token[slot] = 3
        state.generated_tokens[slot] = 1
        state.histogram[slot, 3] = 1
    return state, slots, epochs, live, inputs, records, frame, sequence


def snapshot(state):
    saved = {name: getattr(state, name).clone() for name in state.__dataclass_fields__}
    def restore():
        for name, value in saved.items():
            getattr(state, name).copy_(value)
    return restore


@pytest.mark.parametrize("temperature", [.7, 1.])
@pytest.mark.parametrize("width", [1, 3, 8])
def test_native_copy_conditional_grammar_and_stale_metadata(temperature, width):
    args = tensors(2, width)
    state, slots, epochs, live, inputs, records, frame, sequence = args
    state.parameters[:, 0] = temperature
    copy_ids = torch.arange(4, 11, device="cuda").expand(4, -1).clone()
    copy_count = torch.full((4,), 7, dtype=torch.int32, device="cuda")
    copy_expected = torch.full((4,), 2, dtype=torch.int64, device="cuda")
    copy_generation = torch.full((4,), 7, dtype=torch.int64, device="cuda")
    masks = torch.zeros((4, 8, VOCAB // 32), dtype=torch.int32, device="cuda")
    mask_count = torch.zeros(4, dtype=torch.int32, device="cuda")
    mask_count[3] = width
    state.grammar_enabled[3] = 1
    masks[3, :width, 0] = torch.tensor([1 << (4 + j) for j in range(width)], dtype=torch.int32, device="cuda")
    drafts = torch.zeros((2, width - 1), dtype=torch.int64, device="cuda")
    probabilities = torch.zeros((2, width - 1, VOCAB), dtype=torch.float32, device="cuda")
    valid = torch.zeros((2, width - 1), dtype=torch.bool, device="cuda")
    copy_mask = torch.zeros(2, dtype=torch.int64, device="cuda")
    previous = torch.full((4,), 8, dtype=torch.int32, device="cuda")
    accepted_generation = torch.zeros(4, dtype=torch.int64, device="cuda")
    proposals = PreparedCopyProposals(state, slots, epochs, live, copy_ids=copy_ids,
        copy_count=copy_count, copy_expected=copy_expected, copy_generation=copy_generation,
        grammar_mask_count=mask_count, draft_ids=drafts, draft_probabilities=probabilities,
        draft_valid=valid, copy_mask=copy_mask, previous_accepted=previous,
        accepted_generation=accepted_generation)
    logits = torch.zeros((2, width, VOCAB), dtype=torch.float32, device="cuda")
    # Unconstrained lane one has a deterministic target; lane zero follows masks.
    logits[1].fill_(-torch.inf)
    for index in range(width):
        logits[1, index, 4 + index] = 0
    distribution = PreparedSpeculativeDistributions(logits, state, slots, epochs, live, drafts,
        grammar_masks=masks, grammar_mask_count=mask_count)
    rng = SpeculativeRngState.allocate("cuda")
    distribution.prepare()
    verify = PreparedStochasticVerification(probabilities, drafts, distribution.probabilities,
        state, slots, epochs, live, rng=rng, draft_valid=valid, target_valid=distribution.valid)
    verify.prepare()
    logits = torch.zeros((*inputs.shape, state.vocab_size), device=inputs.device)
    commit = PreparedSpeculativeCommit(state, verify.output, inputs, slots, epochs, live,
        completion_outputs(records), frame, sequence, rng=rng, copy_offered_mask=copy_mask, draft_context_lag=None, logits=logits)
    commit.prepare()
    state_restore = snapshot(state)

    def restore():
        state_restore()
        rng.target_offset.zero_(); rng.generation.zero_()
        previous.fill_(8); accepted_generation.zero_()

    def run():
        proposals()
        inputs[:, 0].copy_(state.pending_token[slots.long()])
        inputs[:, 1:].copy_(drafts)
        distribution()
        verify(); commit()

    with captured(run, (*args, proposals, distribution, verify, commit, rng),
                  {"copy_ids": copy_ids, "masks": masks, "records": records}, restore) as replay:
        replay()
        assert records[0, 3, 16:16 + width].cpu().tolist() == list(range(4, 4 + width))
        assert records[0, 3, 8:12].cpu().tolist() == [width, width - 1, width - 1, (1 << (width - 1)) - 1]
        assert records[0, 3, 13] == -1 and previous[3] == 1
        assert rng.target_offset[3] == verify.draw_reservation
        restore()
        masks[3, 0].zero_(); masks[3, 0, 1] = 1 << (42 - 32)
        state.eos_count[3] = 1; state.eos_token_ids[3, 0] = 42
        replay()
        assert records[0, 3, 8] == 1 and records[0, 3, 16] == 42
        assert state.status[3] == FINISHED
        for fault in ("generation", "position", "mask_count"):
            restore()
            copy_generation.fill_(7); copy_expected.fill_(2); mask_count[3] = width
            if fault == "generation": copy_generation[3] = 6
            elif fault == "position": copy_expected[3] = 1
            else: mask_count[3] = width - 1
            replay()
            assert records[0, 3, 5:9].cpu().tolist() == [3, 6, 0, 0]
            assert state.consumed_tokens[3] == 2 and rng.target_offset[3] == 0
            assert previous[3] == 8 and accepted_generation[3] == 0


@contextmanager
def captured(run, resources, buffers, restore=lambda: None):
    run()
    restore()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    restore()
    bundle = PreparedGraphBundle(device=0, graphs={"speculation": graph}, buffers=buffers,
        resources=tuple(resources))
    worker = bundle.start_worker()
    try:
        def replay():
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            ticket = worker.submit(0)
            completion = worker.poll(timeout_ms=30000)
            assert completion is not None and completion.ticket == ticket and completion.error is None
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocated_bytes.all.allocated"):
                assert before[key] == after[key]
        yield replay
    finally:
        worker.close()
        graph.reset()


@pytest.mark.parametrize("width", range(1, 9))
def test_native_flashinfer_one_hot_all_prefix_lengths_and_clipping(width):
    args = tensors(4, width)
    state, slots, epochs, live, inputs, records, frame, sequence = args
    draft_probs = torch.zeros((4, width - 1, VOCAB), device="cuda")
    target_probs = torch.zeros((4, width, VOCAB), device="cuda")
    for column in range(width - 1):
        draft_probs[:, column, 4 + column] = 1
    def reset_probabilities():
        target_probs.zero_()
        for column in range(width):
            target_probs[:, column, 4 + column] = 1
    reset_probabilities()
    rng = SpeculativeRngState.allocate("cuda")
    verify = PreparedStochasticVerification(draft_probs, inputs[:, 1:].contiguous(),
        target_probs, state, slots, epochs, live, rng=rng)
    verify.prepare()
    logits = torch.zeros((*inputs.shape, state.vocab_size), device=inputs.device)
    commit = PreparedSpeculativeCommit(state, verify.output, inputs, slots, epochs, live,
        completion_outputs(records), frame, sequence, rng=rng, logits=logits)
    commit.prepare()
    restore_state = snapshot(state)
    def restore():
        restore_state(); rng.generation.zero_(); rng.target_offset.zero_()
    def run():
        verify(); commit()
    with captured(run, (*args, verify, commit, rng, draft_probs, target_probs), {"inputs": inputs, "records": records}, restore) as replay:
        for accepted in range(width):
            restore()
            reset_probabilities()
            target_probs[:, accepted].zero_()
            target_probs[:, accepted, 100] = 1
            state.consumed_tokens.fill_(63)
            frame[0] = accepted % 2
            sequence[0] = accepted + 1
            replay()
            assert commit.emitted_count.cpu().tolist() == [accepted + 1] * 4
            assert commit.commit_column.cpu().tolist() == [accepted] * 4
            assert state.pending_token.cpu().tolist() == [100] * 4
            assert state.consumed_tokens.cpu().tolist() == [64 + accepted] * 4
        restore()
        reset_probabilities()
        state.max_new_tokens.fill_(2)
        if width > 1:
            state.eos_count.fill_(1)
            state.eos_token_ids[:, 0] = 4
        replay()
        assert commit.emitted_count.cpu().tolist() == [1] * 4
        assert state.status.cpu().tolist() == [FINISHED] * 4
        assert commit.accepted_drafts.cpu().tolist() == [int(width > 1)] * 4
        before = state.histogram.cpu()
        replay()
        assert commit.emitted_count.eq(0).all()
        assert torch.equal(state.histogram.cpu(), before)


def penalized_scores(state, slots, logits, candidates, prefix_offset):
    """Reference penalized, tempered scores from the committed history and preceding drafts.

    Counts gather every valid candidate before each column's prefix boundary on
    top of the slot's committed histogram. Grammar masking is part of the
    prepared scores; the comparison runs with grammar disabled.
    """
    batch, width, vocab = logits.shape
    result = torch.empty((batch, width, vocab), dtype=torch.float32)
    for row in range(batch):
        slot = slots[row].item()
        history = state.histogram[slot].cpu().clone()
        temperature, _, _, penalty, presence, frequency = state.parameters[slot].tolist()
        for column in range(width):
            counts = history.clone()
            for token in candidates[row, :column + prefix_offset].tolist():
                if 0 <= token < vocab:
                    counts[token] += 1
            values = logits[row, column].cpu()
            adjusted = torch.where(values > 0, values / penalty, values * penalty)
            values = torch.where(counts > 0, adjusted, values)
            values = values - (counts > 0) * presence - counts * frequency
            result[row, column] = values / (temperature if temperature > 0 else 1)
    return result.to(logits.device)


@pytest.mark.parametrize("batch,width,prefix_offset", [(1, 1, 0), (1, 1, 1), (2, 1, 5), (1, 4, 0), (2, 4, 0)])
@torch.no_grad()
def test_compiled_distributions_match_eager_with_changed_parameters(batch, width, prefix_offset):
    from ignition.preparation.compilation import compile_forward

    args = tensors(batch, width)
    state, slots, epochs, live, inputs, *_ = args
    generator = torch.Generator(device="cuda").manual_seed(1803)
    logits = torch.randn((batch, width, VOCAB), generator=generator, device="cuda") * 3
    candidates = torch.arange(4, 11, device="cuda").expand(batch, -1).clone()
    candidates[:, width - 1 + prefix_offset:] = -1
    op = PreparedSpeculativeDistributions(logits, state, slots, epochs, live,
        candidates, prefix_offset=prefix_offset)
    op.prepare()
    run = compile_forward(op)
    with captured(run, (*args, op, run), {"probabilities": op.probabilities}) as replay:
        for top_k, top_p, min_p in ((0, 1., 0.), (0, .95, 0.), (20, .95, .1), (1, 1., .5)):
            state.top_k.fill_(top_k)
            state.parameters[:, 1] = top_p
            state.parameters[:, 2] = min_p
            state.parameters[:, 3] = 1.2
            state.parameters[:, 4] = .15
            state.parameters[:, 5] = .05
            expected = op().clone()
            expected_valid = op.valid.clone()
            op.prepare_scores()
            reference = penalized_scores(state, slots, logits, candidates, prefix_offset)
            torch.testing.assert_close(op.scores[expected_valid], reference[expected_valid], rtol=2e-5, atol=1e-5)
            op.probabilities.fill_(float("nan"))
            op.scores.fill_(float("nan"))
            replay()
            torch.testing.assert_close(op.probabilities, expected, rtol=2e-5, atol=1e-7)
            assert torch.equal(op.valid, expected_valid)
            if top_k == 0:
                # Only an unmasked top-k leaves the replayed scores reportable.
                torch.testing.assert_close(op.scores[expected_valid], reference[expected_valid], rtol=2e-5, atol=1e-5)
        state.grammar_enabled.fill_(1)
        state.grammar_mask.zero_()
        replay()
        assert not op.valid.any()
        assert op.probabilities[:, :, 0].eq(1).all()
        assert op.probabilities[:, :, 1:].count_nonzero() == 0


def test_native_conditional_filtering_full_vocabulary_ties_and_masks():
    args = tensors(4, 8)
    state, slots, epochs, live, inputs, records, frame, sequence = args
    logits = torch.full((4, 8, VOCAB), -torch.inf, device="cuda")
    low, high, third = 4095, 4096, VOCAB - 1
    logits[:, :, [low, high]] = 4
    logits[:, :, third] = 2
    candidates = torch.tensor([[low, high, low, high, low, high, low]] * 4, device="cuda")
    state.top_k.fill_(1)
    state.parameters[:, 3] = 2
    state.parameters[1, 0] = .7
    op = PreparedSpeculativeDistributions(logits, state, slots, epochs, live, candidates)
    op.prepare()
    with captured(op, (*args, op, logits, candidates), {"probabilities": op.probabilities}) as replay:
        replay()
        assert op.valid.all()
        probabilities = op.probabilities.cpu()
        torch.testing.assert_close(probabilities[0, 0, [low, high]], torch.tensor([1., 0.]), rtol=0, atol=0)
        assert probabilities[0, 1, high] == 1
        torch.testing.assert_close(probabilities[1, 0, [low, high]], torch.tensor([1., 0.]), rtol=0, atol=0)
        assert probabilities.count_nonzero() == 4 * 8
        state.grammar_enabled[3] = 1
        live[0] = 3
        replay()
        assert op.valid[0].tolist() == [True] + [False] * 7
        assert not op.valid[3].any()
        torch.testing.assert_close(op.probabilities[0, 0].cpu(), probabilities[0, 0], rtol=0, atol=0)
        assert op.probabilities[0, 1:, 0].eq(1).all() and op.probabilities[3, :, 0].eq(1).all()
        # Invalid prefixes have finite original maxima but sanitized token-zero
        # scores. A missing maximum match must preserve that dummy distribution.
        assert op.scores[0, 1:, 0].eq(0).all()
        assert op.scores[0, 1:, 1:].isneginf().all()
        # A zero original maximum matches the sanitized dummy's token zero;
        # selecting that winner must still leave these prefixes invalid.
        logits.fill_(-torch.inf)
        logits[:, :, [low, high]] = 0.
        replay()
        assert op.maximum.eq(0).all()
        assert op.valid[0].tolist() == [True] + [False] * 7
        assert not op.valid[3].any()
        assert op.scores[0, 1:, 0].eq(0).all()
        assert op.scores[0, 1:, 1:].isneginf().all()
        assert op.probabilities[0, 1:, 0].eq(1).all()
        assert op.probabilities[3, :, 0].eq(1).all()


@pytest.mark.parametrize("compiled", [False, True])
@torch.no_grad()
def test_width_four_mixed_top_one_and_twenty_ties_preserve_greedy_verification(compiled):
    args = tensors(4, 4)
    state, slots, epochs, live, *_ = args
    tied = [4095, 4096, VOCAB - 1]
    logits = torch.full((4, 4, VOCAB), -torch.inf, device="cuda")
    logits[:, :, tied] = -2.
    drafts = torch.full((4, 3), tied[0], dtype=torch.int64, device="cuda")
    draft_probs = torch.zeros((4, 3, VOCAB), device="cuda")
    draft_probs[:, :, tied[0]] = 1.
    target = PreparedSpeculativeDistributions(logits, state, slots, epochs, live, drafts)
    target.prepare()
    rng = SpeculativeRngState.allocate("cuda")
    rng.generation.fill_(7)
    verify = PreparedStochasticVerification(draft_probs, drafts, target.probabilities,
        state, slots, epochs, live, rng=rng, target_valid=target.valid)
    verify.prepare()

    def run():
        target()
        verify()

    if compiled:
        from ignition.preparation.compilation import compile_forward
        run = compile_forward(run)
    resources = (*args, logits, drafts, draft_probs, target, rng, verify, run)
    with captured(run, resources, {"probabilities": target.probabilities}) as replay:
        for iteration, policies in enumerate(((1, 20, 1, 20), (20, 1, 20, 1), (1, 20, 1, 20))):
            state.top_k[slots.long()] = torch.tensor(policies, dtype=torch.int32, device="cuda")
            rng.target_offset.fill_(iteration * 12)
            target.top_one_candidates.fill_(VOCAB + 9)
            replay()
            assert target.valid.all() and verify.output.valid.all()
            for row, top_k in enumerate(policies):
                expected_ids = tied[:1] if top_k == 1 else tied
                for column in range(4):
                    probabilities = target.probabilities[row, column].cpu()
                    assert probabilities.gt(0).nonzero().flatten().tolist() == expected_ids
                count = int(verify.output.proposed_count[row])
                tokens = verify.output.token_ids[row, :count].cpu().tolist()
                assert all(token in expected_ids for token in tokens)
                if top_k == 1:
                    assert target.probabilities[row, :, tied[0]].eq(1).all()
                    assert verify.output.accepted_drafts[row] == 3
                    assert count == 4 and tokens == [tied[0]] * 4
                else:
                    assert target.top_one_candidates[row * 4:(row + 1) * 4].eq(VOCAB + 9).all()


@pytest.mark.parametrize("width", range(1, 9))
@pytest.mark.parametrize("compiled", [False, True])
def test_native_flashinfer_chain_accept_reject_rng_and_budget(width, compiled):
    args = tensors(4, width)
    state, slots, epochs, live, inputs, records, frame, sequence = args
    drafts = inputs[:, 1:].contiguous()
    draft_probs = torch.zeros((4, width - 1, VOCAB), device="cuda")
    target_probs = torch.zeros((4, width, VOCAB), device="cuda")
    for column in range(width - 1):
        draft_probs[:, column, 4 + column] = 1
        target_probs[:, column, 4 + column] = 1
    target_probs[:, -1, 100] = 1
    rng = SpeculativeRngState.allocate("cuda")
    verify = PreparedStochasticVerification(draft_probs, drafts, target_probs, state, slots, epochs, live, rng=rng)
    verify.prepare()
    logits = torch.zeros((*inputs.shape, state.vocab_size), device=inputs.device)
    commit = PreparedSpeculativeCommit(state, verify.output, inputs, slots, epochs, live, completion_outputs(records), frame, sequence, rng=rng, logits=logits)
    commit.prepare()
    restore_state = snapshot(state)
    def restore():
        restore_state(); rng.generation.zero_(); rng.target_offset.zero_()
    def run():
        verify(); commit()
    if compiled:
        from ignition.preparation.compilation import compile_forward
        run = compile_forward(run)
    with captured(run, (*args, verify, commit, rng), {"records": records}, restore) as replay:
        replay()
        assert commit.emitted_count.cpu().tolist() == [width] * 4
        assert verify.independent_accepted.cpu().tolist() == [width - 1] * 4
        assert rng.target_offset.cpu().tolist() == [verify.draw_reservation] * 4
        expected_seed = state.seed.view(torch.int64)[slots.long()] ^ TARGET_RNG_DOMAIN
        assert torch.equal(verify.seed_bits, expected_seed)
        restore()
        target_probs[:, 0].zero_(); target_probs[:, 0, 101] = 1
        replay()
        assert commit.emitted_count.cpu().tolist() == [1] * 4
        assert state.pending_token.cpu().tolist() == [101] * 4
        assert verify.emitted_drafts.eq(0).all()
        restore()
        target_probs[:, 0].fill_(float("nan"))
        replay()
        assert commit.emitted_count.eq(0).all()
        assert rng.target_offset.eq(0).all()
        assert state.generated_tokens.eq(1).all()


@pytest.mark.parametrize("compiled", [False, True])
def test_native_chain_one_row_stream_is_invariant_to_packed_lane(compiled):
    args = tensors(4, 4)
    state, slots, epochs, live, inputs, records, frame, sequence = args
    draft_probs = torch.zeros((4, 3, VOCAB), device="cuda")
    target_probs = torch.zeros((4, 4, VOCAB), device="cuda")
    draft_probs[:, :, 4:7] = torch.tensor([.2, .3, .5], device="cuda")
    target_probs[:, :, 4:7] = torch.tensor([.3, .4, .3], device="cuda")
    rng = SpeculativeRngState.allocate("cuda")
    rng.generation.fill_(7); rng.target_offset.fill_(20)
    many = PreparedStochasticVerification(draft_probs, inputs[:, 1:].contiguous(), target_probs, state, slots, epochs, live, rng=rng)
    one_slots, one_epochs, one_live = slots[2:3].clone(), epochs[2:3].clone(), torch.ones(1, device="cuda", dtype=torch.int32)
    one = PreparedStochasticVerification(draft_probs[2:3], inputs[2:3, 1:].contiguous(), target_probs[2:3],
        state, one_slots, one_epochs, one_live, rng=rng)
    many.prepare(); one.prepare()
    def run():
        many(); one()
    if compiled:
        from ignition.preparation.compilation import compile_forward
        run = compile_forward(run)
    with captured(run, (*args, many, one, rng), {"many": many.output.token_ids, "one": one.output.token_ids}) as replay:
        for offset in (0, 4, 20, 100):
            rng.target_offset.fill_(offset)
            replay()
            assert torch.equal(many.output.token_ids[2], one.output.token_ids[0])
            assert many.output.proposed_count[2] == one.output.proposed_count[0]
            assert many.independent_accepted[2] == one.independent_accepted[0]


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("bonus", [100, 0])
@torch.no_grad()
def test_fixed_graph_commits_bounded_prefixes_at_batch_four(compiled, bonus):
    args = tensors(4, 4)
    state, slots, epochs, live, inputs, records, frame, sequence = args
    limits = torch.tensor([4, 1, 2, 3], dtype=torch.int32, device="cuda")
    columns = torch.arange(4, device="cuda")
    drafts = inputs[:, 1:].contiguous()
    draft_probs = torch.zeros((4, 3, VOCAB), dtype=torch.float32, device="cuda")
    draft_probs.scatter_(2, drafts[:, :, None], 1.)
    target_probs = torch.zeros((4, 4, VOCAB), dtype=torch.float32, device="cuda")
    for row in range(4):
        length = int(limits[row])
        for column in range(length):
            target_probs[row, column, int(inputs[row, column + 1]) if column + 1 < length else bonus] = 1
        target_probs[row, length:] = torch.nan
    target_valid = columns[None, :] < limits[:, None]
    draft_valid = columns[None, :-1] < limits[:, None] - 1
    rng = SpeculativeRngState.allocate("cuda")
    verify = PreparedStochasticVerification(draft_probs, drafts, target_probs,
        state, slots, epochs, live, rng=rng, draft_valid=draft_valid,
        target_valid=target_valid, token_limits=limits)
    verify.prepare()
    logits = torch.zeros((*inputs.shape, state.vocab_size), device=inputs.device)
    commit = PreparedSpeculativeCommit(state, verify.output, inputs, slots, epochs, live,
        completion_outputs(records), frame, sequence, rng=rng, token_limits=limits, logits=logits)
    commit.prepare()
    restore = snapshot(state)

    def run():
        verify()
        commit()

    if compiled:
        from ignition.preparation.compilation import compile_forward
        run = compile_forward(run)
    with captured(run, (*args, limits, verify, commit, rng), {"records": records}, restore) as replay:
        replay()
        assert verify.output.valid.all()
        assert verify.output.proposed_count.tolist() == [4, 1, 2, 3]
        assert verify.output.accepted_drafts.tolist() == [3, 0, 1, 2]
        assert verify.output.offered_drafts.tolist() == [3, 0, 1, 2]
        assert state.error_code.eq(0).all()
        assert commit.emitted_count.tolist() == [4, 1, 2, 3]
        assert state.consumed_tokens.tolist() == [4, 3, 5, 6]
        assert state.pending_token.tolist() == [bonus] * 4
        assert records[0, :, 15].tolist() == [4] * 4
