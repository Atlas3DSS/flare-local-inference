"""Trace FlashInfer sampling operators with explicit persistent workspace inputs."""

import torch
import triton
import triton.language as tl
from flashinfer import sampling
from flashinfer.utils import device_support_pdl

TOP_ONE_BLOCK = 4096


@triton.jit
def _top_one_candidates(Scores, Maximum, TopK, Candidates, V: tl.constexpr, BLOCKS: tl.constexpr,
                        BLOCK: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    maximum = tl.load(Maximum + row)
    finite = (maximum == maximum) & (tl.abs(maximum) < float("inf"))
    if (tl.load(TopK + row) == 1) & finite:
        token = block * BLOCK + tl.arange(0, BLOCK)
        score = tl.load(Scores + row * V + token, token < V, other=-float("inf"))
        first = tl.min(tl.where((token < V) & (score == maximum), token, V), 0)
        tl.store(Candidates + row * BLOCKS + block, first)


@triton.jit
def _top_one_mask(Scores, Maximum, TopK, Candidates, V: tl.constexpr, BLOCKS: tl.constexpr,
                  REDUCE: tl.constexpr, BLOCK: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    maximum = tl.load(Maximum + row)
    finite = (maximum == maximum) & (tl.abs(maximum) < float("inf"))
    if (tl.load(TopK + row) == 1) & finite:
        part = tl.arange(0, REDUCE)
        winner = tl.min(tl.load(Candidates + row * BLOCKS + part, part < BLOCKS, other=V), 0)
        # Invalid conditional prefixes may already contain a dummy distribution
        # whose score differs from Maximum. Preserve that sanitized row.
        if winner < V:
            token = block * BLOCK + tl.arange(0, BLOCK)
            tl.store(Scores + row * V + token, -float("inf"), (token < V) & (token != winner))


def prepare_top_one(scores, maximum, top_k, candidates):
    """Prepare lowest-ID argmax masking using explicit per-block Int32 scratch."""
    from ignition.preparation.triton import prepare_triton

    rows, vocab = scores.shape
    blocks = triton.cdiv(vocab, TOP_ONE_BLOCK)
    args = (scores, maximum, top_k, candidates)
    grid = (rows, blocks, 1)
    return (
        prepare_triton(_top_one_candidates, args, grid, V=vocab, BLOCKS=blocks,
                      BLOCK=TOP_ONE_BLOCK, num_warps=4),
        prepare_triton(_top_one_mask, args, grid, V=vocab, BLOCKS=blocks,
                      REDUCE=triton.next_power_of_2(blocks), BLOCK=TOP_ONE_BLOCK, num_warps=4),
    )


def probability_filter_workspace(rows, vocabulary, device):
    """Reserve FlashInfer's deterministic top-p workspace before graph capture."""
    return torch.empty(sampling._top_p_renorm_workspace_size(rows, vocabulary, True),
                       dtype=torch.uint8, device=device)


def filter_probabilities_out(logits, top_k, top_p, row_states, softmax_workspace,
                             output, top_p_workspace, *, top_one):
    """Top-k one chooses the lowest maximum token ID; larger k retains cutoff ties."""
    for kernel in top_one:
        kernel()
    _flashinfer_probabilities_out(logits, top_k, top_p, row_states, softmax_workspace,
                                 output, top_p_workspace)


@torch.compiler.allow_in_graph
def _flashinfer_probabilities_out(logits, top_k, top_p, row_states, softmax_workspace,
                                  output, top_p_workspace):
    module = sampling.get_sampling_module()
    torch.ops.flashinfer.top_k_mask_logits_out(logits, logits, top_k, 0, row_states)
    probabilities = module.softmax(softmax_workspace, logits, None, 1.,
                                   device_support_pdl(logits.device))
    torch.ops.flashinfer.top_p_renorm_probs_out(
        probabilities, output, top_p, 0., True, top_p_workspace)


@torch.compiler.allow_in_graph
def sample_probabilities(probabilities, seed, offset, return_valid=False):
    return sampling.sampling_from_probs(probabilities, seed=seed, offset=offset,
                                       deterministic=True, check_nan=False, return_valid=return_valid)


@torch.compiler.allow_in_graph
def verify_chain(draft_probs, draft_ids, target_probs, accepted, emitted, seed, offset):
    # Each row draws from its own seed and offset only, so its tokens do not depend on its batch slot.
    return sampling.chain_speculative_sampling(draft_probs, draft_ids, target_probs,
        accepted, emitted, deterministic=True, seed=seed, offset=offset, row_independent_rng=True)
