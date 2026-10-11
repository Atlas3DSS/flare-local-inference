"""Bounded top-k draft proposals: filtering and sampling over at most DRAFT_CANDIDATES tokens.

Proposals follow the request policy after penalties, grammar and temperature:
top-k, softmax, top-p and min-p, each retaining boundary ties among the
candidates, then renormalization. A request whose top-k exceeds
DRAFT_CANDIDATES, including disabled top-k, proposes from the
DRAFT_CANDIDATES highest scores. Chain verification receives exactly the
distribution each draft was drawn from, so verified output is unchanged;
only acceptance depends on how closely proposals follow the target policy.

The first kernel writes zeros over the dense proposal row and keeps each
block's highest scores. The second selects the row's candidates, applies the
policy, draws one Philox uniform, and scatters the surviving probabilities.
"""

import torch
import triton
import triton.language as tl

from flare.components.speculation import PreparedSpeculativeDistributions

DRAFT_CANDIDATES = 64
BLOCK = 4096


@triton.jit
def _orderable(scores):
    # Signed integers ordered like the scores; NaN orders above infinity and is rejected later.
    bits = scores.to(tl.int32, bitcast=True)
    return tl.where(bits < 0, bits ^ 2147483647, bits)


@triton.jit
def _kth_largest(ordered, k):
    # Largest t with at least k values >= t, by bisection over the 32-bit order.
    low = tl.full((), -2147483648, tl.int64)
    high = tl.full((), 2147483647, tl.int64)
    wide = ordered.to(tl.int64)
    for _ in range(32):
        middle = (low + high + 1) >> 1
        enough = tl.sum((wide >= middle).to(tl.int32), 0) >= k
        low = tl.where(enough, middle, low)
        high = tl.where(enough, high, middle - 1)
    return low


@triton.jit
def _block_candidates(Scores, Probabilities, TopK, Candidates, V: tl.constexpr, BLOCKS: tl.constexpr,
                      BLOCK: tl.constexpr, K: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    tokens = block * BLOCK + tl.arange(0, BLOCK)
    inside = tokens < V
    scores = tl.load(Scores + row * V + tokens, inside, other=-float("inf"))
    tl.store(Probabilities + row * V + tokens, tl.zeros((BLOCK,), tl.float32), inside)
    k = tl.minimum(tl.maximum(tl.load(TopK + row), 1), K)
    ordered = tl.where(inside, _orderable(scores), -2147483648)
    take = inside & (ordered.to(tl.int64) >= _kth_largest(ordered, k))
    position = tl.cumsum(take.to(tl.int32), 0) - 1
    keys = (ordered.to(tl.int64) << 32) | (2147483647 - tokens).to(tl.int64)
    base = Candidates + (row * BLOCKS + block) * K
    slots = tl.arange(0, K)
    count = tl.minimum(tl.sum(take.to(tl.int32), 0), K)
    tl.store(base + slots, tl.full((K,), -9223372036854775808, tl.int64), slots >= count)
    tl.store(base + position, keys, take & (position < K))


@triton.jit
def _select_and_sample(Candidates, Selected, Probabilities, Valid, TopK, TopP, Params, Slots, PointMass, Seed,
                       Offset, Sampled, V: tl.constexpr, BLOCKS: tl.constexpr, POOL: tl.constexpr, K: tl.constexpr,
                       SLOTS: tl.constexpr):
    row = tl.program_id(0)
    index = tl.arange(0, POOL)
    pool = tl.load(Candidates + row * BLOCKS * K + index, index < BLOCKS * K, other=-9223372036854775808)
    top_k = tl.minimum(tl.maximum(tl.load(TopK + row), 1), K)
    present = pool != -9223372036854775808
    ordered = (pool >> 32).to(tl.int32)
    take = present & (ordered.to(tl.int64) >= _kth_largest(ordered, top_k))
    position = tl.cumsum(take.to(tl.int32), 0) - 1
    rank = tl.arange(0, K)
    tl.store(Selected + row * K + rank, tl.full((K,), -9223372036854775808, tl.int64))
    tl.debug_barrier()
    tl.store(Selected + row * K + position, pool, take & (position < K))
    tl.debug_barrier()
    keys = tl.sort(tl.load(Selected + row * K + rank), descending=True)
    ordered = (keys >> 32).to(tl.int32)
    scores = tl.where(ordered < 0, ordered ^ 2147483647, ordered).to(tl.float32, bitcast=True)
    tokens = 2147483647 - (keys & 2147483647).to(tl.int32)
    present = keys != -9223372036854775808
    finite = present & (scores == scores) & (scores != float("inf")) & (scores != -float("inf"))  # noqa: PLR0124

    slot = tl.minimum(tl.maximum(tl.load(Slots + row), 0), SLOTS - 1)
    temperature = tl.load(Params + slot * 6)
    min_p = tl.load(Params + slot * 6 + 2)
    top_p = tl.load(TopP + row)

    # Top-k keeps every candidate tied with the k-th score.
    pivot = tl.max(tl.where(rank == top_k - 1, scores, -float("inf")), 0)
    kept = finite & (scores >= pivot)
    maximum = tl.max(tl.where(kept, scores, -float("inf")), 0)
    weights = tl.where(kept, tl.exp(scores - maximum), 0.)
    probabilities = weights / tl.sum(weights, 0)
    # Top-p keeps the shortest descending prefix reaching top_p and ties with its last member.
    before = tl.cumsum(probabilities, 0) - probabilities
    cutoff = tl.min(tl.where(kept & (before < top_p), probabilities, float("inf")), 0)
    kept &= probabilities >= cutoff
    probabilities = tl.where(kept, probabilities, 0.)
    probabilities = probabilities / tl.sum(probabilities, 0)
    # Min-p against the largest remaining probability, then renormalize.
    kept &= (probabilities > 0.) & (probabilities >= tl.max(probabilities, 0) * min_p)
    probabilities = tl.where(kept, probabilities, 0.)
    mass = tl.sum(probabilities, 0)
    valid = (tl.load(Valid + row) != 0) & (mass > 0.) & (mass == mass) & (mass != float("inf"))  # noqa: PLR0124
    probabilities = probabilities / tl.where(mass > 0., mass, 1.)

    greedy = temperature == 0.
    point_mass = tl.load(PointMass + row).to(tl.int32)
    seed = tl.load(Seed + row)
    offset = (tl.load(Offset + row) & 4294967295).to(tl.uint32)
    uniform = tl.rand(seed, offset)
    cumulative = tl.cumsum(probabilities, 0)
    chosen = tl.sum(((cumulative <= uniform) & kept).to(tl.int32), 0)
    last = tl.max(tl.where(kept, rank, 0), 0)
    chosen = tl.minimum(chosen, last)
    sampled = tl.sum(tl.where(rank == chosen, tokens, 0), 0)
    sampled = tl.where(greedy, point_mass, sampled)
    sampled = tl.where(valid, sampled, 0)

    tl.store(Probabilities + row * V + tokens, probabilities, kept & valid & ~greedy)
    tl.store(Probabilities + row * V + sampled, 1., ~valid | greedy)
    tl.store(Valid + row, valid)
    tl.store(Sampled + row, sampled.to(tl.int64))


class PreparedDraftProposal:
    """Two captured launches over explicit request-policy inputs for one draft position."""

    def __init__(self, scores, probabilities, valid, top_k, top_p, params, slots, point_mass, seed, offset, sampled,
                 *, capacity):
        rows, vocab = scores.shape[0], scores.shape[-1]
        self.blocks = triton.cdiv(vocab, BLOCK)
        self.candidates = torch.empty((rows, self.blocks, DRAFT_CANDIDATES), dtype=torch.int64, device=scores.device)
        self.selected = torch.empty((rows, DRAFT_CANDIDATES), dtype=torch.int64, device=scores.device)
        pool = triton.next_power_of_2(self.blocks * DRAFT_CANDIDATES)
        self._block_args = (scores, probabilities, top_k, self.candidates, vocab, self.blocks, BLOCK, DRAFT_CANDIDATES)
        self._select_args = (self.candidates, self.selected, probabilities, valid, top_k, top_p,
                             params, slots,
                             point_mass, seed, offset, sampled, vocab, self.blocks, pool, DRAFT_CANDIDATES, capacity)
        self._grids = ((rows, self.blocks, 1), (rows, 1, 1))
        self._kernels = None

    def prepare(self):
        from flare.preparation.triton import prepare_triton

        self._kernels = (
            prepare_triton(_block_candidates, self._block_args, self._grids[0], num_warps=8),
            prepare_triton(_select_and_sample, self._select_args, self._grids[1], num_warps=4),
        )
        self()

    def __call__(self):
        if self._kernels is None:
            raise RuntimeError("draft proposal prepare() must complete before capture")
        for kernel in self._kernels:
            kernel()



class PreparedDraftDistribution(PreparedSpeculativeDistributions):
    """A one-position conditional proposal whose filtering and draw use PreparedDraftProposal."""

    def __init__(self, logits, requests, slots, generations, live_rows, candidate_prefix_ids, *,
                 seed, offset, sampled, prefix_offset=0):
        super().__init__(logits, requests, slots, generations, live_rows, candidate_prefix_ids,
                         prefix_offset=prefix_offset)
        if logits.shape[1] != 1:
            raise ValueError("draft proposals cover one position")
        # Contiguous [B,1,...] storage is addressed by row; kernels receive the base tensors.
        self.proposal = PreparedDraftProposal(
            self.scores, self.probabilities, self.valid, self.packed_top_k, self.packed_top_p,
            requests.parameters, slots, self.point_mass_ids, seed, offset, sampled, capacity=requests.capacity)

    def prepare(self):
        if self.logits.device.type != "cuda":
            raise ValueError("draft proposals require CUDA")
        self._prepared = True
        self.prepare_scores()
        self.proposal.prepare()

    def __call__(self):
        if not self._prepared:
            raise RuntimeError("draft proposal requires prepare() before capture")
        self.prepare_scores()
        self.proposal()
        return self.probabilities
