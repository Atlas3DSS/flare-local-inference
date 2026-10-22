"""Captured Sol-13c S1L decode attention: four-agent packing, replay identity, padded rows and an FP64 reference."""

import os

import pytest
import torch

from ignition.components import sm70_decode_attention as component
from ignition.components.sm70_decode_attention import DecodeAgents, PreparedDecodeAttention, admits

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned V100")

HEADS, KV, DIM, PAGE = component.HEADS, component.KV_HEADS, component.DIM, component.PAGE_TOKENS


def _reference(q, kc, vc, pages, lengths, cu):
    """FP64 packed causal attention: local row j of an agent with R rows sees positions < length - R + j + 1."""
    out = torch.zeros_like(q, dtype=torch.float64)
    k64 = kc.view(torch.bfloat16).double().cpu()
    v64 = vc.view(torch.bfloat16).double().cpu()
    q64, pages, lengths, cu = q.double().cpu(), pages.cpu(), lengths.cpu().tolist(), cu.cpu().tolist()
    for agent in range(len(lengths)):
        begin, end = cu[agent], cu[agent + 1]
        R = end - begin
        for j in range(R):
            visible = lengths[agent] - R + j + 1
            positions = torch.arange(visible)
            physical = pages[agent, positions // PAGE].long()
            keys, values = k64[physical, positions % PAGE], v64[physical, positions % PAGE]     # [visible, KV, DIM]
            for head in range(HEADS):
                kv = head // (HEADS // KV)
                scores = keys[:, kv] @ q64[begin + j, head] / DIM ** 0.5
                out[begin + j, head] = (scores.softmax(0) @ values[:, kv]).to(out.device)
    return out


@pytest.mark.parametrize("b, w", [(2, 4), (4, 5)])          # (4, 5): zero-length slice paths and variant 5 | 128
@torch.inference_mode()                                      # preparation runs under inference mode (functionalized mutation copies)
def test_packed_agents_replay_identity_padded_rows_and_reference(b, w):
    from b12x.preparation import PreparationSession
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena

    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires V100")
    torch.manual_seed(13)
    device = torch.device("cuda")
    physical, blocks = 24 * b, 24                                # 3,072-token page tables, 24 physical pages per agent
    rows = b * w
    q = torch.full((rows, HEADS, DIM), float("nan"), device=device)
    output = torch.full_like(q, float("nan"))
    kc = (torch.randn((physical, PAGE, KV, DIM), device=device) * .2).bfloat16().view(torch.uint16)
    vc = torch.randn((physical, PAGE, KV, DIM), device=device).bfloat16().view(torch.uint16)
    page_table = torch.full((b, blocks), -1, dtype=torch.int32, device=device)
    lengths = torch.zeros(b, dtype=torch.int32, device=device)        # metadata.lengths (query lengths)
    starts = torch.zeros(b, dtype=torch.int32, device=device)         # committed context
    cu = torch.zeros(b + 1, dtype=torch.int32, device=device)
    agents = DecodeAgents.allocate(rows, blocks, device)
    assert admits(q, kc, page_table)
    arena = ScratchArena(device, overlap=False)
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        attention = PreparedDecodeAttention(q, kc, vc, agents, output, session=session, name="decode-attention",
                                            workspace=arena, rows_per_agent=w)
        compiled = compile_forward(attention)
    arena.freeze()
    attention.bind()

    def load(case):
        """case: per agent (committed context, rows this step); rows 0 = inactive agent."""
        perm = torch.randperm(physical, device=device).to(torch.int32)
        page_table.fill_(-1)
        ends = [0]
        for agent, (context, r) in enumerate(case):
            total = context + r
            npg = (total + PAGE - 1) // PAGE
            page_table[agent, :npg] = perm[agent * (physical // b): agent * (physical // b) + npg]
            starts[agent], lengths[agent] = context, r
            ends.append(ends[-1] + r)
        cu.copy_(torch.tensor(ends, dtype=torch.int32, device=device))
        q.fill_(float("nan"))
        q[:ends[-1]] = (torch.randn((ends[-1], HEADS, DIM), device=device) * .2).bfloat16().float()
        output.fill_(float("nan"))
        agents.pack(lengths, starts, cu, page_table)
        return ends[-1]

    graph = torch.cuda.CUDAGraph()
    load([(300, w)] * b)
    compiled()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        compiled()
    cases = ([(300, w), (1281, w)],          # agents full, one on the 256-token ladder rung
             [(1275, w), (2561, w - 1)],     # ladder cutoffs; second agent with a partial row count
             [(2555, 1), (0, 0)],            # ordinary single row next to an inactive agent (R = 0, length 0)
             [(1280, w), (2560, w)])
    cases = tuple(list(case) + [(1276 + 1285 * a, w if a % 2 else 1) for a in range(b - 2)] for case in cases)
    for case in cases:
        active = load(case)
        packed = (agents.lengths.cpu().tolist(), agents.cu.cpu().tolist())
        assert packed[0][b:] == [0] * (4 - b) and packed[1] == [*cu.cpu().tolist(), *([active] * (4 - b))]
        assert all(packed[0][a] == 0 for a, (_, r) in enumerate(case) if r == 0)
        graph.replay()
        torch.cuda.synchronize()
        captured = output.clone()
        output.fill_(float("nan"))
        compiled()                                                     # eager launch, same inputs
        torch.cuda.synchronize()
        assert torch.equal(captured.view(torch.int32), output.view(torch.int32))
        assert torch.equal(output[active:].view(torch.int32), torch.zeros_like(output[active:], dtype=torch.int32))
        assert torch.isfinite(output[:active]).all()
        expected = _reference(q[:active], kc, vc, page_table, starts + lengths, cu)
        torch.testing.assert_close(output[:active].double().cpu(), expected.cpu(), rtol=2e-5, atol=2e-6)
    graph.reset()


@torch.no_grad()
def test_refuses_page_tables_beyond_the_admitted_context():
    """Option (a) of the S1L review: 34,816 tokens (272 pages) must be refused at preparation, never served by a mixture."""
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires V100")
    device = torch.device("cuda")
    q = torch.zeros((4, HEADS, DIM), device=device)
    kc = torch.zeros((2, PAGE, KV, DIM), dtype=torch.uint16, device=device)
    agents = DecodeAgents.allocate(4, 272, device)
    assert not admits(q, kc, agents.pages)
    with pytest.raises(ValueError, match="--context 32768"):
        PreparedDecodeAttention(q, kc, kc.clone(), agents, torch.zeros_like(q), session=None, name="too-wide",
                                rows_per_agent=1)
