"""Captured-graph Sol-11s prefill attention: bit identity with the host entry, status/NaN policy, clean recovery."""

import ctypes as C
import os

import pytest
import torch

from ignition.components import sm70_prefill_attention as component
from ignition.components.sm70_prefill_attention import PreparedPrefillAttention, admits, library

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned V100")

HEADS, KV, DIM, PAGE = component.HEADS, component.KV_HEADS, component.DIM, component.PAGE_TOKENS


def _host(lib, q, kc, vc, pages, M, context):
    """The shipped Sol-11s host path (validated by its FP64 oracle) on the same physical pool and page table."""
    P, I = C.c_void_p, C.c_int
    lib.sol11s_meta.argtypes = [P] * 5 + [I] * 3 + [P]
    lib.sol11s_attention.argtypes = [P] * 7 + [I] * 4 + [P]
    lib.sol11s_meta.restype = lib.sol11s_attention.restype = I
    length = context + M
    npg = (length + PAGE - 1) // PAGE
    out = torch.empty((M, HEADS, DIM), device=q.device)
    ws = torch.zeros(component.workspace_bytes(), dtype=torch.uint8, device=q.device)
    meta = torch.zeros((kc.shape[0], KV, 2), dtype=torch.int32, device=q.device)
    stream = P(torch.cuda.current_stream().cuda_stream)
    ptr = lambda t: P(t.data_ptr())
    assert lib.sol11s_attention_init(ptr(ws), stream) == 0
    assert lib.sol11s_meta(ptr(kc), ptr(vc), ptr(pages), ptr(meta), ptr(ws), length, 0, npg, stream) == 0
    assert lib.sol11s_attention(ptr(q), ptr(kc), ptr(vc), ptr(pages), ptr(meta), ptr(out), ptr(ws), M, context, PAGE, 0, stream) == 0
    torch.cuda.synchronize()
    assert int(ws[:4].view(torch.int32)[0]) == 0
    return out


@torch.no_grad()
def test_captured_graph_matches_host_bits_and_fails_loudly():
    from b12x.preparation import PreparationSession
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena

    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires V100")
    torch.manual_seed(11)
    device = torch.device("cuda")
    capacity, physical, table = 128, 24, 20
    q = torch.full((capacity, HEADS, DIM), float("nan"), device=device)
    output = torch.full_like(q, float("nan"))
    kc = (torch.randn((physical, PAGE, KV, DIM), device=device) * .2).bfloat16().view(torch.uint16)
    vc = torch.randn((physical, PAGE, KV, DIM), device=device).bfloat16().view(torch.uint16)
    pages = torch.full((1, table), -1, dtype=torch.int32, device=device)
    lengths = torch.zeros(1, dtype=torch.int32, device=device)
    cu = torch.zeros(2, dtype=torch.int32, device=device)
    assert admits(q, kc, pages)
    arena = ScratchArena(device, overlap=False)
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        attention = PreparedPrefillAttention(q, kc, vc, pages, lengths, cu, output, session=session,
                                             name="prefill-attention", workspace=arena)
        compiled = compile_forward(attention)
    arena.freeze()
    attention.bind()
    lib = library(getattr(session, "kernel_cache", None))

    def load(M, context, kind="random"):
        length = context + M
        npg = (length + PAGE - 1) // PAGE
        perm = torch.randperm(physical, device=device)[:npg].to(torch.int32)
        q.fill_(float("nan"))
        q[:M] = (torch.randn((M, HEADS, DIM), device=device) * .2).bfloat16().float()
        if kind == "nan_k":
            kc.view(torch.int16)[perm[1].long(), 5, 1, 7] = torch.tensor(0x7FC0, dtype=torch.int16)   # a BF16 NaN
        pages.fill_(-1)
        pages[0, :npg] = perm
        lengths[0], cu[0], cu[1] = length, 0, M
        output.fill_(float("nan"))
        return perm

    graph = torch.cuda.CUDAGraph()
    load(37, 300)
    compiled()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        compiled()
    for M, context in ((37, 300), (128, 1000), (1, 0), (100, 2047)):
        perm = load(M, context)
        graph.replay()
        torch.cuda.synchronize()
        expected = _host(lib, q[:M].contiguous(), kc, vc, pages[:, :perm.numel()].contiguous(), M, context)
        assert int(attention.status) == 0
        assert torch.equal(output[:M].view(torch.int32), expected.view(torch.int32))
        assert torch.equal(output[M:].view(torch.int32), torch.zeros_like(output[M:], dtype=torch.int32))
    # Inadmissible plan (M above the graph capacity): status bit 2, every row NaN.
    load(37, 300)
    cu[1] = capacity + 1
    lengths[0] = 300 + capacity + 1
    graph.replay()
    torch.cuda.synchronize()
    assert int(attention.status) == 4 and torch.isnan(output).all()
    # Non-finite K on a visible page: status bit 0, NaN rows for the request, zeros beyond it, then clean recovery.
    saved = kc.clone()
    load(37, 300, kind="nan_k")
    graph.replay()
    torch.cuda.synchronize()
    assert int(attention.status) & 1 and torch.isnan(output[:37]).all()
    assert torch.equal(output[37:].view(torch.int32), torch.zeros_like(output[37:], dtype=torch.int32))
    kc.copy_(saved)
    perm = load(64, 777)
    graph.replay()
    torch.cuda.synchronize()
    expected = _host(lib, q[:64].contiguous(), kc, vc, pages[:, :perm.numel()].contiguous(), 64, 777)
    assert int(attention.status) == 0 and torch.equal(output[:64].view(torch.int32), expected.view(torch.int32))
    graph.reset()


@torch.no_grad()
def test_g128_group_rows_equal_standalone_128_row_chunks():
    """prefill-lab EXP-10 (G128): in a capacity-512 graph, rows [128g, 128g + M_g) of a chunk of M rows at `context`
    are bit-identical to the shipped host path run on those rows alone as an M_g-row chunk at context + 128g, i.e. to
    a 128-token-chunk prefill of the same prompt. Includes page-unaligned contexts (straddling pages)."""
    from b12x.preparation import PreparationSession
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena

    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires V100")
    torch.manual_seed(13)
    device = torch.device("cuda")
    capacity, physical, table = 512, 40, 36
    q = torch.full((capacity, HEADS, DIM), float("nan"), device=device)
    output = torch.full_like(q, float("nan"))
    kc = (torch.randn((physical, PAGE, KV, DIM), device=device) * .2).bfloat16().view(torch.uint16)
    vc = torch.randn((physical, PAGE, KV, DIM), device=device).bfloat16().view(torch.uint16)
    pages = torch.full((1, table), -1, dtype=torch.int32, device=device)
    lengths = torch.zeros(1, dtype=torch.int32, device=device)
    cu = torch.zeros(2, dtype=torch.int32, device=device)
    assert admits(q, kc, pages)
    arena = ScratchArena(device, overlap=False)
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        attention = PreparedPrefillAttention(q, kc, vc, pages, lengths, cu, output, session=session,
                                             name="prefill-attention-g128", workspace=arena)
        compiled = compile_forward(attention)
    arena.freeze()
    attention.bind()
    lib = library(getattr(session, "kernel_cache", None))

    def load(M, context):
        length = context + M
        npg = (length + PAGE - 1) // PAGE
        perm = torch.randperm(physical, device=device)[:npg].to(torch.int32)
        q.fill_(float("nan"))
        q[:M] = (torch.randn((M, HEADS, DIM), device=device) * .2).bfloat16().float()
        pages.fill_(-1)
        pages[0, :npg] = perm
        lengths[0], cu[0], cu[1] = length, 0, M
        output.fill_(float("nan"))
        return perm

    graph = torch.cuda.CUDAGraph()
    load(300, 777)
    compiled()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        compiled()
    for M, context in ((512, 0), (512, 1024), (300, 777), (256, 128), (200, 2047), (129, 3000), (128, 1000), (37, 300)):
        perm = load(M, context)
        graph.replay()
        torch.cuda.synchronize()
        assert int(attention.status) == 0
        for g in range((M + 127) // 128):
            rows = min(128, M - 128 * g)
            npg = (context + 128 * g + rows + PAGE - 1) // PAGE
            expected = _host(lib, q[128 * g:128 * g + rows].contiguous(), kc, vc, pages[:, :npg].contiguous(),
                             rows, context + 128 * g)
            assert torch.equal(output[128 * g:128 * g + rows].view(torch.int32), expected.view(torch.int32)), (M, context, g)
        assert torch.equal(output[M:].view(torch.int32), torch.zeros_like(output[M:], dtype=torch.int32))
    graph.reset()
