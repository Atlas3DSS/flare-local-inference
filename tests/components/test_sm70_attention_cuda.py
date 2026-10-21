"""Real Volta cache writes, packed causal attention and compiled graph replay."""

import math
import os

import pytest
import torch

from ignition.components.sm70_attention import PreparedAttention, PreparedWriter


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned V100")


@pytest.mark.parametrize("dim", [32, 256])
@pytest.mark.parametrize("window", [-1, 3])
@torch.no_grad()
def test_packed_pages_causality_inactive_rows_and_replay(dim, window):
    from b12x.preparation import PreparationSession
    from ignition.preparation.compilation import compile_forward

    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires V100")
    torch.manual_seed(51)
    heads, kvheads, page, tokens = 4, 2, 8, 8
    q = torch.randn(tokens, heads, dim, device="cuda") * .2
    key = torch.randn(tokens, kvheads, dim, device="cuda") * .2
    value = torch.randn_like(key) * .2
    initial_k = (torch.randn(5, page, kvheads, dim) * .2).bfloat16()
    initial_v = (torch.randn(5, page, kvheads, dim) * .2).bfloat16()
    kc, vc = (value.view(torch.uint16).cuda() for value in (initial_k, initial_v))
    pages = torch.tensor([[3, 1], [0, 0], [2, 4]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([11, 0, 3], dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 4, 4, 6], dtype=torch.int32, device="cuda")
    slots = torch.tensor([3 * page + 7, page, page + 1, page + 2,
                          2 * page + 1, 2 * page + 2, -1, -1], dtype=torch.int64, device="cuda")
    output = torch.full_like(q, float("nan"))
    expected_k, expected_v = initial_k.clone(), initial_v.clone()
    for token, slot in enumerate(slots.cpu().tolist()):
        if slot >= 0:
            expected_k.reshape(-1, kvheads, dim)[slot] = key[token].cpu().bfloat16()
            expected_v.reshape(-1, kvheads, dim)[slot] = value[token].cpu().bfloat16()
    with PreparationSession(device=q.device, autotune=False, compile_workers=0) as session:
        writer = PreparedWriter(key, value, kc, vc, slots, session=session, name="writer")
        attention = PreparedAttention(q, kc, vc, pages, lengths, cu, output,
                                      session=session, name="attention", window=window)

        def forward():
            writer()
            attention()

        compiled = compile_forward(forward)
        compiled()
        torch.cuda.synchronize()
        assert torch.equal(kc.cpu(), expected_k.view(torch.uint16))
        assert torch.equal(vc.cpu(), expected_v.view(torch.uint16))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        for factor in (1., -.5):
            q.mul_(factor)
            output.fill_(float("nan"))
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            expected = torch.zeros_like(q, device="cpu")
            query_cpu, page_cpu = q.cpu(), pages.cpu()
            for row, (begin, end, length) in enumerate(((0, 4, 11), (4, 4, 0), (4, 6, 3))):
                for token in range(begin, end):
                    visible = length - (end - begin) + token - begin + 1
                    first = max(0, visible - window - 1) if window >= 0 else 0
                    positions = torch.arange(first, visible)
                    indices = page_cpu[row, positions // page].long()
                    keys = expected_k[indices, positions % page].float()
                    values = expected_v[indices, positions % page].float()
                    for head in range(heads):
                        kv = head // (heads // kvheads)
                        scores = (keys[:, kv] @ query_cpu[token, head]) / math.sqrt(dim)
                        expected[token, head] = scores.softmax(0) @ values[:, kv]
            torch.testing.assert_close(output.cpu(), expected, rtol=1e-5, atol=2e-6)
            assert torch.equal(output[6:], torch.zeros_like(output[6:]))
            assert torch.equal(kc.cpu(), expected_k.view(torch.uint16))
            assert torch.equal(vc.cpu(), expected_v.view(torch.uint16))
        graph.reset()


@torch.no_grad()
def test_recorded_preparation_preserves_live_pages_and_full_bf16_range(tmp_path):
    from b12x._lib.compile_plan import observe_programs
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.tuning import RecordedPreparationSession

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign the test V100 explicitly"
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires V100")
    q = torch.zeros((4, 2, 32), device="cuda")
    key_pattern = torch.tensor([2.**80, -(2.**80), 2.**-80, -(2.**-80),
                                2.**-133, -(2.**-133), 1.00390625, 1.01171875]).repeat(4)
    value_pattern = torch.tensor([2.**80, -(2.**80), 2.**-80, -(2.**-80)]).repeat(8)
    key = key_pattern[None, None].repeat(4, 1, 1).cuda()
    value = value_pattern[None, None].repeat(4, 1, 1).cuda()
    initial = torch.full((4, 2, 1, 32), .25, dtype=torch.bfloat16)
    kc, vc = initial.view(torch.uint16).cuda(), initial.view(torch.uint16).cuda()
    slots = torch.full((4,), -1, dtype=torch.int64, device="cuda")
    pages = torch.tensor([[2, 0], [1, 3]], dtype=torch.int32, device="cuda")
    lengths = torch.zeros((2,), dtype=torch.int32, device="cuda")
    cu = torch.zeros((3,), dtype=torch.int32, device="cuda")
    output = torch.empty_like(q)
    with RecordedPreparationSession(device=q.device, checkpoint=str(tmp_path),
            workload_schema="sm70-bf16-cache", numerical_policy="bf16-bits") as session:
        with observe_programs() as declarations:
            writer = PreparedWriter(key, value, kc, vc, slots, session=session, name="writer")
            attention = PreparedAttention(q, kc, vc, pages, lengths, cu, output,
                                          session=session, name="attention")
        assert not declarations
        report = session.tune()
        assert report["benchmarked_candidates"] > 0
        # Both the writer and attention private tuning fixtures must leave all
        # real pages intact, including pages outside the live request interval.
        assert torch.equal(kc.cpu(), initial.view(torch.uint16))
        assert torch.equal(vc.cpu(), initial.view(torch.uint16))

        def forward():
            writer()
            attention()

        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        slots.copy_(torch.tensor([4, 5, 0, -1], dtype=torch.int64, device="cuda"))
        lengths[0] = 3
        cu[1:] = 3
        for scale in (1., -.5):
            value.copy_((value_pattern * scale)[None, None].repeat(4, 1, 1))
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            expected_k, expected_v = initial.clone(), initial.clone()
            for token, slot in enumerate((4, 5, 0)):
                expected_k.reshape(-1, 1, 32)[slot] = key[token].cpu().bfloat16()
                expected_v.reshape(-1, 1, 32)[slot] = value[token].cpu().bfloat16()
            assert torch.equal(kc.cpu(), expected_k.view(torch.uint16))
            assert torch.equal(vc.cpu(), expected_v.view(torch.uint16))
            expected = torch.zeros_like(q, device="cpu")
            for token in range(3):
                # Zero queries yield uniform attention over the causal prefix.
                positions = torch.arange(token + 1)
                page = pages[0].cpu()[positions // 2].long()
                visible = expected_v[page, positions % 2, 0].float()
                expected[token] = visible.mean(0)
            torch.testing.assert_close(output.cpu(), expected, rtol=1e-5, atol=0)
        graph.reset()
