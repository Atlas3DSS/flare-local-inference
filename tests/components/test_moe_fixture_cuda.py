"""Model-route tuning inputs preserve text routing with an image-aware plan."""

import os

import pytest
import torch
from b12x.preparation import PreparationSession

from ignition.components.moe import PreparedRouter, RouterScratch
from ignition.preparation.fixtures.fused_moe import ExpertBenchmarkInputs

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


@torch.no_grad()
def test_image_aware_plan_can_produce_text_benchmark_routes():
    device = torch.device("cuda:0")
    torch.manual_seed(97)
    tokens, hidden, experts, top_k = 64, 128, 384, 6
    source = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device)
    weight = torch.randn((experts, hidden), dtype=torch.bfloat16, device=device)
    bias = torch.randn(experts, dtype=torch.float32, device=device)
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        router = PreparedRouter(torch.zeros((tokens, experts), device=device),
            RouterScratch.allocate(tokens, device, experts=experts, top_k=top_k),
            session=session, name="image_aware", score_func="sqrtsoftplus",
            routed_scaling_factor=1.5, correction_bias=bias, image_correction_bias=-bias,
            image_mask=torch.ones(tokens, dtype=torch.bool, device=device))
        fixture = ExpertBenchmarkInputs(source, experts=experts, top_k=top_k,
            allocate_router=RouterScratch.allocate, router_weight=weight,
            router_plan=router.plan, correction_bias=bias)
        for produce in fixture.producers[-2:]:
            produce()
            scores = torch.nn.functional.softplus(fixture.logits).sqrt()
            expected = (scores + bias).topk(top_k, dim=-1).indices
            assert torch.equal(fixture.router.ids.long(), expected)
            selected = scores.gather(1, expected)
            torch.testing.assert_close(fixture.router.weights, selected / selected.sum(-1, keepdim=True) * 1.5)
