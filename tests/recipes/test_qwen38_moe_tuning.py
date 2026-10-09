"""Serialized expert candidates share bounded scratch and exact routing inputs."""

from types import SimpleNamespace

import pytest
import torch
from b12x._lib.scratch import ScratchBufferSpec, scratch_tensor

from sparkgap.recipes.qwen38 import moe
from sparkgap.preparation.scratch import ScratchArena


@pytest.mark.parametrize("tokens", [4, 17])
@pytest.mark.parametrize("storage", ["arena", "tensor"])
def test_candidate_cohort_reuses_exact_operands_and_one_arena(monkeypatch, tokens, storage):
    source = torch.full((tokens, 32), 7, dtype=torch.bfloat16)
    ids = torch.zeros((tokens, 10), dtype=torch.int32)
    weights = torch.zeros((tokens, 10), dtype=torch.float32)
    output = torch.zeros_like(source)
    router_weight = torch.randn((512, 32), generator=torch.Generator().manual_seed(23)).bfloat16()
    experts = SimpleNamespace(num_experts=512)
    arena = ScratchArena("cpu")
    arena.reserve(256)
    arena.freeze()
    workspace = arena if storage == "arena" else arena.tensor
    resources, requests, runs = {}, [], []
    session = SimpleNamespace(prepare=lambda declarations: requests.extend(declarations))

    def specs(size):
        return (ScratchBufferSpec(name="workspace", shape=(size,),
            dtype=torch.uint8, device=torch.device("cpu")),)

    plan = SimpleNamespace(scratch_specs=lambda: specs(16),
        request=lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(moe.fused_moe, "plan_execution", lambda **kwargs: plan)
    monkeypatch.setattr(moe.fused_moe, "bind", lambda plan, **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(moe, "register_finalizer", lambda *args, **kwargs: None)
    router_plan = object()

    def route(selected_plan, logits, selected, routed_ids, routed_weights):
        assert selected_plan is router_plan
        values, indices = logits.float().topk(10, -1)
        selected.copy_(values)
        routed_ids.copy_(indices)
        routed_weights.copy_(values.softmax(-1))

    monkeypatch.setattr(moe.fused_moe, "route_topk", route)

    def resource(session, key, factory):
        if key not in resources:
            resources[key] = factory()
        return resources[key]

    monkeypatch.setattr(moe, "benchmark_resource", resource)
    operator = moe.PreparedRoutedExperts(source, experts, ids, weights, output,
        session=session, name="prefill.layer0.experts", workspace=workspace,
        router_weight=router_weight, router_plan=router_plan)
    default_scratch = operator.scratch
    assert operator.scratch.data_ptr() == arena.tensor.data_ptr()

    def candidate(index, size):
        def bind(**kwargs):
            scratch_tensor(kwargs["scratch"], specs(size), owner="TP MoE")
            return SimpleNamespace(**kwargs)

        def run(binding):
            runs.append(index)
            binding.output.copy_(binding.a)
            binding.scratch.fill_(index + 1)

        return SimpleNamespace(scratch=SimpleNamespace(scratch_specs=lambda: specs(size)),
            bind=bind, run=run)

    calls = [requests[0].benchmark_call(candidate(index, size))
             for index, size in enumerate((32, 128, 256))]
    fixture = resources[("qwen_moe_inputs", "prefill.layer0.experts")]
    reference = moe._ExpertBenchmarkInputs(source, experts=512, top_k=10,
        router_weight=router_weight, router_plan=router_plan)
    assert len(resources) == 1
    assert all(call.owners[0] is fixture and call.owners[1].data_ptr() == arena.tensor.data_ptr() for call in calls)
    assert all(call.output is fixture.output and call.owners[2].scratch.data_ptr() == arena.tensor.data_ptr() for call in calls)
    assert all(call.owners[3] is experts and call.owners[2].input_scales_static for call in calls)
    assert all(call.reset is None and call.restore is None for call in calls)
    assert all(call.benchmark_producers is fixture.producers for call in calls)
    assert operator.scratch.data_ptr() == arena.tensor.data_ptr() and operator.scratch is default_scratch
    assert len(fixture.producers) == (5 if tokens == 4 else 3)
    for index in (0, 1, 2, 0):
        call = calls[index]
        for produce, expected_produce in zip(call.benchmark_producers, reference.producers, strict=True):
            fixture.source.fill_(float("nan"))
            fixture.router.ids.fill_(-1)
            fixture.router.weights.fill_(float("nan"))
            fixture.output.fill_(float("nan"))
            produce()
            expected_produce()
            assert torch.equal(fixture.source, reference.source)
            assert torch.equal(fixture.router.ids, reference.router.ids)
            assert torch.equal(fixture.router.weights, reference.router.weights)
            call.run()
            assert torch.equal(fixture.output, reference.source)
    assert runs == [index for index in (0, 1, 2, 0) for _ in fixture.producers]
    assert torch.all(source == 7) and not ids.count_nonzero() and not weights.count_nonzero()
    assert not output.count_nonzero()
    with pytest.raises((ValueError, RuntimeError), match=f"requires {512 if storage == 'arena' else 257}"):
        requests[0].benchmark_call(candidate(3, 257))
    tensors = (fixture.fixture, fixture.source, fixture.output, fixture.routes,
        *(getattr(fixture.router, field) for field in fixture.router.__dataclass_fields__),
        fixture.route_weights, fixture.logits)
    storage = {value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
               for value in tensors if value is not None}
    assert fixture.nbytes == sum(storage.values())
    assert router_weight.untyped_storage().data_ptr() not in storage
