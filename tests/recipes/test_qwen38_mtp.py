from types import SimpleNamespace

import pytest
import torch

from flare.preparation.fixtures import mtp_feedback as feedback_fixtures
from flare.preparation.scratch import ScratchArena
from flare.recipes.qwen38 import mtp, moe


@pytest.mark.parametrize("tokens,rows", [(0, 1), (2, 3), (True, 1), (2, 0)])
def test_invalid_persistent_geometry(tokens, rows):
    with pytest.raises(ValueError, match="positive output rows"):
        mtp.MTPScratch.allocate(tokens, rows, "cpu")


def test_persistent_views_keep_full_residual_and_bounded_logits_separate():
    scratch = mtp.MTPScratch.allocate(16, 4, "cpu")
    view = scratch.rows(8, 2)
    assert view.residual.shape == (8, 10240)
    assert view.hidden.shape == (8, 2560)
    assert view.logits.shape == (2, 248320)
    assert view.residual.data_ptr() == scratch.residual.data_ptr()
    assert view.logits.data_ptr() == scratch.logits.data_ptr()
    outputs = (view.embedding, view.residual, view.hidden, view.selected_hidden, view.logits)
    assert len({x.untyped_storage().data_ptr() for x in outputs}) == len(outputs)
    with pytest.raises(ValueError, match="exceeds"):
        scratch.rows(17, 2)


def test_feedback_refresh_and_private_benchmark_preserve_input_state(monkeypatch):
    from b12x._lib.scratch import ScratchBufferSpec

    arena = ScratchArena("cpu")
    spec = lambda n: (ScratchBufferSpec(name="feedback", shape=(n,), dtype=torch.uint8, device=torch.device("cpu")),)
    plan = SimpleNamespace(size=64, version=0)
    plan.scratch_specs = lambda: spec(plan.size)
    plan.request = lambda **kwargs: SimpleNamespace(**kwargs)
    requests, resources = [], {}
    monkeypatch.setattr(mtp.mtp_feedback, "Caps", lambda **kwargs: kwargs)
    monkeypatch.setattr(mtp.mtp_feedback, "plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(mtp.mtp_feedback, "bind", lambda p, **kwargs: SimpleNamespace(version=p.version, **kwargs))
    monkeypatch.setattr(mtp, "register_finalizer", lambda s, fn, *, workspace: workspace.register_finalizer(fn))

    def resource(session, key, factory):
        if key not in resources:
            resources[key] = factory()
        return resources[key]

    monkeypatch.setattr(feedback_fixtures, "benchmark_resource", resource)
    token = torch.full((2, 2560), 8, dtype=torch.bfloat16)
    hidden = torch.full((2, 10240), 4, dtype=torch.bfloat16)
    output = torch.full_like(hidden, -1)
    module = SimpleNamespace(config=SimpleNamespace(rms_norm_eps=1e-6),
        pre_fc_norm_embedding=SimpleNamespace(weight=torch.zeros(2560, dtype=torch.bfloat16)),
        pre_fc_norm_hidden=SimpleNamespace(weight=torch.zeros(10240, dtype=torch.bfloat16)),
        fc_embedding=SimpleNamespace(weight=torch.empty((2560, 2560), device="meta", dtype=torch.bfloat16)),
        fc_hidden=SimpleNamespace(weight=torch.empty((2560, 2560), device="meta", dtype=torch.bfloat16)))
    op = mtp.PreparedMTPFeedback(module, token, hidden, output, arena,
                                session=SimpleNamespace(prepare=requests.extend), name="draft.feedback")
    first = op.binding
    plan.size, plan.version = 512, 1
    arena.reserve(1024)
    arena.trim(512)
    arena.finalize()
    arena.freeze()
    assert op.binding is not first and op.binding.version == 1
    assert op.scratch.data_ptr() == arena.tensor.data_ptr()
    assert op.binding.output.untyped_storage().data_ptr() == output.untyped_storage().data_ptr()
    assert op.binding.multi_state.untyped_storage().data_ptr() == hidden.untyped_storage().data_ptr()
    state = SimpleNamespace(layout=SimpleNamespace(scratch_specs=lambda: spec(256)),
        bind=lambda **kwargs: SimpleNamespace(**kwargs), run=lambda binding, **kwargs: binding.output.copy_(binding.multi_state))
    one = requests[0].benchmark_call(state)
    two = requests[0].benchmark_call(state)
    assert one.owners[0] is two.owners[0]
    fixture = one.owners[0]
    fixture.token_embedding.fill_(float("nan")); fixture.multi_state.zero_()
    one.produce(); one.run()
    assert torch.all(fixture.token_embedding == .125)
    assert torch.all(fixture.output == .25)
    assert torch.all(token == 8) and torch.all(hidden == 4) and torch.all(output == -1)
    assert one.owners[1].data_ptr() == two.owners[1].data_ptr() == arena.tensor.data_ptr()


def test_mtp_forward_preserves_feedback_alignment_and_all_streams(monkeypatch):
    scratch = mtp.MTPScratch.allocate(3, 2, "cpu")
    ids = torch.tensor([8, -1, 9], dtype=torch.int64)
    target = torch.arange(12, dtype=torch.bfloat16).view(3, 4, 1).expand(3, 4, 2560).reshape(3, 10240).contiguous()
    original = target.clone()
    positions = torch.tensor([[3, 3, 3], [-1, -1, -1], [9, 9, 9]])
    metadata = SimpleNamespace(request_ids=torch.tensor([0, -1, 1]), rope_positions=positions)
    selected = torch.tensor([2, 0], dtype=torch.int64)
    plan = SimpleNamespace(request=lambda **kwargs: kwargs)
    monkeypatch.setattr(mtp.embedding, "query_from_call", lambda *args, **kwargs: None)
    monkeypatch.setattr(mtp.embedding, "plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(mtp.embedding, "run", lambda weight, ids, *, out, plan: out.copy_(ids[:, None]))

    class Feedback:
        def __init__(self, module, token, hidden, output, workspace, **kwargs):
            self.token, self.hidden, self.output = token, hidden, output

        def __call__(self):
            self.output.view(-1, 4, 2560).copy_(self.hidden.view(-1, 4, 2560) + self.token[:, None])

    class HC:
        def __init__(self, module, buffers, **kwargs):
            self.buffers = buffers

        def mix(self, source):
            self.buffers.block_input.copy_(source.view(-1, 4, 2560).mean(1))
            return self.buffers.block_input

        def combine(self, source, block):
            return (source.view(-1, 4, 2560) + block[:, None]).flatten(1)

    class Attention:
        def __init__(self, module, weights, source, buffers, state, meta, rotary, **kwargs):
            assert meta.rope_positions is positions
            assert kwargs["draft_tokens"] == 7
            self.buffer = buffers.output

        def __call__(self, source):
            return self.buffer.copy_(source * 2)

        def rebind_pages_for_preparation(self, state):
            self.state = state

    class Moe:
        def __init__(self, source, weights, **kwargs):
            self.buffer = kwargs["scratch"].output

        def __call__(self, source):
            return self.buffer.copy_(source * 3)

    class Head:
        def __init__(self, source, weight, output, **kwargs):
            self.output = output

        def __call__(self, source):
            return self.output.copy_(source[:, :1])

    for name, value in (("PreparedMTPFeedback", Feedback), ("PreparedHyperconnection", HC),
                        ("PreparedAttention", Attention), ("PreparedQwenMoe", Moe), ("BF16Projection", Head)):
        monkeypatch.setattr(mtp, name, value)
    layer = SimpleNamespace(attn_hyper_connection=object(), mlp_hyper_connection=object(), self_attn=object())
    weights = SimpleNamespace(module=SimpleNamespace(layers=[layer], hyper_connection_mixer=object()),
                              attention={}, moe=object(), embedding=object(), lm_head=object())
    operator = mtp.PreparedMTP(weights, scratch, ids, target, object(), metadata, object(),
        ScratchArena("cpu"), output_indices=selected, session=SimpleNamespace(prepare=lambda _: None), name="draft")
    for _ in range(2):
        out = operator()
        expected = original.view(3, 4, 2560) + ids.clamp_min(0)[:, None, None]
        expected[1].zero_()
        expected += 2 * expected.mean(1, keepdim=True)
        expected += 3 * expected.mean(1, keepdim=True)
        assert out is operator.output
        assert torch.equal(out.residual, expected.flatten(1))
        assert torch.equal(out.hidden, expected.mean(1))
        assert torch.equal(out.logits, expected.mean(1)[selected, :1].expand_as(out.logits))
        assert torch.equal(target, original)
        assert scratch.embedding_ids.tolist() == ids.clamp_min(0).tolist()
        ids[0] = 3
        selected.copy_(torch.tensor([0, 2]))
    replacement = object()
    operator.rebind_pages_for_preparation(replacement)
    assert operator.attention.state is replacement


def test_shared_bf16_expert_uses_bf16_projections_without_activation_quantization(monkeypatch):
    source = torch.ones((1, 2560), dtype=torch.bfloat16)
    weights = moe.SharedExpertWeights(torch.empty((1280, 2560), dtype=torch.bfloat16, device="meta"),
        torch.empty((2560, 640), dtype=torch.bfloat16, device="meta"),
        torch.empty((1, 2560), dtype=torch.bfloat16, device="meta"))
    calls = []
    monkeypatch.setattr(moe, "BF16Projection", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(moe, "QuantizedProjection", lambda *a, **kw: pytest.fail("BF16 MTP must not use quantized projections"))
    plan = SimpleNamespace(request=lambda **kwargs: kwargs)
    monkeypatch.setattr(moe.hyperconnection, "Caps", lambda **kwargs: kwargs)
    monkeypatch.setattr(moe.hyperconnection, "plan", lambda *args, **kwargs: plan)
    moe.PreparedSharedExpert(source, weights, session=SimpleNamespace(prepare=lambda _: None), name="mtp.shared")
    assert [kw["name"] for _, kw in calls] == ["mtp.shared.gate_up", "mtp.shared.down", "mtp.shared.shared_gate"]
