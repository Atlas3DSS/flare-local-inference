"""Dense learned feedback, ordinary residuals and cache absorption against BF16."""

from dataclasses import replace
import math
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ignition.recipes.qwen38_dense import mtp
from ignition.recipes.qwen38_dense.config import ModelConfig
from ignition.recipes.qwen38_dense.mtp_weights import MTPWeights, tensor_shapes, validate_checkpoint


def _config(*, full=False, attention=False):
    # attention=True: the real attention geometry (24 q / 4 KV heads, head_dim 256) that the SM70 S1L draft attention
    # admits, with a toy hidden size and vocabulary.
    hidden, intermediate, dim, heads, kvheads, vocabulary = (
        (5120, 17408, 256, 24, 4, 248320) if full else (32, 64, 256, 24, 4, 64) if attention else (32, 64, 8, 4, 2, 64))
    return ModelConfig(dict(hidden_size=hidden, intermediate_size=intermediate, head_dim=dim,
        num_attention_heads=heads, num_key_value_heads=kvheads, vocab_size=vocabulary,
        mtp_num_hidden_layers=1, mtp_use_dedicated_embeddings=False, rms_norm_eps=1e-6,
        rope_parameters=dict(partial_rotary_factor=.25, rope_theta=10000000)), {})


def _weights(config, device="cpu"):
    generator = torch.Generator().manual_seed(3870)
    tensors = {}
    for name, shape in tensor_shapes(config).items():
        value = (torch.randn(shape, generator=generator) * .125).bfloat16()
        tensors[name] = (value.float() if "norm" in name else value).to(device)
    embedding = torch.randn((config.vocabulary, config.hidden), generator=generator).bfloat16().to(device)
    head = (torch.randn((config.vocabulary, config.hidden), generator=generator) * .125).bfloat16().to(device)
    return MTPWeights(config, tensors, embedding, head, 0, 0)


def _metadata(device="cpu"):
    # Two independent requests, first crossing a page boundary; two padded tokens.
    pages = torch.tensor([[2, 0], [1, 3]], dtype=torch.int32, device=device)
    return SimpleNamespace(
        positions=torch.tensor([127, 128, 0, 1, -1, -1], dtype=torch.int64, device=device),
        starts=torch.tensor([127, 0], dtype=torch.int64, device=device),
        lengths=torch.tensor([2, 2], dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32, device=device),
        page_table=pages, token_page_table=pages[[0, 0, 1, 1, 1, 1]],
    )


class _Projection:
    def __init__(self, source, weight, output, **kwargs):
        self.weight, self.output = weight, output

    def __call__(self, source):
        torch.mm(source, self.weight.float().T, out=self.output)
        return self.output


class _Embedding:
    def __init__(self, weight, ids, output, **kwargs):
        self.weight, self.ids, self.output = weight, ids, output

    def __call__(self):
        self.output.copy_(self.weight[self.ids])
        return self.output


class _Vocabulary(_Projection):
    def __init__(self, source, weight, local_output, output, **kwargs):
        super().__init__(source, weight, output)


class _RMSNorm:
    def __init__(self, source, weight, output, epsilon, **kwargs):
        self.weight, self.output, self.epsilon = weight, output, epsilon

    def __call__(self, source):
        self.output.copy_(_norm(source, self.weight, self.epsilon).float())
        return self.output


class _Writer:
    def __init__(self, key, value, key_cache, value_cache, slots, **kwargs):
        self.args = key, value, key_cache, value_cache, slots

    def __call__(self):
        key, value, kc, vc, slots = self.args
        for row, slot in enumerate(slots.tolist()):
            if slot >= 0:
                kc.view(torch.bfloat16).flatten(0, 1)[slot].copy_(key[row])
                vc.view(torch.bfloat16).flatten(0, 1)[slot].copy_(value[row])


def _attend(query, key, value, pages, lengths, cu):
    output = torch.zeros_like(query)
    for request in range(pages.shape[0]):
        begin, end = cu[request:request + 2].tolist()
        for token in range(begin, end):
            visible = int(lengths[request]) - end + token + 1
            positions = torch.arange(visible)
            physical = pages[request, positions // 128].long()
            keys, values = key[physical, positions % 128].float(), value[physical, positions % 128].float()
            for head in range(query.shape[1]):
                kvhead = head // (query.shape[1] // key.shape[2])
                scores = keys[:, kvhead] @ query[token, head] / math.sqrt(query.shape[2])
                output[token, head] = scores.softmax(0) @ values[:, kvhead]
    return output


class _DecodeAttention:
    """CPU stand-in for the SM70 S1L draft attention: reads the real four-agent DecodeAgents packing."""

    def __init__(self, query, key, value, agents, output, *, rows_per_agent, **kwargs):
        self.args, self.agents, self.output, self.rows_per_agent = (query, key, value), agents, output, rows_per_agent

    def __call__(self):
        query, key, value = self.args
        a = self.agents
        self.output.copy_(_attend(query, key.view(torch.bfloat16), value.view(torch.bfloat16), a.pages, a.lengths, a.cu))


@pytest.fixture
def cpu_primitives(monkeypatch):
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CPU test initialized CUDA"))
    for name, replacement in (("UnquantizedProjection", _Projection), ("PreparedEmbedding", _Embedding),
                              ("PreparedVocabularyProjection", _Vocabulary), ("PreparedWriter", _Writer),
                              ("PreparedDecodeAttention", _DecodeAttention), ("PreparedRMSNorm", _RMSNorm)):
        monkeypatch.setattr(mtp, name, replacement)


def _norm(value, weight, epsilon):
    value = value.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + epsilon)
            * (1 + weight.float())).bfloat16()


def _reference(weights, ids, hidden, metadata, initial_k, initial_v, indices):
    """Independent eager BF16 decoder; no software rounding or prepared ops."""
    c, w = weights.config, weights.tensors
    epsilon = c.text["rms_norm_eps"]
    heads, kvheads, dim = (c.text[field] for field in ("num_attention_heads", "num_key_value_heads", "head_dim"))
    active = metadata.positions >= 0
    embedding = weights.embedding[torch.where(active, ids, 0)]
    combined = torch.cat((_norm(embedding, w["pre_fc_norm_embedding.weight"], epsilon),
                          _norm(hidden, w["pre_fc_norm_hidden.weight"], epsilon)), -1)
    residual = F.linear(combined, w["fc.weight"]).masked_fill(~active[:, None], 0)
    normalized = _norm(residual, w["layers.0.input_layernorm.weight"], epsilon)
    qg, key, value = (F.linear(normalized, w["layers.0.self_attn." + field + ".weight"])
                     for field in ("q_proj", "k_proj", "v_proj"))
    query, gate = qg.view(-1, heads, 2 * dim).chunk(2, -1)
    query = _norm(query, w["layers.0.self_attn.q_norm.weight"], epsilon)
    key = _norm(key.view(-1, kvheads, dim), w["layers.0.self_attn.k_norm.weight"], epsilon)
    value = value.view_as(key)
    rotary = c.text["rope_parameters"]
    width = int(dim * rotary["partial_rotary_factor"])
    inverse = 1 / rotary["rope_theta"] ** (torch.arange(0, width, 2).float() / width)
    angle = metadata.positions.float()[:, None] * inverse
    angle = torch.cat((angle, angle), -1)[:, None]
    cos, sin = angle.cos().bfloat16(), angle.sin().bfloat16()

    def rotate(tensor):
        first, second = tensor[..., :width].chunk(2, -1)
        rotated = torch.cat((-second, first), -1)
        return torch.cat((tensor[..., :width] * cos + rotated * sin, tensor[..., width:]), -1)

    query, key = rotate(query), rotate(key)
    kc, vc = initial_k.clone(), initial_v.clone()
    for token in range(ids.numel()):
        position = int(metadata.positions[token])
        if position >= 0:
            page = int(metadata.token_page_table[token, position // 128])
            kc[page, position % 128], vc[page, position % 128] = key[token], value[token]
    attended = _attend(query.float(), kc, vc, metadata.page_table,
                       (metadata.starts + metadata.lengths).int(), metadata.query_start_loc).bfloat16()
    attended = (attended * gate.sigmoid()).reshape(ids.numel(), -1)
    residual = residual + F.linear(attended, w["layers.0.self_attn.o_proj.weight"])
    normalized = _norm(residual, w["layers.0.post_attention_layernorm.weight"], epsilon)
    gate, up = (F.linear(normalized, w["layers.0.mlp." + field + ".weight"])
                for field in ("gate_proj", "up_proj"))
    residual = (residual + F.linear(F.silu(gate) * up, w["layers.0.mlp.down_proj.weight"]))
    residual.masked_fill_(~active[:, None], 0)
    output_hidden = _norm(residual, w["norm.weight"], epsilon)
    logits = F.linear(output_hidden[indices], weights.lm_head)
    return residual.float(), output_hidden.float(), logits.float(), kc, vc


def test_exact_checkpoint_geometry_and_reject_incomplete_or_quantized_head():
    config = _config(full=True)
    tensors = {"mtp." + name: SimpleNamespace(shape=shape, dtype="BF16", nbytes=2 * math.prod(shape))
               for name, shape in tensor_shapes(config).items()}
    manifest = SimpleNamespace(tensors=tensors)
    assert len(tensors) == 15
    assert validate_checkpoint(manifest, config) == 849398784
    assert tensors["mtp.fc.weight"].shape == (5120, 10240)
    assert tensors["mtp.layers.0.self_attn.q_proj.weight"].shape == (12288, 5120)
    tensors["mtp.norm.weight"].dtype = "F32"
    with pytest.raises(ValueError, match="BF16"):
        validate_checkpoint(manifest, config)
    del tensors["mtp.norm.weight"]
    with pytest.raises(ValueError, match="missing"):
        validate_checkpoint(manifest, config)
    with pytest.raises(ValueError, match="one layer"):
        tensor_shapes(replace(config, text={**config.text, "mtp_num_hidden_layers": 2}))


def test_quantized_draft_uses_weight_only_activation_policy(monkeypatch):
    calls = []
    def projection(source, weight, output, **options):
        calls.append(options)
        return output
    monkeypatch.setattr(mtp, "QuantizedProjection", projection)
    output = torch.empty(1, 32)
    assert mtp._projection(torch.zeros_like(output), object(), output,
                           session=None, name="draft", row_invariant=True) is output
    assert calls == [dict(session=None, name="draft", row_invariant=True, activation_mode="a16")]


@torch.no_grad()
def test_feedback_decoder_and_cache_only_match_bf16_reference(cpu_primitives):
    config, metadata = _config(), _metadata()
    weights = _weights(config)
    scratch = mtp.MTPScratch.allocate(6, 2, config, "cpu")
    ids = torch.tensor([2, 3, 4, 5, -200, 999999])
    indices = torch.tensor([1, 3])
    torch.manual_seed(3880)
    hidden = torch.randn((6, config.hidden)).bfloat16().float()
    initial_k = (torch.randn((4, 128, 2, 8)) * .1).bfloat16()
    initial_v = (torch.randn((4, 128, 2, 8)) * .1).bfloat16()
    key, value = initial_k.view(torch.uint16).clone(), initial_v.view(torch.uint16).clone()
    operation = mtp.PreparedMTP(weights, scratch, ids, hidden, metadata, key, value, None,
                               output_indices=indices, session=None, name="full")
    for recursive in range(2):
        reference = _reference(weights, ids, hidden, metadata, initial_k, initial_v, indices)
        output = operation()
        for actual, expected in zip((output.residual, output.hidden, output.logits), reference[:3], strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(key, reference[3].view(torch.uint16))
        assert torch.equal(value, reference[4].view(torch.uint16))
        assert torch.equal(output.hidden[4:], torch.zeros_like(output.hidden[4:]))
        assert output.hidden.data_ptr() == operation.hidden.data_ptr()
        assert not torch.equal(output.hidden, output.residual)
        # Recursive proposals consume final normalized hidden, same as target input.
        hidden.copy_(output.hidden)
    # Four-agent packing of the draft metadata: two live agents (starts + query lengths), two padded agents.
    agents = operation.attention.agents
    assert agents.lengths.tolist() == [129, 2, 0, 0] and agents.cu.tolist() == [0, 2, 4, 4, 4]
    assert agents.pages[2:].eq(-1).all() and torch.equal(agents.pages[:2], metadata.page_table)
    assert operation.attention.attention.rows_per_agent == 3
    absorbed_key, absorbed_value = initial_k.view(torch.uint16).clone(), initial_v.view(torch.uint16).clone()
    absorption = mtp.PreparedMTP(weights, scratch, ids, hidden, metadata, absorbed_key, absorbed_value, None,
        output_indices=indices, session=None, name="absorb", cache_only=True, mode="prefill")
    assert set(absorption.attention.projections) == {"k_proj", "v_proj"}
    assert not hasattr(absorption, "mlp") and not hasattr(absorption, "head")
    operation()
    assert absorption() is None
    assert torch.equal(absorbed_key, key) and torch.equal(absorbed_value, value)


def test_draft_attention_admission_matches_the_target_decoder(cpu_primitives):
    """The draft attention takes the target decoder's S1L admission: <= 32,768 tokens, <= 4 agents, <= 5 rows each."""
    config, weights = _config(), _weights(_config())
    for tokens, agents, blocks in ((6, 2, 257), (6, 1, 2), (10, 5, 2), (5, 2, 2)):
        scratch = mtp.MTPScratch.allocate(tokens, 1, config, "cpu")
        metadata = SimpleNamespace(positions=torch.zeros(tokens, dtype=torch.int64),
            starts=torch.zeros(agents, dtype=torch.int64), lengths=torch.zeros(agents, dtype=torch.int32),
            query_start_loc=torch.zeros(agents + 1, dtype=torch.int32),
            page_table=torch.full((agents, blocks), -1, dtype=torch.int32),
            token_page_table=torch.full((tokens, blocks), -1, dtype=torch.int32))
        cache = torch.zeros((4, 128, 2, 8), dtype=torch.uint16)
        with pytest.raises(ValueError, match="SM70 decode attention contract"):
            mtp.PreparedMTP(weights, scratch, torch.zeros(tokens, dtype=torch.int64),
                torch.zeros((tokens, config.hidden)), metadata, cache, cache.clone(), None,
                output_indices=torch.zeros(1, dtype=torch.int64), session=None, name="admission")


def test_scratch_views_and_feedback_alias_rejected(cpu_primitives):
    config, weights = _config(), _weights(_config())
    scratch = mtp.MTPScratch.allocate(6, 2, config, "cpu")
    assert scratch.rows(4, 1).residual.data_ptr() == scratch.residual.data_ptr()
    with pytest.raises(ValueError, match="capacity"):
        scratch.rows(7, 2)
    with pytest.raises(ValueError, match="alias"):
        mtp.PreparedMTP(weights, scratch, torch.zeros(6, dtype=torch.int64), scratch.hidden, _metadata(),
            torch.zeros((4, 128, 2, 8), dtype=torch.uint16), torch.zeros((4, 128, 2, 8), dtype=torch.uint16),
            None, output_indices=torch.tensor([1, 3]), session=None, name="alias")


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned V100")
@torch.no_grad()
def test_dense_mtp_compiles_and_replays_with_existing_sm70_operators():
    from b12x.preparation import PreparationSession
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign the test V100 explicitly"
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires V100")
    device = torch.device("cuda", torch.cuda.current_device())
    # astra-13: the draft attention is the SM70 S1L decode attention, which admits only the real attention geometry.
    config, cpu_weights = _config(attention=True), _weights(_config(attention=True))
    weights, metadata = _weights(config, device), _metadata(device)
    ids_cpu, indices_cpu = torch.tensor([2, 3, 4, 5, -200, 999999]), torch.tensor([1, 3])
    ids, indices = ids_cpu.to(device), indices_cpu.to(device)
    torch.manual_seed(3880)
    hidden = torch.randn((6, config.hidden)).bfloat16().float().to(device)
    initial = torch.zeros((4, 128, 4, 256), dtype=torch.bfloat16)
    key, value = initial.view(torch.uint16).to(device), initial.view(torch.uint16).to(device)
    scratch = mtp.MTPScratch.allocate(6, 2, config, device)
    workspace = ScratchArena(device)
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        operation = mtp.PreparedMTP(weights, scratch, ids, hidden, metadata, key, value, workspace,
            output_indices=indices, session=session, name="mtp")
        compiled = compile_forward(operation)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        for factor in (1., -.5):
            hidden.mul_(factor)
            expected = _reference(cpu_weights, ids_cpu, hidden.cpu(), _metadata(), initial, initial, indices_cpu)
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            torch.testing.assert_close(scratch.hidden.cpu(), expected[1], rtol=.02, atol=.015625)
            torch.testing.assert_close(scratch.logits.cpu(), expected[2], rtol=.02, atol=.015625)
            assert torch.equal(key.cpu(), expected[3].view(torch.uint16))
            assert torch.equal(value.cpu(), expected[4].view(torch.uint16))
        graph.reset()


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned V100")
@pytest.mark.parametrize("batch", [1, 4])
@torch.inference_mode()
def test_dense_mtp_captured_dynamic_draft_rows_pages_and_empty_replay(batch):
    """One row/request as in PreparedRound; replay changes packed row ownership and physical page mapping."""
    from b12x.preparation import PreparationSession
    from ignition.components.paged_batch import Metadata
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign the test V100 explicitly"
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires V100")
    device = torch.device("cuda", torch.cuda.current_device())
    config = _config(attention=True)
    weights, cpu_weights = _weights(config, device), _weights(config)
    metadata, cpu_metadata = (Metadata.allocate(batch, batch, 2, where) for where in (device, "cpu"))
    scratch = mtp.MTPScratch.allocate(batch, batch, config, device)
    hidden = torch.zeros((batch, config.hidden), device=device)
    generator = torch.Generator().manual_seed(3884 + batch)
    initial_k = (torch.randn((2 * batch + 1, 128, 4, 256), generator=generator) * .2).bfloat16()
    initial_v = (torch.randn(initial_k.shape, generator=generator) * .2).bfloat16()
    key, value = initial_k.view(torch.uint16).to(device), initial_v.view(torch.uint16).to(device)
    workspace = ScratchArena(device)
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        operation = mtp.PreparedMTP(weights, scratch, metadata.token_ids, hidden, metadata, key, value, workspace,
            output_indices=metadata.sample_indices, session=session, name="dynamic-mtp")
        compiled = compile_forward(operation)
    workspace.freeze()
    # Materialize the same complete forward before capture; initially every metadata row is inactive.
    compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()

    all_live = [1] * batch
    middle_inactive = [1, 0, 1, 1] if batch == 4 else [0]
    suffix_live = [0, 1, 0, 1] if batch == 4 else [1]
    # Full -> ragged -> sparse -> empty -> full: positions 127/128 cross the draft page boundary.
    cases = [(127, all_live), (128, middle_inactive), (129, suffix_live),
             (130, [0] * batch), (131, all_live)]
    for ordinal, (position, lengths) in enumerate(cases):
        pages = torch.randperm(2 * batch, generator=generator).int().view(batch, 2)
        starts = torch.tensor([position - row for row in range(batch)], dtype=torch.int64)
        lengths_cpu = torch.tensor(lengths, dtype=torch.int32)
        ids = torch.arange(batch, dtype=torch.int64)[:, None] + ordinal + 2
        cpu_metadata.set(starts, lengths_cpu, torch.arange(1, batch + 1), pages, ids, torch.tensor([batch]))
        for field in cpu_metadata.__dataclass_fields__:
            source = getattr(cpu_metadata, field)
            if isinstance(source, torch.Tensor):
                getattr(metadata, field).copy_(source)
        feedback = torch.randn((batch, config.hidden), generator=generator).bfloat16().float()
        hidden.copy_(feedback[cpu_metadata.requests])
        before_k, before_v = key.cpu().view(torch.bfloat16), value.cpu().view(torch.bfloat16)
        expected = _reference(cpu_weights, cpu_metadata.token_ids, hidden.cpu(), cpu_metadata,
                              before_k, before_v, cpu_metadata.sample_indices)
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        captured = (scratch.hidden.clone(), scratch.logits.clone(), key.clone(), value.clone())
        # Independent BF16 arithmetic keeps the existing operator-specific bounds. Complete cache comparisons
        # additionally establish that inactive, old-prefix and unmapped sentinel bytes are untouched.
        torch.testing.assert_close(scratch.hidden.cpu(), expected[1], rtol=.02, atol=.015625)
        live_requests = lengths_cpu.bool()
        torch.testing.assert_close(scratch.logits.cpu()[live_requests], expected[2][live_requests],
                                   rtol=.02, atol=.015625)
        assert torch.equal(key.cpu(), expected[3].view(torch.uint16))
        assert torch.equal(value.cpu(), expected[4].view(torch.uint16))
        active = sum(lengths)
        assert torch.equal(scratch.hidden[active:].view(torch.int32),
                           torch.zeros_like(scratch.hidden[active:], dtype=torch.int32))
        agents = operation.attention.agents
        assert agents.cu.cpu().tolist() == [*cpu_metadata.query_start_loc.tolist(), *([active] * (4 - batch))]
        assert agents.lengths.cpu().tolist() == [
            *[int(starts[row]) + length if length else 0 for row, length in enumerate(lengths)],
            *([0] * (4 - batch))]
        # Same compiled operators, inputs and initial state: captured and eager executions must agree in bytes.
        key.copy_(before_k.view(torch.uint16))
        value.copy_(before_v.view(torch.uint16))
        compiled()
        torch.cuda.synchronize()
        for actual, saved in zip((scratch.hidden, scratch.logits, key, value), captured, strict=True):
            assert torch.equal(actual.view(torch.uint8), saved.view(torch.uint8))
    graph.reset()
