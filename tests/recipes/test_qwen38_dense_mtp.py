"""Dense Qwen MTP checkpoint arithmetic and shifted continuation histories."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ignition.recipes import resolve


SNAPSHOT = resolve("qwen3.8-27b").snapshot(Path.home() / ".cache/huggingface/hub")
CUDA = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned GPU")


def test_dense_mtp_configuration_and_ordinary_mode():
    from ignition.recipes.qwen38_dense.bootstrap import PreparationConfig

    config = PreparationConfig(max_active=4, speculative_widths=tuple(range(1, 9)))
    assert {(4, width) for width in range(1, 9)} <= set(config.speculative_shapes)
    assert PreparationConfig(mtp=False).speculative_shapes == ()
    with pytest.raises(ValueError):
        PreparationConfig(speculative_widths=(1, 9))
    with pytest.raises(ValueError, match="fixed verification"):
        PreparationConfig(speculative_fixed_width=8)


@pytest.fixture(scope="module")
def weights():
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.weights import DirectWeightLoader
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.recipes.qwen38_dense.config import ModelConfig, TEXT_PREFIX
    from ignition.recipes.qwen38_dense.weights import MTPWeights

    if not (SNAPSHOT / "model.safetensors.index.json").exists():
        pytest.skip("the pinned NVIDIA Qwen3.8-27B checkpoint is required")
    manifest = inspect_indexed_checkpoint(SNAPSHOT)
    config = ModelConfig.load(SNAPSHOT)
    device = torch.device("cuda", 0)
    with DirectWeightLoader(manifest, device, prefixes=(TEXT_PREFIX + ".embed_tokens.", "lm_head.")) as loader:
        embedding = loader.load_tensor(TEXT_PREFIX + ".embed_tokens.weight")
        head = loader.load_linear("lm_head")
    return SimpleNamespace(config=config, tree=SimpleNamespace(embed_tokens=SimpleNamespace(weight=embedding)),
        head=head, draft=MTPWeights.load(manifest, config, device), shard=TensorParallelShard(0, 1))


def reference_predictor(weights, token, previous, position, key, value):
    """Canonical dense layer with the prepared path's FP8 KV arithmetic."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    c, w = weights.config, weights.draft
    residual = torch.nn.functional.linear(torch.cat((w.embedding_norm(weights.tree.embed_tokens.weight[token]),
        w.hidden_norm(previous)), -1), w.fc)
    layer, a = w.layer, w.layer.self_attn
    source = layer.input_layernorm(residual)
    q, gate = a.q_proj(source).view(1, c.text["num_attention_heads"], -1).chunk(2, dim=-1)
    k = a.k_proj(source).view(1, c.text["num_key_value_heads"], -1)
    v = a.v_proj(source).view_as(k)
    rotary = c.text["rope_parameters"]
    dim = int(c.text["head_dim"] * rotary["partial_rotary_factor"])
    inverse = 1. / (rotary["rope_theta"] ** (torch.arange(0, dim, 2, device=source.device).float() / dim))
    angles = torch.cat((inverse * position, inverse * position))[None]
    q, k = apply_rotary_pos_emb(a.q_norm(q), a.k_norm(k), angles.cos().bfloat16(), angles.sin().bfloat16())
    key = torch.cat((key, k.clamp(-448, 448).to(torch.float8_e4m3fn).bfloat16()))
    value = torch.cat((value, v.clamp(-448, 448).to(torch.float8_e4m3fn).bfloat16()))
    repeats = q.shape[1] // k.shape[1]
    probabilities = torch.einsum("qhd,khd->hqk", q.float(), key.repeat_interleave(repeats, 1).float())
    probabilities = (probabilities * c.text["head_dim"] ** -.5).softmax(-1)
    attended = torch.einsum("hqk,khd->qhd", probabilities, value.repeat_interleave(repeats, 1).float()).bfloat16()
    attended = attended.flatten(1) * gate.flatten(1).sigmoid()
    residual = residual + a.o_proj(attended)
    return w.norm(residual + layer.mlp(layer.post_attention_layernorm(residual))), key, value


@CUDA
@torch.no_grad()
def test_checkpoint_predictor_and_captured_inactive_rows(weights):
    from ignition.components.paged_batch import Metadata
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.qwen38_dense.mtp import PreparedPredictor
    from ignition.recipes.qwen38_dense.pool import ContinuationPool

    device = torch.device("cuda", 0)
    pool = ContinuationPool.for_model(weights.config, 5, 8, device, mtp=True)
    metadata = Metadata.allocate(4, 4, 2, device)
    arena = ScratchArena(device)
    with RecordedPreparationSession(device=device, checkpoint="qwen38-dense-mtp", workload_schema="predictor",
                                    numerical_policy="default") as session:
        session.register_scratch(arena)
        predictor = PreparedPredictor(weights, metadata, pool, session=session, workspace=arena, name="mtp")
        session.tune()
        torch.manual_seed(7127)
        previous = torch.randn_like(predictor.previous)
        starts = torch.tensor([0, 127, 128, 6], device=device)
        lengths = torch.tensor([1, 1, 0, 1], dtype=torch.int32, device=device)
        tables = torch.arange(8, dtype=torch.int32, device=device).view(4, 2)
        tokens = torch.tensor([[3], [5], [7], [11]], device=device)
        metadata.set(starts, lengths, torch.tensor([3, 1, 4, 2], device=device), tables, tokens,
                     torch.tensor([4], dtype=torch.int32, device=device))
        predictor.previous.copy_(previous[metadata.requests])
        cache = pool.attention("draft", weights.config)
        for tensor in cache:
            tensor.view(torch.uint8).zero_()
        expected = []
        for row in (0, 1, 3):
            history = torch.zeros((int(starts[row]), 4, 256), dtype=torch.bfloat16, device=device)
            result, _, _ = reference_predictor(weights, tokens[row], previous[row:row+1], int(starts[row]) + 1,
                                               history, history)
            expected.append(result[0])
        expected = torch.stack(expected)
        predictor()
        actual = predictor.hidden[:3].clone()
        error = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
        print(f"MTP BF16/FP8 reference relative L2: {error:.6g}")
        assert error < .008
        unused = tuple(t.view(torch.uint8)[4:6].clone() for t in cache)
        compiled = compile_forward(predictor)
        compiled()
        torch.testing.assert_close(predictor.hidden[:3], actual, atol=.015625, rtol=.0078125)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=stream):
            compiled()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(predictor.hidden[:3], actual, atol=.015625, rtol=.0078125)
        for tensor, saved in zip(cache, unused, strict=True):
            assert torch.equal(tensor.view(torch.uint8)[4:6], saved)
        graph.reset()


@CUDA
@torch.no_grad()
def test_feedback_keeps_only_accepted_pairs_across_page_boundaries(weights):
    from ignition.components.paged_batch import BatchInputs, Metadata
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.qwen38_dense.mtp import PreparedFeedback
    from ignition.recipes.qwen38_dense.pool import ContinuationPool

    device = torch.device("cuda", 0)
    pool = ContinuationPool.for_model(weights.config, 5, 6, device, mtp=True)
    inputs = BatchInputs.allocate(16, 3, device, capacity=2)
    inputs.row_slot_ids.copy_(torch.tensor([1, 0], device=device))
    inputs.live_rows.fill_(2)
    tables = torch.arange(6, dtype=torch.int32, device=device).view(2, 3)
    source = Metadata.allocate(16, 2, 3, device, samples=16)
    auxiliary = torch.randn((16, weights.config.hidden), dtype=torch.bfloat16, device=device)
    counts = torch.zeros(2, dtype=torch.int32, device=device)
    pool.last_hidden.normal_()
    saved_hidden = pool.last_hidden.clone()
    arena = ScratchArena(device)
    with RecordedPreparationSession(device=device, checkpoint="qwen38-dense-mtp", workload_schema="feedback",
                                    numerical_policy="default") as session:
        session.register_scratch(arena)
        feedback = PreparedFeedback(weights, auxiliary, source, pool, inputs=inputs, tables=tables,
                                    session=session, workspace=arena, name="feedback")
        session.tune()
        compiled = compile_forward(lambda: feedback(auxiliary, counts))
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        for start in (0, 127, 128, 129):
            for accepted in range(9):
                tokens = torch.arange(16, device=device).view(2, 8) + 10
                source.set(torch.tensor([start, 17], device=device), torch.tensor([8, 8], device=device),
                    torch.tensor([3, 1], device=device), tables.flip(0), tokens, inputs.live_rows)
                counts.copy_(torch.tensor([accepted, 0], device=device))
                cache = pool.attention("draft", weights.config)
                for tensor in cache:
                    tensor.view(torch.uint8).zero_()
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(pool.last_hidden, saved_hidden)
                assert not cache[0][:3].view(torch.uint8).count_nonzero()
                assert not cache[1][:3].view(torch.uint8).count_nonzero()
                skip = int(start == 0)
                for index in range(skip, accepted):
                    previous = auxiliary[index-1:index] if index else saved_hidden[3:4]
                    empty = torch.empty((0, 4, 256), dtype=torch.bfloat16, device=device)
                    _, key, value = reference_predictor(weights, tokens[0, index:index+1], previous, start + index,
                                                       empty, empty)
                    position = start + index - 1
                    page = 3 + position // 128
                    for tensor, expected in ((cache[0], key), (cache[1], value)):
                        actual = tensor[page, position % 128].float()
                        error = (actual - expected[0].float()).norm() / expected[0].float().norm()
                        assert error < .01
                expected_positions = set(range(max(start-1, 0), start + accepted - 1)) if accepted > skip else set()
                for position in range(384):
                    if position not in expected_positions:
                        for tensor in cache:
                            assert not tensor[3 + position // 128, position % 128].view(torch.uint8).count_nonzero()
        graph.reset()


@CUDA
@pytest.mark.parametrize("endpoint", [1, 127, 128, 129, 256])
@torch.no_grad()
def test_draft_partial_page_snapshot_branch_restore(weights, endpoint):
    from ignition.components.sampling import DeviceRequestState
    from ignition.components.state_transfer import PreparedStateTransfer, StateCommands
    from ignition.recipes.qwen38_dense.pool import ContinuationPool

    device = torch.device("cuda", 0)
    pool = ContinuationPool.for_model(weights.config, 4, 6, device, mtp=True)
    commands = StateCommands.allocate(device, capacity=2)
    tables = torch.arange(6, device=device, dtype=torch.int32).view(2, 3)
    requests = DeviceRequestState.allocate(device, vocab_size=128, capacity=2)
    requests.consumed_tokens.fill_(endpoint)
    rows = torch.tensor([[1] * 8, [2] * 8], device=device)
    transfer = PreparedStateTransfer(pool, commands, tables, pages=pool.partial_pages(), page_tokens=128,
        state_rows=rows, requests=requests, trailing_pages=pool.draft_pages(), trailing_tables=tables)
    transfer.prepare()
    pool.backing[1].random_(0, 256)
    pool.region("consumed_tokens")[1] = endpoint
    for _, page in pool.pages.items():
        page.random_(0, 256)
    expected = {name: page[(endpoint-1)//128].clone() for name, page in pool.draft_pages()}
    commands.request_slots[0] = 0
    commands.source_slots[0], commands.destination_slots[0] = 1, 3
    commands.expected_consumed[0], commands.num_ops[0] = endpoint, 1
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        transfer.validate()
        transfer.apply()
    graph.replay()
    torch.cuda.synchronize()
    assert not commands.errors.count_nonzero()
    saved = pool.backing[3].clone()
    commands.request_slots[0] = 1
    commands.source_slots[0], commands.destination_slots[0] = 3, 2
    graph.replay()
    torch.cuda.synchronize()
    assert not commands.errors.count_nonzero()
    assert torch.equal(pool.backing[3], saved)
    if (endpoint-1) % 128:
        for name, page in pool.draft_pages():
            assert torch.equal(page[3 + (endpoint-1)//128], expected[name])
    graph.reset()
