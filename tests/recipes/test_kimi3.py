"""Kimi K3 configuration, residual composition, and checkpoint-backed execution."""

from copy import deepcopy
import os
from pathlib import Path

import pytest

CHECKPOINT = Path("/data/models/Kimi-K3-MXFP4-CSF")


@pytest.fixture
def config():
    from ignition.recipes.kimi3.config import ModelConfig

    if not CHECKPOINT.is_dir():
        pytest.skip("Kimi K3 checkpoint is not installed")
    return ModelConfig.load(CHECKPOINT)


def test_checkpoint_geometry(config):
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.kimi3.pool import continuation_layout, page_bytes

    config.validate_shard(TensorParallelShard(0, 16))
    assert config.layers.count("kda") == 69
    assert config.layers.count("mla") == 24
    assert config.layers[0] == "kda" and config.layers[3] == "mla"
    layout = continuation_layout(config, 16)
    assert layout.region("layer0.recurrent").shape == (6, 128, 128)
    assert page_bytes(config) == 24 * 128 * 576 * 2


def test_quantization_and_layer_partition_rejected(config):
    from ignition.recipes.kimi3.config import ModelConfig

    raw = dict(model_type="kimi_k3", architectures=["KimiK3ForConditionalGeneration"],
               text_config=deepcopy(config.text), vision_config=config.vision,
               media_placeholder_token_id=163605, image_placeholder="<|kimi_image_placeholder|>")
    raw["text_config"]["linear_attn_config"]["kda_layers"].append(4)
    with pytest.raises(ValueError, match="partition"):
        ModelConfig.from_dict(raw)
    raw["text_config"] = deepcopy(config.text)
    raw["text_config"]["activation_situ_beta"] = 1.
    with pytest.raises(ValueError, match="activation_situ_beta"):
        ModelConfig.from_dict(raw)


def test_attention_residual_and_situ():
    import torch
    from ignition.recipes.kimi3.model import attention_residual, situ

    generator = torch.Generator().manual_seed(41)
    prefix, a, b = (torch.randn((3, 32), generator=generator).bfloat16() for _ in range(3))
    weight = torch.ones(32, dtype=torch.bfloat16)
    projection = torch.zeros((1, 32), dtype=torch.bfloat16)
    expected = torch.stack((a, b, prefix), 1).float().mean(1).bfloat16()
    torch.testing.assert_close(attention_residual(prefix, (a, b), weight, projection, 1e-5), expected)
    x = torch.tensor([[100., -100., 0., 100., 100., 100.]], dtype=torch.bfloat16)
    y = situ(x)
    assert torch.isfinite(y).all() and y[0, 0] == 100 and y[0, 2] == 0
    assert y[0, 1].abs() < 1e-30


@pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1", reason="requires CUDA checkpoint tests")
@pytest.mark.parametrize("mode,tokens", [("decode", 1), ("prefill", 65), ("prefill", 256)])
def test_checkpoint_kda(config, mode, tokens):
    import torch
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.kimi3.weights import KimiWeightLoader
    from ignition.recipes.kimi3.kda import PreparedKDA
    from ignition.components.recurrent import DeltaState, DeltaMetadata
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    manifest = inspect_indexed_checkpoint(CHECKPOINT)
    with KimiWeightLoader(manifest, device, shard=TensorParallelShard(0, 16),
                         prefixes=("language_model.model.layers.0.",)) as loader:
        weights = loader.load_layer(0)
    with RecordedPreparationSession(device=device, checkpoint=str(CHECKPOINT),
            workload_schema="kimi3-kda", numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        state = DeltaState.allocate(3, device, heads=6, channels=2304)
        metadata = DeltaMetadata.allocate(1, device)
        torch.manual_seed(41)
        source = torch.randn((tokens, 7168), device=device, dtype=torch.bfloat16)
        op = PreparedKDA(weights["attention"], source, state, metadata, mode=mode, eps=1e-5,
                        session=session, name="kda", workspace=arena, buffers={})
        session.tune(show_progress=False)
        metadata.query_start_loc[1] = tokens
        metadata.convolution_slots[0] = 1
        metadata.state_indices[0, 0] = 1
        metadata.final_state_indices[0] = 1
        metadata.num_seqs.fill_(1)
        metadata.num_tokens.fill_(tokens)
        metadata.fresh.fill_(True)
        state.convolution.zero_()
        state.recurrent.zero_()
        output = op(source).clone()
        assert torch.isfinite(output).all() and output.abs().max() > 0
        expected_state = state.recurrent[1].clone()
        from b12x.sequence.kda_prefill.reference import recurrent_kda
        from torch.nn import functional as F

        w = weights["attention"]
        projected = F.linear(source, w["input"])
        qkv, beta, forget = projected.split((2304, 6, 128), -1)
        convolution = F.conv1d(F.pad(qkv.T.float()[None], (3, 0)),
                               w["convolution"][:, None], groups=2304).bfloat16()
        q, k, v = F.silu(convolution)[0].T.reshape(tokens, 3, 6, 128).unbind(1)
        raw_g = F.linear(forget, w["forget"]).view(tokens, 6, 128)
        recurrent, reference_state, _ = recurrent_kda(q, k, v, raw_g, beta,
            w["A_log"], w["dt_bias"].view(6, 128), lower_bound=-5.,
            initial_state=torch.zeros_like(expected_state))
        normalized = recurrent.float() * torch.rsqrt(recurrent.float().square().mean(-1, keepdim=True) + 1e-5)
        gate = F.linear(source, w["gate"]).view(tokens, 6, 128).float().sigmoid()
        reference = F.linear((normalized * w["norm"] * gate).bfloat16().flatten(1), w["output"])
        for actual, expected in ((output, reference), (expected_state, reference_state)):
            relative_rms = (actual.float() - expected.float()).square().mean().sqrt() / expected.float().square().mean().sqrt()
            assert relative_rms.item() < .015
        state.convolution.zero_()
        state.recurrent.zero_()
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            replayed = op(source)
        graph.replay()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(replayed, output, rtol=0, atol=0)
        torch.testing.assert_close(state.recurrent[1], expected_state, rtol=0, atol=0)
        assert not state.recurrent[0].count_nonzero()


@pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1", reason="requires CUDA checkpoint tests")
@pytest.mark.parametrize("mode,tokens", [("decode", 1), ("prefill", 65)])
def test_checkpoint_mla(config, mode, tokens):
    import torch
    from torch.nn import functional as F
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.paged_batch import Metadata
    from ignition.recipes.kimi3.weights import KimiWeightLoader
    from ignition.recipes.kimi3.attention import PreparedMLA
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    manifest = inspect_indexed_checkpoint(CHECKPOINT)
    with KimiWeightLoader(manifest, device, shard=TensorParallelShard(0, 16),
                         prefixes=("language_model.model.layers.3.self_attn.",)) as loader:
        weights = loader.load_attention(3)
    with RecordedPreparationSession(device=device, checkpoint=str(CHECKPOINT),
            workload_schema="kimi3-mla", numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        metadata = Metadata.allocate(tokens, 1, 2, device)
        cache = torch.zeros((2, 128, 576), dtype=torch.bfloat16, device=device)
        torch.manual_seed(41)
        source = torch.randn((tokens, 7168), device=device, dtype=torch.bfloat16)
        op = PreparedMLA(weights, source, metadata, cache, mode=mode, session=session,
                         name="mla", workspace=arena, buffers={})
        session.tune(show_progress=False)
        metadata.query_start_loc[1] = tokens
        metadata.lengths.fill_(tokens)
        metadata.page_table.copy_(torch.tensor([[0, 1]], device=device))
        metadata.positions.copy_(torch.arange(tokens, device=device))
        output = op(source).clone()
        w = weights
        projected = F.linear(source, w["input"])
        def norm(x, weight):
            return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * weight
        query = F.linear(norm(projected[:, :1536], w["query_norm"]), w["q_b"]).view(tokens, 6, 192)
        latent = norm(projected[:, 1536:-64], w["kv_norm"])
        keys = torch.einsum("tk,hdk->thd", latent.float(), w["key"].float()).bfloat16()
        keys = torch.cat((keys, projected[:, None, -64:].expand(-1, 6, -1)), -1)
        values = torch.einsum("tk,hkd->thd", latent.float(), w["value"].float()).bfloat16()
        expected = F.scaled_dot_product_attention(query.transpose(0, 1)[None], keys.transpose(0, 1)[None],
            values.transpose(0, 1)[None], is_causal=True).squeeze(0).transpose(0, 1).flatten(1)
        expected = F.linear(expected * F.linear(source, w["gate"]).sigmoid(), w["output"])
        error = (output.float() - expected.float()).square().mean().sqrt() / expected.float().square().mean().sqrt()
        assert error.item() < .02
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            replayed = op(source)
        graph.replay()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(replayed, output, rtol=0, atol=0)
