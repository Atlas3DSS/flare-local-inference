"""Actual-checkpoint Qwen GDN components through prepared CUDA graphs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from flare.components.recurrent import DeltaMetadata
from flare.preparation import inspect_qwen38_checkpoint
from flare.preparation.qwen38 import TEXT_PREFIX
from flare.recipes.qwen38.gdn import (
    GDNScratch,
    GDNState,
    PreparedGDN,
    _validate_geometry,
)
from flare.recipes.qwen38.weights import QwenWeightLoader
from flare.reference.qwen38_quant import (
    CheckpointTensors,
    decode_mxfp8,
    quantize_mxfp8_activation,
)

CUDA = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires explicit CUDA_VISIBLE_DEVICES and FLARE_RUN_CUDA_TESTS=1",
)
PREFIX = f"{TEXT_PREFIX}.layers.0.linear_attn"
PROJECTIONS = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")


def test_gdn_cpu_buffer_contracts():
    state = GDNState.allocate(3, torch.device("cpu"), speculative_tokens=2)
    metadata = DeltaMetadata.allocate(2, torch.device("cpu"), verification_columns=3)
    scratch = GDNScratch.allocate(6, torch.device("cpu"))
    assert state.convolution.shape == (3, 10240, 5)
    assert state.recurrent.shape == (3, 48, 128, 128)
    assert metadata.state_indices.shape == (2, 3)
    assert metadata.num_tokens.item() == metadata.num_seqs.item() == 0
    assert torch.all(metadata.accepted_tokens == 1)
    assert torch.all(metadata.convolution_slots == -1)
    assert scratch.rows(2).projected_qkv.data_ptr() == scratch.projected_qkv.data_ptr()
    assert scratch.rows(2).output.shape == (2, 2560)


@pytest.mark.parametrize(
    "factory,args,kwargs",
    [
        (GDNState.allocate, (1, "cpu"), {}),
        (GDNState.allocate, (3, "cpu"), {"speculative_tokens": 8}),
        (DeltaMetadata.allocate, (0, "cpu"), {}),
        (DeltaMetadata.allocate, (2, "cpu"), {"verification_columns": 0}),
        (GDNScratch.allocate, (0, "cpu"), {}),
    ],
)
def test_gdn_invalid_capacities_fail_before_preparation(factory, args, kwargs):
    with pytest.raises(ValueError):
        factory(*args, **kwargs)


@pytest.fixture(scope="module")
def manifest():
    path = Path("/data/checkpoint-releases/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD")
    path = Path(os.environ.get("FLARE_QWEN38_SNAPSHOT", path))
    if not path.exists():
        pytest.skip("exact Qwen checkpoint snapshot is unavailable")
    return inspect_qwen38_checkpoint(path)


def _module(manifest):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextGatedDeltaNet,
    )

    configuration = json.loads((manifest.snapshot / "config.json").read_text())[
        "text_config"
    ]
    config = Qwen4ExpTextConfig.from_dict(configuration)
    with torch.device("meta"):
        result = Qwen4ExpTextGatedDeltaNet(config, layer_idx=0)
    return result.eval().requires_grad_(False)


def test_pinned_canonical_gdn_geometry(manifest):
    module = _module(manifest)
    _validate_geometry(module)
    assert tuple(module.conv1d.weight.shape) == (10240, 1, 4)
    assert tuple(module.norm.weight.shape) == (128,)
    module.num_v_heads = 16
    with pytest.raises(ValueError, match="geometry"):
        _validate_geometry(module)


def _parameters(module, load):
    for name in ("A_log", "dt_bias", "conv1d.weight", "norm.weight"):
        parent, _, attribute = name.rpartition(".")
        destination = module.get_submodule(parent) if parent else module
        setattr(
            destination,
            attribute,
            torch.nn.Parameter(load(f"{PREFIX}.{name}"), requires_grad=False),
        )


class _ReferenceProjection(torch.nn.Module):
    def __init__(self, weight, scales):
        super().__init__()
        self.register_buffer("weight", decode_mxfp8(weight, scales))

    def forward(self, source):
        return (quantize_mxfp8_activation(source) @ self.weight.T).to(torch.bfloat16)


@pytest.fixture(scope="module")
def reference_module(manifest):
    module = _module(manifest)
    with CheckpointTensors(manifest, byte_budget=256 << 20) as reader:
        _parameters(module, reader.read)
        for projection in PROJECTIONS:
            prefix = f"{PREFIX}.{projection}"
            setattr(
                module,
                projection,
                _ReferenceProjection(
                    reader.read(prefix + ".weight"),
                    reader.read(prefix + ".weight_scale"),
                ),
            )
    return module


def _cache():
    from transformers.cache_utils import Cache, LinearAttentionLayer

    return Cache(layers=[LinearAttentionLayer()])


def _numerical(actual, expected, *, limit=0.015):
    actual, expected = actual.float().cpu(), expected.float().cpu()
    error = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(
        expected
    ).clamp_min(1e-12)
    assert float(error) < limit, f"relative L2 error {float(error)} >= {limit}"
    torch.testing.assert_close(
        actual, expected, rtol=0.025, atol=float(expected.abs().max()) * 0.015 + 1e-6
    )
    return float(error)


def _metadata(metadata, lengths, slots, *, fresh, initial=None, columns=None):
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    device = metadata.query_start_loc.device
    metadata.query_start_loc.fill_(-999)
    metadata.query_start_loc[: len(starts)].copy_(
        torch.tensor(starts, device=device, dtype=torch.int32)
    )
    metadata.num_seqs.fill_(len(lengths))
    metadata.num_tokens.fill_(sum(lengths))
    metadata.convolution_slots.fill_(-1)
    metadata.convolution_slots[: len(slots)].copy_(
        torch.tensor(slots, device=device, dtype=torch.int64)
    )
    metadata.fresh.fill_(fresh)
    metadata.accepted_tokens.fill_(1)
    metadata.initial_state_indices.zero_()
    metadata.final_state_indices.zero_()
    metadata.checkpoint_state_indices.zero_()
    metadata.checkpoint_offsets.zero_()
    metadata.state_indices.zero_()
    if initial is not None:
        metadata.initial_state_indices[: len(initial)].copy_(
            torch.tensor(initial, device=device, dtype=torch.int64)
        )
    metadata.final_state_indices[: len(slots)].copy_(
        torch.tensor(slots, device=device, dtype=torch.int64)
    )
    if columns is None:
        metadata.state_indices[: len(slots), 0].copy_(
            torch.tensor(slots, device=device, dtype=torch.int64)
        )
    else:
        metadata.state_indices[: len(columns)].copy_(
            torch.tensor(columns, device=device, dtype=torch.int64)
        )


def _load_cuda(manifest):
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("assign a CUDA device explicitly")
    assert torch.cuda.get_device_capability(0) in ((12, 0), (12, 1))
    module = _module(manifest)
    with QwenWeightLoader(manifest, "cuda:0", prefixes=(PREFIX + ".",)) as loader:
        _parameters(module, loader.load_tensor)
        projections = loader.load_gdn_projections(PREFIX)
    return module, projections


@CUDA
@torch.no_grad()
@pytest.mark.parametrize("initial_tokens", (17, 65))
def test_actual_checkpoint_prefill_decode_continuation_graphs(
    manifest, reference_module, initial_tokens, record_property
):
    from b12x.preparation import PreparationSession

    module, projections = _load_cuda(manifest)
    device = torch.device("cuda", 0)
    state = GDNState.allocate(4, device)
    prefill_metadata, decode_metadata = (
        DeltaMetadata.allocate(3, device),
        DeltaMetadata.allocate(3, device),
    )
    prefill_scratch, decode_scratch = (
        GDNScratch.allocate(max(initial_tokens + 1, 32), device),
        GDNScratch.allocate(3, device),
    )
    prefill_input = torch.zeros(
        (prefill_scratch.output.shape[0], 2560), dtype=torch.bfloat16, device=device
    )
    decode_input = torch.zeros((3, 2560), dtype=torch.bfloat16, device=device)
    with PreparationSession(
        device=device, autotune=False, compile_workers=1
    ) as session:
        prefill = PreparedGDN(
            module,
            projections,
            prefill_input,
            prefill_scratch,
            state,
            prefill_metadata,
            mode="prefill",
            session=session,
            name="gdn.prefill",
            checkpoint_export=True,
        )
        decode = PreparedGDN(
            module,
            projections,
            decode_input,
            decode_scratch,
            state,
            decode_metadata,
            mode="decode",
            session=session,
            name="gdn.decode",
        )
        prefill(prefill_input)
        decode(decode_input)
        torch.cuda.synchronize()
        session.freeze()
        pool = torch.cuda.graph_pool_handle()
        prefill_graph, decode_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(prefill_graph, pool=pool):
            prefill(prefill_input)
        with torch.cuda.graph(decode_graph, pool=pool):
            decode(decode_input)
        caches = [_cache(), _cache()]
        generator = torch.Generator().manual_seed(741)
        errors = {}
        state.convolution.fill_(9)
        state.recurrent.zero_()
        for mode, lengths in (
            ("prefill", (initial_tokens, 1)),
            ("decode", (1, 1)),
            ("prefill", (3, 2)),
        ):
            source = (
                torch.randn((sum(lengths), 2560), generator=generator).to(
                    torch.bfloat16
                )
                * 0.1
            )
            graph, inputs, metadata, scratch = (
                (prefill_graph, prefill_input, prefill_metadata, prefill_scratch)
                if mode == "prefill"
                else (decode_graph, decode_input, decode_metadata, decode_scratch)
            )
            first = mode == "prefill" and lengths == (initial_tokens, 1)
            _metadata(
                metadata,
                list(lengths),
                [1, 2],
                fresh=first,
                initial=[0, 0] if first else [1, 2],
            )
            if first:
                metadata.checkpoint_state_indices[0] = 3
                metadata.checkpoint_offsets[0] = 16
            inputs.zero_()
            inputs[: source.shape[0]].copy_(source)
            outputs = []
            start = 0
            for index, length in enumerate(lengths):
                outputs.append(
                    reference_module(
                        source[start : start + length].unsqueeze(0),
                        cache_params=caches[index],
                    )[0]
                )
                start += length
            expected = torch.cat(outputs)
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            label = f"{mode}_{lengths[0]}"
            errors[label + ".output"] = _numerical(
                scratch.output[: sum(lengths)], expected
            )
            assert torch.count_nonzero(scratch.output[sum(lengths) :]) == 0
            for request, slot in enumerate((1, 2)):
                reference_state = caches[request].layers[0]
                errors[f"{label}.state_{slot}"] = _numerical(
                    state.recurrent[slot].transpose(-1, -2),
                    reference_state.recurrent_states[0][0],
                    limit=0.012,
                )
                errors[f"{label}.convolution_{slot}"] = _numerical(
                    state.convolution[slot], reference_state.conv_states[0][0, :, -3:]
                )
            if first:
                checkpoint_cache = _cache()
                reference_module(
                    source[:16].unsqueeze(0), cache_params=checkpoint_cache
                )
                errors["checkpoint_16.state"] = _numerical(
                    state.recurrent[3].transpose(-1, -2),
                    checkpoint_cache.layers[0].recurrent_states[0][0],
                    limit=0.012,
                )
        prior = state.recurrent.clone()
        convolution = state.convolution.clone()
        _metadata(prefill_metadata, [], [], fresh=False)
        prefill_graph.replay()
        torch.cuda.synchronize()
        assert torch.count_nonzero(prefill_scratch.output) == 0
        torch.testing.assert_close(state.recurrent, prior, rtol=0, atol=0)
        torch.testing.assert_close(state.convolution, convolution, rtol=0, atol=0)
        record_property("relative_l2", json.dumps(errors, sort_keys=True))
        prefill_graph.reset()
        decode_graph.reset()


@CUDA
@torch.no_grad()
def test_actual_checkpoint_decode_accepts_selected_speculative_state(
    manifest, reference_module, record_property
):
    from b12x.preparation import PreparationSession

    module, projections = _load_cuda(manifest)
    device = torch.device("cuda", 0)
    state = GDNState.allocate(5, device, speculative_tokens=2)
    metadata = DeltaMetadata.allocate(1, device, verification_columns=3)
    scratch = GDNScratch.allocate(3, device)
    inputs = torch.zeros((3, 2560), dtype=torch.bfloat16, device=device)
    source = (
        torch.randn((4, 2560), generator=torch.Generator().manual_seed(192)) * 0.1
    ).to(torch.bfloat16)
    cache = _cache()
    expected_tokens = [
        reference_module(source[i : i + 1].unsqueeze(0), cache_params=cache)[0]
        for i in range(2)
    ]
    expected_final = reference_module(source[3:4].unsqueeze(0), cache_params=cache)[0]
    with PreparationSession(
        device=device, autotune=False, compile_workers=1
    ) as session:
        adapter = PreparedGDN(
            module,
            projections,
            inputs,
            scratch,
            state,
            metadata,
            mode="decode",
            session=session,
            name="gdn.verify",
        )
        adapter(inputs)
        session.freeze()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            adapter(inputs)
        _metadata(metadata, [3], [1], fresh=True, columns=[[1, 2, 3]])
        inputs.copy_(source[:3])
        graph.replay()
        torch.cuda.synchronize()
        errors = {
            "verification.output": _numerical(
                scratch.output[:2], torch.cat(expected_tokens)
            )
        }
        _metadata(metadata, [1], [1], fresh=False, columns=[[1, 2, 3]])
        metadata.accepted_tokens.fill_(2)
        inputs.zero_()
        inputs[:1].copy_(source[3:4])
        graph.replay()
        torch.cuda.synchronize()
        errors["accepted_2.output"] = _numerical(scratch.output[:1], expected_final)
        errors["accepted_2.state"] = _numerical(
            state.recurrent[1].transpose(-1, -2),
            cache.layers[0].recurrent_states[0][0],
            limit=0.012,
        )
        errors["accepted_2.convolution"] = _numerical(
            state.convolution[1, :, :3], cache.layers[0].conv_states[0][0, :, -3:]
        )
        assert torch.count_nonzero(scratch.output[1:]) == 0
        record_property("relative_l2", json.dumps(errors, sort_keys=True))
        graph.reset()
