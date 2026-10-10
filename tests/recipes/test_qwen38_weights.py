from __future__ import annotations

import os
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch

from flare.components.weights import join_nvfp4_csf
from flare.preparation import inspect_qwen38_checkpoint
from flare.preparation.qwen38 import MODEL_ID, REVISION, TEXT_PREFIX
from flare.recipes.qwen38.weights import (
    QwenWeightLoader,
    _expert_ids,
    _expert_scale_vectors,
    allocate_expert_scale_scratch,
    expert_memory_plan,
)
from flare.reference.qwen38_quant import CheckpointTensors, decode_csf_scales


@pytest.fixture(scope="module")
def manifest():
    default = Path.home() / ".cache/huggingface/hub" / f"models--{MODEL_ID.replace('/', '--')}" / "snapshots" / REVISION
    snapshot = Path(os.environ.get("FLARE_QWEN38_SNAPSHOT", default))
    if not snapshot.exists():
        pytest.skip("exact Qwen checkpoint snapshot is unavailable")
    return inspect_qwen38_checkpoint(snapshot)


def test_csf_join_preserves_up_gate_order_and_exception_payload():
    up = torch.zeros((1, 48), dtype=torch.uint8)
    gate = torch.full((1, 48), 0x11, dtype=torch.uint8)
    up[:, :16] = 20
    gate[:, :16] = 40
    up_ex = torch.tensor([(101 << 24) | 0, (102 << 24) | 63], dtype=torch.uint32)
    gate_ex = torch.tensor([(201 << 24) | 1, (202 << 24) | 62], dtype=torch.uint32)
    fixed, exceptions = join_nvfp4_csf(up, up_ex, gate, gate_ex, rows=16, columns=4)
    decoded = decode_csf_scales(fixed, exceptions, rows=32, columns=4)
    expected = torch.cat((
        decode_csf_scales(up, up_ex, rows=16, columns=4),
        decode_csf_scales(gate, gate_ex, rows=16, columns=4),
    ))
    assert torch.equal(decoded, expected)
    assert (exceptions.to(torch.int64) & 0xFFFFFF).tolist() == [0, 63, 65, 126]


@pytest.mark.parametrize("positions", [[2, 1], [3, 3], [64]])
def test_csf_join_rejects_corrupt_exception_positions(positions):
    fixed = torch.zeros((1, 48), dtype=torch.uint8)
    exceptions = torch.tensor(positions, dtype=torch.uint32)
    with pytest.raises(ValueError, match="unique, sorted, in-range"):
        join_nvfp4_csf(fixed, exceptions, fixed, exceptions, rows=16, columns=4)


def _scalars():
    values = {}
    for expert, base in ((3, 1.0), (5, 2.0)):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"{TEXT_PREFIX}.layers.0.mlp.experts.{expert}.{projection}"
            values[f"{prefix}.weight_scale_2"] = torch.tensor(base * .01)
            values[f"{prefix}.input_scale"] = torch.tensor(base * .125)
    return values


def test_expert_scale_vectors_preserve_order_and_per_expert_values():
    values = _expert_scale_vectors(_scalars(), 0, (5, 3))
    torch.testing.assert_close(values[0], torch.tensor([.02, .01]), rtol=0, atol=0)
    torch.testing.assert_close(values[1], torch.tensor([.02, .01]), rtol=0, atol=0)
    assert values[2].tolist() == [4, 8]
    assert values[3].tolist() == [4, 8]


@pytest.mark.parametrize("suffix", ["weight_scale_2", "input_scale"])
def test_fused_experts_reject_differing_gate_up_scales(suffix):
    sources = _scalars()
    sources[f"{TEXT_PREFIX}.layers.0.mlp.experts.3.gate_proj.{suffix}"] *= 2
    with pytest.raises(ValueError, match="gate/up.*differ"):
        _expert_scale_vectors(sources, 0, (3, 5))


@pytest.mark.parametrize("invalid", [0, -1, float("inf"), float("nan")])
def test_expert_scales_reject_invalid_values(invalid):
    sources = _scalars()
    sources[f"{TEXT_PREFIX}.layers.0.mlp.experts.5.down_proj.input_scale"] = torch.tensor(float(invalid))
    with pytest.raises(ValueError, match="finite and positive"):
        _expert_scale_vectors(sources, 0, (3, 5))


def test_expert_scales_reject_reciprocal_overflow():
    sources = _scalars()
    sources[f"{TEXT_PREFIX}.layers.0.mlp.experts.5.down_proj.input_scale"] = torch.tensor(1e-40)
    with pytest.raises(ValueError, match="reciprocal scales must remain finite"):
        _expert_scale_vectors(sources, 0, (3, 5))


@pytest.mark.parametrize("layer, ids", [(48, None), (True, None), (0, []), (0, [0, 0]), (0, [512]), (0, [False])])
def test_invalid_expert_selection(layer, ids):
    with pytest.raises(ValueError):
        _expert_ids(layer, ids)


def test_loader_construction_and_memory_plan_are_cpu_only(manifest):
    loader = QwenWeightLoader(manifest, "cuda:2")
    with pytest.raises(RuntimeError, match="open QwenWeightLoader"):
        loader.load_tensor(f"{TEXT_PREFIX}.embed_tokens.weight")
    plan = expert_memory_plan(manifest, 0)
    assert plan.expert_ids == tuple(range(512))
    assert plan.packed_bytes == 1_258_291_200
    assert plan.scalar_bytes == 12_288
    assert plan.scale_scratch_bytes == 157_286_400
    assert 75 << 20 < plan.csf_bytes < 90 << 20
    with pytest.raises(ValueError, match="CUDA"):
        allocate_expert_scale_scratch("cpu")


def test_actual_layer_scales_and_csf_order(manifest):
    sources = {}
    with CheckpointTensors(manifest, byte_budget=2 << 20) as reader:
        for expert in range(512):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                prefix = f"{TEXT_PREFIX}.layers.0.mlp.experts.{expert}.{projection}"
                for suffix in ("weight_scale_2", "input_scale"):
                    name = f"{prefix}.{suffix}"
                    sources[name] = reader.read(name)
        scale_vectors = _expert_scale_vectors(sources, 0, tuple(range(512)))
        assert scale_vectors[0].unique().numel() > 1
        assert scale_vectors[3].unique().numel() > 1
        assert scale_vectors[2].shape == (512,)
        prefix = f"{TEXT_PREFIX}.layers.0.mlp.experts.511"
        planes = [tuple(reader.read(f"{prefix}.{projection}.weight_scale.nvfp4_csf_{suffix}")
                        for suffix in ("fixed", "exceptions")) for projection in ("up_proj", "gate_proj")]
        fixed, exceptions = join_nvfp4_csf(*planes[0], *planes[1], rows=640, columns=160)
        expected = torch.cat([decode_csf_scales(*plane, rows=640, columns=160) for plane in planes])
        assert torch.equal(decode_csf_scales(fixed, exceptions, rows=1280, columns=160), expected)


def test_mapped_scale_planes_match_checkpoint_and_own_storage(manifest):
    loader = QwenWeightLoader(manifest, "cpu")
    loader._session = object()
    prefix = f"{TEXT_PREFIX}.layers.0.mlp.experts.511.up_proj.weight_scale.nvfp4_csf_"
    with ExitStack() as stack:
        loader._stack = stack
        tensors = {suffix: loader._cpu_csf(prefix + suffix) for suffix in ("fixed", "exceptions")}
    with CheckpointTensors(manifest, byte_budget=2 << 20) as reader:
        for suffix, tensor in tensors.items():
            assert torch.equal(tensor, reader.read(prefix + suffix))
            tensor.zero_()
            assert torch.count_nonzero(reader.read(prefix + suffix).view(torch.uint8)) > 0
