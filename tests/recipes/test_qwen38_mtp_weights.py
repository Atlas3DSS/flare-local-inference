from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sparkgap.preparation import inspect_qwen38_checkpoint
from sparkgap.recipes.qwen38.mtp_weights import MTPWeightLoader, _global_scales, create_mtp_modules
from sparkgap.recipes.qwen38.weights import QwenWeightLoader


@pytest.fixture(scope="module")
def manifest():
    snapshot = Path("/data/checkpoints-masters/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD")
    if not snapshot.exists():
        pytest.skip("exact MTP checkpoint unavailable")
    return inspect_qwen38_checkpoint(snapshot)


def test_exact_module_geometry_and_mtp_only_selection(manifest):
    module = create_mtp_modules(manifest)
    assert len(module.layers) == 1
    assert module.layers[0].self_attn.layer_idx == 48
    assert not hasattr(module.layers[0], "ple")
    assert not hasattr(module.layers[0], "linear_attn")
    assert module.pre_fc_norm_hidden.group_size is None
    assert module.layers[0].attn_hyper_connection.hc_norm.group_size == 2560
    names = {"mtp." + name for name, parameter in module.named_parameters()
             if parameter.is_meta and not parameter.requires_grad}
    assert len(names) == 24
    for name, parameter in module.named_parameters():
        assert tuple(parameter.shape) == manifest.tensors[f"mtp.{name}"].shape
    selected = MTPWeightLoader(manifest, "cpu")._selected_names()
    assert len(selected) == 4637
    assert sum(manifest.tensors[n].nbytes for n in selected) == 1596720640
    assert all(name.startswith("mtp.") for name in selected)
    assert not selected & QwenWeightLoader(manifest, "cpu")._selected_names()
    assert "mtp.layers.48.mlp.experts" not in selected
    assert not torch.cuda.is_initialized()


def scalar_sources():
    return {f"mtp.layers.0.mlp.experts.{expert}.{projection}.weight_scale_2": torch.tensor(value)
            for expert, value in ((4, .25), (7, .5))
            for projection in ("gate_proj", "up_proj", "down_proj")}


def test_expert_scales_preserve_requested_order():
    one, two = _global_scales(scalar_sources(), (7, 4))
    assert one.tolist() == two.tolist() == [.5, .25]


@pytest.mark.parametrize("invalid", [0., -1., float("inf"), float("nan")])
def test_bad_mtp_scalar_fails_before_preparation(invalid):
    sources = scalar_sources()
    sources["mtp.layers.0.mlp.experts.4.down_proj.weight_scale_2"] = torch.tensor(invalid)
    with pytest.raises(ValueError, match="finite and positive"):
        _global_scales(sources, (4, 7))


def test_unequal_gate_up_and_nonscalar_rejected():
    sources = scalar_sources()
    name = "mtp.layers.0.mlp.experts.4.up_proj.weight_scale_2"
    sources[name] *= 2
    with pytest.raises(ValueError, match="gate/up global scales differ"):
        _global_scales(sources, (4,))
    sources[name] = torch.tensor([.25])
    with pytest.raises(ValueError, match="CPU float32 scalar"):
        _global_scales(sources, (4,))


def test_actual_packed_experts_use_a16_and_plain_scales_in_up_gate_order(manifest, monkeypatch):
    from b12x.moe import fused_moe
    from safetensors import safe_open

    ids = (7, 4)
    names = [n for n in manifest.tensors if any(n.startswith(f"mtp.layers.0.mlp.experts.{i}.") for i in ids)]
    sources = {}
    for shard in {manifest.tensors[name].shard for name in names}:
        with safe_open(manifest.snapshot / shard, framework="pt", device="cpu") as handle:
            sources.update((name, handle.get_tensor(name).clone()) for name in names
                           if manifest.tensors[name].shard == shard)
    captured = {}

    def plan_weights(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(fused_moe, "plan_weights", plan_weights)
    monkeypatch.setattr(fused_moe, "prepare_weights", lambda **kwargs: kwargs["weights"])
    loader = MTPWeightLoader(manifest, "cpu")
    loader._session = SimpleNamespace(flush=lambda: None)
    loader._sources = sources
    loader._copy = lambda destination, source: destination.copy_(source)
    weights = loader.load_experts(expert_ids=ids)
    assert captured["activation"].mode.value == "a16"
    assert captured["source"].format.value == "modelopt_nvfp4"
    assert captured["source"].w13_layout.value == "w13"
    assert weights.input_scale is None and weights.intermediate_scale is None
    assert captured["geometry"].num_experts == 2
    for local, expert in enumerate(ids):
        for projection, values, scales in (("up_proj", weights.w13[local, :640], weights.w13_block_scales[local, :640]),
                                          ("gate_proj", weights.w13[local, 640:], weights.w13_block_scales[local, 640:]),
                                          ("down_proj", weights.w2[local], weights.w2_block_scales[local])):
            prefix = f"mtp.layers.0.mlp.experts.{expert}.{projection}"
            assert torch.equal(values, sources[f"{prefix}.weight"])
            assert torch.equal(scales.view(torch.uint8), sources[f"{prefix}.weight_scale"].view(torch.uint8))
        assert weights.w13_global_scales[local] == sources[f"mtp.layers.0.mlp.experts.{expert}.up_proj.weight_scale_2"]
        assert weights.w2_global_scales[local] == sources[f"mtp.layers.0.mlp.experts.{expert}.down_proj.weight_scale_2"]
    assert not torch.cuda.is_initialized()
