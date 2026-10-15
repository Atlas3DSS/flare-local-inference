"""Checkpoint quantization dispatch and projection fusion constraints."""

import json
from types import SimpleNamespace

import pytest

from ignition.components.weights import DirectWeightLoader
from ignition.preparation.quantization import expert_format, linear_format, projection_group_format, quantized_layers
from ignition.recipes.qwen38.weights import QwenWeightLoader


def config(layers):
    return {"quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                                    "quantized_layers": layers}}


def test_mxfp4_storage_selection_preserves_projection_order_and_shared_scale_lifetime(monkeypatch):
    names = [f"layer{layer}.expert{expert}" for layer in (0, 1) for expert in (0, 1)]
    compressed = {f"{name}.{projection}.scale" for name in names for projection in ("w1", "w3", "w2")}
    loader = DirectWeightLoader(SimpleNamespace(compressed_scales=compressed), "cpu")
    loader._session = object()
    monkeypatch.setattr(loader, "load_mxfp4_csf_experts", lambda prefixes, **kwargs: kwargs)
    monkeypatch.setattr(loader, "load_fp4_experts", lambda prefixes, **kwargs: kwargs)
    scratch = {}
    options = dict(hidden_size=128, intermediate_size=64, intermediate_range=(32, 64),
                   scale_scratch=scratch, projection_names=("w3", "w1", "w2"), scale_name="scale")
    first = loader.load_mxfp4_experts(names[:2], **options)
    second = loader.load_mxfp4_experts(names[2:], **options)
    assert first["projection_names"] == ("w1", "w3", "w2")
    assert first["scale_scratch"] is second["scale_scratch"]
    assert [tuple(tensor.shape) for tensor in first["scale_scratch"]] == [(2, 4, 64), (2, 1, 128)]
    ordinary = loader.load_mxfp4_experts(["draft.expert0"], **options)
    assert ordinary["projection_names"] == ("w3", "w1", "w2")
    assert ordinary["format"] == "mxfp4" and ordinary["activation_mode"] == "a8"
    assert "scale_scratch" not in ordinary and len(scratch) == 1
    with pytest.raises(ValueError, match="complete MXFP4 scale encoding"):
        loader.load_mxfp4_experts([names[0], "draft.expert0"], **options)


@pytest.mark.parametrize("loader_type", [DirectWeightLoader, QwenWeightLoader])
@pytest.mark.parametrize("algo,group,selected", [("MXFP8", 32, "mxfp8"), ("MXFP4", 32, "mxfp4"), ("NVFP4", 16, "nvfp4"),
                                                ("W4A16_NVFP4", 16, "nvfp4")])
def test_checkpoint_selects_single_projection_format(tmp_path, monkeypatch, loader_type, algo, group, selected):
    prefix = "model.language_model.layers.0.linear_attn.out_proj"
    (tmp_path / "config.json").write_text(json.dumps(config({prefix: {"quant_algo": algo, "group_size": group}})))
    loader = loader_type(SimpleNamespace(snapshot=tmp_path, metadata_path=tmp_path), "cuda:0")
    loader._session = object()
    monkeypatch.setattr(loader, "load_linear_group", lambda prefixes, **kwargs: (projection_group_format(loader.quantization, prefixes), kwargs))
    result, slices = loader.load_linear(prefix, rows=(0, 32), columns=(16, 144))
    assert result == selected
    assert slices == {"row_ranges": ((0, 32),), "column_range": (16, 144)}


@pytest.mark.parametrize("formats", [("NVFP4", "NVFP4"), ("MXFP8", "NVFP4"),
                                      ("W4A16_NVFP4", "MXFP8")])
def test_nvfp4_fusion_rejected_before_tensor_access(tmp_path, formats):
    layers = {name: {"quant_algo": algo, "group_size": 32 if algo == "MXFP8" else 16}
              for name, algo in zip(("q", "k"), formats)}
    (tmp_path / "config.json").write_text(json.dumps(config(layers)))
    loader = DirectWeightLoader(SimpleNamespace(snapshot=tmp_path, metadata_path=tmp_path), "cuda:0")
    loader._session = object()
    with pytest.raises(ValueError, match="cannot fuse NVFP4.*only MoE expert w1/w3"):
        loader.load_linear_group(("q", "k"))


@pytest.mark.parametrize("entry", [None, {}, {"quant_algo": "AWQ", "group_size": 128},
    {"quant_algo": "MXFP8", "group_size": 16},
    {"quant_algo": "NVFP4", "group_size": 32},
    {"quant_algo": "NVFP4", "group_size": 16, "weight_scale_encoding": "csf_32x2_v1"}])
def test_unsupported_projection_encodings_rejected(entry):
    with pytest.raises(ValueError, match="projection|quantization|declaration"):
        linear_format({"proj": entry}, "proj")


def test_quantization_config_requires_supported_schema():
    with pytest.raises(ValueError, match="modelopt"):
        quantized_layers({"quantization_config": {"quant_method": "compressed-tensors"}})


@pytest.mark.parametrize("algo,group,encoding,expected", [
    ("NVFP4", 16, None, ("nvfp4", "a4")),
    ("NVFP4", 16, "csf_32x2_v1", ("nvfp4_csf", "a4")),
    ("NVFP4", 16, "csf", ("nvfp4_csf", "a4")),
    ("W4A16_NVFP4", 16, None, ("nvfp4", "a16")),
    ("MXFP4", 32, None, ("mxfp4", "a8")),
])
def test_expert_storage_and_activation_contract(algo, group, encoding, expected):
    entry = {"quant_algo": algo, "group_size": group}
    if encoding is not None:
        entry["weight_scale_encoding"] = encoding
    assert expert_format({"experts": entry}, "experts") == expected


def test_projection_fusion_requires_matching_formats():
    with pytest.raises(ValueError, match="same quantization format"):
        projection_group_format({"q": {"quant_algo": "MXFP4", "group_size": 32},
                                 "k": {"quant_algo": "MXFP8", "group_size": 32}}, ("q", "k"))


@pytest.mark.parametrize("algo,group,scale_dtype", [("NVFP4", 16, "F8_E4M3"),
    ("W4A16_NVFP4", 16, "F8_E4M3"), ("MXFP4", 32, "U8")])
def test_qwen_declares_ordinary_expert_storage(algo, group, scale_dtype):
    from ignition.preparation.qwen38 import _mlp

    layers = {"layer.mlp.experts": {"quant_algo": algo, "group_size": group}}
    tensors = {name: (dtype, shape) for name, dtype, shape in _mlp("layer.mlp", quantization=layers)}
    prefix = "layer.mlp.experts.0.gate_proj"
    assert tensors[prefix + ".weight"] == ("U8", (640, 1280))
    assert tensors[prefix + ".weight_scale"] == (scale_dtype, (640, 2560 // group))
    assert (prefix + ".input_scale" in tensors) == (algo == "NVFP4")
    assert (prefix + ".weight_scale_2" in tensors) == (algo != "MXFP4")
    assert not any("csf" in name for name in tensors)


@pytest.mark.parametrize("algo,group,mode", [("NVFP4", 16, "a4"),
    ("W4A16_NVFP4", 16, "a16"), ("MXFP4", 32, "a8")])
def test_mtp_experts_use_declared_format_and_mtp_scales(tmp_path, monkeypatch, algo, group, mode):
    import torch
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.qwen38.mtp_weights import MTPWeightLoader

    prefix = "mtp.layers.0.mlp.experts"
    (tmp_path / "config.json").write_text(json.dumps(config({prefix: {"quant_algo": algo, "group_size": group}})))
    loader = MTPWeightLoader(SimpleNamespace(snapshot=tmp_path, metadata_path=tmp_path), "cpu", shard=TensorParallelShard(1, 2))
    loader._session = object()
    loader._sources = {f"{prefix}.{expert}.{proj}.{suffix}": torch.tensor(value)
        for expert, value in ((7, .5), (4, .25))
        for proj in ("gate_proj", "up_proj", "down_proj")
        for suffix in ("weight_scale_2", "input_scale")}
    monkeypatch.setattr(loader, "load_fp4_experts", lambda prefixes, **kwargs: (prefixes, kwargs))
    prefixes, options = loader.load_experts(expert_ids=(7, 4))
    assert prefixes == (prefix + ".7", prefix + ".4")
    assert options["format"] == ("mxfp4" if algo == "MXFP4" else "nvfp4")
    assert options["activation_mode"] == mode
    assert options["intermediate_range"] == (320, 640)
    if mode == "a4":
        assert [v.tolist() for v in options["scale_vectors"]] == [[.5, .25], [.5, .25], [2., 4.], [2., 4.]]
    else:
        assert options["scale_vectors"] is None
