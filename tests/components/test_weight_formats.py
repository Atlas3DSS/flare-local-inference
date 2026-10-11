"""Checkpoint quantization dispatch and projection fusion constraints."""

import json
from types import SimpleNamespace

import pytest

from flare.components.weights import DirectWeightLoader
from flare.preparation.quantization import linear_format, quantized_layers
from flare.recipes.qwen38.weights import QwenWeightLoader


def config(layers):
    return {"quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                                    "quantized_layers": layers}}


@pytest.mark.parametrize("loader_type", [DirectWeightLoader, QwenWeightLoader])
@pytest.mark.parametrize("algo,group,selected", [("MXFP8", 32, "mxfp8"), ("NVFP4", 16, "nvfp4"),
                                                ("W4A16_NVFP4", 16, "nvfp4")])
def test_checkpoint_selects_single_projection_format(tmp_path, monkeypatch, loader_type, algo, group, selected):
    prefix = "model.language_model.layers.0.linear_attn.out_proj"
    (tmp_path / "config.json").write_text(json.dumps(config({prefix: {"quant_algo": algo, "group_size": group}})))
    loader = loader_type(SimpleNamespace(snapshot=tmp_path), "cuda:0")
    loader._session = object()
    monkeypatch.setattr(loader, "load_mxfp8_group", lambda *args, **kwargs: ("mxfp8", kwargs))
    monkeypatch.setattr(loader, "_load_nvfp4_linear", lambda *args, **kwargs: ("nvfp4", kwargs))
    result, slices = loader.load_linear(prefix, rows=(0, 32), columns=(16, 144))
    assert result == selected
    assert slices == ({"row_ranges": ((0, 32),), "column_range": (16, 144)} if selected == "mxfp8"
                      else {"rows": (0, 32), "columns": (16, 144)})


@pytest.mark.parametrize("formats", [("NVFP4", "NVFP4"), ("MXFP8", "NVFP4"),
                                      ("W4A16_NVFP4", "MXFP8")])
def test_nvfp4_fusion_rejected_before_tensor_access(tmp_path, formats):
    layers = {name: {"quant_algo": algo, "group_size": 32 if algo == "MXFP8" else 16}
              for name, algo in zip(("q", "k"), formats)}
    (tmp_path / "config.json").write_text(json.dumps(config(layers)))
    loader = DirectWeightLoader(SimpleNamespace(snapshot=tmp_path), "cuda:0")
    loader._session = object()
    with pytest.raises(ValueError, match="cannot fuse NVFP4.*only MoE expert w1/w3"):
        loader.load_mxfp8_group(("q", "k"))


@pytest.mark.parametrize("entry", [None, {}, {"quant_algo": "AWQ", "group_size": 128},
    {"quant_algo": "MXFP8", "group_size": 16},
    {"quant_algo": "NVFP4", "group_size": 32},
    {"quant_algo": "NVFP4", "group_size": 16, "weight_scale_encoding": "csf"}])
def test_unsupported_projection_encodings_rejected(entry):
    with pytest.raises(ValueError, match="projection|quantization|declaration"):
        linear_format({"proj": entry}, "proj")


def test_quantization_config_requires_supported_schema():
    with pytest.raises(ValueError, match="modelopt"):
        quantized_layers({"quantization_config": {"quant_method": "compressed-tensors"}})
