"""Supported ModelOpt projection storage declarations."""

from collections.abc import Mapping


def quantized_layers(config):
    quant = config.get("quantization_config")
    if (not isinstance(quant, Mapping) or quant.get("quant_method") != "modelopt"
            or quant.get("quant_algo") != "MIXED_PRECISION"
            or quant.get("kv_cache_quant_algo") is not None):
        raise ValueError("expected modelopt MIXED_PRECISION with unquantized KV")
    layers = quant.get("quantized_layers")
    if not isinstance(layers, Mapping):
        raise ValueError("missing quantized_layers mapping")
    return layers


def linear_format(layers, prefix):
    entry = layers.get(prefix)
    if not isinstance(entry, Mapping):
        raise ValueError(f"{prefix}: missing quantized_layers declaration")
    algo = entry.get("quant_algo")
    if algo == "FP8" and entry.get("group_size") is None and entry.get("weight_scale_encoding") is None:
        return "tensor_fp8"
    group = {"MXFP8": 32, "MXFP4": 32, "NVFP4": 16, "W4A16_NVFP4": 16}.get(algo)
    if group is None or entry.get("group_size") != group:
        raise ValueError(f"{prefix}: unsupported linear quantization {entry!r}")
    if entry.get("weight_scale_encoding") is not None:
        raise ValueError(f"{prefix}: dense projections require ordinary block scales")
    return algo.lower() if algo in ("MXFP8", "MXFP4") else "nvfp4"


def projection_group_format(layers, prefixes):
    if not prefixes:
        raise ValueError("projection group must be nonempty")
    formats = [linear_format(layers, prefix) for prefix in prefixes]
    if "tensor_fp8" in formats and len(prefixes) > 1:
        raise ValueError("tensor-FP8 projections retain separate checkpoint activation and weight scales")
    if "nvfp4" in formats and len(prefixes) > 1:
        names = ", ".join(prefixes)
        raise ValueError(f"cannot fuse NVFP4 projections ({names}); only MoE expert w1/w3 fusion is supported")
    if len(set(formats)) != 1:
        raise ValueError("fused projections must use the same quantization format")
    return formats[0]


def expert_format(layers, prefix):
    entry = layers.get(prefix)
    if not isinstance(entry, Mapping):
        raise ValueError(f"{prefix}: missing expert quantization declaration")
    if entry.get("weight_scale_encoding") in ("csf", "csf_32x2_v1"):
        if entry.get("quant_algo") == "NVFP4" and entry.get("group_size") == 16:
            return "nvfp4_csf", "a4"
        raise ValueError(f"{prefix}: unsupported compressed expert scales")
    fmt = linear_format(layers, prefix)
    if fmt not in ("nvfp4", "mxfp4"):
        raise ValueError(f"{prefix}: expected NVFP4 or MXFP4 experts")
    mode = "a16" if entry["quant_algo"] == "W4A16_NVFP4" else "a8" if fmt == "mxfp4" else "a4"
    return fmt, mode
