"""Dense NextN BF16 weights or the checkpoint NVFP4 draft overlay."""

from dataclasses import dataclass
from math import prod

import torch

from ignition.components.weights import DirectWeightLoader

from .config import ModelConfig


def tensor_shapes(config):
    """Exact single-layer dense NextN geometry, without creating a target model."""
    text = config.text
    if text.get("mtp_num_hidden_layers") != 1 or text.get("mtp_use_dedicated_embeddings", False):
        raise ValueError("dense MTP requires one layer and shared token embeddings")
    hidden, intermediate = config.hidden, text["intermediate_size"]
    dim = text["head_dim"]
    query, key = text["num_attention_heads"] * dim, text["num_key_value_heads"] * dim
    return {
        "fc.weight": (hidden, 2 * hidden),
        "pre_fc_norm_embedding.weight": (hidden,),
        "pre_fc_norm_hidden.weight": (hidden,),
        "norm.weight": (hidden,),
        "layers.0.input_layernorm.weight": (hidden,),
        "layers.0.post_attention_layernorm.weight": (hidden,),
        "layers.0.self_attn.q_norm.weight": (dim,),
        "layers.0.self_attn.k_norm.weight": (dim,),
        "layers.0.self_attn.q_proj.weight": (2 * query, hidden),
        "layers.0.self_attn.k_proj.weight": (key, hidden),
        "layers.0.self_attn.v_proj.weight": (key, hidden),
        "layers.0.self_attn.o_proj.weight": (hidden, query),
        "layers.0.mlp.gate_proj.weight": (intermediate, hidden),
        "layers.0.mlp.up_proj.weight": (intermediate, hidden),
        "layers.0.mlp.down_proj.weight": (hidden, intermediate),
    }


def validate_checkpoint(manifest, config):
    """Reject an incompatible/partial drafter before transferring any weights."""
    expected = {}
    quantization = config.quantization
    for suffix, shape in tensor_shapes(config).items():
        name = "mtp." + suffix
        prefix = name.removesuffix(".weight")
        if prefix in quantization:
            if quantization[prefix]["quant_algo"] != "NVFP4" or len(shape) != 2:
                raise ValueError(f"{prefix}: MTP overlay requires NVFP4 matrices")
            n, k = shape
            expected.update({name: ((n, k // 2), "U8", n * k // 2),
                prefix + ".weight_scale": ((n, k // 16), "F8_E4M3", n * k // 16),
                prefix + ".weight_scale_2": ((), "F32", 4),
                prefix + ".input_scale": ((), "F32", 4)})
        else:
            expected[name] = (shape, "BF16", 2 * prod(shape))
    actual = {name for name in manifest.tensors if name.startswith("mtp.")}
    if actual != expected.keys():
        missing, extra = sorted(expected.keys() - actual), sorted(actual - expected.keys())
        raise ValueError(f"dense MTP checkpoint tensor mismatch: missing={missing}, extra={extra}")
    for name, (shape, dtype, size) in expected.items():
        entry = manifest.tensors[name]
        if tuple(entry.shape) != shape or entry.dtype != dtype or entry.nbytes != size:
            raise ValueError(f"{name}: dense MTP requires {dtype} {shape}")
    return sum(manifest.tensors[name].nbytes for name in expected)


@dataclass(frozen=True)
class MTPWeights:
    config: ModelConfig
    tensors: dict[str, object]
    embedding: torch.Tensor
    lm_head: object
    stored_bytes: int
    allocated_bytes: int

    @classmethod
    def load(cls, manifest, device, *, embedding_weight, lm_head_weight, config=None, progress=None, package_storage=None):
        config = ModelConfig.load(manifest.snapshot) if config is None else config
        stored = validate_checkpoint(manifest, config)
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("dense MTP weights require a CUDA device")
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        from ignition.components.sm70_projection import is_sm70

        if not is_sm70(device):
            raise ValueError("dense MTP preparation currently requires SM70")
        if (embedding_weight.shape != (config.vocabulary, config.hidden)
                or embedding_weight.dtype != torch.bfloat16 or embedding_weight.device != device
                or not embedding_weight.is_contiguous()):
            raise ValueError("dense MTP requires the target's contiguous BF16 embedding table")
        shape = (tuple(lm_head_weight.shape) if isinstance(lm_head_weight, torch.Tensor)
                 else (lm_head_weight.out_features, lm_head_weight.in_features))
        if shape != (config.vocabulary, config.hidden):
            raise ValueError("dense MTP shared vocabulary projection shape differs from the target")
        before = torch.cuda.memory_allocated(device)
        tensors = {}
        with DirectWeightLoader(manifest, device, prefixes=("mtp.",), progress=progress,
                                progress_label="Load dense MTP weights") as loader:
            for name in tensor_shapes(config):
                # Keep matrices in their checkpoint BF16 storage. Only the tiny
                # RMSNorm vectors enter SM70's FP32 arithmetic graph directly.
                dtype = torch.float32 if "norm" in name else None
                prefix = "mtp." + name.removesuffix(".weight")
                if prefix in config.quantization:
                    from ignition.components.sm70_projection_tc import load_weight
                    tensors[name] = load_weight(loader, prefix, storage=package_storage)
                else:
                    tensors[name] = loader.load_tensor("mtp." + name, dtype=dtype)
        return cls(config, tensors, embedding_weight, lm_head_weight, stored,
                   torch.cuda.memory_allocated(device) - before)
