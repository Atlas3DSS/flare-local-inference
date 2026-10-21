"""Canonical dense decoder parameters and checkpoint-calibrated projections."""

from dataclasses import dataclass

import torch

from ignition.components.model import canonical_tree
from ignition.components.parallel import TensorParallelShard
from ignition.components.tensor_fp8 import TensorFP8Weight
from ignition.components.weights import DirectWeightLoader

from .config import ModelConfig, TEXT_PREFIX


@dataclass(frozen=True)
class ModelWeights:
    config: ModelConfig
    tree: torch.nn.Module
    projections: dict
    head: object
    allocated_bytes: int
    shard: TensorParallelShard = TensorParallelShard(0, 1)
    draft: object = None

    @classmethod
    def load(cls, manifest, device, *, progress=None, row_invariant=False, package_storage=None):
        from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

        config = ModelConfig.load(manifest.snapshot)
        from ignition.components.sm70_projection import is_sm70
        sm70 = is_sm70(device)
        tree = canonical_tree(Qwen3_5TextModel, Qwen3_5TextConfig.from_dict(config.text))
        projections = {}
        before = torch.cuda.memory_allocated(device)
        if row_invariant:
            from ignition.components.sm70_projection_tc import load_weight
        with DirectWeightLoader(manifest, device, prefixes=(TEXT_PREFIX + ".", "lm_head."),
                                progress=progress) as loader:
            for name, parameter in tree.named_parameters():
                prefix = TEXT_PREFIX + "." + name.removesuffix(".weight")
                if name.endswith(".weight") and prefix in config.quantization:
                    weight = load_weight(loader, prefix, storage=package_storage) if row_invariant else loader.load_linear(prefix)
                    packed = weight.packed if isinstance(weight, TensorFP8Weight) else weight
                    if (packed.out_features, packed.in_features) != tuple(parameter.shape):
                        raise ValueError(f"{prefix}: checkpoint projection shape differs from the model")
                    projections[name.removesuffix(".weight")] = weight
                    continue
                tensor = loader.load_tensor(TEXT_PREFIX + "." + name)
                if tensor.shape != parameter.shape:
                    raise ValueError(f"{name}: checkpoint parameter shape differs from the model")
                parent, _, field = name.rpartition(".")
                # Keep large embedding/projection weights packed. Small canonical
                # parameters participate in Inductor graphs, which require FP32 on Volta.
                if sm70 and ("norm" in name or name.endswith(("A_log", "dt_bias", "conv1d.weight"))):
                    if name.endswith("linear_attn.norm.weight"):
                        tree.get_submodule(parent.rpartition(".")[0]).norm_weight_bf16 = tensor.dtype == torch.bfloat16
                    tensor = tensor.float()
                setattr(tree.get_submodule(parent), field, torch.nn.Parameter(tensor, requires_grad=False))
            head = load_weight(loader, "lm_head", storage=package_storage) if row_invariant else loader.load_linear("lm_head")
            if (head.out_features, head.in_features) != (config.vocabulary, config.hidden):
                raise ValueError("Qwen vocabulary projection shape differs from the model")
        for index, layer in enumerate(tree.layers):
            if layer.block_type == "linear_attention":
                module = layer.linear_attn
                projections[f"layers.{index}.linear_attn.in_proj_ba"] = torch.cat(
                    (module.in_proj_b.weight, module.in_proj_a.weight))
        return cls(config, tree, projections, head, torch.cuda.memory_allocated(device) - before)
