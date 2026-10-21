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
    def load(cls, manifest, device, *, progress=None, row_invariant=False, package_storage=None, kernel_cache=None):
        from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

        config = ModelConfig.load(manifest.snapshot)
        from ignition.components.sm70_projection import is_sm70
        sm70 = is_sm70(device)
        tree = canonical_tree(Qwen3_5TextModel, Qwen3_5TextConfig.from_dict(config.text))
        projections = {}
        before = torch.cuda.memory_allocated(device)
        if row_invariant:
            from ignition.components.sm70_projection_tc import load_weight, load_fused
        with DirectWeightLoader(manifest, device, prefixes=(TEXT_PREFIX + ".", "lm_head."),
                                progress=progress) as loader:
            fused_members = set()
            if row_invariant:
                for index, layer in enumerate(tree.layers):
                    groups = [(f'layers.{index}.mlp.', ('gate_proj', 'up_proj'), 'gate_up_proj')]
                    groups.append((f'layers.{index}.linear_attn.', ('in_proj_qkv', 'in_proj_z'), 'in_proj_qkvz')
                        if layer.block_type == 'linear_attention' else
                        (f'layers.{index}.self_attn.', ('q_proj', 'k_proj', 'v_proj'), 'qkv_proj'))
                    for base, roles, fused in groups:
                        names = tuple(base + role for role in roles)
                        weight = load_fused(loader,
                            tuple(TEXT_PREFIX + '.' + name for name in names), storage=package_storage,
                            kernel_cache=kernel_cache)
                        packed = weight.packed if isinstance(weight, TensorFP8Weight) else weight
                        for name, width in zip(names, packed.member_widths, strict=True):
                            if (width, packed.in_features) != tuple(tree.get_parameter(name + '.weight').shape):
                                raise ValueError(f"{name}: checkpoint projection shape differs from the model")
                        projections[base + fused] = weight
                        fused_members.update(name + '.weight' for name in names)
            for name, parameter in tree.named_parameters():
                if name in fused_members:
                    continue
                prefix = TEXT_PREFIX + "." + name.removesuffix(".weight")
                if name.endswith(".weight") and prefix in config.quantization:
                    weight = load_weight(loader, prefix, storage=package_storage,
                                         kernel_cache=kernel_cache) if row_invariant else loader.load_linear(prefix)
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
            head = load_weight(loader, "lm_head", storage=package_storage,
                               kernel_cache=kernel_cache) if row_invariant else loader.load_linear("lm_head")
            if (head.out_features, head.in_features) != (config.vocabulary, config.hidden):
                raise ValueError("Qwen vocabulary projection shape differs from the model")
        for index, layer in enumerate(tree.layers):
            if layer.block_type == "linear_attention":
                module = layer.linear_attn
                projections[f"layers.{index}.linear_attn.in_proj_ba"] = torch.cat(
                    (module.in_proj_b.weight, module.in_proj_a.weight))
        return cls(config, tree, projections, head, torch.cuda.memory_allocated(device) - before)
