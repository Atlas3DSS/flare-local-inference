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
    def load(cls, manifest, device, *, progress=None, mtp=False):
        from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

        config = ModelConfig.load(manifest.snapshot)
        tree = canonical_tree(Qwen3_5TextModel, Qwen3_5TextConfig.from_dict(config.text))
        projections = {}
        before = torch.cuda.memory_allocated(device)
        with DirectWeightLoader(manifest, device, prefixes=(TEXT_PREFIX + ".", "lm_head."),
                                progress=progress) as loader:
            for name, parameter in tree.named_parameters():
                prefix = TEXT_PREFIX + "." + name.removesuffix(".weight")
                if name.endswith(".weight") and prefix in config.quantization:
                    weight = loader.load_linear(prefix)
                    packed = weight.packed if isinstance(weight, TensorFP8Weight) else weight
                    if (packed.out_features, packed.in_features) != tuple(parameter.shape):
                        raise ValueError(f"{prefix}: checkpoint projection shape differs from the model")
                    projections[name.removesuffix(".weight")] = weight
                    continue
                tensor = loader.load_tensor(TEXT_PREFIX + "." + name)
                if tensor.shape != parameter.shape:
                    raise ValueError(f"{name}: checkpoint parameter shape differs from the model")
                parent, _, field = name.rpartition(".")
                setattr(tree.get_submodule(parent), field, torch.nn.Parameter(tensor, requires_grad=False))
            head = loader.load_linear("lm_head")
            if (head.out_features, head.in_features) != (config.vocabulary, config.hidden):
                raise ValueError("Qwen vocabulary projection shape differs from the model")
        for index, layer in enumerate(tree.layers):
            if layer.block_type == "linear_attention":
                module = layer.linear_attn
                projections[f"layers.{index}.linear_attn.in_proj_ba"] = torch.cat(
                    (module.in_proj_b.weight, module.in_proj_a.weight))
        draft = MTPWeights.load(manifest, config, device, progress=progress) if mtp else None
        return cls(config, tree, projections, head, torch.cuda.memory_allocated(device) - before, draft=draft)


@dataclass(frozen=True)
class MTPWeights:
    layer: torch.nn.Module
    embedding_norm: torch.nn.Module
    hidden_norm: torch.nn.Module
    norm: torch.nn.Module
    fc: torch.Tensor

    @classmethod
    def load(cls, manifest, config, device, *, progress=None):
        from transformers import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm

        if (config.text.get("mtp_num_hidden_layers") != 1
                or config.text.get("mtp_use_dedicated_embeddings") is not False):
            raise ValueError("Qwen dense MTP requires one predictor with shared embeddings and vocabulary head")
        text = Qwen3_5TextConfig.from_dict({**config.text, "num_hidden_layers": 1, "layer_types": ["full_attention"]})
        text._attn_implementation = "eager"
        with torch.device("meta"):
            layer = Qwen3_5DecoderLayer(text, 0)
            norms = [Qwen3_5RMSNorm(config.hidden, eps=text.rms_norm_eps) for _ in range(3)]
        with DirectWeightLoader(manifest, device, prefixes=("mtp.",), progress=progress) as loader:
            for prefix, module in (("mtp.layers.0", layer), ("mtp.pre_fc_norm_embedding", norms[0]),
                                   ("mtp.pre_fc_norm_hidden", norms[1]), ("mtp.norm", norms[2])):
                for name, parameter in module.named_parameters():
                    tensor = loader.load_tensor(f"{prefix}.{name}")
                    if tensor.shape != parameter.shape or tensor.dtype != torch.bfloat16:
                        raise ValueError(f"{prefix}.{name}: MTP requires checkpoint BF16 parameters with canonical shapes")
                    parent, _, field = name.rpartition(".")
                    setattr(module.get_submodule(parent), field, torch.nn.Parameter(tensor, requires_grad=False))
                module.eval().requires_grad_(False)
            fc = loader.load_tensor("mtp.fc.weight")
            if fc.shape != (config.hidden, 2 * config.hidden) or fc.dtype != torch.bfloat16:
                raise ValueError("mtp.fc.weight: expected a BF16 hidden-by-double-hidden feedback projection")
        return cls(layer, *norms, fc)
