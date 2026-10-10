"""GLM checkpoint weights partitioned for b12x TP2 and TP4 execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from flare.components.parallel import TensorParallelShard
from flare.components.weights import DirectWeightLoader, MLPWeights
from flare.preparation.checkpoint import IndexedCheckpoint

if TYPE_CHECKING:
    import torch
    from b12x.moe.fused_moe import PreparedExperts

TEXT_PREFIX = "model.language_model"


@dataclass(frozen=True)
class MoEWeights:
    router: torch.Tensor
    correction: torch.Tensor
    experts: PreparedExperts
    shared: MLPWeights


class GlmWeightLoader(DirectWeightLoader):
    """Head and intermediate sharding with replicated latent attention and mHC.

    KDA and MLA output projections produce partial hidden vectors that require
    an all-reduce. Indexer projections, low-rank inputs, normalization and mHC
    remain replicated. Routed experts partition their intermediate dimension,
    preserving all 288 experts on every rank.
    """

    def __init__(self, manifest: IndexedCheckpoint, device, *, shard: TensorParallelShard,
                 prefixes=None, csf_byte_budget=128 << 20):
        super().__init__(manifest, device, prefixes=prefixes, csf_byte_budget=csf_byte_budget)
        self.shard = shard
        config = json.loads((manifest.snapshot / "config.json").read_text())
        if (config.get("model_type") != "glm5_next"
                or config.get("architectures") != ["Glm5NextForConditionalGeneration"]):
            raise ValueError("GLM weights require Glm5NextForConditionalGeneration")
        text = config["text_config"]
        for key, expected in {
            "hidden_size": 4096, "num_hidden_layers": 45, "intermediate_size": 12288,
            "moe_intermediate_size": 2048, "n_routed_experts": 288, "n_shared_experts": 1,
            "first_k_dense_replace": 3, "num_attention_heads": 64, "q_lora_rank": 1536,
            "kv_lora_rank": 512, "qk_nope_head_dim": 256, "qk_rope_head_dim": 0,
            "v_head_dim": 256, "linear_num_heads": 64, "linear_head_dim": 128,
            "linear_conv_kernel_dim": 4, "index_n_heads": 32, "index_head_dim": 128,
            "swiglu_limit": 10.0, "hc_mult": 4,
        }.items():
            if text.get(key) != expected:
                raise ValueError(f"GLM {key} requires {expected}, got {text.get(key)}")
        self.quantization = config["quantization_config"]["quantized_layers"]

    def _selected_names(self):
        return {name for name in super()._selected_names()
                if name.startswith(TEXT_PREFIX + ".") or name == "lm_head.weight"}

    def _layer(self, layer):
        if type(layer) is not int or not 0 <= layer < 45:
            raise ValueError("GLM decoder layer must be in [0, 45)")
        return f"{TEXT_PREFIX}.layers.{layer}"

    def load_mxfp8_group(self, prefixes, *, row_ranges=None, column_range=None):
        prefixes = tuple(prefixes)
        for prefix in prefixes:
            if self.quantization.get(prefix, {}).get("quant_algo") != "MXFP8":
                raise ValueError(f"{prefix}: expected declared MXFP8 quantization")
        return super().load_mxfp8_group(prefixes, row_ranges=row_ranges, column_range=column_range)

    def _projection(self, prefix, *, rows=None, columns=None):
        return self.load_mxfp8_group((prefix,), row_ranges=(rows,), column_range=columns)

    def load_attention(self, layer):
        prefix = self._layer(layer) + ".self_attn"
        if layer % 4 != 3:
            return self._load_kda(prefix)
        head_channels = self.shard.partition(64 * 256, alignment=256)
        result = {
            name: self._projection(f"{prefix}.{name}")
            for name in ("q_a_proj", "kv_a_proj_with_mqa", "indexer.wq_b", "indexer.wk", "indexer.weights_proj")
        }
        result["q_b_proj"] = self._projection(f"{prefix}.q_b_proj", rows=head_channels)
        result["kv_b_proj"] = self._projection(
            f"{prefix}.kv_b_proj", rows=self.shard.partition(64 * 512, alignment=512))
        result["o_proj"] = self._projection(f"{prefix}.o_proj", columns=head_channels)
        for name in ("q_a_layernorm.weight", "kv_a_layernorm.weight", "indexer.k_norm.weight",
                     "indexer.k_norm.bias", "indexer.index_kpool_compress_ape", "indexer.index_kpool_compress_gate"):
            result[name] = self.load_tensor(f"{prefix}.{name}")
        return result

    def _load_kda(self, prefix):
        import torch

        channels = self.shard.partition(8192, alignment=128)
        heads = self.shard.partition(64)
        result = {name: self._projection(f"{prefix}.{name}", rows=channels)
                  for name in ("f_b_proj", "g_b_proj")}
        result["qkv_proj"] = self.load_mxfp8_group(
            tuple(f"{prefix}.{name}" for name in ("q_proj", "k_proj", "v_proj")),
            row_ranges=(channels,) * 3)
        result.update({name: self._projection(f"{prefix}.{name}") for name in ("f_a_proj", "g_a_proj")})
        result["b_proj"] = self._projection(f"{prefix}.b_proj", rows=heads)
        result["o_proj"] = self._projection(f"{prefix}.o_proj", columns=channels)
        result["A_log"] = self.load_tensor(f"{prefix}.A_log", extent=slice(*heads))
        result["dt_bias"] = self.load_tensor(f"{prefix}.dt_bias", extent=slice(*channels))
        result["o_norm.weight"] = self.load_tensor(f"{prefix}.o_norm.weight")
        result["conv1d.weight"] = torch.cat([
            self.load_tensor(f"{prefix}.{name}_conv1d.weight",
                             extent=slice(*channels)) for name in ("q", "k", "v")
        ]).squeeze(1)
        return result

    def load_mlp(self, layer, *, scale_scratch=None):
        import torch

        prefix = self._layer(layer) + ".mlp"
        if layer < 3:
            extent = self.shard.partition(12288)
            gate_up = torch.cat([self.load_tensor(f"{prefix}.{name}.weight", extent=slice(*extent))
                                 for name in ("gate_proj", "up_proj")])
            down = self.load_tensor(f"{prefix}.down_proj.weight", extent=(slice(None), slice(*extent)))
            if gate_up.dtype != torch.bfloat16 or down.dtype != torch.bfloat16:
                raise ValueError("GLM dense MLPs require stored BF16 weights")
            return MLPWeights(gate_up, down)
        extent = self.shard.partition(2048, alignment=128)
        shared_prefix = prefix + ".shared_experts"
        shared = MLPWeights(
            self.load_mxfp8_group(tuple(f"{shared_prefix}.{name}" for name in ("gate_proj", "up_proj")),
                                 row_ranges=(extent, extent)),
            self._projection(f"{shared_prefix}.down_proj", columns=extent),
        )
        return MoEWeights(
            self.load_tensor(f"{prefix}.gate.weight", dtype=torch.float32),
            self.load_tensor(f"{prefix}.gate.e_score_correction_bias"),
            self.load_experts(layer, scale_scratch=scale_scratch), shared,
        )

    def load_experts(self, layer, *, expert_ids=None, scale_scratch=None):
        if layer < 3:
            raise ValueError("GLM routed experts begin at layer 3")
        prefix = self._layer(layer) + ".mlp.experts"
        ids = tuple(range(288)) if expert_ids is None else tuple(expert_ids)
        if (not ids or len(set(ids)) != len(ids)
                or any(type(i) is not int or not 0 <= i < 288 for i in ids)):
            raise ValueError("GLM expert IDs must be distinct integers in [0, 288)")
        prefixes = tuple(f"{prefix}.{i}" for i in ids)
        recipe = self.quantization.get(prefix, {})
        if (recipe.get("quant_algo") != "NVFP4" or recipe.get("group_size") != 16
                or recipe.get("weight_scale_encoding") != "csf"):
            raise ValueError(f"{prefix}: expected declared NVFP4 CSF quantization")
        return self.load_nvfp4_experts(
            prefixes, hidden_size=4096, intermediate_size=2048,
            intermediate_range=self.shard.partition(2048, alignment=128),
            scale_vectors=_expert_scales(self._sources, prefixes), scale_scratch=scale_scratch,
            swiglu_limit=10.0,
        )

    def allocate_expert_scale_scratch(self, expert_count=288):
        import torch

        if type(expert_count) is not int or not 1 <= expert_count <= 288:
            raise ValueError("GLM expert count must be in [1, 288]")
        local = 2048 // self.shard.world_size
        return tuple(torch.empty(shape, dtype=torch.float8_e4m3fn, device=self.device) for shape in (
            (expert_count, 2 * local, 256), (expert_count, 4096, local // 16)))

    def load_residual(self, layer):
        prefix = self._layer(layer)
        return {name: self.load_tensor(f"{prefix}.{name}") for name in (
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "hc_attn_base", "hc_attn_fn", "hc_attn_scale", "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale")}


def _expert_scales(sources, prefixes):
    import torch

    w13, w2, a1, a2 = [], [], [], []
    for prefix in prefixes:
        values = {}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight_scale_2", "input_scale"):
                name = f"{prefix}.{projection}.{suffix}"
                value = sources[name]
                if (value.device.type != "cpu" or value.dtype != torch.float32 or value.numel() != 1
                        or not bool((torch.isfinite(value) & (value > 0)).all())):
                    raise ValueError(f"{name}: expected a finite positive FP32 scalar")
                values[projection, suffix] = value.item()
        if values["gate_proj", "weight_scale_2"] != values["up_proj", "weight_scale_2"]:
            raise ValueError(f"{prefix}: gate/up weight global scales differ")
        w13.append(values["up_proj", "weight_scale_2"])
        w2.append(values["down_proj", "weight_scale_2"])
        a1.append(max(values["gate_proj", "input_scale"], values["up_proj", "input_scale"]))
        a2.append(values["down_proj", "input_scale"])
    return (torch.tensor(w13, dtype=torch.float32), torch.tensor(w2, dtype=torch.float32),
            torch.tensor(a1, dtype=torch.float32).reciprocal(), torch.tensor(a2, dtype=torch.float32).reciprocal())
