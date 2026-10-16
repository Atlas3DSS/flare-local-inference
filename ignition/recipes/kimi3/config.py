"""Kimi K3 geometry and checkpoint quantization contracts."""

from dataclasses import dataclass
import json
from pathlib import Path
import re

from ignition.components.parallel import TensorParallelShard

PAGE_TOKENS = 128


@dataclass(frozen=True)
class ModelConfig:
    text: dict
    vision: dict
    draft: object = None

    @classmethod
    def load(cls, snapshot):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()))

    @classmethod
    def from_dict(cls, raw):
        if raw.get("model_type") != "kimi_k3" or raw.get("architectures") != ["KimiK3ForConditionalGeneration"]:
            raise ValueError("Kimi K3 requires KimiK3ForConditionalGeneration")
        c = raw["text_config"]
        required = dict(model_type="kimi_linear", hidden_act="situ", activation_situ_beta=4.0,
            activation_situ_linear_beta=25.0, mla_use_nope=True, mla_use_output_gate=True,
            latent_moe_use_norm=True, moe_renormalize=True, moe_router_activation_func="sigmoid",
            num_expert_group=1, topk_group=1, topk_method="noaux_tc", moe_layer_freq=1,
            num_nextn_predict_layers=0, tie_word_embeddings=False)
        for name, value in required.items():
            if c.get(name) != value:
                raise ValueError(f"unsupported Kimi K3 {name}: {c.get(name)!r}")
        for name in ("hidden_size", "num_hidden_layers", "intermediate_size", "moe_intermediate_size",
                     "num_attention_heads", "routed_expert_hidden_size", "num_experts", "num_experts_per_token",
                     "num_shared_experts", "attn_res_block_size", "vocab_size"):
            if type(c.get(name)) is not int or c[name] <= 0:
                raise ValueError(f"Kimi K3 requires positive {name}")
        if not 0 < c["num_experts_per_token"] <= c["num_experts"] <= 1024:
            raise ValueError("Kimi K3 routing requires top_k <= experts <= 1024")
        if not 0 <= c["first_k_dense_replace"] <= c["num_hidden_layers"]:
            raise ValueError("dense layer count exceeds the decoder")
        if (c["kv_lora_rank"], c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]) != (512, 128, 64, 128):
            raise ValueError("Kimi K3 requires MLA dimensions (512,128,64,128)")
        kda = c["linear_attn_config"]
        if any(kda.get(k) != v for k, v in dict(head_dim=128, short_conv_kernel_size=4,
                use_full_rank_gate=True, gate_lower_bound=-5.0).items()):
            raise ValueError("unsupported Kimi K3 KDA geometry")
        linear, full = kda["kda_layers"], kda["full_attn_layers"]
        if sorted(linear + full) != list(range(1, c["num_hidden_layers"] + 1)):
            raise ValueError("Kimi K3 attention layers must partition the one-based decoder indices")
        q = c["quantization_config"]
        if (q.get("quant_method") != "compressed-tensors" or q.get("format") != "mxfp4-pack-quantized"
                or q.get("quantization_status") != "compressed" or q.get("kv_cache_scheme") is not None
                or q.get("weight_scale_encoding") != "csf_32x2_v1"):
            raise ValueError("Kimi K3 requires compressed-tensors MXFP4 with csf_32x2_v1 scales")
        groups = list(q.get("config_groups", {}).values())
        if len(groups) != 1:
            raise ValueError("Kimi K3 requires one MXFP4 quantization group")
        group, w = groups[0], groups[0].get("weights", {})
        if (group.get("targets") != ["Linear"] or group.get("format") != q["format"]
                or group.get("input_activations") is not None or group.get("output_activations") is not None
                or any(w.get(k) != v for k, v in dict(num_bits=4, group_size=32, strategy="group",
                    type="float", symmetric=True, dynamic=False, scale_dtype="torch.uint8").items())):
            raise ValueError("unsupported Kimi K3 MXFP4 group")
        vision = raw["vision_config"]
        expected = dict(vt_hidden_size=1024, vt_intermediate_size=4096, vt_num_attention_heads=12,
            vt_num_hidden_layers=27, qkv_hidden_size=1536, patch_size=14, norm_type="rmsnorm",
            mlp_type="mlp2", activation_func="gelu_pytorch_tanh", attn_bias=False, linear_bias=False,
            patch_embed_proj_bias=False, init_pos_emb_height=64, init_pos_emb_width=64, init_pos_emb_time=4,
            pos_emb_type="divided_fixed", pos_emb_interpolation_mode="bilinear", merge_kernel_size=[2, 2],
            merge_type="sd2_tpool", mm_projector_type="patchmergerv2", mm_hidden_size=1024,
            text_hidden_size=c["hidden_size"], projector_hidden_act="gelu", projector_ln_eps=1e-5)
        for name, value in expected.items():
            if vision.get(name) != value:
                raise ValueError(f"unsupported Kimi K3 vision {name}: {vision.get(name)!r}")
        if raw.get("media_placeholder_token_id") != 163605 or raw.get("image_placeholder") != "<|kimi_image_placeholder|>":
            raise ValueError("unsupported Kimi K3 image placeholder")
        return cls(c, vision)

    @property
    def hidden(self):
        return self.text["hidden_size"]

    @property
    def vocabulary(self):
        return self.text["vocab_size"]

    @property
    def layers(self):
        linear = set(self.text["linear_attn_config"]["kda_layers"])
        return tuple("kda" if i + 1 in linear else "mla" for i in range(self.text["num_hidden_layers"]))

    def validate_shard(self, shard: TensorParallelShard):
        c = self.text
        for extent, alignment in ((c["num_attention_heads"], 1),
                (c["linear_attn_config"]["num_heads"], 1), (c["hidden_size"], 8),
                (c["routed_expert_hidden_size"], 8), (c["moe_intermediate_size"], 32),
                (c["intermediate_size"], 32), (c["num_shared_experts"] * c["moe_intermediate_size"], 32),
                (c["num_experts"], 8),
                (c["vocab_size"], 64)):
            shard.partition(extent, alignment=alignment)

    def projection_format(self, name):
        ignore = self.text["quantization_config"]["ignore"]
        if any(re.search(pattern[3:], name) if pattern.startswith("re:") else pattern == name for pattern in ignore):
            return "bf16"
        return "mxfp4_csf"
