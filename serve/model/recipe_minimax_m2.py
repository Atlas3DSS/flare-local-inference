"""Layer surgery recipe for MiniMax-M2.5-NVFP4.

Extracts weights from a vanilla HuggingFace MiniMaxM2ForCausalLM via
direct safetensor loading, applies TP sharding, packs expert weights
into b12x format, and returns generic FusedMoELayer objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from b12x.cute.fp4 import swizzle_block_scale
from b12x.integration.tp_moe import B12XFP4ExpertWeights

from serve.model.attention import B12xPagedAttention
from serve.model.ffn import MoEFFN
from serve.model.layer import TransformerLayer
from serve.model.ops import make_norm
from serve.tp.group import TPGroup


@dataclass(frozen=True)
class MiniMaxM2ModelConfig:
    """Resolved model geometry after TP sharding."""

    hidden_size: int        # 3072.
    num_q_heads: int        # Per-GPU after TP.
    num_kv_heads: int       # Per-GPU after TP.
    head_dim: int           # 128.
    rotary_dim: int         # 64.
    num_experts: int        # 256.
    top_k: int              # 8.
    intermediate_size: int  # Per-GPU after TP.
    num_layers: int         # 62.
    vocab_size: int         # 200064.
    rope_base: float        # 5e6.
    rms_norm_eps: float     # 1e-6.
    routing_fn: str = "sigmoid"  # "sigmoid" or "softmax".


def build_config(hf_config, tp_world_size: int = 1) -> MiniMaxM2ModelConfig:
    """Build resolved config from HF config + TP parameters."""
    return MiniMaxM2ModelConfig(
        hidden_size=hf_config.hidden_size,
        num_q_heads=hf_config.num_attention_heads // tp_world_size,
        num_kv_heads=hf_config.num_key_value_heads // tp_world_size,
        head_dim=hf_config.head_dim,
        rotary_dim=getattr(hf_config, "rotary_dim", hf_config.head_dim // 2),
        num_experts=hf_config.num_local_experts,
        top_k=hf_config.num_experts_per_tok,
        intermediate_size=hf_config.intermediate_size // tp_world_size,
        num_layers=hf_config.num_hidden_layers,
        vocab_size=hf_config.vocab_size,
        rope_base=getattr(hf_config, "rope_theta", None) or hf_config.rope_scaling.get("rope_theta", 10000.0),
        rms_norm_eps=hf_config.rms_norm_eps,
        routing_fn=getattr(hf_config, "scoring_func", "sigmoid"),
    )


# -- surgery ---------------------------------------------------------------


def extract_layer(
    hf_layer,
    layer_idx: int,
    cfg: MiniMaxM2ModelConfig,
    tp_group: Optional[TPGroup],
    device: str,
    loader,
) -> TransformerLayer:
    """Extract one layer into a TransformerLayer with paged attention + MoE FFN."""
    rank = tp_group.rank if tp_group is not None else 0
    world_size = tp_group.world_size if tp_group is not None else 1
    prefix = f"model.layers.{layer_idx}"

    # -- attention ---------------------------------------------------------
    q_weight = _load_sharded_dim0(loader, f"{prefix}.self_attn.q_proj.weight", rank, world_size)
    k_weight = _load_sharded_dim0(loader, f"{prefix}.self_attn.k_proj.weight", rank, world_size)
    v_weight = _load_sharded_dim0(loader, f"{prefix}.self_attn.v_proj.weight", rank, world_size)
    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0).contiguous()
    o_proj_weight = _load_sharded_dim1(loader, f"{prefix}.self_attn.o_proj.weight", rank, world_size)
    q_norm_w = _load_sharded_dim0(loader, f"{prefix}.self_attn.q_norm.weight", rank, world_size)
    k_norm_w = _load_sharded_dim0(loader, f"{prefix}.self_attn.k_norm.weight", rank, world_size)

    attention = B12xPagedAttention(
        num_q_heads=cfg.num_q_heads, num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim, hidden_size=cfg.hidden_size,
        rotary_dim=cfg.rotary_dim, rms_norm_eps=cfg.rms_norm_eps,
        qkv_weight=qkv_weight, o_proj_weight=o_proj_weight,
        q_norm_weight=q_norm_w, k_norm_weight=k_norm_w,
        tp_group=tp_group,
    )

    # -- MoE FFN -----------------------------------------------------------
    gate_weight = loader.tensor(f"{prefix}.block_sparse_moe.gate.weight")
    gate_bias_key = f"{prefix}.block_sparse_moe.e_score_correction_bias"
    gate_bias = loader.optional(gate_bias_key)
    experts = _pack_experts(loader, prefix, cfg, rank, world_size, device)

    ffn = MoEFFN(
        gate_weight=gate_weight, gate_bias=gate_bias, experts=experts,
        top_k=cfg.top_k, routing_fn=cfg.routing_fn, renormalize_topk=True,
        tp_group=tp_group,
    )

    # -- norms -------------------------------------------------------------
    input_ln = loader.tensor(f"{prefix}.input_layernorm.weight")
    post_attn_ln = loader.tensor(f"{prefix}.post_attention_layernorm.weight")

    return TransformerLayer(
        attn=attention, ffn=ffn,
        norm1=make_norm(input_ln, cfg.rms_norm_eps),
        norm2=make_norm(post_attn_ln, cfg.rms_norm_eps),
    )


def _pack_experts(
    loader,
    prefix: str,
    cfg: MiniMaxM2ModelConfig,
    rank: int,
    world_size: int,
    device: str,
) -> B12XFP4ExpertWeights:
    """Pack expert weights into b12x format via direct safetensor loading.

    MiniMax M2 uses w1 (gate_proj), w2 (down_proj), w3 (up_proj) naming.
    b12x convention: w13 = cat([up, gate], dim=1).
    """
    E = cfg.num_experts
    K = cfg.hidden_size
    I_tp = cfg.intermediate_size

    gate_w = torch.empty(E, I_tp, K // 2, dtype=torch.uint8, device=device)
    up_w = torch.empty(E, I_tp, K // 2, dtype=torch.uint8, device=device)
    down_w = torch.empty(E, K, I_tp // 2, dtype=torch.uint8, device=device)

    gate_sf = torch.empty(E, I_tp, K // 16, dtype=torch.float8_e4m3fn, device=device)
    up_sf = torch.empty(E, I_tp, K // 16, dtype=torch.float8_e4m3fn, device=device)
    down_sf = torch.empty(E, K, I_tp // 16, dtype=torch.float8_e4m3fn, device=device)

    gate_gs = torch.empty(E, dtype=torch.float32, device=device)
    down_gs = torch.empty(E, dtype=torch.float32, device=device)
    gate_is = torch.empty(E, dtype=torch.float32, device=device)
    down_is = torch.empty(E, dtype=torch.float32, device=device)

    ep = f"{prefix}.block_sparse_moe.experts"
    for eid in range(E):
        # w1 = gate_proj, w3 = up_proj, w2 = down_proj.
        loader.load_into_dim0_shard(gate_w[eid], f"{ep}.{eid}.w1.weight", unit=I_tp, pad=False)
        loader.load_into_dim0_shard(gate_sf[eid], f"{ep}.{eid}.w1.weight_scale", unit=I_tp, pad=False)
        gate_gs[eid] = loader.scalar(f"{ep}.{eid}.w1.weight_scale_2")
        gate_is_key = f"{ep}.{eid}.w1.input_scale"
        gate_is[eid] = loader.scalar(gate_is_key, default=1.0)

        loader.load_into_dim0_shard(up_w[eid], f"{ep}.{eid}.w3.weight", unit=I_tp, pad=False)
        loader.load_into_dim0_shard(up_sf[eid], f"{ep}.{eid}.w3.weight_scale", unit=I_tp, pad=False)

        loader.load_into_dim1_shard(down_w[eid], f"{ep}.{eid}.w2.weight", unit=I_tp // 2, pad=False)
        loader.load_into_dim1_shard(down_sf[eid], f"{ep}.{eid}.w2.weight_scale", unit=I_tp // 16, pad=False)
        down_gs[eid] = loader.scalar(f"{ep}.{eid}.w2.weight_scale_2")
        down_is_key = f"{ep}.{eid}.w2.input_scale"
        down_is[eid] = loader.scalar(down_is_key, default=1.0)

    # Pack: up first, gate second (matching b12x convention).
    w13 = torch.cat([up_w, gate_w], dim=1).contiguous()
    w13_sf = torch.cat([up_sf, gate_sf], dim=1).contiguous()
    w13_blockscale_swizzled = swizzle_block_scale(w13_sf)
    w2_blockscale_swizzled = swizzle_block_scale(down_sf)

    # Per-expert reciprocal input scales and fused alphas.
    g1_alphas = (gate_is * gate_gs).to(torch.float32)
    g2_alphas = (down_is * down_gs).to(torch.float32)

    return B12XFP4ExpertWeights(
        a1_gscale=(1.0 / gate_is).to(torch.float32).contiguous(),
        w1_fp4=w13,
        w1_blockscale=w13_blockscale_swizzled,
        w1_alphas=g1_alphas,
        a2_gscale=(1.0 / down_is).to(torch.float32).contiguous(),
        w2_fp4=down_w.contiguous(),
        w2_blockscale=w2_blockscale_swizzled,
        w2_alphas=g2_alphas,
    )


# -- helpers ---------------------------------------------------------------


def _load_sharded_dim0(loader, key: str, rank: int, world_size: int) -> torch.Tensor:
    """Load a tensor and shard along dim 0."""
    del rank, world_size
    return loader.dim0_shard(key)


def _load_sharded_dim1(loader, key: str, rank: int, world_size: int) -> torch.Tensor:
    """Load a tensor and shard along dim 1."""
    del rank, world_size
    return loader.dim1_shard(key)
