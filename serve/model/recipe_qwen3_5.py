"""Layer surgery recipe for Qwen3.5-397B-A17B-NVFP4.

Hybrid architecture: 15 self-attention layers + 45 GDN linear attention
layers, all with 512-expert MoE + shared expert.

Weight prefix: model.language_model.layers.N.{self_attn,linear_attn,mlp}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from b12x.cute.fp4 import swizzle_block_scale
from b12x.integration.tp_moe import B12XFP4ExpertWeights

from serve.logging import get_logger
from serve.model.attention import B12xPagedAttention
from serve.model.ffn import MoEFFN
from serve.model.gdn import GDNLinearAttention
from serve.model.layer import TransformerLayer
from serve.model.loader import LoadedModel, ShardedLoader, register_recipe
from serve.model.ops import make_norm, precompute_rope_freqs
from serve.tp.group import tp_shard_dim0, tp_shard_dim1, TPGroup

LOGGER = get_logger(__name__)

@dataclass(frozen=True)
class Qwen35ModelConfig:
    """Resolved model geometry."""
    hidden_size: int         # 4096.
    num_q_heads: int         # Per-GPU after TP.
    num_kv_heads: int        # Per-GPU after TP.
    total_num_kv_heads: int  # Global KV head count before TP.
    kv_head_replicas: int    # How many TP ranks share each KV head shard.
    head_dim: int            # 256.
    rotary_dim: int          # 64 (partial_rotary_factor=0.25).
    # Linear attention.
    linear_num_k_heads: int  # Per-GPU after TP.
    linear_num_v_heads: int  # Per-GPU after TP.
    linear_head_k_dim: int   # 128.
    linear_head_v_dim: int   # 128.
    linear_conv_kernel: int  # 4.
    # MoE.
    num_experts: int         # 512.
    top_k: int               # 10.
    moe_intermediate_size: int  # 1024.
    shared_expert_intermediate_size: int  # 1024.
    # Model.
    num_layers: int          # 60.
    vocab_size: int          # 248320.
    rope_base: float         # 1e7.
    rms_norm_eps: float      # 1e-6.
    full_attention_interval: int  # 4.
    gemma_norm: bool = True  # Qwen3.5 uses (1+weight) RMSNorm.
    layer_types: list[str] = field(default_factory=list)


def build_config(hf_config, tp_world_size: int = 1) -> Qwen35ModelConfig:
    tc = hf_config.text_config if hasattr(hf_config, 'text_config') else hf_config

    full_attn_interval = getattr(tc, 'full_attention_interval', 4)
    num_layers = tc.num_hidden_layers
    total_num_kv_heads = tc.num_key_value_heads
    if tp_world_size >= total_num_kv_heads:
        num_kv_heads = 1
        kv_head_replicas = tp_world_size // total_num_kv_heads
    else:
        num_kv_heads = total_num_kv_heads // tp_world_size
        kv_head_replicas = 1

    layer_types = []
    for i in range(num_layers):
        if (i + 1) % full_attn_interval == 0:
            layer_types.append("attention")
        else:
            layer_types.append("linear_attention")

    rope_params = getattr(tc, 'rope_parameters', {}) or {}

    def _heads_per_rank(total_heads, tp):
        """Heads per rank, padded so total is divisible by tp."""
        return ((total_heads + tp - 1) // tp)

    return Qwen35ModelConfig(
        hidden_size=tc.hidden_size,
        # Q heads per rank must be a multiple of KV heads for GQA.
        num_q_heads=_heads_per_rank(tc.num_attention_heads, tp_world_size)
                    if _heads_per_rank(tc.num_attention_heads, tp_world_size) % tc.num_key_value_heads == 0
                    else _heads_per_rank(tc.num_attention_heads // tc.num_key_value_heads, tp_world_size) * tc.num_key_value_heads,
        num_kv_heads=num_kv_heads,
        total_num_kv_heads=total_num_kv_heads,
        kv_head_replicas=kv_head_replicas,
        head_dim=tc.head_dim,
        rotary_dim=int(tc.head_dim * getattr(tc, 'partial_rotary_factor', 0.25)),
        # GDN K and V heads must maintain their ratio (V = K * ratio) for GQA grouping.
        linear_num_k_heads=_heads_per_rank(getattr(tc, 'linear_num_key_heads', 16), tp_world_size),
        linear_num_v_heads=_heads_per_rank(getattr(tc, 'linear_num_key_heads', 16), tp_world_size)
                           * (getattr(tc, 'linear_num_value_heads', 64) // getattr(tc, 'linear_num_key_heads', 16)),
        linear_head_k_dim=getattr(tc, 'linear_key_head_dim', 128),
        linear_head_v_dim=getattr(tc, 'linear_value_head_dim', 128),
        linear_conv_kernel=getattr(tc, 'linear_conv_kernel_dim', 4),
        num_experts=tc.num_experts,
        top_k=tc.num_experts_per_tok,
        moe_intermediate_size=getattr(tc, 'moe_intermediate_size', 1024),
        shared_expert_intermediate_size=getattr(tc, 'shared_expert_intermediate_size', 1024),
        num_layers=num_layers,
        vocab_size=tc.vocab_size,
        rope_base=rope_params.get('rope_theta', 1e7),
        rms_norm_eps=tc.rms_norm_eps,
        full_attention_interval=full_attn_interval,
        layer_types=layer_types,
    )


# -- surgery ---------------------------------------------------------------


def _load(loader, key):
    """Load a weight, returning None if not found."""
    return loader.optional(key)


def _load_sharded_dim0(loader, key, rank, world_size):
    del rank, world_size
    return loader.dim0_shard(key)


def _load_sharded_dim1(loader, key, rank, world_size):
    del rank, world_size
    return loader.dim1_shard(key)


def extract_attention_layer(
    layer_idx: int, cfg: Qwen35ModelConfig,
    tp_group: Optional[TPGroup], device: str, loader: ShardedLoader,
) -> HybridMoELayer:
    """Build a self-attention + MoE layer."""
    rank = tp_group.rank if tp_group else 0
    world_size = tp_group.world_size if tp_group else 1
    prefix = f"model.language_model.layers.{layer_idx}"

    # QKV — Q is sharded by heads. K/V follow QKVParallelLinear semantics:
    # when TP > total_kv_heads, each KV head shard is replicated across a
    # group of ranks instead of loading all KV heads on every rank.
    # Q is doubled for output gate: [Q_real || Q_gate] interleaved per head.
    hd = cfg.head_dim
    q_unit = 2 * hd * cfg.num_kv_heads
    q_weight = loader.dim0_shard(f"{prefix}.self_attn.q_proj.weight", unit=q_unit)
    kv_shard = cfg.num_kv_heads * hd
    kv_logical_world = world_size // cfg.kv_head_replicas
    k_weight = loader.dim0_shard(
        f"{prefix}.self_attn.k_proj.weight",
        unit=kv_shard,
        shard_world_size=kv_logical_world,
        replica_group_size=cfg.kv_head_replicas,
        pad=False,
    )
    v_weight = loader.dim0_shard(
        f"{prefix}.self_attn.v_proj.weight",
        unit=kv_shard,
        shard_world_size=kv_logical_world,
        replica_group_size=cfg.kv_head_replicas,
        pad=False,
    )
    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0).contiguous()

    # o_proj maps from Q_real heads only (not doubled). Must match Q head count.
    o_proj_weight = loader.dim1_shard(f"{prefix}.self_attn.o_proj.weight", unit=hd * cfg.num_kv_heads)
    q_norm_w = loader.tensor(f"{prefix}.self_attn.q_norm.weight")  # Per-head, not sharded.
    k_norm_w = loader.tensor(f"{prefix}.self_attn.k_norm.weight")  # Per-head, not sharded.

    attention = B12xPagedAttention(
        num_q_heads=cfg.num_q_heads, num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim, hidden_size=cfg.hidden_size,
        rotary_dim=cfg.rotary_dim, rms_norm_eps=cfg.rms_norm_eps,
        qkv_weight=qkv_weight.to(device), o_proj_weight=o_proj_weight.to(device),
        q_norm_weight=q_norm_w.to(device), k_norm_weight=k_norm_w.to(device),
        tp_group=tp_group, gemma_norm=True, output_gate=True,
    )

    ffn = _build_moe_ffn(loader, prefix, cfg, device, rank, world_size, tp_group)
    input_ln = loader.tensor(f"{prefix}.input_layernorm.weight").to(device)
    post_attn_ln = loader.tensor(f"{prefix}.post_attention_layernorm.weight").to(device)
    gemma = cfg.gemma_norm

    return TransformerLayer(
        attn=attention, ffn=ffn,
        norm1=make_norm(input_ln, cfg.rms_norm_eps, gemma),
        norm2=make_norm(post_attn_ln, cfg.rms_norm_eps, gemma),
    ).to(device)


def extract_linear_layer(
    layer_idx: int, cfg: Qwen35ModelConfig,
    tp_group: Optional[TPGroup], device: str, loader: ShardedLoader,
) -> HybridMoELayer:
    """Build a GDN linear attention + MoE layer."""
    rank = tp_group.rank if tp_group else 0
    world_size = tp_group.world_size if tp_group else 1
    prefix = f"model.language_model.layers.{layer_idx}"

    # GDN weights — shard heads across TP (pad-and-shard for uneven TP).
    # in_proj_qkv is [Q_heads || K_heads || V_heads, hidden]. Split using checkpoint
    # tensor shape (not config, which has padded per-rank counts).
    full_qkv_shape = loader.shape(f"{prefix}.linear_attn.in_proj_qkv.weight")
    full_z_shape = loader.shape(f"{prefix}.linear_attn.in_proj_z.weight")
    v_total_full = full_z_shape[0]  # in_proj_z has [V_total, hidden].
    k_total_full = (full_qkv_shape[0] - v_total_full) // 2  # QKV = [2*K_total + V_total, hidden].
    kd = cfg.linear_head_k_dim
    vd = cfg.linear_head_v_dim
    # V heads must be sharded in groups that maintain the K:V head ratio.
    full_num_k = k_total_full // kd
    full_num_v = v_total_full // vd
    v_per_k = full_num_v // full_num_k  # GQA ratio (e.g. 4).
    v_unit = vd * v_per_k  # Shard V in groups of v_per_k heads.

    qkv_key = f"{prefix}.linear_attn.in_proj_qkv.weight"
    q_tp = loader.dim0_shard(qkv_key, start=0, length=k_total_full, unit=kd)
    k_tp = loader.dim0_shard(qkv_key, start=k_total_full, length=k_total_full, unit=kd)
    v_tp = loader.dim0_shard(qkv_key, start=2 * k_total_full, length=v_total_full, unit=v_unit)
    in_proj_qkv = torch.cat([q_tp, k_tp, v_tp], dim=0).contiguous()
    in_proj_z = loader.dim0_shard(f"{prefix}.linear_attn.in_proj_z.weight", unit=v_unit)
    in_proj_a = loader.dim0_shard(f"{prefix}.linear_attn.in_proj_a.weight", unit=v_per_k)
    in_proj_b = loader.dim0_shard(f"{prefix}.linear_attn.in_proj_b.weight", unit=v_per_k)
    # conv1d weight has same [Q || K || V] structure on dim 0.
    conv_key = f"{prefix}.linear_attn.conv1d.weight"
    conv1d_w = torch.cat(
        [
            loader.dim0_shard(conv_key, start=0, length=k_total_full, unit=kd),
            loader.dim0_shard(conv_key, start=k_total_full, length=k_total_full, unit=kd),
            loader.dim0_shard(conv_key, start=2 * k_total_full, length=v_total_full, unit=v_unit),
        ],
        dim=0,
    ).contiguous()
    out_proj = loader.dim1_shard(f"{prefix}.linear_attn.out_proj.weight", unit=v_unit)
    norm_w = loader.tensor(f"{prefix}.linear_attn.norm.weight")  # Per-head, not sharded.
    A_log = loader.dim0_shard(f"{prefix}.linear_attn.A_log", unit=v_per_k)
    dt_bias = loader.dim0_shard(f"{prefix}.linear_attn.dt_bias", unit=v_per_k)

    linear_attn = GDNLinearAttention(
        num_k_heads=cfg.linear_num_k_heads,
        num_v_heads=cfg.linear_num_v_heads,
        head_k_dim=cfg.linear_head_k_dim,
        head_v_dim=cfg.linear_head_v_dim,
        hidden_size=cfg.hidden_size,
        conv_kernel=cfg.linear_conv_kernel,
        in_proj_qkv_weight=in_proj_qkv.to(device),
        in_proj_z_weight=in_proj_z.to(device),
        in_proj_a_weight=in_proj_a.to(device),
        in_proj_b_weight=in_proj_b.to(device),
        conv1d_weight=conv1d_w.to(device),
        out_proj_weight=out_proj.to(device),
        norm_weight=norm_w.to(device),
        A_log=A_log.to(device),
        dt_bias=dt_bias.to(device),
        rms_norm_eps=cfg.rms_norm_eps,
        tp_group=tp_group,
    )

    # MoE + norms (same for both layer types).
    input_ln = loader.tensor(f"{prefix}.input_layernorm.weight").to(device)
    ffn = _build_moe_ffn(loader, prefix, cfg, device, rank, world_size, tp_group)
    post_attn_ln = loader.tensor(f"{prefix}.post_attention_layernorm.weight").to(device)
    gemma = cfg.gemma_norm

    return TransformerLayer(
        attn=linear_attn, ffn=ffn,
        norm1=make_norm(input_ln, cfg.rms_norm_eps, gemma),
        norm2=make_norm(post_attn_ln, cfg.rms_norm_eps, gemma),
    ).to(device)


def _build_moe_ffn(loader, prefix, cfg, device, rank, world_size, tp_group):
    """Build the MoEFFN block shared by both attention and linear layers."""
    gate_weight = loader.tensor(f"{prefix}.mlp.gate.weight").to(device)
    experts = _load_experts(loader, prefix, cfg, device, rank, world_size)
    shared_expert = _load_shared_expert(loader, prefix, device, rank, world_size)
    shared_gate = _load(loader, f"{prefix}.mlp.shared_expert_gate.weight")
    if shared_gate is not None:
        shared_gate = shared_gate.to(device)
    return MoEFFN(
        gate_weight=gate_weight, experts=experts,
        top_k=cfg.top_k, routing_fn="softmax", renormalize_topk=True,
        tp_group=tp_group,
        shared_expert=shared_expert, shared_expert_gate_weight=shared_gate,
    )


def _load_experts(loader, prefix, cfg, device, rank=0, world_size=1):
    """Load all routed experts into B12XFP4ExpertWeights.

    TP shards the intermediate dimension: gate/up sharded on dim0,
    down sharded on dim1. Pads for uneven TP via tp_shard_dim0/dim1.
    """
    E = cfg.num_experts

    # Probe expert 0 for shapes.
    ep0 = f"{prefix}.mlp.experts.0"
    gate_rows_tp, gate_cols = loader.dim0_shard_shape(f"{ep0}.gate_proj.weight", unit=8)
    down_rows, down_cols_tp = loader.dim1_shard_shape(f"{ep0}.down_proj.weight", unit=8)
    gate_s_rows_tp, gate_s_cols = loader.dim0_shard_shape(f"{ep0}.gate_proj.weight_scale", unit=8)
    down_s_rows, down_s_cols_tp = loader.dim1_shard_shape(f"{ep0}.down_proj.weight_scale")
    gate_w0 = loader.dim0_shard(f"{ep0}.gate_proj.weight", unit=8)
    up_w0 = loader.dim0_shard(f"{ep0}.up_proj.weight", unit=8)
    gate_s0 = loader.dim0_shard(f"{ep0}.gate_proj.weight_scale", unit=8)
    up_s0 = loader.dim0_shard(f"{ep0}.up_proj.weight_scale", unit=8)
    down_w0 = loader.dim1_shard(f"{ep0}.down_proj.weight", unit=8)
    down_s0 = loader.dim1_shard(f"{ep0}.down_proj.weight_scale")

    # Compute per-rank shapes.
    # Shard gate and up SEPARATELY, then cat [up, gate] per rank.
    # This matches sglang's MergedColumnParallelLinear with load_up_proj_weight_first=True
    # where each rank's w13 = [up_shard, gate_shard].
    w13_rows = gate_rows_tp * 2
    w13_weight = torch.empty(E, w13_rows, gate_cols, dtype=gate_w0.dtype, device=device)
    w13_scale = torch.empty(E, gate_s_rows_tp * 2, gate_s_cols, dtype=gate_s0.dtype, device=device)
    w2_weight = torch.empty(E, down_rows, down_cols_tp, dtype=down_w0.dtype, device=device)
    w2_scale = torch.empty(E, down_s_rows, down_s_cols_tp, dtype=down_s0.dtype, device=device)
    a1_gscale = torch.ones(E, dtype=torch.float32, device=device)
    a2_gscale = torch.ones(E, dtype=torch.float32, device=device)
    g1_alphas = torch.ones(E, dtype=torch.float32, device=device)
    g2_alphas = torch.ones(E, dtype=torch.float32, device=device)

    for e in range(E):
        ep = f"{prefix}.mlp.experts.{e}"
        if e == 0:
            gate_w = gate_w0
            up_w = up_w0
            gate_s = gate_s0
            up_s = up_s0
            down_w = down_w0
            down_s = down_s0
        else:
            gate_w = loader.dim0_shard(f"{ep}.gate_proj.weight", unit=8)
            up_w = loader.dim0_shard(f"{ep}.up_proj.weight", unit=8)
            gate_s = loader.dim0_shard(f"{ep}.gate_proj.weight_scale", unit=8)
            up_s = loader.dim0_shard(f"{ep}.up_proj.weight_scale", unit=8)
            down_w = loader.dim1_shard(f"{ep}.down_proj.weight", unit=8)
            down_s = loader.dim1_shard(f"{ep}.down_proj.weight_scale")
        w13_weight[e] = torch.cat([up_w, gate_w], dim=0).to(device)

        w13_scale[e] = torch.cat([up_s, gate_s], dim=0).to(device)

        w2_weight[e] = down_w.to(device)
        w2_scale[e] = down_s.to(device)

        gate_is = loader.scalar(f"{ep}.gate_proj.input_scale", default=None)
        down_is = loader.scalar(f"{ep}.down_proj.input_scale", default=None)
        gate_s2 = loader.scalar(f"{ep}.gate_proj.weight_scale_2", default=None)
        down_s2 = loader.scalar(f"{ep}.down_proj.weight_scale_2", default=None)

        # Store reciprocal input scales (matching sglang's w13_input_scale_quant).
        if gate_is is not None:
            a1_gscale[e] = float(1.0 / gate_is)
        if down_is is not None:
            a2_gscale[e] = float(1.0 / down_is)
        if gate_is is not None and gate_s2 is not None:
            g1_alphas[e] = float(gate_is * gate_s2)
        if down_is is not None and down_s2 is not None:
            g2_alphas[e] = float(down_is * down_s2)

    w13_blockscale = swizzle_block_scale(w13_scale)
    w2_blockscale = swizzle_block_scale(w2_scale)

    return B12XFP4ExpertWeights(
        a1_gscale=a1_gscale,
        w1_fp4=w13_weight,
        w1_blockscale=w13_blockscale,
        w1_alphas=g1_alphas,
        a2_gscale=a2_gscale,
        w2_fp4=w2_weight.contiguous(),
        w2_blockscale=w2_blockscale,
        w2_alphas=g2_alphas,
    )


def _dequant_fp4(weight, scale, scale2):
    """Dequantize NVFP4 weight to BF16: unpack uint8 → float, apply scales."""
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=weight.device,
    )
    packed = weight.to(torch.int32)
    lo = lut[packed & 0xF]
    hi = lut[(packed >> 4) & 0xF]
    vals = torch.stack([lo, hi], dim=-1).reshape(weight.shape[0], weight.shape[1] * 2)
    grouped_scale = scale.float().unsqueeze(-1).expand(-1, -1, 16).reshape(weight.shape[0], -1)
    return (vals * grouped_scale * float(scale2)).to(torch.bfloat16)


def _load_shared_expert(loader, prefix, device, rank=0, world_size=1):
    """Load shared expert weights (TP-sharded), dequantizing FP4 to BF16 if needed."""
    ep = f"{prefix}.mlp.shared_expert"
    gate_key = f"{ep}.gate_proj.weight"
    if not loader.exists(gate_key):
        return None

    if loader.dtype(gate_key) == torch.uint8:
        return _load_shared_expert_fp4(loader, prefix, device, rank, world_size)

    del rank, world_size
    gate = loader.dim0_shard(gate_key, unit=8)
    up = loader.dim0_shard(f"{ep}.up_proj.weight", unit=8)
    down = loader.dim1_shard(f"{ep}.down_proj.weight", unit=8)

    # Fuse gate+up into single weight (matching sglang's MergedColumnParallelLinear).
    # SiluAndMul splits at midpoint: silu(first_half) * second_half.
    gate_up = torch.cat([gate, up], dim=0).contiguous()

    return {
        "gate_up_proj": gate_up.to(device, dtype=torch.bfloat16),
        "down_proj": down.to(device, dtype=torch.bfloat16),
    }


def _load_shared_expert_fp4(loader, prefix, device, rank=0, world_size=1):
    """Load a quantized shared expert and dequantize it once to BF16."""
    ep = f"{prefix}.mlp.shared_expert"
    # Down-proj is column-packed in FP4. Dequantize the full tensor first, then
    # shard the BF16 result so TP columns stay aligned with the dense path.
    gate = _dequant_fp4(
        loader.tensor(f"{ep}.gate_proj.weight"),
        loader.tensor(f"{ep}.gate_proj.weight_scale"),
        loader.scalar(f"{ep}.gate_proj.weight_scale_2"),
    )
    up = _dequant_fp4(
        loader.tensor(f"{ep}.up_proj.weight"),
        loader.tensor(f"{ep}.up_proj.weight_scale"),
        loader.scalar(f"{ep}.up_proj.weight_scale_2"),
    )
    down = _dequant_fp4(
        loader.tensor(f"{ep}.down_proj.weight"),
        loader.tensor(f"{ep}.down_proj.weight_scale"),
        loader.scalar(f"{ep}.down_proj.weight_scale_2"),
    )

    gate = tp_shard_dim0(gate, rank, world_size, unit=8)
    up = tp_shard_dim0(up, rank, world_size, unit=8)
    down = tp_shard_dim1(down, rank, world_size, unit=8)
    gate_up = torch.cat([gate, up], dim=0).contiguous()

    return {
        "gate_up_proj": gate_up.to(device, dtype=torch.bfloat16),
        "down_proj": down.to(device, dtype=torch.bfloat16),
    }


# -- top-level recipe ------------------------------------------------------


@register_recipe("qwen3_5_moe")
def recipe_qwen3_5_moe(hf_model, hf_config, loader, device, tp_group):
    """Build a Qwen3.5-397B-A17B MoE model from HF weights."""
    import torch.nn as nn

    world_size = tp_group.world_size if tp_group else 1
    rank = tp_group.rank if tp_group else 0
    cfg = build_config(hf_config, world_size)

    attn_layers = sum(1 for t in cfg.layer_types if t == "attention")
    linear_layers = sum(1 for t in cfg.layer_types if t == "linear_attention")
    if rank == 0:
        LOGGER.info(
            f"Qwen3.5: {cfg.num_layers} layers ({attn_layers} attn + {linear_layers} linear), "
            f"{cfg.num_experts} experts, TP={world_size}"
        )

    layers = nn.ModuleList()
    loader.start_layer_progress("Qwen3.5 layers", total=cfg.num_layers)
    for i in range(cfg.num_layers):
        if cfg.layer_types[i] == "attention":
            layer = extract_attention_layer(i, cfg, tp_group, device, loader)
        else:
            layer = extract_linear_layer(i, cfg, tp_group, device, loader)
        layers.append(layer)
        loader.advance_layer_progress(description=f"Qwen3.5 layers [{i + 1}/{cfg.num_layers}]")

    # Embedding and head.
    embed_weight = loader.tensor("model.language_model.embed_tokens.weight").to(device)
    embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)

    final_norm = loader.tensor("model.language_model.norm.weight").to(device)
    lm_head = loader.tensor("lm_head.weight").to(device)

    # RoPE (only for self-attention layers).
    cos, sin = precompute_rope_freqs(
        cfg.head_dim, cfg.rotary_dim, 262144, cfg.rope_base, device,
    )

    return LoadedModel(
        layers=layers,
        embed_tokens=embed,
        final_norm_weight=final_norm,
        lm_head_weight=lm_head,
        cos=cos,
        sin=sin,
        config=cfg,
    )
