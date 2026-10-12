"""Tensor-parallel DeepSeek text stack with lagged hyperconnections."""

from dataclasses import dataclass, fields, is_dataclass

import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.embedding import PreparedEmbedding
from ignition.components.expert_overlap import ConcurrentSharedExpert
from ignition.components.linear import UnquantizedProjection
from ignition.components.mhc import MHCWeights, MHCScratch, PreparedMHC, PreparedMHCPost
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.moe import RouterScratch, PreparedRouter, PreparedRoutedExperts
from ignition.components.vocabulary import PreparedVocabularyProjection

from .attention import AttentionScratch, PreparedAttention, Rotary, normalize
from .engram import PreparedEngram, PreparedTableRequests


@dataclass(frozen=True)
class Scratch:
    embedding: torch.Tensor
    residual: torch.Tensor
    engram_residual: torch.Tensor
    pre_mix: torch.Tensor
    attn_mix: torch.Tensor
    attention_hc: MHCScratch
    ffn_hc: MHCScratch
    attention: AttentionScratch
    attention_output: torch.Tensor
    shared: MLPScratch
    router_logits: torch.Tensor
    router: RouterScratch
    routed: torch.Tensor
    mlp_partial: torch.Tensor
    mlp_output: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    def rows(self, tokens, samples):
        def select(value, count):
            if isinstance(value, torch.Tensor):
                return value[:count]
            if is_dataclass(value):
                return type(value)(*(select(getattr(value, field.name), count) for field in fields(value)))
            raise TypeError("decoder scratch must contain tensors and tensor dataclasses")
        return Scratch(*(select(getattr(self, field.name), samples if field.name in
            ("hidden", "local_logits", "logits") else tokens) for field in fields(self)))

    @classmethod
    def allocate(cls, tokens, samples, world, device, *, experts=384, top_k=6):
        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)
        return cls(zeros(tokens, 5120), zeros(tokens, 4, 5120), zeros(tokens, 4, 5120),
            zeros(tokens, 4, dtype=torch.float32), zeros(tokens, 4, dtype=torch.float32),
            MHCScratch.allocate(tokens, 5120, device), MHCScratch.allocate(tokens, 5120, device),
            AttentionScratch.allocate(tokens, world, device), zeros(tokens, 5120),
            MLPScratch.allocate(tokens, 5120, 2304 // world, device), zeros(tokens, experts, dtype=torch.float32),
            RouterScratch.allocate(tokens, device, experts=experts, top_k=top_k), zeros(tokens, 5120),
            zeros(tokens, 5120), zeros(tokens, 5120), zeros(samples, 5120),
            zeros(samples, 129280 // world), zeros(samples, 129280))


class PreparedLayer:
    def __init__(self, index, weights, tables, scratch, metadata, pool, rotary, *, shard, mode,
                 runtime, coordinator, session, name, workspace, carry, defer_post):
        s, text = scratch, pool.config.text
        self.scratch, self.carry = s, carry
        self.engram = None
        if weights.engram is not None:
            self.engram = PreparedEngram(index, weights.engram, tables, s.residual, s.engram_residual,
                metadata, pool,
                session=session, name=name + ".engram", workspace=workspace)
        source = s.engram_residual if self.engram is not None else s.residual
        options = dict(rms_eps=text["rms_norm_eps"], norm_eps=text["rms_norm_eps"], hc_eps=text["hc_eps"],
                       sinkhorn_iters=text["hc_sinkhorn_iters"])
        def mhc_weights(kind, norm):
            r = weights.residual
            return MHCWeights(r[f"hc_{kind}_fn"], r[f"hc_{kind}_scale"], r[f"hc_{kind}_base"], r[norm + ".weight"])
        incoming = ((s.mlp_output, s.ffn_hc.residual, s.ffn_hc.post, s.ffn_hc.comb) if carry else (source,))
        self.attention_hc = PreparedMHC(incoming, mhc_weights("attn", "attn_norm"), s.attention_hc,
            session=session, name=name + ".attn_hc", workspace=workspace,
            pre_mix=s.pre_mix, pre_out=s.attn_mix, **options)
        self.attention = PreparedAttention(index, weights.attention, s.attention_hc.normalized, s.attention,
            metadata, pool, rotary, mode=mode, world=shard.world_size,
            session=session, name=name + ".attention", workspace=workspace)
        self.attention_reduce = PreparedAllReduce(s.attention.output, s.attention_output, runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".attention_reduce")
        self.ffn_hc = PreparedMHC((s.attention_output, s.attention_hc.residual, s.attention_hc.post, s.attention_hc.comb),
            mhc_weights("ffn", "ffn_norm"), s.ffn_hc, session=session, name=name + ".ffn_hc", workspace=workspace,
            pre_mix=s.attn_mix, pre_out=s.pre_mix, **options)
        self.router_projection = UnquantizedProjection(s.ffn_hc.normalized, weights.experts.router, s.router_logits,
            session=session, name=name + ".router")
        self.router = PreparedRouter(s.router_logits, s.router, session=session, name=name + ".topk",
            score_func=text["scoring_func"], renormalize=text["norm_topk_prob"],
            routed_scaling_factor=text["routed_scaling_factor"], correction_bias=weights.experts.correction)
        self.experts = PreparedRoutedExperts(s.ffn_hc.normalized, weights.experts.experts, s.router.ids,
            s.router.weights, s.routed, session=session, name=name + ".experts", workspace=workspace,
            router_weight=weights.experts.router, router_plan=self.router.plan,
            correction_bias=weights.experts.correction)
        self.shared = PreparedMLP(s.ffn_hc.normalized, weights.experts.shared, s.shared.gate_up,
            s.shared.intermediate, s.shared.output, session=session, name=name + ".shared", workspace=workspace,
            swiglu_limit=text["swiglu_limit"], round_silu=False, activation_mode="auto")
        self.concurrent_shared = None
        if mode == "decode" and s.ffn_hc.normalized.shape[0] == 1:
            self.concurrent_shared = ConcurrentSharedExpert(self.shared,
                parallel_plans=(self.router_projection.plan, self.router.plan, self.experts.plan),
                session=session, workspace=workspace, defer_launch=True)
        self.mlp_reduce = PreparedAllReduce(s.mlp_partial, s.mlp_output, runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".mlp_reduce")
        self.post = None if defer_post else PreparedMHCPost(s.mlp_output, s.ffn_hc.residual,
            s.ffn_hc.post, s.ffn_hc.comb, s.residual, session=session, name=name + ".post")

    def __call__(self):
        s = self.scratch
        if self.engram is not None:
            source = self.engram(s.residual)
        else:
            source = s.residual
        incoming = ((s.mlp_output, s.ffn_hc.residual, s.ffn_hc.post, s.ffn_hc.comb)
                    if self.carry else (source,))
        self.attention_hc(*incoming)
        self.attention_reduce(self.attention(s.attention_hc.normalized))
        self.ffn_hc(s.attention_output, s.attention_hc.residual, s.attention_hc.post, s.attention_hc.comb)
        if self.concurrent_shared is not None:
            self.concurrent_shared.start(s.ffn_hc.normalized)
        self.router(self.router_projection(s.ffn_hc.normalized))
        self.experts()
        if self.concurrent_shared is None:
            self.shared(s.ffn_hc.normalized)
        else:
            self.concurrent_shared.finish(s.ffn_hc.normalized, s.routed)
        s.mlp_partial.copy_(s.routed + s.shared.output)
        self.mlp_reduce(s.mlp_partial)
        if self.post is not None:
            self.post(s.mlp_output, s.ffn_hc.residual, s.ffn_hc.post, s.ffn_hc.comb)


class PreparedDecoder:
    def __init__(self, weights, tables, scratch, metadata, pool, *, mode, runtime,
                 vocabulary_runtime, coordinator, session, name, workspace):
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        self.mode = mode
        self.table_requests = PreparedTableRequests(tables, metadata, pool, session=session, name=name + ".table_requests")
        self.aux_layers = () if weights.draft is None else weights.config.draft.target_layers
        self.auxiliary = torch.zeros((metadata.token_ids.numel(), len(self.aux_layers) * 5120),
                                    dtype=torch.bfloat16, device=scratch.embedding.device)
        self.embedding = PreparedEmbedding(weights.embedding, metadata.token_ids, scratch.embedding,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        self.rotary = Rotary(weights.config, scratch.embedding.device)
        self.layers, carry = [], False
        for index, layer in enumerate(weights.layers):
            defer = (index + 1 < len(weights.layers) and weights.layers[index + 1].engram is None
                     and index + 1 not in self.aux_layers)
            self.layers.append(PreparedLayer(index, layer, tables, scratch, metadata, pool, self.rotary,
                shard=weights.shard, mode=mode, runtime=runtime,
                coordinator=coordinator, session=session, name=f"{name}.layer{index}", workspace=workspace,
                carry=carry, defer_post=defer))
            carry = defer
        self.head = PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head")

    def __call__(self):
        s, m = self.scratch, self.metadata
        self.table_requests()
        s.residual.copy_(self.embedding()[:, None, :])
        s.pre_mix.zero_()
        s.pre_mix[:, 0].fill_(1)
        for index, layer in enumerate(self.layers):
            if index in self.aux_layers:
                column = self.aux_layers.index(index)
                self.auxiliary[:, column * 5120:(column + 1) * 5120].copy_(s.residual.float().mean(1))
            layer()
        residual = s.residual[m.sample_indices].float()
        premix = s.pre_mix[m.sample_indices, :, None]
        collapsed = (residual * premix).sum(1).bfloat16()
        normalize(collapsed, self.weights.norm, s.hidden, self.weights.config.text["rms_norm_eps"])
        self.head(s.hidden)
        if self.mode != "verify":
            self.table_requests.commit(m.lengths)
        return s.hidden

    def commit(self, counts):
        for layer in self.layers:
            layer.attention.commit(counts)
        self.table_requests.commit(counts)
