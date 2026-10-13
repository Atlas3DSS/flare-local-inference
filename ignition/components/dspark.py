"""Captured DSpark proposals, packed target verification, and accepted-state commits."""

from dataclasses import dataclass

import torch

from ignition.components.speculation import SpeculativeRngState

from .collectives import PreparedAllGather
from .compressed_mla import PreparedCompressedCacheWriter
from .embedding import PreparedEmbedding
from .linear import QuantizedProjection, UnquantizedProjection
from .rms_norm import PreparedRMSNorm
from .rotary import CompressedRotary as Rotary


@dataclass(frozen=True)
class DSparkForwards:
    decoder: type
    scratch: type
    context: type
    draft: type
    markov: type


class SpeculativePreparation:
    def __init__(
        self,
        weights,
        tables,
        pool,
        requests,
        inputs,
        scratch,
        *,
        draft_tokens,
        records,
        frame,
        sequence,
        session,
        workspace,
        runtime,
        vocabulary_runtime,
        markov_runtime,
        coordinator,
        forwards,
    ):
        self.forwards = forwards
        self.weights, self.tables, self.pool = weights, tables, pool
        self.requests, self.inputs, self.scratch = requests, inputs, scratch
        self.records, self.frame, self.sequence = records, frame, sequence
        self.session, self.workspace = session, workspace
        self.collectives = {
            "runtime": runtime, "vocabulary_runtime": vocabulary_runtime, "coordinator": coordinator
        }
        self.markov_runtime = markov_runtime
        self.rng = SpeculativeRngState.allocate(pool.backing.device, capacity=requests.capacity)
        self.generation = torch.zeros_like(requests.generation)
        self.offset = torch.zeros_like(requests.generation)
        draft = weights.config.draft
        self.logit_offset = 0
        self.block_size = max(1, draft_tokens)
        tokens = requests.capacity * self.block_size
        self.draft_scratch = forwards.scratch.allocate(
            tokens,
            tokens,
            weights.shard.world_size,
            pool.backing.device,
            experts=draft.experts,
            top_k=draft.top_k,
        )


class PreparedContext:
    """Project accepted target feedback into each draft stage's sliding window."""

    def __init__(
        self, weights, auxiliary, metadata, pool, *, session, name, workspace, cache_format="deepseek_v41"
    ):
        self.metadata, self.pool = metadata, pool
        tokens, device = auxiliary.shape[0], auxiliary.device
        draft = weights.draft
        self.projected = torch.zeros(
            (tokens, weights.config.text["hidden_size"]), dtype=torch.bfloat16, device=device
        )
        self.main = torch.zeros_like(self.projected)
        self.kv = torch.zeros((tokens, 512), dtype=torch.bfloat16, device=device)
        self.normalized = torch.zeros_like(self.kv)
        self.slots = torch.full((tokens,), -1, dtype=torch.int64, device=device)
        self.project = QuantizedProjection(
            auxiliary,
            draft.main_projection,
            self.projected,
            session=session,
            name=name + ".main",
            workspace=workspace,
            activation_mode="auto",
        )
        self.norm = PreparedRMSNorm(
            self.projected,
            draft.main_norm,
            self.main,
            eps=weights.config.text["rms_norm_eps"],
            session=session,
            name=name + ".main_norm",
        )
        self.rotary = Rotary(weights.config, device)
        self.stages = []
        for index, layer in enumerate(draft.layers):
            projection = QuantizedProjection(
                self.main,
                layer.attention["main_kv"],
                self.kv,
                session=session,
                name=f"{name}.{index}.kv",
                workspace=workspace,
                activation_mode="auto",
            )
            norm = PreparedRMSNorm(
                self.kv,
                layer.attention["kv_norm"],
                self.normalized,
                eps=weights.config.text["rms_norm_eps"],
                session=session,
                name=f"{name}.{index}.norm",
            )
            writer = PreparedCompressedCacheWriter(
                self.normalized,
                pool.window(index, draft=True),
                self.slots,
                page_size=128,
                kind="swa",
                session=session,
                name=f"{name}.{index}.write",
                cache_format=cache_format,
            )
            self.stages.append((projection, norm, writer))

    def __call__(self, auxiliary, counts):
        m = self.metadata
        offset = m.positions - m.starts[m.requests]
        count = counts[m.requests]
        live = (m.positions >= 0) & (offset < count) & (offset >= (count - 128).clamp_min(0))
        self.slots.copy_(torch.where(live, m.state_rows[m.requests] * 128 + m.positions % 128, -1))
        self.norm(self.project(auxiliary))
        for projection, norm, writer in self.stages:
            norm(projection(self.main))
            self.rotary.apply(self.normalized, m.positions.clamp_min(0), compressed=False)
            writer(self.normalized)


class PreparedMarkov:
    """Preserve FP32 conditional logits across the bitwise TP gather."""

    def __init__(self, weights, tokens, *, runtime, coordinator, session, name):
        rows, device = tokens.numel(), tokens.device
        self.embedding = torch.zeros(
            (rows, weights.config.draft.markov_rank), dtype=torch.bfloat16, device=device
        )
        self.local = torch.zeros(
            (rows, 129280 // weights.shard.world_size), dtype=torch.float32, device=device
        )
        self.logits = torch.zeros((rows, 129280), dtype=torch.float32, device=device)
        self.embed = PreparedEmbedding(
            weights.draft.markov_embedding, tokens, self.embedding, session=session, name=name + ".embed"
        )
        self.projection = UnquantizedProjection(
            self.embedding, weights.draft.markov_head, self.local, session=session, name=name + ".head"
        )
        self.gather = PreparedAllGather(
            self.local.view(torch.bfloat16),
            self.logits.view(torch.bfloat16),
            runtime=runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".gather",
        )
        self.confidence_weight = weights.draft.confidence

    def __call__(self):
        self.projection(self.embed())
        self.gather(self.local.view(torch.bfloat16))
        return self.logits

    def confidence(self, hidden):
        source = torch.cat((hidden, self.embedding), dim=-1).float()
        return (source * self.confidence_weight.float()).sum(-1).sigmoid()
