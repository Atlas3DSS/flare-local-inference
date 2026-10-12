"""Block drafts conditioned on accepted target states, with sequential Markov logits."""

import torch

from ignition.components.collectives import PreparedAllGather
from ignition.components.compressed_mla import PreparedCompressedCacheWriter
from ignition.components.embedding import PreparedEmbedding
from ignition.components.linear import QuantizedProjection, UnquantizedProjection
from ignition.components.rms_norm import PreparedRMSNorm
from ignition.components.vocabulary import PreparedVocabularyProjection

from .attention import Rotary, normalize
from .decoder import PreparedLayer


class PreparedContext:
    """Project accepted target feedback into each draft stage's sliding window."""

    def __init__(self, weights, auxiliary, metadata, pool, *, session, name, workspace):
        self.metadata, self.pool = metadata, pool
        tokens, device = auxiliary.shape[0], auxiliary.device
        draft = weights.draft
        self.projected = torch.zeros((tokens, 5120), dtype=torch.bfloat16, device=device)
        self.main = torch.zeros_like(self.projected)
        self.kv = torch.zeros((tokens, 512), dtype=torch.bfloat16, device=device)
        self.normalized = torch.zeros_like(self.kv)
        self.slots = torch.full((tokens,), -1, dtype=torch.int64, device=device)
        self.project = QuantizedProjection(auxiliary, draft.main_projection, self.projected,
            session=session, name=name + ".main", workspace=workspace, activation_mode="auto")
        self.norm = PreparedRMSNorm(self.projected, draft.main_norm, self.main,
            eps=weights.config.text["rms_norm_eps"], session=session, name=name + ".main_norm")
        self.rotary = Rotary(weights.config, device)
        self.stages = []
        for index, layer in enumerate(draft.layers):
            projection = QuantizedProjection(self.main, layer.attention["main_kv"], self.kv,
                session=session, name=f"{name}.{index}.kv", workspace=workspace, activation_mode="auto")
            norm = PreparedRMSNorm(self.kv, layer.attention["kv_norm"], self.normalized,
                eps=weights.config.text["rms_norm_eps"], session=session, name=f"{name}.{index}.norm")
            writer = PreparedCompressedCacheWriter(self.normalized, pool.window(index, draft=True), self.slots,
                page_size=128, kind="swa", session=session, name=f"{name}.{index}.write")
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


class PreparedDraft:
    def __init__(self, weights, scratch, metadata, pool, *, runtime, vocabulary_runtime,
                 coordinator, session, name, workspace):
        self.weights, self.scratch, self.metadata = weights, scratch, metadata
        self.embedding = PreparedEmbedding(weights.embedding, metadata.token_ids, scratch.embedding,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        rotary = Rotary(weights.config, scratch.embedding.device)
        self.layers = tuple(PreparedLayer(i, layer, None, scratch, metadata, pool, rotary,
            shard=weights.shard, mode="draft", runtime=runtime, coordinator=coordinator,
            session=session, name=f"{name}.layer{i}", workspace=workspace, carry=i > 0,
            defer_post=i + 1 < len(weights.draft.layers)) for i, layer in enumerate(weights.draft.layers))
        self.head_hidden = torch.zeros_like(scratch.hidden)
        self.head = PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head")

    def __call__(self):
        s = self.scratch
        s.residual.copy_(self.embedding()[:, None, :])
        s.pre_mix.zero_()
        s.pre_mix[:, 0].fill_(1)
        for layer in self.layers:
            layer()
        self.head_hidden.copy_((s.residual[self.metadata.sample_indices].float()
                               * s.pre_mix[self.metadata.sample_indices, :, None]).sum(1))
        normalize(self.head_hidden, self.weights.draft.norm, s.hidden, self.weights.config.text["rms_norm_eps"])
        self.head(s.hidden)


class PreparedMarkov:
    """Preserve FP32 conditional logits across the bitwise TP gather."""

    def __init__(self, weights, tokens, *, runtime, coordinator, session, name):
        rows, device = tokens.numel(), tokens.device
        self.embedding = torch.zeros((rows, weights.config.draft.markov_rank), dtype=torch.bfloat16, device=device)
        self.local = torch.zeros((rows, 129280 // weights.shard.world_size), dtype=torch.float32, device=device)
        self.logits = torch.zeros((rows, 129280), dtype=torch.float32, device=device)
        self.embed = PreparedEmbedding(weights.draft.markov_embedding, tokens, self.embedding,
            session=session, name=name + ".embed")
        self.projection = UnquantizedProjection(self.embedding, weights.draft.markov_head, self.local,
            session=session, name=name + ".head")
        self.gather = PreparedAllGather(self.local.view(torch.bfloat16), self.logits.view(torch.bfloat16),
            runtime=runtime, coordinator=coordinator, session=session, name=name + ".gather")
        self.confidence_weight = weights.draft.confidence

    def __call__(self):
        self.projection(self.embed())
        self.gather(self.local.view(torch.bfloat16))
        return self.logits

    def confidence(self, hidden):
        source = torch.cat((hidden, self.embedding), dim=-1).float()
        return (source * self.confidence_weight.float()).sum(-1).sigmoid()
