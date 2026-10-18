"""Block drafts conditioned on accepted target states, with sequential Markov logits."""

import torch

from ignition.components.embedding import PreparedEmbedding
from ignition.components.vocabulary import PreparedVocabularyProjection

from .attention import Rotary, normalize
from .decoder import PreparedLayer


class PreparedDraft:
    def __init__(self, weights, scratch, metadata, pool, *, runtime, vocabulary_runtime,
                 coordinator, session, name, workspace, mla_compute_mode="auto"):
        self.weights, self.scratch, self.metadata = weights, scratch, metadata
        self.embedding = PreparedEmbedding(weights.embedding, metadata.token_ids, scratch.embedding,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        rotary = Rotary(weights.config, scratch.embedding.device)
        self.layers = tuple(PreparedLayer(i, layer, None, scratch, metadata, pool, rotary,
            shard=weights.shard, mode="draft", runtime=runtime, coordinator=coordinator,
            session=session, name=f"{name}.layer{i}", workspace=workspace, carry=i > 0,
            defer_post=i + 1 < len(weights.draft.layers), mla_compute_mode=mla_compute_mode)
            for i, layer in enumerate(weights.draft.layers))
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
