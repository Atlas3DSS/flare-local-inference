"""Qwen3.5 MTP feedback, causal proposals, and target-derived draft KV."""

from functools import partial

import torch

from ignition.components.causal_draft import PreparedCausalDraftRound
from ignition.components.embedding import PreparedEmbedding
from ignition.components.linear import UnquantizedProjection
from ignition.components.paged_batch import Metadata
from ignition.components.state import gather_rows
from ignition.components.vocabulary import PreparedVocabularyProjection

from .decoder import FullAttention, PreparedDecoder, Projection, module_copy


class PreparedPredictor:
    def __init__(self, weights, metadata, pool, *, session, workspace, name, cache_only=False, buffers=None):
        self.weights, self.metadata, self.cache_only = weights, metadata, cache_only
        c, w, device = weights.config, weights.draft, metadata.token_ids.device
        tokens, rows = metadata.token_ids.numel(), metadata.sample_indices.numel()
        def zeros(shape):
            return (torch.zeros(shape, dtype=torch.bfloat16, device=device) if buffers is None
                    else buffers.zeros(shape, torch.bfloat16))
        self.previous, self.embedded, self.residual = (zeros((tokens, c.hidden)) for _ in range(3))
        self.joined = zeros((tokens, 2 * c.hidden))
        self.embedding = PreparedEmbedding(weights.tree.embed_tokens.weight, metadata.token_ids, self.embedded,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        self.fc = UnquantizedProjection(self.joined, w.fc, self.residual, session=session, name=name + ".fc")
        self.layer = module_copy(w.layer)
        attention_buffers = {field: zeros((tokens, width)) for field, width in (
            ("qg", 2 * c.text["num_attention_heads"] * c.text["head_dim"]),
            ("q", c.text["num_attention_heads"] * c.text["head_dim"]),
            ("attended", c.text["num_attention_heads"] * c.text["head_dim"]),
            ("k", c.text["num_key_value_heads"] * c.text["head_dim"]),
            ("v", c.text["num_key_value_heads"] * c.text["head_dim"]),
            ("key", c.text["num_key_value_heads"] * c.text["head_dim"]), ("output", c.hidden))}
        self.layer.self_attn = FullAttention(w.layer.self_attn, weights, self.residual, metadata, pool, "draft",
            mode="prefill" if cache_only else "decode", session=session, workspace=workspace,
            name=name + ".attention", buffers=attention_buffers, cache_only=cache_only)
        rotary = c.text["rope_parameters"]
        dim = int(c.text["head_dim"] * rotary["partial_rotary_factor"])
        self.inverse = 1.0 / (rotary["rope_theta"] ** (torch.arange(0, dim, 2, device=device).float() / dim))
        if not cache_only:
            self.layer.mlp = module_copy(w.layer.mlp)
            mlp_input = zeros((tokens, c.text["intermediate_size"]))
            for field in ("gate_proj", "up_proj", "down_proj"):
                parameter = getattr(w.layer.mlp, field).weight
                output = zeros((tokens, parameter.shape[0]))
                source = mlp_input if field == "down_proj" else self.residual
                setattr(self.layer.mlp, field, Projection(UnquantizedProjection(source, parameter, output,
                    session=session, name=name + "." + field)))
            self.hidden, self.selected = zeros((tokens, c.hidden)), zeros((rows, c.hidden))
            self.logits = zeros((rows, c.vocabulary))
            self.head = PreparedVocabularyProjection(self.selected, weights.head, self.logits, self.logits,
                runtime=None, coordinator=None, session=session, name=name + ".head", activation_mode="auto")

    def __call__(self):
        w, m = self.weights.draft, self.metadata
        self.joined.copy_(torch.cat((w.embedding_norm(self.embedding()), w.hidden_norm(self.previous)), dim=-1))
        residual = self.fc(self.joined)
        # Draft cache position t-1 combines target hidden t-1 with token t; RoPE uses token t.
        frequency = (m.positions + 1).float()[:, None] * self.inverse[None, :]
        angles = torch.cat((frequency, frequency), dim=-1)
        rotary = angles.cos().bfloat16(), angles.sin().bfloat16()
        if self.cache_only:
            self.layer.self_attn.write_cache(self.layer.input_layernorm(residual), rotary)
        else:
            self.hidden.copy_(w.norm(self.layer(residual, position_embeddings=rotary)))
            self.selected.copy_(self.hidden[m.sample_indices])
            self.head(self.selected)


class PreparedFeedback:
    """Absorb accepted (preceding target hidden, next token) pairs into shifted pages."""

    def __init__(self, weights, auxiliary, metadata, pool, *, inputs, tables, session, name, workspace, buffers=None):
        self.source, self.pool, self.inputs, self.tables = metadata, pool, inputs, tables
        tokens, rows = auxiliary.shape[0], metadata.starts.numel()
        self.metadata = Metadata.allocate(tokens, rows, tables.shape[1], auxiliary.device, buffers=buffers)
        self.predictor = PreparedPredictor(weights, self.metadata, pool, session=session, workspace=workspace,
                                           name=name, cache_only=True, buffers=buffers)
        self.previous = (torch.zeros((rows, weights.config.hidden), dtype=auxiliary.dtype, device=auxiliary.device)
                         if buffers is None else buffers.zeros((rows, weights.config.hidden), auxiliary.dtype))
        self.indices = torch.arange(tokens, device=auxiliary.device)

    def __call__(self, auxiliary, counts):
        source, m = self.source, self.metadata
        skip = ((source.starts == 0) & (counts > 0)).int()
        lengths = counts - skip
        ends = lengths.cumsum(0)
        begins = ends - lengths
        request = (self.indices[:, None] >= ends[None]).sum(1).clamp_max(lengths.numel() - 1)
        offset = self.indices - begins[request]
        index = (source.query_start_loc[request] + skip[request] + offset).clamp(0, auxiliary.shape[0] - 1).long()
        live = self.indices < ends[-1]
        m.starts.copy_((source.starts - 1).clamp_min(0))
        m.lengths.copy_(lengths)
        m.state_rows.copy_(torch.where(lengths > 0, source.state_rows, 0))
        m.query_start_loc[0].zero_()
        m.query_start_loc[1:].copy_(ends)
        m.live_counts[0:1].copy_(ends[-1:])
        m.live_counts[1:].copy_(source.live_counts[1:])
        m.requests.copy_(request)
        m.positions.copy_(torch.where(live, m.starts[request] + offset, -1))
        m.token_ids.copy_(torch.where(live, source.token_ids[index], 0))
        slots = self.inputs.row_slot_ids[:lengths.numel()].long().clamp(0, self.tables.shape[0] - 1)
        m.page_table.copy_(self.tables[slots])
        if lengths.numel() > 1:
            m.token_page_table.copy_(m.page_table[request])
        gather_rows(self.pool.last_hidden, source.state_rows, self.previous)
        previous = self.previous[request]
        self.predictor.previous.copy_(torch.where((index == source.query_start_loc[request])[:, None],
                                                  previous, auxiliary[(index - 1).clamp_min(0)]))
        self.predictor()

    def scratch_tensors(self):
        return self.metadata, self.previous, self.predictor.previous, self.predictor.embedded, self.predictor.joined


class PreparedProposals:
    def __init__(self, weights, pool, scratch, batch, depth, *, inputs, tables,
                 session, workspace, name, buffers=None, **collectives):
        self.pool, self.inputs, self.tables = pool, inputs, tables
        self.metadata = Metadata.allocate(batch, batch, tables.shape[1], pool.backing.device, buffers=buffers)
        self.predictor = PreparedPredictor(weights, self.metadata, pool, session=session, workspace=workspace,
                                           name=name, buffers=buffers)
        self.feedback = torch.zeros_like(self.predictor.previous) if buffers is None else buffers.zeros(
            self.predictor.previous.shape, self.predictor.previous.dtype)
        self.logits = self.predictor.logits

    def begin(self, states):
        gather_rows(self.pool.last_hidden, states, self.feedback)

    def step(self, index, tokens, before, states, active, proposing=None):
        m = self.metadata
        slots = self.inputs.row_slot_ids[:states.numel()].long().clamp(0, self.tables.shape[0] - 1)
        m.set(before - 1 + index, proposing.int(), states, self.tables[slots], tokens[:, None], self.inputs.live_rows)
        self.predictor.previous.copy_(self.feedback[m.requests])
        self.predictor()
        self.feedback.copy_(torch.where(proposing[:, None], self.predictor.selected, self.feedback))

    def finish(self, before, states, active):
        pass

    def scratch_tensors(self):
        return self.metadata, self.feedback, self.logits


class PreparedRound(PreparedCausalDraftRound):
    def __init__(self, owner, batch, width, name, *, buffers=None):
        def decoder(weights, scratch, metadata, pool, *, overlap, runtime, buffers=None, **kwargs):
            return PreparedDecoder(weights, scratch, metadata, pool, records=owner.records, buffers=buffers, **kwargs)

        def context(weights, metadata, pool, *, runtime, vocabulary_runtime, coordinator, **kwargs):
            return PreparedFeedback(weights, self.decoder.auxiliary, metadata, pool,
                inputs=owner.inputs, tables=owner.draft_tables, name=name + ".feedback", **kwargs)

        super().__init__(owner, batch, width, name, decoder_factory=decoder, context_factory=context,
            proposal_factory=partial(PreparedProposals, inputs=owner.inputs, tables=owner.draft_tables),
            draft_context_lag=1, buffers=buffers)
