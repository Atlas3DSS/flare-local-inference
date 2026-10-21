"""Prepared dense NextN feedback and one BF16 decoder using existing SM70 ops."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ignition.components.embedding import PreparedEmbedding
from ignition.components.linear import UnquantizedProjection
from ignition.components.sm70_attention import PreparedAttention, PreparedWriter
from ignition.components.sm70_precision import apply_rotary, rms_norm, round_bf16
from ignition.components.vocabulary import PreparedVocabularyProjection

from .config import PAGE_TOKENS


@dataclass(frozen=True)
class MTPScratch:
    embedding_ids: torch.Tensor
    embedding: torch.Tensor
    feedback: torch.Tensor
    residual: torch.Tensor
    hidden: torch.Tensor
    selected_hidden: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, tokens, output_rows, config, device):
        if type(tokens) is not int or type(output_rows) is not int or not 0 < output_rows <= tokens:
            raise ValueError("dense MTP requires positive output rows no greater than token capacity")

        def zeros(rows, width):
            return torch.zeros((rows, width), dtype=torch.float32, device=device)

        hidden = config.hidden
        return cls(torch.zeros(tokens, dtype=torch.int64, device=device),
                   zeros(tokens, hidden), zeros(tokens, 2 * hidden), zeros(tokens, hidden),
                   zeros(tokens, hidden), zeros(output_rows, hidden), zeros(output_rows, config.vocabulary))

    def rows(self, tokens, output_rows):
        if (type(tokens) is not int or type(output_rows) is not int
                or not 0 < output_rows <= min(tokens, self.logits.shape[0])
                or tokens > self.residual.shape[0]):
            raise ValueError("dense MTP view exceeds persistent token/output capacity")
        return type(self)(self.embedding_ids[:tokens], self.embedding[:tokens], self.feedback[:tokens],
                          self.residual[:tokens], self.hidden[:tokens], self.selected_hidden[:output_rows],
                          self.logits[:output_rows])


@dataclass(frozen=True)
class MTPOutput:
    residual: torch.Tensor
    hidden: torch.Tensor
    logits: torch.Tensor


torch.utils._pytree.register_dataclass(MTPOutput)


class PreparedMTPFeedback:
    """Normalize [next-token embedding, normalized target hidden], then project."""

    def __init__(self, weights, scratch, target_hidden, workspace, *, session, name):
        self.weights, self.scratch, self.target_hidden = weights, scratch, target_hidden
        self.projection = UnquantizedProjection(scratch.feedback, weights.tensors["fc.weight"],
            scratch.residual, workspace=workspace, session=session, name=name + ".fc")

    def __call__(self):
        s, w = self.scratch, self.weights.tensors
        epsilon, hidden = self.weights.config.text["rms_norm_eps"], self.weights.config.hidden
        s.feedback[:, :hidden].copy_(rms_norm(s.embedding, w["pre_fc_norm_embedding.weight"], epsilon))
        s.feedback[:, hidden:].copy_(rms_norm(self.target_hidden, w["pre_fc_norm_hidden.weight"], epsilon))
        s.residual.copy_(round_bf16(self.projection(s.feedback)))
        return s.residual


class MTPAttention:
    def __init__(self, weights, source, metadata, key_cache, value_cache, workspace, *,
                 session, name, cache_only):
        self.metadata, self.cache_only = metadata, cache_only
        self.weights = weights
        text, tokens = weights.config.text, source.shape[0]
        self.heads, self.kv_heads, self.dim = (text[field] for field in
            ("num_attention_heads", "num_key_value_heads", "head_dim"))
        expected = (PAGE_TOKENS, self.kv_heads, self.dim)
        if (key_cache.ndim != 4 or tuple(key_cache.shape[1:]) != expected
                or value_cache.shape != key_cache.shape
                or key_cache.dtype != torch.uint16 or value_cache.dtype != torch.uint16):
            raise ValueError("dense MTP caches require UInt16 BF16-bit [pages,128,kv_heads,head_dim]")

        def zeros(width):
            return torch.zeros((tokens, width), dtype=source.dtype, device=source.device)

        self.key, self.value = (zeros(self.kv_heads * self.dim).view(tokens, self.kv_heads, self.dim)
                               for _ in range(2))
        self.k = zeros(self.kv_heads * self.dim)
        self.slots = torch.full((tokens,), -1, dtype=torch.int64, device=source.device)
        prefix = "layers.0.self_attn."
        self.projections = {
            field: UnquantizedProjection(source, weights.tensors[prefix + field + ".weight"], output,
                session=session, name=name + "." + field, workspace=workspace)
            for field, output in (("k_proj", self.k), ("v_proj", self.value.view(tokens, -1)))
        }
        self.writer = PreparedWriter(self.key, self.value, key_cache, value_cache, self.slots,
                                     session=session, name=name + ".write")
        if cache_only:
            return
        self.qg = zeros(2 * self.heads * self.dim)
        self.query = zeros(self.heads * self.dim).view(tokens, self.heads, self.dim)
        self.attended, self.output = zeros(self.heads * self.dim), zeros(weights.config.hidden)
        self.lengths = torch.zeros_like(metadata.lengths)
        for field, inputs, output in (("q_proj", source, self.qg),
                                      ("o_proj", self.attended, self.output)):
            self.projections[field] = UnquantizedProjection(inputs, weights.tensors[prefix + field + ".weight"],
                output, session=session, name=name + "." + field, workspace=workspace)
        self.attention = PreparedAttention(self.query, key_cache, value_cache, metadata.page_table,
            self.lengths, metadata.query_start_loc, self.attended.view_as(self.query), session=session,
            name=name + ".attention")

    def __call__(self, source, position_embeddings):
        w, m = self.weights.tensors, self.metadata
        epsilon = self.weights.config.text["rms_norm_eps"]
        self.k.copy_(round_bf16(self.projections["k_proj"](source)))
        self.value.copy_(round_bf16(self.projections["v_proj"](source)).view_as(self.value))
        key = rms_norm(self.k.view_as(self.key), w["layers.0.self_attn.k_norm.weight"], epsilon)
        if self.cache_only:
            _, key = apply_rotary(key, key, *position_embeddings)
        else:
            self.qg.copy_(round_bf16(self.projections["q_proj"](source)))
            query, gate = self.qg.view(-1, self.heads, 2 * self.dim).chunk(2, dim=-1)
            query = rms_norm(query, w["layers.0.self_attn.q_norm.weight"], epsilon)
            query, key = apply_rotary(query, key, *position_embeddings)
            self.query.copy_(query)
        self.key.copy_(key)
        page = m.token_page_table.gather(1, (m.positions.clamp(min=0) // PAGE_TOKENS)[:, None]).squeeze(1)
        self.slots.copy_(torch.where((m.positions >= 0) & (page >= 0),
            page.long() * PAGE_TOKENS + m.positions % PAGE_TOKENS, -1))
        self.writer()
        if self.cache_only:
            return None
        self.lengths.copy_(torch.where(m.lengths > 0, m.starts + m.lengths, 0))
        self.attention()
        self.attended.copy_(round_bf16(round_bf16(self.attended)
            * round_bf16(gate.reshape_as(self.attended).sigmoid())))
        self.output.copy_(round_bf16(self.projections["o_proj"](self.attended)))
        return self.output


class PreparedMTP:
    """One learned dense draft step or accepted-history cache absorption.

    ``target_hidden[i]`` is AFTER the target final norm (or this drafter's
    ``output.hidden`` during recursive drafting), paired with token ``x[i+1]``
    and position ``i`` in ordinary paged metadata. This matches the checkpoint's
    NextN path used by llama.cpp. ``output.residual`` is before the MTP final norm
    and must not be used for recursive feedback. The caller owns alignment,
    page allocation and rollback. Cache-only absorption writes K/V without
    preparing or executing query/output projections, attention, MLP or head.
    """

    def __init__(self, weights, scratch, token_ids, target_hidden, metadata, key_cache, value_cache,
                 workspace, *, output_indices, session, name, cache_only=False, mode="decode",
                 vocabulary_runtime=None, coordinator=None):
        tokens, hidden = scratch.residual.shape
        device = scratch.residual.device
        if mode not in ("prefill", "decode"):
            raise ValueError("dense MTP mode must be prefill or decode")
        if (hidden != weights.config.hidden or target_hidden.shape != (tokens, hidden)
                or target_hidden.dtype != torch.float32 or target_hidden.device != device
                or not target_hidden.is_contiguous()):
            raise ValueError("dense MTP feedback requires contiguous FP32 [tokens,hidden] BF16 values")
        if (token_ids.shape != (tokens,) or token_ids.dtype != torch.int64
                or token_ids.device != device or not token_ids.is_contiguous()):
            raise ValueError("dense MTP token IDs require contiguous device int64 [tokens]")
        if (output_indices.shape != (scratch.logits.shape[0],) or output_indices.dtype != torch.int64
                or output_indices.device != device or not output_indices.is_contiguous()):
            raise ValueError("dense MTP output indices require contiguous device int64 [output_rows]")
        if metadata.positions.shape != (tokens,) or metadata.positions.device != device:
            raise ValueError("dense MTP metadata must match feedback capacity and device")
        if any(torch._C._overlaps(target_hidden, output) for output in
               (scratch.embedding, scratch.feedback, scratch.residual, scratch.hidden)):
            raise ValueError("dense MTP feedback input must not alias intermediate/output storage")
        self.weights, self.scratch, self.metadata, self.cache_only = weights, scratch, metadata, cache_only
        self.token_ids, self.output_indices = token_ids, output_indices
        self.output = MTPOutput(scratch.residual, scratch.hidden, scratch.logits)
        self.residual, self.hidden, self.logits = scratch.residual, scratch.hidden, scratch.logits
        self.embedding = PreparedEmbedding(weights.embedding, scratch.embedding_ids, scratch.embedding,
            session=session, name=name + ".embedding")
        self.feedback = PreparedMTPFeedback(weights, scratch, target_hidden, workspace,
                                            session=session, name=name + ".feedback")
        self.attention = MTPAttention(weights, scratch.residual, metadata, key_cache, value_cache, workspace,
                                      session=session, name=name + ".attention", cache_only=cache_only)
        rotary = weights.config.text["rope_parameters"]
        dim = int(weights.config.text["head_dim"] * rotary["partial_rotary_factor"])
        self.inverse_frequency = 1.0 / (rotary["rope_theta"] ** (
            torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        if cache_only:
            return
        self.mlp = {}
        intermediate = weights.config.text["intermediate_size"]
        self.product = torch.zeros((tokens, intermediate), dtype=torch.float32, device=device)
        for field, source, width in (("gate_proj", scratch.residual, intermediate),
                                      ("up_proj", scratch.residual, intermediate),
                                      ("down_proj", self.product, hidden)):
            output = torch.zeros((tokens, width), dtype=torch.float32, device=device)
            self.mlp[field] = UnquantizedProjection(source, weights.tensors["layers.0.mlp." + field + ".weight"],
                output, session=session, name=name + ".mlp." + field, workspace=workspace)
        self.head = PreparedVocabularyProjection(scratch.selected_hidden, weights.lm_head, scratch.logits,
            scratch.logits, runtime=vocabulary_runtime, coordinator=coordinator, session=session,
            name=name + ".head", activation_mode="auto", workspace=workspace)

    def __call__(self):
        s, w = self.scratch, self.weights.tensors
        active = self.metadata.positions >= 0
        s.embedding_ids.copy_(torch.where(active, self.token_ids, 0))
        self.embedding()
        self.feedback()
        s.residual.masked_fill_(~active[:, None], 0)
        frequency = self.metadata.positions.float()[:, None] * self.inverse_frequency[None, :]
        angles = torch.cat((frequency, frequency), dim=-1)
        position_embeddings = (round_bf16(angles.cos()), round_bf16(angles.sin()))
        epsilon = self.weights.config.text["rms_norm_eps"]
        source = rms_norm(s.residual, w["layers.0.input_layernorm.weight"], epsilon)
        attention = self.attention(source, position_embeddings)
        if self.cache_only:
            return None
        s.residual.copy_(round_bf16(s.residual + attention))
        source = rms_norm(s.residual, w["layers.0.post_attention_layernorm.weight"], epsilon)
        gate, up = (round_bf16(self.mlp[field](source)) for field in ("gate_proj", "up_proj"))
        self.product.copy_(round_bf16(round_bf16(F.silu(gate)) * up))
        s.residual.copy_(round_bf16(s.residual + round_bf16(self.mlp["down_proj"](self.product))))
        s.residual.masked_fill_(~active[:, None], 0)
        s.hidden.copy_(rms_norm(s.residual, w["norm.weight"], epsilon))
        torch.index_select(s.hidden, 0, self.output_indices, out=s.selected_hidden)
        self.head(s.selected_hidden)
        s.logits.copy_(round_bf16(s.logits))
        return self.output
