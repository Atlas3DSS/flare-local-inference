"""Canonical Qwen dense layers with prepared projections and persistent GPU caches."""

from copy import copy
from dataclasses import dataclass

import torch

from ignition.components.embedding import PreparedEmbedding
from ignition.components.gated_delta import GDNScratch, PreparedGDN
from ignition.components.linear import QuantizedProjection
from ignition.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter
from ignition.components.recurrent import DeltaMetadata
from ignition.components.sm70_precision import apply_rotary, rms_norm, round_bf16
from ignition.components.tensor_fp8 import TensorFP8Projection
from ignition.components.vocabulary import PreparedVocabularyProjection

from .config import PAGE_TOKENS


@dataclass(frozen=True)
class Scratch:
    activations: torch.Tensor
    hidden: torch.Tensor
    logits: torch.Tensor
    gdn: GDNScratch

    @classmethod
    def allocate(cls, tokens, rows, config, device):
        from ignition.components.sm70_projection import is_sm70
        dtype = torch.float32 if is_sm70(device) else torch.bfloat16
        return cls(torch.zeros((tokens, config.hidden), dtype=dtype, device=device),
                   torch.zeros((rows, config.hidden), dtype=dtype, device=device),
                   torch.zeros((rows, config.vocabulary), dtype=dtype, device=device),
                   GDNScratch.allocate(tokens, device, hidden_size=config.hidden, dtype=dtype))

    def rows(self, tokens, rows):
        return Scratch(self.activations[:tokens], self.hidden[:rows], self.logits[:rows], self.gdn.rows(tokens))


class Projection(torch.nn.Module):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def forward(self, source):
        return self.operation(source)


def module_copy(module):
    result = copy(module)
    result._modules = module._modules.copy()
    return result


class RecurrentAttention(torch.nn.Module):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def forward(self, hidden_states, **kwargs):
        return self.operation(hidden_states)


class SM70DecoderLayer(torch.nn.Module):
    """Canonical Qwen layer arithmetic with BF16 values stored as FP32."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states, position_embeddings):
        layer = self.layer
        residual = hidden_states
        norm = layer.input_layernorm
        hidden_states = rms_norm(hidden_states, norm.weight, norm.eps)
        if layer.block_type == "linear_attention":
            hidden_states = layer.linear_attn(hidden_states=hidden_states)
        else:
            hidden_states, _ = layer.self_attn(hidden_states=hidden_states,
                                              position_embeddings=position_embeddings)
        hidden_states = round_bf16(residual + round_bf16(hidden_states))
        residual = hidden_states
        norm = layer.post_attention_layernorm
        hidden_states = rms_norm(hidden_states, norm.weight, norm.eps)
        gate = round_bf16(layer.mlp.gate_proj(hidden_states))
        up = round_bf16(layer.mlp.up_proj(hidden_states))
        activated = round_bf16(layer.mlp.act_fn(gate))
        product = round_bf16(activated * up)
        hidden_states = round_bf16(layer.mlp.down_proj(product))
        return round_bf16(residual + hidden_states)


class FullAttention(torch.nn.Module):
    def __init__(self, module, weights, source, metadata, pool, layer, *, mode, session, workspace, name, buffers):
        super().__init__()
        self.canonical, self.metadata = module, metadata
        config = weights.config
        self.heads, self.kv_heads = config.text["num_attention_heads"], config.text["num_key_value_heads"]
        self.dim = config.text["head_dim"]
        tokens, device = source.shape[0], source.device

        def empty(field, width):
            if field not in buffers:
                buffers[field] = torch.zeros((tokens, width), dtype=source.dtype, device=device)
            return buffers[field]

        self.qg = empty("qg", 2 * self.heads * self.dim)
        self.k, self.v = (empty(field, self.kv_heads * self.dim) for field in ("k", "v"))
        self.q = empty("q", self.heads * self.dim).view(tokens, self.heads, self.dim)
        self.key = empty("key", self.kv_heads * self.dim).view(tokens, self.kv_heads, self.dim)
        self.attended, self.output = empty("attended", self.heads * self.dim), empty("output", config.hidden)
        prefix = f"layers.{layer}.self_attn."
        self.projections = {
            field: TensorFP8Projection(inputs, weights.projections[prefix + field], output,
                                       session=session, workspace=workspace, name=name + "." + field)
            for field, inputs, output in (("q_proj", source, self.qg), ("k_proj", source, self.k),
                                         ("v_proj", source, self.v), ("o_proj", self.attended, self.output))
        }
        self.slots = torch.full((tokens,), -1, dtype=torch.int64, device=device)
        self.lengths = torch.zeros_like(metadata.lengths)
        key_cache, value_cache = pool.attention(layer, config)
        from ignition.components.sm70_projection import is_sm70
        self.sm70 = is_sm70(device)
        writer, attention = PreparedPagedKVWriter, PreparedPagedAttention
        options = dict(mode=mode, workspace=workspace)
        if self.sm70:
            from ignition.components.sm70_attention import PreparedWriter, PreparedAttention
            writer, attention, options = PreparedWriter, PreparedAttention, {}
        self.writer = writer(self.key, self.v.view(tokens, self.kv_heads, self.dim),
            key_cache, value_cache, self.slots, session=session, name=name + ".write")
        self.attention = attention(self.q, key_cache, value_cache, metadata.page_table,
            self.lengths, metadata.query_start_loc, self.attended.view(tokens, self.heads, self.dim),
            session=session, name=name + ".attention", **options)

    def forward(self, hidden_states, position_embeddings, **kwargs):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

        for field in ("q_proj", "k_proj", "v_proj"):
            self.projections[field](hidden_states)
        if self.sm70:
            for projected in (self.qg, self.k, self.v):
                projected.copy_(round_bf16(projected))
        query, gate = self.qg.view(-1, self.heads, 2 * self.dim).chunk(2, dim=-1)
        key = self.k.view(-1, self.kv_heads, self.dim)
        if self.sm70:
            q_norm, k_norm = self.canonical.q_norm, self.canonical.k_norm
            q, k = apply_rotary(rms_norm(query, q_norm.weight, q_norm.eps),
                rms_norm(key, k_norm.weight, k_norm.eps), *position_embeddings)
        else:
            q, k = apply_rotary_pos_emb(self.canonical.q_norm(query),
                self.canonical.k_norm(key), *position_embeddings)
        self.q.copy_(q)
        self.key.copy_(k)
        m = self.metadata
        page = m.token_page_table.gather(1, (m.positions.clamp(min=0) // PAGE_TOKENS)[:, None]).squeeze(1)
        self.slots.copy_(torch.where(m.positions >= 0, page.long() * PAGE_TOKENS + m.positions % PAGE_TOKENS, -1))
        self.lengths.copy_(torch.where(m.lengths > 0, m.starts + m.lengths, 0))
        self.writer()
        self.attention()
        if self.sm70:
            rounded_gate = round_bf16(gate.reshape_as(self.attended).sigmoid())
            self.attended.copy_(round_bf16(round_bf16(self.attended) * rounded_gate))
        else:
            self.attended.mul_(gate.reshape_as(self.attended).sigmoid())
        output = self.projections["o_proj"](self.attended)
        if self.sm70:
            output.copy_(round_bf16(output))
        return output, None


class PreparedDecoder:
    def __init__(self, weights, scratch, metadata, pool, *, mode, vocabulary_runtime,
                 coordinator, session, name, workspace, delta=None, feedback_output=None):
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        from ignition.components.sm70_projection import is_sm70
        self.sm70 = is_sm70(scratch.activations.device)
        self.embedding = PreparedEmbedding(weights.tree.embed_tokens.weight, metadata.token_ids,
            scratch.activations, num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        self.delta = delta or DeltaMetadata.allocate(metadata.starts.numel(), scratch.activations.device)
        self.external_delta = delta is not None
        self.feedback_output = feedback_output
        config, source = weights.config, scratch.activations
        attention_buffers = {}
        mlp_buffers = {field: torch.zeros((source.shape[0], width), dtype=source.dtype, device=source.device)
                       for field, width in (("gate_proj", config.text["intermediate_size"]),
                                            ("up_proj", config.text["intermediate_size"]),
                                            ("down_proj", config.hidden))}
        mlp_input = torch.zeros_like(mlp_buffers["gate_proj"])
        self.layers = []
        for index, original in enumerate(weights.tree.layers):
            layer = module_copy(original)
            prefix = f"layers.{index}."
            if original.block_type == "linear_attention":
                projections = {field: weights.projections[prefix + "linear_attn." + field]
                               for field in ("in_proj_qkv", "in_proj_z", "in_proj_ba", "out_proj")}
                layer.linear_attn = RecurrentAttention(PreparedGDN(original.linear_attn, projections, source,
                    scratch.gdn, pool.recurrent(index), self.delta, mode=mode, session=session,
                    name=name + "." + prefix + "gdn", workspace=workspace, projection_workspace=workspace))
            else:
                layer.self_attn = FullAttention(original.self_attn, weights, source, metadata, pool, index,
                    mode=mode, session=session, workspace=workspace, name=name + "." + prefix + "attention",
                    buffers=attention_buffers)
            layer.mlp = module_copy(original.mlp)
            for field in ("gate_proj", "up_proj", "down_proj"):
                weight = weights.projections[prefix + "mlp." + field]
                inputs = source if field != "down_proj" else mlp_input
                output = mlp_buffers[field]
                setattr(layer.mlp, field, Projection(QuantizedProjection(inputs, weight, output,
                    activation_mode="auto", session=session, workspace=workspace, name=name + "." + prefix + field)))
            self.layers.append(SM70DecoderLayer(layer) if self.sm70 else layer)
        rotary = config.text["rope_parameters"]
        dim = int(config.text["head_dim"] * rotary["partial_rotary_factor"])
        self.inverse_frequency = 1.0 / (rotary["rope_theta"] ** (
            torch.arange(0, dim, 2, device=source.device, dtype=torch.float32) / dim))
        self.head = PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head",
            activation_mode="auto", workspace=workspace)

    def __call__(self):
        m, d = self.metadata, self.delta
        active = m.lengths > 0
        if not self.external_delta:
            d.query_start_loc.copy_(m.query_start_loc)
            d.convolution_slots.copy_(torch.where(active, m.state_rows, -1))
            d.fresh.copy_(m.starts == 0)
            d.state_indices[:, 0].copy_(m.state_rows)
            d.initial_state_indices.copy_(torch.where(m.starts > 0, m.state_rows, 0))
            d.final_state_indices.copy_(m.state_rows)
            d.num_tokens.copy_(m.live_counts[:1])
            d.num_seqs.copy_(m.live_counts[1:])
        frequency = m.positions.float()[:, None] * self.inverse_frequency[None, :]
        embedding = torch.cat((frequency, frequency), dim=-1)
        dtype = self.scratch.activations.dtype
        position_embeddings = ((round_bf16(embedding.cos()), round_bf16(embedding.sin())) if self.sm70
                               else (embedding.cos().to(dtype), embedding.sin().to(dtype)))
        hidden = self.embedding()
        for layer in self.layers:
            hidden = layer(hidden, position_embeddings=position_embeddings)
        selected = hidden[m.sample_indices]
        norm = self.weights.tree.norm
        if self.feedback_output is not None:
            self.feedback_output.copy_(rms_norm(hidden, norm.weight, norm.eps) if self.sm70 else norm(hidden))
        self.scratch.hidden.copy_(rms_norm(selected, norm.weight, norm.eps) if self.sm70 else norm(selected))
        self.head(self.scratch.hidden)
        if self.sm70:
            self.scratch.logits.copy_(round_bf16(self.scratch.logits))
