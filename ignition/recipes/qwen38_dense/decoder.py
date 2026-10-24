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

# Completion error code of a prefill chunk the reviewed SM70 prefill attention refused (status word != 0):
# bit 0 non-finite K/V, bit 1 V range above 2^80 across pages, bit 2 inadmissible (M, context). Codes 1, 6, 32 and 33
# are taken by sampling, speculation, batch validation and resume.
PREFILL_ATTENTION_ERROR = 34


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

    def __init__(self, layer, norms=None, swiglu=False):
        super().__init__()
        self.layer = layer
        self.norms = norms
        # astra-21 (SM70 prefill graphs): the down projection reads the gate|up output and applies the BF16 SwiGLU in
        # its producer (input_transform="swiglu_bf16"), bit-identical to the pointwise path below.
        self.swiglu = swiglu

    def forward(self, hidden_states, position_embeddings):
        layer = self.layer
        residual = hidden_states
        norm = layer.input_layernorm
        hidden_states = (self.norms[0](hidden_states) if self.norms is not None
                         else rms_norm(hidden_states, norm.weight, norm.eps))
        if layer.block_type == "linear_attention":
            hidden_states = layer.linear_attn(hidden_states=hidden_states)
        else:
            hidden_states, _ = layer.self_attn(hidden_states=hidden_states,
                                              position_embeddings=position_embeddings)
        hidden_states = round_bf16(residual + round_bf16(hidden_states))
        residual = hidden_states
        norm = layer.post_attention_layernorm
        hidden_states = (self.norms[1](hidden_states) if self.norms is not None
                         else rms_norm(hidden_states, norm.weight, norm.eps))
        if self.swiglu:
            return round_bf16(residual + round_bf16(layer.mlp.down_proj(layer.mlp.gate_up_proj(hidden_states))))
        if hasattr(layer.mlp, 'gate_up_proj'):
            gate, up = layer.mlp.gate_up_proj(hidden_states).chunk(2, dim=-1)
            gate, up = round_bf16(gate), round_bf16(up)
        else:
            gate = round_bf16(layer.mlp.gate_proj(hidden_states))
            up = round_bf16(layer.mlp.up_proj(hidden_states))
        activated = round_bf16(layer.mlp.act_fn(gate))
        product = round_bf16(activated * up)
        hidden_states = round_bf16(layer.mlp.down_proj(product))
        return round_bf16(residual + hidden_states)


class FullAttention(torch.nn.Module):
    def __init__(self, module, weights, source, metadata, pool, layer, *, mode, session, workspace, name, buffers,
                 row_invariant=False, status_sink=None, decode_agents=None, rows_per_agent=None):
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

        self.fused_qkv = f"layers.{layer}.self_attn.qkv_proj" in weights.projections
        if self.fused_qkv:
            widths = (2 * self.heads * self.dim, self.kv_heads * self.dim, self.kv_heads * self.dim)
            self.qkv = empty('qkv', sum(widths))
            self.qkv_members = self.qkv.split(widths, dim=-1)
            # Head-wise RMSNorm flattens token/head rows; keep its existing
            # contiguous geometry while the rounding copies read fused strides.
            self.qg = empty('qg', widths[0])
            self.k, self.v = (empty(field, width) for field, width in zip(('k', 'v'), widths[1:], strict=True))
        else:
            self.qg = empty("qg", 2 * self.heads * self.dim)
            self.k, self.v = (empty(field, self.kv_heads * self.dim) for field in ("k", "v"))
        self.q = empty("q", self.heads * self.dim).view(tokens, self.heads, self.dim)
        self.key = empty("key", self.kv_heads * self.dim).view(tokens, self.kv_heads, self.dim)
        self.attended, self.output = empty("attended", self.heads * self.dim), empty("output", config.hidden)
        prefix = f"layers.{layer}.self_attn."
        from ignition.components.sm70_projection import is_sm70
        # SM70 prefill graphs: projections read the device live-row count (rows >= it are written +0.0; sm70_prefill).
        live_count = metadata.live_counts[:1] if mode == "prefill" and is_sm70(device) else None
        self.projections = {
            field: TensorFP8Projection(inputs, weights.projections[prefix + field], output,
                                       session=session, workspace=workspace, name=name + "." + field, row_invariant=row_invariant,
                                       live_count=live_count)
            for field, inputs, output in (
                (("qkv_proj", source, self.qkv), ("o_proj", self.attended, self.output)) if self.fused_qkv else
                (("q_proj", source, self.qg), ("k_proj", source, self.k),
                 ("v_proj", source, self.v), ("o_proj", self.attended, self.output)))
        }
        self.slots = torch.full((tokens,), -1, dtype=torch.int64, device=device)
        self.lengths = torch.zeros_like(metadata.lengths)
        key_cache, value_cache = pool.attention(layer, config)
        self.sm70 = is_sm70(device)
        writer, attention = PreparedPagedKVWriter, PreparedPagedAttention
        options = dict(mode=mode, workspace=workspace)
        self.status_sink = None
        if self.sm70:
            from ignition.components.sm70_attention import PreparedWriter, PreparedAttention
            writer, attention, options = PreparedWriter, PreparedAttention, dict(row_invariant=row_invariant)
        self.writer = writer(self.key, self.v.view(tokens, self.kv_heads, self.dim),
            key_cache, value_cache, self.slots, session=session, name=name + ".write")
        attended = self.attended.view(tokens, self.heads, self.dim)
        if self.sm70 and mode == "prefill" and status_sink is not None:
            # Reviewed Sol-11s tensor-core prefill attention (one request per prefill graph). Decode, resume and
            # speculative verification keep the scalar kernel. Its sticky status word is OR-ed into the decoder's
            # prefill status so an inadmissible chunk fails the request instead of producing silent NaN.
            from ignition.components.sm70_prefill_attention import PreparedPrefillAttention, admits
            if not admits(self.q, key_cache, metadata.page_table):
                raise ValueError("SM70 prefill attention: the prefill graph is outside the reviewed Sol-11s contract")
            self.attention = PreparedPrefillAttention(self.q, key_cache, value_cache, metadata.page_table,
                self.lengths, metadata.query_start_loc, attended, session=session, name=name + ".attention",
                workspace=workspace)
            self.status_sink = status_sink
        elif self.sm70 and mode == "decode" and decode_agents is not None:
            # Reviewed Sol-13c S1L tensor-core verification attention (raw BF16 mode, four-agent metadata packed once
            # per forward by the decoder); ordinary decode and speculative verification rows, never prefill.
            from ignition.components.sm70_decode_attention import PreparedDecodeAttention
            self.attention = PreparedDecodeAttention(self.q, key_cache, value_cache, decode_agents, attended,
                session=session, name=name + ".attention", workspace=workspace, rows_per_agent=rows_per_agent)
        else:
            self.attention = attention(self.q, key_cache, value_cache, metadata.page_table,
                self.lengths, metadata.query_start_loc, attended, session=session, name=name + ".attention", **options)
        self.norms = None
        if self.sm70 and row_invariant:
            from ignition.components.sm70_norm import PreparedRMSNorm
            query = self.qg.view(tokens, self.heads, 2 * self.dim)[..., :self.dim]
            key = self.k.view(tokens, self.kv_heads, self.dim)
            self.norms = tuple(PreparedRMSNorm(value, norm.weight, output, norm.eps,
                session=session, name=name + "." + field) for value, norm, output, field in (
                    (query, module.q_norm, self.q, "q_norm"), (key, module.k_norm, self.key, "k_norm")))

    def forward(self, hidden_states, position_embeddings, **kwargs):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

        for field in (("qkv_proj",) if self.fused_qkv else ("q_proj", "k_proj", "v_proj")):
            self.projections[field](hidden_states)
        if self.sm70:
            outputs = (self.qg, self.k, self.v)
            for projected, raw in zip(outputs, self.qkv_members if self.fused_qkv else outputs, strict=True):
                projected.copy_(round_bf16(raw))
        query, gate = self.qg.view(-1, self.heads, 2 * self.dim).chunk(2, dim=-1)
        key = self.k.view(-1, self.kv_heads, self.dim)
        if self.sm70:
            q_norm, k_norm = self.canonical.q_norm, self.canonical.k_norm
            q, k = apply_rotary(
                self.norms[0](query) if self.norms is not None else rms_norm(query, q_norm.weight, q_norm.eps),
                self.norms[1](key) if self.norms is not None else rms_norm(key, k_norm.weight, k_norm.eps),
                *position_embeddings)
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
        if self.status_sink is not None:
            self.status_sink.bitwise_or_(self.attention.status_word)
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
    """``replay`` (SM70 speculative verification only) is the ``[linear layers, tokens, operand_floats]`` replay
    operand buffer; each linear-attention layer stores its recurrence operands in its own plane and writes no state."""

    def __init__(self, weights, scratch, metadata, pool, *, mode, vocabulary_runtime,
                 coordinator, session, name, workspace, delta=None, feedback_output=None, row_invariant=False,
                 prefill_errors=None, replay=None):
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        if replay is not None and (delta is None or mode != "decode"):
            raise ValueError("replay-commit operand storage belongs to an SM70 verification decoder with external delta metadata")
        linear_layers = sum(kind == "linear_attention" for kind in weights.config.layers)
        if replay is not None and (replay.ndim != 3 or replay.shape[0] != linear_layers):
            raise ValueError("replay operand buffer needs one plane per linear-attention layer")
        from ignition.components.sm70_projection import is_sm70
        self.sm70 = is_sm70(scratch.activations.device)
        # SM70 prefill graphs: the reviewed tensor-core attention reports an inadmissible chunk through a sticky status
        # word; the OR over the 16 layers fails the request (status ERROR, error code PREFILL_ATTENTION_ERROR) before
        # completion reads the records, so no NaN state is ever sampled or published as a prefix-cache boundary.
        self.prefill_errors, self.prefill_status = None, None
        if self.sm70 and mode == "prefill" and prefill_errors is not None:
            self.prefill_errors = prefill_errors
            self.prefill_status = torch.zeros(1, dtype=torch.int32, device=scratch.activations.device)
        # SM70 decode graphs: the reviewed Sol-13c S1L attention reads four-agent metadata packed once per forward.
        # Option (a) of REVIEW-ARGON-S1L.md: the kernel admits a served context of 32,768 tokens; a wider page table is
        # refused at preparation (prepare with --context 32768) rather than served by a slower, unreviewed mixture.
        self.decode_agents, self.rows_per_agent = None, None
        if self.sm70 and mode == "decode":
            from ignition.components import sm70_decode_attention as decode_attention
            tokens, agents, blocks = scratch.activations.shape[0], metadata.page_table.shape[0], metadata.page_table.shape[1]
            if blocks * PAGE_TOKENS > decode_attention.MAX_LENGTH:
                raise ValueError(f"SM70 decode attention admits {decode_attention.MAX_LENGTH} tokens of context; the "
                                 f"page table serves {blocks * PAGE_TOKENS}: prepare with --context 32768")
            if not (1 <= agents <= decode_attention.AGENTS and tokens % agents == 0
                    and 1 <= tokens // agents <= decode_attention.MAX_ROWS_PER_AGENT):
                raise ValueError(f"SM70 decode attention: {tokens} rows over {agents} requests is outside the contract")
            self.decode_agents = decode_attention.DecodeAgents.allocate(tokens, blocks, scratch.activations.device)
            self.rows_per_agent = tokens // agents
        self.embedding = PreparedEmbedding(weights.tree.embed_tokens.weight, metadata.token_ids,
            scratch.activations, num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        self.delta = delta or DeltaMetadata.allocate(metadata.starts.numel(), scratch.activations.device)
        self.external_delta = delta is not None
        self.feedback_output = feedback_output
        config, source = weights.config, scratch.activations
        attention_buffers = {}
        fused_projections = "layers.0.mlp.gate_up_proj" in weights.projections
        # SM70 prefill graphs: every quantized projection reads the device live-row count; padded rows are +0.0.
        # Nothing reads a padded row into a live row (embedding zeroes them; attention, GDN conv/recurrence and the
        # sampled/feedback rows are bounded by query_start_loc / sample_indices / positions >= 0).
        live_count = metadata.live_counts[:1] if self.sm70 and mode == "prefill" else None
        mlp_buffers = {field: torch.zeros((source.shape[0], width), dtype=source.dtype, device=source.device)
                       for field, width in (("gate_proj", config.text["intermediate_size"]),
                                            ("up_proj", config.text["intermediate_size"]),
                                            ("down_proj", config.hidden))}
        if fused_projections:
            mlp_buffers['gate_up_proj'] = torch.zeros((source.shape[0], 2 * config.text['intermediate_size']),
                dtype=source.dtype, device=source.device)
        # astra-21: SM70 prefill graphs fuse the SwiGLU into the down projection's producer (sm70_prefill.cu
        # atlas_produce_swiglu); its source is the gate|up parent output and no FP32 product buffer exists.
        swiglu = self.sm70 and mode == "prefill" and fused_projections
        mlp_input = None if swiglu else torch.zeros_like(mlp_buffers["gate_proj"])
        norm_buffers = [torch.zeros_like(source) for _ in range(2)] if self.sm70 and row_invariant else None
        self.layers, ordinal = [], 0
        for index, original in enumerate(weights.tree.layers):
            layer = module_copy(original)
            prefix = f"layers.{index}."
            if original.block_type == "linear_attention":
                projections = {field: weights.projections[prefix + "linear_attn." + field]
                               for field in (("in_proj_qkvz", "in_proj_ba", "out_proj") if fused_projections else
                                             ("in_proj_qkv", "in_proj_z", "in_proj_ba", "out_proj"))}
                layer.linear_attn = RecurrentAttention(PreparedGDN(original.linear_attn, projections, source,
                    scratch.gdn, pool.recurrent(index), self.delta, mode=mode, session=session,
                    name=name + "." + prefix + "gdn", workspace=workspace, projection_workspace=workspace, row_invariant=row_invariant,
                    live_count=live_count, replay=None if replay is None else replay[ordinal]))
                ordinal += 1
            else:
                layer.self_attn = FullAttention(original.self_attn, weights, source, metadata, pool, index,
                    mode=mode, session=session, workspace=workspace, name=name + "." + prefix + "attention",
                    buffers=attention_buffers, row_invariant=row_invariant, status_sink=self.prefill_status,
                    decode_agents=self.decode_agents, rows_per_agent=self.rows_per_agent)
            layer.mlp = module_copy(original.mlp)
            for field in (("gate_up_proj", "down_proj") if fused_projections else ("gate_proj", "up_proj", "down_proj")):
                weight = weights.projections[prefix + "mlp." + field]
                transform = "swiglu_bf16" if swiglu and field == "down_proj" else None
                inputs = source if field != "down_proj" else (mlp_buffers["gate_up_proj"] if transform else mlp_input)
                output = mlp_buffers[field]
                setattr(layer.mlp, field, Projection(QuantizedProjection(inputs, weight, output,
                    activation_mode="auto", session=session, workspace=workspace, name=name + "." + prefix + field, row_invariant=row_invariant,
                    live_count=live_count, input_transform=transform)))
            norms = None
            if norm_buffers is not None:
                from ignition.components.sm70_norm import PreparedRMSNorm
                norms = tuple(PreparedRMSNorm(source, norm.weight, output, norm.eps,
                    session=session, name=name + "." + prefix + field) for norm, output, field in zip(
                        (layer.input_layernorm, layer.post_attention_layernorm), norm_buffers,
                        ("input_norm", "post_attention_norm"), strict=True))
            self.layers.append(SM70DecoderLayer(layer, norms, swiglu=swiglu) if self.sm70 else layer)
        rotary = config.text["rope_parameters"]
        dim = int(config.text["head_dim"] * rotary["partial_rotary_factor"])
        self.inverse_frequency = 1.0 / (rotary["rope_theta"] ** (
            torch.arange(0, dim, 2, device=source.device, dtype=torch.float32) / dim))
        self.head = PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head",
            activation_mode="auto", workspace=workspace, row_invariant=row_invariant)
        self.final_norm = None
        if self.sm70 and row_invariant:
            from ignition.components.sm70_norm import PreparedRMSNorm
            norm = weights.tree.norm
            self.final_norm = PreparedRMSNorm(torch.empty_like(scratch.hidden), norm.weight, scratch.hidden,
                                             norm.eps, session=session, name=name + ".final_norm")

    def report_prefill_status(self):
        """Fail the single prefill request when any full-attention layer reported a non-zero status word.

        Only a live row may be failed: an inactive row (length 0) yields an inadmissible plan by construction and its
        slot id is stale. ``requests.status``/``error_code`` are the existing per-request error channel completion
        copies into the records (same mechanism as the duplicate-slot code 32 in ``paged_batch``).
        """
        from ignition.components.sampling import ERROR

        requests, row_slot_ids = self.prefill_errors
        bad = (self.prefill_status != 0) & (self.metadata.lengths[:1] > 0)
        slot = row_slot_ids[:1].long().clamp(0, requests.capacity - 1)
        requests.status[slot] = torch.where(bad, ERROR, requests.status[slot])
        requests.error_code[slot] = torch.where(bad, PREFILL_ATTENTION_ERROR, requests.error_code[slot])

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
        if self.prefill_status is not None:
            self.prefill_status.zero_()
        if self.decode_agents is not None:
            self.decode_agents.pack(m.lengths, m.starts, m.query_start_loc, m.page_table)
        hidden = self.embedding()
        for layer in self.layers:
            hidden = layer(hidden, position_embeddings=position_embeddings)
        if self.prefill_status is not None:
            self.report_prefill_status()
        selected = hidden[m.sample_indices]
        norm = self.weights.tree.norm
        if self.feedback_output is not None:
            self.feedback_output.copy_(rms_norm(hidden, norm.weight, norm.eps) if self.sm70 else norm(hidden))
        if self.final_norm is not None:
            self.final_norm(selected)
        else:
            self.scratch.hidden.copy_(rms_norm(selected, norm.weight, norm.eps) if self.sm70 else norm(selected))
        self.head(self.scratch.hidden)
        if self.sm70:
            self.scratch.logits.copy_(round_bf16(self.scratch.logits))
