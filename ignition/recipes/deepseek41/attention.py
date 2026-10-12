"""Windowed MLA, source-shared compression, and two-level sparse selection."""

from dataclasses import dataclass
import math

import torch
import triton
import triton.language as tl
from b12x.attention import mla_compress
from b12x.preparation import PreparedCall

from ignition.components.compressed_mla import PreparedCompressedCacheWriter, PreparedCompressedMLA
from ignition.components.attention_output import PreparedGroupedOutputProjection
from ignition.components.linear import QuantizedProjection, UnquantizedProjection
from ignition.components.mxfp4_indexer import PreparedMXFP4Indexer
from ignition.components.rms_norm import PreparedRMSNorm
from ignition.components.state import gather_rows, save_hidden
from ignition.preparation.tuning import declare, register_finalizer

from .pool import PAGE_TOKENS


def normalize(source, weight, out, eps):
    value = source.float()
    out.copy_((value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)) * weight.float())


class Rotary:
    def __init__(self, config, device):
        text = config.text
        scaling = text["rope_scaling"]
        self.frequencies = {}
        dims = torch.arange(32, device=device, dtype=torch.float32)
        for compressed in (False, True):
            base = text["compress_rope_theta"] if compressed else text["rope_theta"]
            frequencies = 1 / base ** (dims / 32)
            if compressed:
                original = scaling["original_max_position_embeddings"]
                def boundary(rotations):
                    return 64 * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))
                low = max(math.floor(boundary(scaling["beta_fast"])), 0)
                high = min(math.ceil(boundary(scaling["beta_slow"])), 63)
                smooth = 1 - ((dims - low) / max(high - low, 1e-3)).clamp(0, 1)
                frequencies = frequencies / scaling["factor"] * (1 - smooth) + frequencies * smooth
            self.frequencies[compressed] = frequencies

    def apply(self, tensor, positions, *, compressed, inverse=False):
        angles = positions.float()[:, None] * self.frequencies[compressed][None]
        cos, sin = angles.cos(), angles.sin() * (-1 if inverse else 1)
        tail = tensor[..., -64:].float().unflatten(-1, (32, 2))
        if tensor.ndim == 3:
            cos, sin = cos[:, None], sin[:, None]
        a, b = tail[..., 0], tail[..., 1]
        tensor[..., -64:].copy_(torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2))


@dataclass(frozen=True)
class AttentionScratch:
    projected: torch.Tensor
    query_norm_input: torch.Tensor
    window_norm_input: torch.Tensor
    query_low: torch.Tensor
    query: torch.Tensor
    window_values: torch.Tensor
    attended: torch.Tensor
    output: torch.Tensor
    selected: torch.Tensor
    candidates: torch.Tensor
    candidate_counts: torch.Tensor
    index_query: torch.Tensor
    index_weights: torch.Tensor
    index_keys: torch.Tensor
    normalized_keys: torch.Tensor

    @classmethod
    def allocate(cls, tokens, world, device):
        def zeros(shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)
        return cls(zeros((tokens, 1792)), zeros((tokens, 1280)), zeros((tokens, 512)),
            zeros((tokens, 1280)), zeros((tokens, 64 // world, 512)),
            zeros((tokens, 512)), zeros((tokens, 64 // world, 512)),
            zeros((tokens, 5120)), zeros((tokens, 512), torch.int32) - 1,
            zeros((tokens, 16384), torch.int32) - 1, zeros(tokens, torch.int32),
            zeros((tokens, 32, 128)), zeros((tokens, 32)), zeros((tokens, 128)), zeros((tokens, 128)))


class PreparedCompressor:
    def __init__(self, layer, weights, source, metadata, pool, slots, *, session, name):
        self.pool, self.metadata, self.layer = pool, metadata, layer
        ratio = pool.config.text["compress_ratios"][layer]
        self.ratio = ratio
        tokens, rows = source.shape[0], metadata.starts.numel()
        device = source.device
        self.values = torch.zeros((tokens, 512), device=device,
            dtype=torch.float32 if ratio == 2 else torch.bfloat16)
        self.project_values = UnquantizedProjection(source, weights["values"], self.values,
            session=session, name=f"{name}.values")
        self.output = torch.zeros((tokens, 512), dtype=torch.bfloat16, device=device)
        self.emitted = torch.zeros(tokens, dtype=torch.bool, device=device)
        self.slots = torch.full((tokens,), -1, dtype=torch.int64, device=device)
        self.state_ids = torch.arange(rows, dtype=torch.int64, device=device)
        self.arguments = dict(values=self.values, weight=weights["norm"], query_start_loc=metadata.query_start_loc,
            positions=metadata.starts, state_ids=self.state_ids, destination_slots=slots,
            live_counts=metadata.live_counts, out=self.output, emitted=self.emitted, emitted_slots=self.slots)
        if ratio == 2:
            self.gates = torch.zeros_like(self.values)
            self.project_gates = UnquantizedProjection(source, weights["gates"], self.gates,
                session=session, name=f"{name}.gates")
            self.pending = {key: torch.zeros((rows, 512) if key != "pending_position" else (rows,),
                dtype=torch.float32 if key != "pending_position" else torch.int64, device=device)
                for key in ("pending_values", "pending_gates", "pending_position")}
            self.arguments.update(gates=self.gates, **self.pending)
        self.plan = mla_compress.plan(mla_compress.Caps(device=device, max_tokens=tokens,
            max_requests=rows, max_states=rows, ratio=ratio))
        declare(session, (self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(state.bind(**self.arguments)))),))
        register_finalizer(session, lambda: setattr(self, "binding", mla_compress.bind(self.plan, **self.arguments)))

    def __call__(self, source):
        m = self.metadata
        if self.ratio == 2:
            for key, value in self.pending.items():
                gather_rows(self.pool.region(f"layer{self.layer}.{key}"), m.state_rows, value.view(value.shape[0], -1))
            self.project_gates(source)
        self.project_values(source)
        mla_compress.run(self.binding)
        if self.ratio == 2:
            for key, value in self.pending.items():
                save_hidden(value.view(value.shape[0], -1), self.pool.region(f"layer{self.layer}.{key}"),
                            m.state_rows, m.lengths > 0)
        return self.output


@triton.jit
def _commit_window(Staged, Window, State, Starts, Lengths, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    item = tl.program_id(0)
    length, start, row = tl.load(Lengths), tl.load(Starts), tl.load(State)
    token = tl.maximum(0, length - 128) + item
    offset = tl.arange(0, BLOCK)
    valid = (token < length) & (row > 0) & (offset < 528)
    value = tl.load(Staged + (128 + token) * 528 + offset, valid, other=0)
    tl.store(Window + row * STRIDE + ((start + token) % 128) * 528 + offset, value, valid)


class PreparedAttention:
    def __init__(self, layer, weights, source, scratch, metadata, pool, rotary, *, mode,
                 world, session, name, workspace):
        self.layer, self.weights, self.scratch, self.metadata = layer, weights, scratch, metadata
        self.pool, self.rotary, self.mode = pool, rotary, mode
        config = pool.config
        self.ratio = config.text["compress_ratios"][layer]
        self.source = config.source(layer)
        self.eps = config.text["rms_norm_eps"]
        tokens, device = source.shape[0], source.device
        self.window = pool.window(layer)
        self.window_cache = self.window if mode == "decode" else torch.zeros(
            (1, (128 + tokens) * 528), dtype=torch.uint8, device=device)
        self.window_slots = torch.full((tokens,), -1, dtype=torch.int64, device=device)
        self.window_indices = torch.full((tokens, 128), -1, dtype=torch.int32, device=device)
        self.window_lengths = torch.zeros(tokens, dtype=torch.int32, device=device)
        self.offsets = torch.arange(128, device=device)
        self.token_offsets = torch.arange(tokens, device=device)
        self.lengths = torch.zeros(tokens, dtype=torch.int32, device=device)
        self.active_width = torch.zeros(1, dtype=torch.int32, device=device)
        self.index_lengths = torch.zeros(tokens, dtype=torch.int32, device=device)
        self.compressed_slots = torch.full((tokens,), -1, dtype=torch.int64, device=device)
        self.projections = {}
        for key, values, out in (
            ("qkv_a", source, scratch.projected), ("q_b", scratch.query_low, scratch.query.view(tokens, -1)),
        ):
            self.projections[key] = QuantizedProjection(values, weights[key], out,
                session=session, name=f"{name}.{key}", workspace=workspace, activation_mode="auto")
        self.query_norm = PreparedRMSNorm(scratch.projected[:, :1280], weights["q_norm"], scratch.query_low,
            eps=self.eps, session=session, name=name + ".q_norm", staging=scratch.query_norm_input)
        self.window_norm = PreparedRMSNorm(scratch.projected[:, 1280:], weights["kv_norm"], scratch.window_values,
            eps=self.eps, session=session, name=name + ".kv_norm", staging=scratch.window_norm_input)
        self.output_projection = PreparedGroupedOutputProjection(scratch.attended.view(tokens, 8 // world, 4096),
            weights["output"], scratch.output, session=session, name=name + ".output", workspace=workspace)
        self.window_writer = PreparedCompressedCacheWriter(scratch.window_values, self.window_cache,
            self.window_slots, page_size=128 if mode == "decode" else 128 + tokens, kind="swa", session=session,
            name=f"{name}.window_write")
        self.compressor = self.compressed_writer = None
        self.selectors = []
        if "compressor" in weights:
            self.compressor = PreparedCompressor(layer, weights["compressor"], source, metadata,
                pool, self.compressed_slots, session=session, name=f"{name}.compress")
            self.index_key = UnquantizedProjection(self.compressor.output, weights["index_k"], scratch.index_keys,
                session=session, name=f"{name}.index_key")
            self.index_norm = PreparedRMSNorm(scratch.index_keys, weights["index_norm"], scratch.normalized_keys,
                eps=self.eps, session=session, name=name + ".index_norm")
            self.compressed_writer = PreparedCompressedCacheWriter(self.compressor.output,
                pool.pages[f"layer{layer}.latent_page"], self.compressor.slots,
                page_size=128 // self.ratio, kind="indexed", session=session, name=f"{name}.compressed_write")
        if "index_q" in weights:
            self.index_query = QuantizedProjection(scratch.query_low, weights["index_q"],
                scratch.index_query.view(tokens, -1), session=session, name=f"{name}.index_query",
                workspace=workspace, activation_mode="auto")
            self.index_weight = UnquantizedProjection(source, weights["index_weights"], scratch.index_weights,
                session=session, name=f"{name}.index_weight")
            candidate_source = config.text["candidate_source_layer_id"]
            for first in range(0, tokens, 32):
                last = min(tokens, first + 32)
                candidates = {}
                if layer == candidate_source:
                    candidates = dict(candidates_out=scratch.candidates[first:last], counts_out=scratch.candidate_counts[first:last])
                elif layer > candidate_source:
                    candidates = dict(candidates=scratch.candidates[first:last], counts=scratch.candidate_counts[first:last])
                selector = PreparedMXFP4Indexer(scratch.index_query[first:last], scratch.index_weights[first:last],
                    pool.pages[f"layer{self.source.layer}.index_page"],
                    metadata.page_table if mode == "prefill" else metadata.page_table[first:last],
                    self.lengths[first:last], self.active_width, scratch.selected[first:last],
                    page_size=128 // self.ratio, session=session, name=f"{name}.index{first}", workspace=workspace, **candidates)
                self.selectors.append(selector)
        indexed = {} if not self.ratio else dict(indexed=pool.pages[f"layer{self.source.layer}.latent_page"],
            indexed_indices=scratch.selected, indexed_lengths=self.index_lengths,
            page_table=metadata.page_table.expand(tokens, -1) if mode == "prefill" else metadata.page_table,
            indexed_page_size=128 // self.ratio)
        self.attention = PreparedCompressedMLA(scratch.query, self.window_cache, self.window_indices,
            self.window_lengths, weights["sink"], scratch.attended,
            window_page_size=128 if mode == "decode" else 128 + tokens,
            mode="decode" if mode == "decode" else "extend", session=session, name=f"{name}.attention",
            workspace=workspace, **indexed)

    def __call__(self, source):
        m, s = self.metadata, self.scratch
        live = m.positions >= 0
        position = m.positions.clamp(min=0)
        rows = m.state_rows[m.requests]
        starts = m.starts[m.requests]
        self.window_lengths.copy_(torch.where(live, (position + 1).clamp(max=128), 0))
        positions = position[:, None] - self.window_lengths[:, None] + 1 + self.offsets
        if self.mode == "decode":
            slots = rows * 128 + position % 128
            indices = rows[:, None] * 128 + positions % 128
        else:
            gather_rows(self.window, m.state_rows, self.window_cache[:, :128 * 528])
            slots = 128 + self.token_offsets
            indices = torch.where(positions < starts[:, None], positions % 128, 128 + positions - starts[:, None])
        self.window_slots.copy_(torch.where(live, slots, -1))
        self.window_indices.copy_(torch.where(self.offsets < self.window_lengths[:, None], indices, -1))
        self.projections["qkv_a"](source)
        self.query_norm(s.projected[:, :1280])
        self.window_norm(s.projected[:, 1280:])
        self.projections["q_b"](s.query_low)
        self.rotary.apply(s.query, position, compressed=bool(self.ratio))
        self.rotary.apply(s.window_values, position, compressed=bool(self.ratio))
        self.window_writer(s.window_values)
        if self.ratio:
            self.lengths.copy_(torch.where(live, (position + 1) // self.ratio, 0))
            self.index_lengths.copy_(self.lengths.clamp(max=512))
            self.active_width.copy_(self.lengths.amax().view(1))
            pages = m.page_table[m.requests, (position // PAGE_TOKENS).clamp(max=m.page_table.shape[1] - 1)]
            self.compressed_slots.copy_(torch.where(live,
                pages.long() * (128 // self.ratio) + (position % 128) // self.ratio, -1))
            if self.compressor is not None:
                latent = self.compressor(source)
                self.index_key(latent)
                self.index_norm(s.index_keys)
                compressed_position = (position + 1) // self.ratio * self.ratio - self.ratio
                self.rotary.apply(s.normalized_keys, compressed_position, compressed=True)
                self.selectors[0].write_keys(s.normalized_keys, self.compressor.slots)
                self.rotary.apply(latent, compressed_position, compressed=True)
                self.compressed_writer(latent)
            if self.selectors:
                self.index_query(s.query_low)
                self.rotary.apply(s.index_query, position, compressed=True)
                self.index_weight(source)
                s.index_weights.mul_(128 ** -.5 * 32 ** -.5)
                for selector in self.selectors:
                    selector()
        self.attention()
        self.rotary.apply(s.attended, position, compressed=bool(self.ratio), inverse=True)
        self.output_projection(s.attended.view(s.attended.shape[0], -1, 4096))
        if self.mode == "prefill":
            torch.library.wrap_triton(_commit_window)[(128,)](self.window_cache, self.window, m.state_rows,
                m.starts, m.lengths, self.window.stride(0), 1024)
        return s.output
