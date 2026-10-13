"""Sliding windows, overlapping C4 retrieval, and dense C128 history."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from ignition.components.attention_output import PreparedGroupedOutputProjection
from ignition.components.compressed_mla import (
    PreparedCompressedCacheWriter,
    PreparedCompressedMLA,
)
from ignition.components.gated_compression import PreparedGatedCompressor
from ignition.components.linear import QuantizedProjection, UnquantizedProjection
from ignition.components.mxfp4_indexer import PreparedMXFP4Indexer
from ignition.components.rms_norm import PreparedRMSNorm

from .config import Compressor
from .pool import PAGE_TOKENS, latent_page_bytes


@triton.jit
def _transfer_window(
    Source,
    Destination,
    StateRows,
    Starts,
    Counts,
    SOURCE_STRIDE: tl.constexpr,
    DEST_STRIDE: tl.constexpr,
    STAGED_TOKENS: tl.constexpr,
    COMMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request, item = tl.program_id(0), tl.program_id(1)
    row = tl.load(StateRows + request)
    offset = tl.arange(0, BLOCK)
    if COMMIT:
        length = tl.load(Counts + request)
        token = tl.maximum(0, length - 128) + item
        live = (row > 0) & (token < length)
        source_row, source_token = request, 128 + token
        dest_row, dest_token = row, (tl.load(Starts + request) + token) % 128
        source_extent: tl.constexpr = STAGED_TOKENS
        dest_extent: tl.constexpr = 128
    else:
        live = row > 0
        source_row, source_token = row, item
        dest_row, dest_token = request, item
        source_extent: tl.constexpr = 128
        dest_extent: tl.constexpr = STAGED_TOKENS
    source_offset = tl.where(
        offset < 576, source_token * 576 + offset, source_extent * 576 + source_token * 8 + offset - 576
    )
    dest_offset = tl.where(
        offset < 576, dest_token * 576 + offset, dest_extent * 576 + dest_token * 8 + offset - 576
    )
    value = tl.load(Source + source_row * SOURCE_STRIDE + source_offset, live & (offset < 584), other=0)
    tl.store(Destination + dest_row * DEST_STRIDE + dest_offset, value, live & (offset < 584))


def transfer_window(source, destination, metadata, staged_tokens, *, counts=None):
    torch.library.wrap_triton(_transfer_window)[(metadata.starts.numel(), 128)](
        source,
        destination,
        metadata.state_rows,
        metadata.starts,
        metadata.lengths if counts is None else counts,
        source.stride(0),
        destination.stride(0),
        staged_tokens,
        counts is not None,
        1024,
    )


@triton.jit
def _hadamard(Values, Output, ROWS: tl.constexpr):
    row = tl.program_id(0)
    dim = tl.arange(0, 128)
    value = tl.load(Values + row * 128 + dim)
    value = value.to(tl.float32)
    for bit in tl.static_range(7):
        other = tl.gather(value, dim ^ (1 << bit), axis=0)
        value = tl.where((dim & (1 << bit)) == 0, value + other, other - value)
    tl.store(Output + row * 128 + dim, value * (128**-0.5))


def rotate_index(source, destination):
    torch.library.wrap_triton(_hadamard)[(source.numel() // 128,)](source, destination, source.numel() // 128)


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
    dense_indices: torch.Tensor
    index_query: torch.Tensor
    rotated_query: torch.Tensor
    index_weights: torch.Tensor
    rotated_keys: torch.Tensor

    @classmethod
    def allocate(cls, tokens, world, device, *, blocks=8192):
        def zeros(shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)

        return cls(
            zeros((tokens, 1536)),
            zeros((tokens, 1024)),
            zeros((tokens, 512)),
            zeros((tokens, 1024)),
            zeros((tokens, 64 // world, 512)),
            zeros((tokens, 512)),
            zeros((tokens, 64 // world, 512)),
            zeros((tokens, 4096)),
            zeros((tokens, 512), torch.int32) - 1,
            zeros((tokens, blocks), torch.int32) - 1,
            zeros((tokens, 64, 128)),
            zeros((tokens, 64, 128)),
            zeros((tokens, 64)),
            zeros((tokens, 128)),
        )


class PreparedAttention:
    def __init__(
        self,
        layer,
        weights,
        source,
        scratch,
        metadata,
        pool,
        rotary,
        *,
        mode,
        world,
        session,
        name,
        workspace,
        visible=None,
    ):
        self.scratch, self.metadata, self.pool = scratch, metadata, pool
        self.rotary, self.mode, self.visible = rotary, mode, visible
        self.ratio = 0 if mode == "draft" else pool.config.text["compress_ratios"][layer]
        self.eps = pool.config.text["rms_norm_eps"]
        tokens, rows, device = source.shape[0], metadata.starts.numel(), source.device
        self.window = pool.window(layer, draft=mode == "draft")
        self.window_page_size = 128 if mode == "decode" else 128 + tokens // rows
        self.window_cache = (
            self.window
            if mode == "decode"
            else torch.zeros(
                (rows, latent_page_bytes(self.window_page_size)), dtype=torch.uint8, device=device
            )
        )
        width = self.window_page_size if mode == "draft" else 128 + (384 if visible is not None else 0)
        self.offsets = torch.arange(width, device=device)
        self.window_slots = torch.full((tokens,), -1, dtype=torch.int64, device=device)
        self.window_indices = torch.full((tokens, width), -1, dtype=torch.int32, device=device)
        self.window_lengths = torch.zeros(tokens, dtype=torch.int32, device=device)
        self.compressed_slots = torch.full_like(self.window_slots, -1)
        self.lengths = torch.zeros_like(self.window_lengths)
        self.active_width = torch.zeros(1, dtype=torch.int32, device=device)
        self.index_lengths = torch.zeros_like(self.window_lengths)
        self.qkv = QuantizedProjection(
            source,
            weights["qkv_a"],
            scratch.projected,
            session=session,
            name=name + ".qkv_a",
            workspace=workspace,
            activation_mode="auto",
        )
        self.qb = QuantizedProjection(
            scratch.query_low,
            weights["q_b"],
            scratch.query.view(tokens, -1),
            session=session,
            name=name + ".q_b",
            workspace=workspace,
            activation_mode="auto",
        )
        self.qnorm = PreparedRMSNorm(
            scratch.projected[:, :1024],
            weights["q_norm"],
            scratch.query_low,
            eps=self.eps,
            session=session,
            name=name + ".q_norm",
            staging=scratch.query_norm_input,
        )
        self.kvnorm = PreparedRMSNorm(
            scratch.projected[:, 1024:],
            weights["kv_norm"],
            scratch.window_values,
            eps=self.eps,
            session=session,
            name=name + ".kv_norm",
            staging=scratch.window_norm_input,
        )
        self.output_projection = PreparedGroupedOutputProjection(
            scratch.attended.view(tokens, 8 // world, 4096),
            weights["output"],
            scratch.output,
            session=session,
            name=name + ".output",
            workspace=workspace,
        )

        def writer(values, cache, slots, size, kind, suffix):
            return PreparedCompressedCacheWriter(
                values,
                cache,
                slots,
                page_size=size,
                kind=kind,
                session=session,
                name=name + suffix,
                cache_format="deepseek_v4",
            )

        self.window_writer = writer(
            scratch.window_values,
            self.window_cache,
            self.window_slots,
            self.window_page_size,
            "swa",
            ".window_write",
        )
        self.compressor = self.index_compressor = None
        self.selectors = []
        self.indexed_indices = None
        indexed = {}
        if self.ratio:
            self.compressor = PreparedGatedCompressor(
                Compressor(layer, self.ratio, 512),
                weights["compressor"],
                source,
                metadata,
                pool,
                self.compressed_slots,
                session=session,
                name=name + ".compress",
                deferred=mode == "verify",
            )
            self.compressed_writer = writer(
                self.compressor.output,
                pool.pages[f"layer{layer}.latent_page"],
                self.compressor.slots,
                PAGE_TOKENS // self.ratio,
                "indexed",
                ".compressed_write",
            )
            if self.ratio == 4:
                self.index_compressor = PreparedGatedCompressor(
                    Compressor(layer, 4, 128, index=True),
                    weights["index_compressor"],
                    source,
                    metadata,
                    pool,
                    self.compressed_slots,
                    session=session,
                    name=name + ".index_compress",
                    deferred=mode == "verify",
                )
                self.index_query = QuantizedProjection(
                    scratch.query_low,
                    weights["index_q"],
                    scratch.index_query.view(tokens, -1),
                    session=session,
                    name=name + ".index_query",
                    workspace=workspace,
                    activation_mode="auto",
                )
                self.index_weight = UnquantizedProjection(
                    source,
                    weights["index_weights"],
                    scratch.index_weights,
                    session=session,
                    name=name + ".index_weight",
                )
                for first in range(0, tokens, 32):
                    last = min(tokens, first + 32)
                    self.selectors.append(
                        PreparedMXFP4Indexer(
                            scratch.rotated_query[first:last],
                            scratch.index_weights[first:last],
                            pool.pages[f"layer{layer}.index_page"],
                            metadata.token_page_table[first:last],
                            self.lengths[first:last],
                            self.active_width,
                            scratch.selected[first:last],
                            page_size=32,
                            session=session,
                            name=f"{name}.index{first}",
                            workspace=workspace,
                        )
                    )
                selected = scratch.selected
            else:
                selected = scratch.dense_indices[:, : metadata.page_table.shape[1]]
            self.indexed_indices = selected
            indexed = {
                "indexed": pool.pages[f"layer{layer}.latent_page"],
                "indexed_indices": selected,
                "indexed_lengths": self.index_lengths,
                "indexed_page_size": PAGE_TOKENS // self.ratio,
            }
        self.attention = PreparedCompressedMLA(
            scratch.query,
            self.window_cache,
            self.window_indices,
            self.window_lengths,
            weights["sink"],
            scratch.attended,
            window_page_size=self.window_page_size,
            mode="decode" if mode == "decode" else "extend",
            session=session,
            name=name + ".attention",
            workspace=workspace,
            cache_format="deepseek_v4",
            **indexed,
        )

    def __call__(self, source):
        m, s = self.metadata, self.scratch
        live, position = m.positions >= 0, m.positions.clamp_min(0)
        rows, starts = m.state_rows[m.requests], m.starts[m.requests]
        left = (position - 127).clamp_min(0)
        right = position
        if self.visible is not None:
            image_left, image_right = self.visible
            left = torch.minimum(left, position - image_left)
            right = position + image_right
        self.window_lengths.copy_(torch.where(live, right - left + 1, 0))
        positions = left[:, None] + self.offsets
        if self.mode == "decode":
            slots = rows * 128 + position % 128
            indices = rows[:, None] * 128 + positions % 128
        else:
            transfer_window(self.window, self.window_cache, m, self.window_page_size)
            base = m.requests * self.window_page_size
            slots = base + 128 + position - starts
            indices = torch.where(
                positions < starts[:, None], positions % 128, 128 + positions - starts[:, None]
            )
            indices = indices + base[:, None]
            if self.mode == "draft":
                context = starts.clamp(max=128)
                self.window_lengths.copy_(torch.where(live, context + m.lengths[m.requests], 0))
                indices = base[:, None] + torch.where(
                    self.offsets < context[:, None], self.offsets, 128 + self.offsets - context[:, None]
                )
        self.window_slots.copy_(torch.where(live, slots, -1))
        self.window_indices.copy_(torch.where(self.offsets < self.window_lengths[:, None], indices, -1))
        self.qkv(source)
        self.qnorm(s.projected[:, :1024])
        self.kvnorm(s.projected[:, 1024:])
        self.qb(s.query_low)
        s.query.mul_(torch.rsqrt(s.query.square().mean(-1, keepdim=True) + self.eps))
        self.rotary.apply(s.query, position, compressed=bool(self.ratio))
        self.rotary.apply(s.window_values, position, compressed=bool(self.ratio))
        self.window_writer(s.window_values)
        if self.ratio:
            self.lengths.copy_(torch.where(live, (position + 1) // self.ratio, 0))
            self.index_lengths.copy_(self.lengths.clamp(max=512) if self.ratio == 4 else self.lengths)
            self.active_width.copy_(self.lengths.amax().view(1))
            pages = m.page_table[m.requests, (position // PAGE_TOKENS).clamp(max=m.page_table.shape[1] - 1)]
            self.compressed_slots.copy_(
                torch.where(
                    live,
                    pages.long() * (PAGE_TOKENS // self.ratio) + (position % PAGE_TOKENS) // self.ratio,
                    -1,
                )
            )
            compressed_position = ((position + 1) // self.ratio * self.ratio - self.ratio).clamp_min(0)
            latent = self.compressor(source)
            self.rotary.apply(latent, compressed_position, compressed=True)
            self.compressed_writer(latent)
            if self.index_compressor is not None:
                keys = self.index_compressor(source)
                self.rotary.apply(keys, compressed_position, compressed=True)
                rotate_index(keys, s.rotated_keys)
                self.selectors[0].write_keys(s.rotated_keys, self.index_compressor.slots)
                self.index_query(s.query_low)
                self.rotary.apply(s.index_query, position, compressed=True)
                rotate_index(s.index_query, s.rotated_query)
                self.index_weight(source)
                s.index_weights.mul_(128**-0.5 * 64**-0.5)
                for selector in self.selectors:
                    selector()
                logical = s.selected.long()
                physical = m.token_page_table.gather(1, (logical // 32).clamp(0, m.page_table.shape[1] - 1))
                s.selected.copy_(
                    torch.where((logical >= 0) & (physical >= 0), physical * 32 + logical % 32, -1)
                )
            else:
                self.indexed_indices.copy_(m.token_page_table)
        self.attention()
        self.rotary.apply(s.attended, position, compressed=bool(self.ratio), inverse=True)
        self.output_projection(s.attended.view(s.attended.shape[0], -1, 4096))
        if self.mode == "prefill":
            self.commit(m.lengths)
        return s.output

    def commit(self, counts):
        transfer_window(self.window_cache, self.window, self.metadata, self.window_page_size, counts=counts)
        if self.mode == "verify":
            if self.compressor is not None:
                self.compressor.commit(counts)
            if self.index_compressor is not None:
                self.index_compressor.commit(counts)
