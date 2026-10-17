"""Gated NoPE MLA with compressed FP8 latent pages."""

import torch
from b12x.attention import dense_mla

from ignition.components.dense_mla import PreparedDenseMLA
from ignition.components.linear import UnquantizedProjection
from ignition.components.paged_attention import PreparedPagedKVWriter

from .config import PAGE_TOKENS
from .model import rms_norm


class PreparedMLA:
    def __init__(self, weights, source, metadata, cache, *, mode, session, name, workspace, buffers):
        tokens, hidden = source.shape
        heads = weights["key"].shape[0]

        def buffer(key, *shape):
            if key not in buffers or buffers[key].shape[0] < tokens:
                buffers[key] = torch.zeros((tokens, *shape), dtype=source.dtype, device=source.device)
            return buffers[key][:tokens]

        self.input = buffer("input", weights["input"].shape[0])
        self.query_norm = buffer("query_norm", self.input.shape[1] - 576)
        self.query = buffer("query", heads * 192).view(tokens, heads, 192)
        self.latent = buffer("latent", 576)
        self.absorbed = buffer("absorbed", heads, 576)
        self.attended = buffer("attended", heads, 512)
        self.expanded = buffer("expanded", heads * 128)
        self.gate = buffer("gate", heads * 128)
        self.weights, self.metadata = weights, metadata
        self.projections = {key: UnquantizedProjection(x, weights[key], y, session=session, name=name + "." + key)
            for key, x, y in (("input", source, self.input), ("q_b", self.query_norm, self.query.view(tokens, -1)),
                             ("gate", source, self.gate), ("output", self.expanded, buffer("output", hidden)))}
        self.query_normalize = rms_norm(weights["query_norm"], 1e-6)
        self.kv_normalize = rms_norm(weights["kv_norm"], 1e-6)
        self.slots = torch.full((tokens,), -1, dtype=torch.int64, device=source.device)
        self.lengths = torch.zeros_like(metadata.lengths)
        self.writer = PreparedPagedKVWriter(self.latent[:, None, :512], self.latent[:, None, 512:],
            cache[:, :, None, :512], cache[:, :, None, 512:], self.slots, session=session, name=name + ".write")
        caps = dense_mla.Caps(device=source.device, mode="decode" if mode == "decode" else "extend",
            kv_dtype=cache.dtype, q_dtype=self.absorbed.dtype, num_q_heads=heads, page_size=PAGE_TOKENS, max_total_q=tokens,
            max_batch=metadata.lengths.numel(), max_cache_tokens=metadata.page_table.shape[1] * PAGE_TOKENS,
            max_page_table_width=metadata.page_table.shape[1], num_cache_pages=cache.shape[0], use_cuda_graph=True,
            partial_dtype=torch.float32)
        self.attention = PreparedDenseMLA(self.absorbed, cache, self.attended, metadata.page_table,
            self.lengths, metadata.query_start_loc, caps=caps, scale=192 ** -.5,
            session=session, name=name + ".attention", workspace=workspace)

    def __call__(self, source):
        p, m = self.projections, self.metadata
        p["input"](source)
        self.query_norm.copy_(self.query_normalize(self.input[:, :-576]))
        self.latent[:, :512].copy_(self.kv_normalize(self.input[:, -576:-64]))
        self.latent[:, 512:].copy_(self.input[:, -64:])
        p["q_b"](self.query_norm)
        self.absorbed[:, :, :512].copy_(torch.bmm(self.query[:, :, :128].transpose(0, 1), self.weights["key"]).transpose(0, 1))
        self.absorbed[:, :, 512:].copy_(self.query[:, :, 128:])
        page = m.token_page_table.gather(1, (m.positions.clamp(min=0) // PAGE_TOKENS)[:, None]).squeeze(1)
        self.slots.copy_(torch.where(m.positions >= 0, page.long() * PAGE_TOKENS + m.positions % PAGE_TOKENS, -1))
        self.lengths.copy_(torch.where(m.lengths > 0, m.starts + m.lengths, 0))
        self.writer()
        self.attention()
        self.expanded.copy_(torch.bmm(self.attended.transpose(0, 1), self.weights["value"]).transpose(0, 1).flatten(1))
        p["gate"](source)
        self.expanded.mul_(self.gate.sigmoid())
        return p["output"](self.expanded)
