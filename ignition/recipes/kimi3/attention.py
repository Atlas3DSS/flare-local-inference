"""Gated NoPE MLA over paged E4M3 latent records, sharded over decode context ranks.

As vLLM's Kimi K3 serving stores its latent cache (--kv-cache-dtype fp8), every record is E4M3 with a
unit scale. Without context ranks a rank attends with its heads over the whole cache. With them
(decode context parallel, vLLM's --decode-context-parallel-size 16), each rank stores the positions it
owns; decode and verification quantize their absorbed queries, gather every rank's heads, attend over
this rank's records with b12x dense MLA and keep this rank's heads by combining the partial outputs by
their LSE (b12x DcpAllToAll). Prefill copies the request's earlier context from every rank's cache into
a slab beside the chunk's own records and attends with this rank's heads.
"""

import torch
from b12x.attention import dense_mla

from ignition.components.collectives import PreparedAllGather, PreparedLseReduceScatter
from ignition.components.dense_mla import PreparedDenseMLA
from ignition.components.linear import UnquantizedProjection

from .config import PAGE_TOKENS
from .dcp import RECORD_BYTES, ContextShards, pull_context, write_records
from .pointwise import rms_norm_columns, sigmoid_gate

FP8 = torch.float8_e4m3fn


class PreparedMLA:
    def __init__(self, weights, source, metadata, cache, *, mode, session, name, workspace, buffers, latents=None,
                 coordinator=None, context=None, peers=None):
        tokens, hidden = source.shape
        heads = weights["key"].shape[0]
        device = source.device

        def buffer(key, *shape, dtype=None, rows=None):
            # Graphs share each buffer at its declared capacity, else their own rows; a larger graph replaces
            # it and earlier graphs keep their bindings.
            if key not in buffers or buffers[key].shape[0] < tokens:
                buffers[key] = torch.zeros((max(rows or tokens, tokens), *shape),
                                           dtype=source.dtype if dtype is None else dtype, device=device)
            return buffers[key][:tokens]

        latent_width = weights["query_norm"].shape[0] + RECORD_BYTES
        self.query_norm = buffer("query_norm", latent_width - RECORD_BYTES)
        self.query = buffer("query", heads * 192).view(tokens, heads, 192)
        self.latent = buffer("latent", RECORD_BYTES)
        self.absorbed = buffer("absorbed", heads, RECORD_BYTES)
        self.attended = buffer("attended", heads, 512)
        self.expanded = buffer("expanded", heads * 128)
        self.weights, self.metadata, self.mode, self.cache = weights, metadata, mode, cache
        # Each rank projects only its query and KV latent rows beside its output gate, and the ranks gather the
        # latents, as vLLM's VLLM_KIMI_SHARD_QKV_A does: every rank keeps and reads a sixteenth of the latent
        # projection. A single rank (``latents`` None) projects every latent row itself.
        self.latent_gather = None
        if latents is None:
            self.local = buffer("local", weights["latent_gate"].shape[0])
            self.input = self.local[:, :latent_width]
            self.gate = self.local[:, latent_width:]
        else:
            rows, width = latents.max_batch_size, latents.query_head_dim
            if not tokens <= rows:
                raise ValueError("the latent exchange must carry every row of the step")
            self.input = buffer("input", latent_width)
            self.local = buffer("local", weights["latent_gate"].shape[0], rows=rows)
            self.gate = self.local[:, width:]
            # The exchange reads contiguous rows, which a single row's latent columns already are.
            self.staged = tokens > 1
            self.send = buffer("send", width, rows=rows) if self.staged else self.local[:, :width]
            self.gathered_latents = buffer("gathered_latents", latents.world_size * width, rows=rows)
            self.latent_gather = PreparedAllGather(self.send, self.gathered_latents, runtime=latents,
                session=session, coordinator=coordinator, name=name + ".gather_latents")
        self.projections = {key: UnquantizedProjection(x, weights[key], y, session=session, name=name + "." + key)
            for key, x, y in (("latent_gate", source, self.local), ("q_b", self.query_norm, self.query.view(tokens, -1)),
                             ("output", self.expanded, buffer("output", hidden)))}
        self.shards = ContextShards() if context is None else context.shards
        self.peers = peers
        self.slots = torch.full((tokens,), -1, dtype=torch.int64, device=device)
        world, width = self.shards.world, metadata.page_table.shape[1]
        common = dict(device=device, kv_dtype=FP8, page_size=PAGE_TOKENS, use_cuda_graph=True,
                      partial_dtype=torch.float32)
        scale = 192 ** -.5
        if world == 1:
            # Every rank holds every record: attend with this rank's heads; b12x quantizes the BF16 query.
            self.lengths = torch.zeros_like(metadata.lengths)
            caps = dense_mla.Caps(mode=mode if mode in ("decode", "verify") else "extend", q_dtype=torch.bfloat16,
                num_q_heads=heads, max_total_q=tokens, max_batch=metadata.lengths.numel(),
                max_cache_tokens=width * PAGE_TOKENS, max_page_table_width=width, num_cache_pages=cache.shape[0],
                **common)
            self.attention = PreparedDenseMLA(self.absorbed, cache, self.attended, metadata.page_table, self.lengths,
                metadata.query_start_loc, caps=caps, scale=scale, session=session, name=name + ".attention",
                workspace=workspace)
        elif mode == "prefill":
            # One packed request: its earlier context arrives from every rank, the chunk's records from this rank.
            slab = context.slab
            pages = slab.shape[0] // PAGE_TOKENS
            if peers is None or peers.shape != (world,) or slab.shape[0] % PAGE_TOKENS or pages < width * world:
                raise ValueError("context-parallel prefill requires every rank's cache address and a context slab")
            self.slab = slab
            self.slab_table = torch.arange(pages, dtype=torch.int32, device=device)[None]
            self.end = torch.zeros(1, dtype=torch.int64, device=device)
            self.lengths = torch.zeros_like(metadata.lengths)
            caps = dense_mla.Caps(mode="extend", q_dtype=torch.bfloat16, num_q_heads=heads, max_total_q=tokens,
                max_batch=1, max_cache_tokens=pages * PAGE_TOKENS, max_page_table_width=pages,
                num_cache_pages=pages, **common)
            self.attention = PreparedDenseMLA(self.absorbed, slab.view(pages, PAGE_TOKENS, RECORD_BYTES),
                self.attended, self.slab_table, self.lengths, metadata.query_start_loc, caps=caps, scale=scale,
                session=session, name=name + ".attention", workspace=workspace)
        else:
            # Each query row attends to the records this rank owns at or before its position, as its own sequence.
            total, rows = heads * world, context.heads.max_batch_size
            self.quantized = buffer("quantized", heads, RECORD_BYTES, dtype=FP8, rows=rows)
            self.gathered = buffer("gathered", total, RECORD_BYTES, dtype=FP8, rows=rows)
            self.partial = buffer("partial", total, 512, rows=rows)
            self.lse = buffer("lse", total, dtype=torch.float32, rows=rows)
            self.token_pages = buffer("token_pages", width, dtype=torch.int32, rows=rows)
            self.token_lengths = buffer("token_lengths", dtype=torch.int32, rows=rows)
            self.token_starts = torch.arange(tokens + 1, dtype=torch.int32, device=device)
            self.gather = PreparedAllGather(self.quantized.view(tokens, -1), self.gathered.view(tokens, -1),
                runtime=context.heads, session=session, coordinator=context.coordinator, name=name + ".gather_heads")
            caps = dense_mla.Caps(mode="decode", q_dtype=FP8, num_q_heads=total, max_total_q=tokens,
                max_batch=tokens, max_cache_tokens=width * PAGE_TOKENS, max_page_table_width=width,
                num_cache_pages=cache.shape[0], **common)
            self.attention = PreparedDenseMLA(self.gathered, cache, self.partial, self.token_pages,
                self.token_lengths, self.token_starts, caps=caps, scale=scale, session=session,
                name=name + ".attention", workspace=workspace, lse=self.lse)
            self.combine = PreparedLseReduceScatter(self.partial, self.lse, self.attended, runtime=context.heads,
                session=session, coordinator=context.coordinator, name=name + ".combine")

    def __call__(self, source):
        p, m, shards = self.projections, self.metadata, self.shards
        p["latent_gate"](source)
        if self.latent_gather is not None:
            tokens, width = self.send.shape
            if self.staged:
                self.send.copy_(self.local[:, :width])
            self.latent_gather(self.send)
            # Rank-major latent shards back into the query and KV latents they partition.
            world = self.gathered_latents.shape[1] // width
            gathered = self.gathered_latents.view(tokens, world, width)
            query = self.input.shape[1] - RECORD_BYTES
            q, kv = query // world, RECORD_BYTES // world
            self.input[:, :query].view(tokens, world, q).copy_(gathered[:, :, :q])
            self.input[:, query:].view(tokens, world, kv).copy_(gathered[:, :, q:q + kv])
        rms_norm_columns(self.input[:, :-RECORD_BYTES], self.weights["query_norm"], self.query_norm, 1e-6)
        rms_norm_columns(self.input[:, -RECORD_BYTES:-64], self.weights["kv_norm"], self.latent[:, :512], 1e-6)
        self.latent[:, 512:].copy_(self.input[:, -64:])
        p["q_b"](self.query_norm)
        self.absorbed[:, :, :512].copy_(torch.bmm(self.query[:, :, :128].transpose(0, 1), self.weights["key"]).transpose(0, 1))
        self.absorbed[:, :, 512:].copy_(self.query[:, :, 128:])
        # Each rank stores the positions it owns; every position is owned when no ranks share the context.
        self.slots.copy_(shards.token_slots(m.positions, m.token_page_table))
        write_records(self.latent, self.slots, self.cache.view(-1, RECORD_BYTES))
        if shards.world == 1:
            self.lengths.copy_(torch.where(m.lengths > 0, m.starts + m.lengths, 0))
            self.attention()
        elif self.mode == "prefill":
            # Earlier chunks' records were written before this forward's collectives ordered the ranks.
            self.end.copy_(m.starts[:1])
            pull_context(self.peers, m.page_table[0], self.end, self.slab.view(torch.uint8).view(torch.int32).view(
                self.slab.shape[0], RECORD_BYTES // 4), shards)
            write_records(self.latent, m.positions, self.slab)
            self.lengths.copy_(torch.where(m.lengths > 0, m.starts + m.lengths, 0))
            self.attention()
        else:
            tokens = self.absorbed.shape[0]
            self.quantized.copy_(self.absorbed.float().clamp(-448., 448.).to(FP8))
            self.gather(self.quantized.view(tokens, -1))
            self.token_pages.copy_(m.token_page_table)
            self.token_lengths.copy_(shards.local_tokens(m.positions + 1).to(torch.int32))
            self.attention()
            self.combine(self.partial, self.lse)
        self.expanded.copy_(torch.bmm(self.attended.transpose(0, 1), self.weights["value"]).transpose(0, 1).flatten(1))
        sigmoid_gate(self.expanded, self.gate)
        return p["output"](self.expanded)
