"""Prepared mixed-format sliding-window and indexed latent attention."""

import torch
from functools import partial
from b12x.attention import compressed_sparse_mla as mla
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle

from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import declare, register_finalizer
from ignition.preparation.fixtures.compressed_mla import benchmark_call


@torch.library.custom_op("flare::compressed_cache_write", mutates_args=("cache",))
def _write(kv: torch.Tensor, cache: torch.Tensor, slots: torch.Tensor, handle: int) -> None:
    mla.write_cache(kv, cache, slots, plan=plan_from_handle(handle))


@_write.register_fake
def _write_fake(kv, cache, slots, handle):
    pass


class PreparedCompressedCacheWriter:
    def __init__(self, values, cache, slots, *, page_size, kind, session, name, cache_format="deepseek_v41"):
        self.plan = mla.plan_cache_writer(mla.CacheWriterQuery(max_rows=values.shape[0],
            page_size=page_size, cache_kind=kind, cache_format=cache_format,
            slot_dtype=str(slots.dtype).removeprefix("torch.")),
            device=values.device)
        self.cache, self.slots = cache, slots
        declare(session, (self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(values, cache, slots))),))

    def __call__(self, values):
        _write(values, self.cache, self.slots, self.plan.handle)


@torch.library.custom_op("flare::compressed_mla", mutates_args=("out", "scratch"))
def _attend(q: torch.Tensor, swa: torch.Tensor, indices: torch.Tensor, lengths: torch.Tensor,
            indexed: torch.Tensor | None, indexed_indices: torch.Tensor | None,
            indexed_lengths: torch.Tensor | None, page_table: torch.Tensor | None,
            sink: torch.Tensor, out: torch.Tensor, scratch: torch.Tensor, scale: float, handle: int) -> None:
    plan = plan_from_handle(handle)
    binding = mla.bind(plan, scratch=scratch, q=q, swa_indices=indices, swa_lengths=lengths,
        indexed_indices=indexed_indices, indexed_lengths=indexed_lengths, indexed_page_table=page_table)
    mla.run(binding=binding, swa_k_cache=swa, indexed_k_cache=indexed,
            attn_sink=sink, sm_scale=scale, out=out)


@_attend.register_fake
def _attend_fake(q, swa, indices, lengths, indexed, indexed_indices, indexed_lengths,
                 page_table, sink, out, scratch, scale, handle):
    pass


class PreparedCompressedMLA:
    def __init__(self, query, window, indices, lengths, sink, output, *, indexed=None,
                 indexed_indices=None, indexed_lengths=None, page_table=None,
                 window_page_size=128, indexed_page_size=64, mode="decode", session, name, workspace,
                 cache_format="deepseek_v41"):
        width = 0 if indexed_indices is None else indexed_indices.shape[1]
        caps = mla.Caps(device=query.device, num_q_heads=query.shape[1], max_q_rows=query.shape[0],
            max_width=indices.shape[1] + width, swa_width=indices.shape[1], indexed_width=width,
            max_page_table_width=1 if page_table is None else page_table.shape[1],
            swa_page_size=window_page_size, indexed_page_size=indexed_page_size,
            cache_format=cache_format, mode=mode)
        self.plan = mla.plan(caps, invocation=mla.invocation_from_tensors(q=query,
            swa_k_cache=window, indexed_k_cache=indexed, attn_sink=sink, out=output))
        self.query, self.window, self.indexed, self.output = query, window, indexed, output
        self.inputs = dict(q=query, swa_indices=indices, swa_lengths=lengths,
            indexed_indices=indexed_indices, indexed_lengths=indexed_lengths, indexed_page_table=page_table)
        self.sink, self.scale = sink, query.shape[-1] ** -.5
        self.workspace = workspace

        def prepare(state):
            scratch = planned_scratch(state.scratch_specs(), workspace)
            binding = state.bind_for_preparation(scratch=scratch, **self.inputs)
            return PreparedCall(run=lambda: state.run(binding, swa_k_cache=window, indexed_k_cache=indexed,
                attn_sink=sink, sm_scale=self.scale, out=output), owners=prime_owners(workspace, scratch, binding))

        declare(session, (self.plan.request(name=name, prepare_call=prepare,
            benchmark_call=partial(benchmark_call, query=query, window=window, indexed=indexed,
                arguments=self.inputs, session=session, workspace=workspace, cache_format=cache_format,
                window_page_size=window_page_size, indexed_page_size=indexed_page_size)),))
        register_finalizer(session, self.rebind, workspace=workspace)

    def rebind(self):
        self.scratch = planned_scratch(self.plan.scratch_specs(), self.workspace)

    def __call__(self):
        args = self.inputs
        _attend(self.query, self.window, args["swa_indices"], args["swa_lengths"], self.indexed,
            args["indexed_indices"], args["indexed_lengths"], args["indexed_page_table"],
            self.sink, self.output, self.scratch, self.scale, self.plan.handle)
        return self.output
