"""Prepared b12x PCIe reductions embedded in compiled CUDA graphs."""

import itertools
import weakref

import torch
from b12x.preparation import CollectiveRequirement, PreparedCall

from ignition.preparation.tuning import declare

_OWNERS = weakref.WeakValueDictionary()
_HANDLES = itertools.count()


@torch.library.custom_op("flare::pcie_all_reduce", mutates_args=("output",))
def _all_reduce(source: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    owner = _OWNERS[handle]
    owner.runtime.all_reduce(source, out=output, plan=owner.plan)


@_all_reduce.register_fake
def _fake_all_reduce(source, output, handle):
    pass


class PreparedAllReduce:
    """One fixed-shape reduction through an established b12x PCIe runtime.

    The bootstrap owns a single-channel runtime, exchange group, and coordinator.
    Every participating rank must declare the same operation name and enter
    preparation together. Graph ownership must retain this component and its
    runtime until every rank has drained native replay.
    """

    def __init__(self, source, output, *, runtime, session, coordinator, name):
        import torch.distributed as dist
        from b12x.comm import pcie

        self.runtime, self.output = runtime, output
        if isinstance(runtime, pcie.DmaAllReduce):
            # Copy-engine ring for prefill-sized activations.
            from b12x.comm.pcie._dma_preparation import query_from_metadata

            self.plan = pcie.plan(query_from_metadata(runtime, shape=tuple(source.shape), dtype=source.dtype),
                                  runtime=runtime)
            group = runtime.group

            def prepare(state):
                return PreparedCall(run=lambda: state.run(source, out=output), output=output,
                                    owners=(runtime, source, output))
        else:
            if (runtime.world_size not in (2, 4) or runtime.algorithm != "oneshot"
                    or not runtime.single_channel):
                raise ValueError("prepared PCIe reductions require single-channel b12x TP2 or TP4 all-reduce")
            self.plan = runtime.plan(source, out=output)
            group = runtime.exchange_group

            def prepare(state):
                state.bind_input(source)
                return PreparedCall(run=lambda: state.run_plain(source, output),
                                    output=output, owners=(runtime, source, output))

        ranks = tuple(dist.get_process_group_ranks(group))
        request = self.plan.request(name=name, prepare_call=prepare,
            collective=CollectiveRequirement(key=name, ranks=ranks))
        declare(session, (request,), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def __call__(self, source):
        _all_reduce(source, self.output, self.handle)
        return self.output


def _gather_columns(source, output, handle):
    owner = _OWNERS[handle]
    runtime = owner.runtime
    runtime.all_gather_heads(
        source.view(source.shape[0], runtime.heads_per_rank, runtime.query_head_dim),
        out=output.view(output.shape[0], runtime.total_heads, runtime.query_head_dim),
        plan=owner.plan)


@torch.library.custom_op("flare::pcie_all_gather", mutates_args=("output",))
def _all_gather(source: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    _gather_columns(source, output, handle)


@_all_gather.register_fake
def _fake_all_gather(source, output, handle):
    pass




class PreparedAllGather:
    """Gather contiguous column partitions through a caller-owned b12x channel.

    Each output row contains rank zero's columns, then rank one's, and so on.
    Operations sharing the channel must execute serially on every rank.
    """

    def __init__(self, source, output, *, runtime, session, coordinator, name):
        import torch.distributed as dist
        from b12x.comm import pcie

        if (runtime.world_size not in (2, 4) or source.ndim != 2
                or source.dtype != torch.bfloat16 or output.dtype != source.dtype
                or source.device != runtime.device or output.device != source.device
                or not source.is_contiguous() or not output.is_contiguous()
                or not 0 < source.shape[0] <= runtime.max_batch_size
                or source.shape[1] != runtime.heads_per_rank * runtime.query_head_dim
                or output.shape != (source.shape[0], source.shape[1] * runtime.world_size)):
            raise ValueError("all-gather requires contiguous BF16 column partitions matching its TP2 or TP4 channel")
        self.runtime, self.output = runtime, output
        call = {"local_input": source.view(source.shape[0], runtime.heads_per_rank, runtime.query_head_dim),
                "out": output.view(output.shape[0], runtime.total_heads, runtime.query_head_dim)}
        query = pcie.query_from_runtime(runtime, surface="DcpAllToAll.all_gather_heads", call=call)
        self.plan = pcie.plan(query, runtime=runtime)
        ranks = tuple(dist.get_process_group_ranks(runtime.exchange_group))
        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: PreparedCall(run=lambda: state.run(**call), output=output,
                                                    owners=(runtime, source, output)),
            collective=CollectiveRequirement(key=name, ranks=ranks)),), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def __call__(self, source):
        _all_gather(source, self.output, self.handle)
        return self.output

    def run_eager(self, source):
        """Gather on the current stream from inside another eager operation."""
        _gather_columns(source, self.output, self.handle)
        return self.output


@torch.library.custom_op("flare::pcie_lse_reduce_scatter", mutates_args=("output",))
def _lse_reduce_scatter(partial: torch.Tensor, lse: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    owner = _OWNERS[handle]
    owner.runtime.lse_reduce_scatter(partial, lse, out=output, plan=owner.plan, is_lse_base_on_e=True)


@_lse_reduce_scatter.register_fake
def _fake_lse_reduce_scatter(partial, lse, output, handle):
    pass


class PreparedLseReduceScatter:
    """Combine every rank's partial attention by natural-log LSE and keep this rank's heads.

    Partial outputs are [rows, heads, dim] over all heads with FP32 [rows, heads] LSE;
    the output holds this rank's contiguous head shard. Operations sharing the
    channel must execute serially on every rank.
    """

    def __init__(self, partial, lse, output, *, runtime, session, coordinator, name):
        import torch.distributed as dist
        from b12x.comm import pcie

        rows = partial.shape[0]
        if (partial.shape != (rows, runtime.total_heads, runtime.query_head_dim) or partial.dtype != torch.bfloat16
                or lse.shape != (rows, runtime.total_heads) or lse.dtype != torch.float32
                or output.shape != (rows, runtime.heads_per_rank, runtime.query_head_dim)
                or output.dtype != torch.bfloat16 or not 0 < rows <= runtime.max_batch_size):
            raise ValueError("LSE reduce-scatter requires BF16 partials, FP32 LSE and a head shard of its channel")
        self.runtime, self.output = runtime, output
        call = {"partial_output": partial, "partial_lse": lse, "out": output, "is_lse_base_on_e": True}
        query = pcie.query_from_runtime(runtime, surface="DcpAllToAll.lse_reduce_scatter", call=call)
        self.plan = pcie.plan(query, runtime=runtime)
        ranks = tuple(dist.get_process_group_ranks(runtime.exchange_group))
        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: PreparedCall(run=lambda: state.run(**call), output=output,
                                                    owners=(runtime, partial, lse, output)),
            collective=CollectiveRequirement(key=name, ranks=ranks)),), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def __call__(self, partial, lse):
        _lse_reduce_scatter(partial, lse, self.output, self.handle)
        return self.output
