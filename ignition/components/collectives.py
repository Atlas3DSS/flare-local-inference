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

        if (runtime.world_size not in (2, 4) or runtime.algorithm != "oneshot"
                or not runtime.single_channel):
            raise ValueError("prepared PCIe reductions require single-channel b12x TP2 or TP4 all-reduce")
        self.runtime, self.output = runtime, output
        self.plan = runtime.plan(source, out=output)

        def prepare(state):
            state.bind_input(source)
            return PreparedCall(run=lambda: state.run_plain(source, output),
                                output=output, owners=(runtime, source, output))

        ranks = tuple(dist.get_process_group_ranks(runtime.exchange_group))
        request = self.plan.request(name=name, prepare_call=prepare,
            collective=CollectiveRequirement(key=name, ranks=ranks))
        declare(session, (request,), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def __call__(self, source):
        _all_reduce(source, self.output, self.handle)
        return self.output


@torch.library.custom_op("flare::pcie_all_gather", mutates_args=("output",))
def _all_gather(source: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    owner = _OWNERS[handle]
    runtime = owner.runtime
    runtime.all_gather_heads(
        source.view(source.shape[0], runtime.heads_per_rank, runtime.query_head_dim),
        out=output.view(output.shape[0], runtime.total_heads, runtime.query_head_dim),
        plan=owner.plan)


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
