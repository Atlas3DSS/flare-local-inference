"""Prepared b12x PCIe reductions embedded in compiled CUDA graphs."""

import itertools
import weakref

import torch
from b12x.preparation import CollectiveRequirement, PreparedCall

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
        session.prepare((request,), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def __call__(self, source):
        _all_reduce(source, self.output, self.handle)
        return self.output
