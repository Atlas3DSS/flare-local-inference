"""Prepared b12x reductions and gathers embedded in compiled CUDA graphs.

Ranks on one host communicate through b12x PCIe runtimes; ranks on separate
DGX Sparks through RoCEnante (b12x.comm.roce), whose one-shot all-reduce and
all-gather are captured in the same graphs. A component given no runtime serves
one rank: the reduction or gather is the rank's own values, copied into the
output its consumers read.
"""

import itertools
import weakref

import torch
from b12x.preparation import CollectiveRequirement, PreparedCall

from ignition.preparation.tuning import declare

_OWNERS = weakref.WeakValueDictionary()
_HANDLES = itertools.count()


def _roce(runtime):
    """True for a RoCEnante runtime joining ranks on separate hosts."""
    from b12x.comm import roce

    return isinstance(runtime, roce.AllReduce)


def _roce_plan(runtime, coordinator, surface):
    """One fixed RoCEnante plan; every operation's geometry is validated by the runtime at launch."""
    from b12x.comm import roce

    query = roce.query_from_runtime(runtime, surface=surface, call={"dtypes": ("bfloat16",)},
                                    topology="roce_rdma", peer_hosts=coordinator.hosts)
    return roce.plan(query, runtime=runtime)


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
        # Traced forwards test this flag, never the runtime: compile workers see a runtime that
        # owns RDMA or IPC resources only as a placeholder.
        self.single = runtime is None
        if runtime is None:
            if source.shape != output.shape or source.dtype != output.dtype:
                raise ValueError("a single-rank reduction copies its source into a matching output")
            return
        if _roce(runtime):
            if (source.shape != output.shape or source.dtype != torch.bfloat16 or output.dtype != source.dtype
                    or not source.is_contiguous() or not output.is_contiguous()
                    or not 0 < source.numel() * 2 <= max(runtime.max_size, runtime.pipe_bytes)
                    or source.numel() * 2 % 16):
                raise ValueError("RoCE reductions require contiguous BF16 16-byte packs within the runtime's capacity")
            self.plan = _roce_plan(runtime, coordinator, "AllReduce.all_reduce")
            group = runtime._group

            def prepare(state):
                return PreparedCall(run=lambda: state.all_reduce(source, out=output), output=output,
                                    owners=(runtime, source, output))
        elif isinstance(runtime, pcie.DmaAllReduce):
            # Copy-engine ring for prefill-sized activations.
            from b12x.comm.pcie._dma_preparation import query_from_metadata

            self.plan = pcie.plan(query_from_metadata(runtime, shape=tuple(source.shape), dtype=source.dtype),
                                  runtime=runtime)
            group = runtime.group

            def prepare(state):
                return PreparedCall(run=lambda: state.run(source, out=output), output=output,
                                    owners=(runtime, source, output))
        elif runtime.algorithm == "hierarchical":
            from b12x.comm.pcie._owner_preparation import prepared_call

            self.plan = runtime.plan(source, out=output)
            group = runtime.for_stream().group

            def prepare(state):
                return prepared_call(state, inp=source, out=output)
        else:
            if (runtime.world_size < 2 or runtime.algorithm != "oneshot"
                    or not runtime.single_channel):
                raise ValueError("prepared PCIe reductions require single-channel b12x direct all-reduce")
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
        if self.single:
            self.output.copy_(source)
            return self.output
        _all_reduce(source, self.output, self.handle)
        return self.output


@torch.library.custom_op("flare::pcie_all_reduce_rmsnorm", mutates_args=("residual", "output"))
def _all_reduce_rmsnorm(source: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                       output: torch.Tensor, handle: int) -> None:
    owner = _OWNERS[handle]
    owner.runtime.all_reduce_fused_add_rms_norm(source, residual, weight, owner.eps, plan=owner.plan,
        out=output, residual_out=residual, zero_centered=owner.zero_centered, reduction_dtype=owner.reduction_dtype)


@_all_reduce_rmsnorm.register_fake
def _fake_all_reduce_rmsnorm(source, residual, weight, output, handle):
    pass


class PreparedAllReduceRMSNorm:
    """One-shot collective with an in-place residual update and independent output dtype."""

    @staticmethod
    def fits(runtime, source):
        from b12x.comm import pcie

        return (isinstance(runtime, pcie.AllReduce) and runtime.algorithm == "oneshot"
                and runtime.single_channel and source.ndim == 2 and 0 < source.shape[0] <= 36)

    def __init__(self, source, residual, weight, output, *, eps, runtime, session, coordinator, name,
                 zero_centered=False, reduction_dtype=None):
        import torch.distributed as dist
        from b12x.comm.pcie._oneshot_preparation import _prepare_fused_call

        if not self.fits(runtime, source):
            raise ValueError("fused all-reduce RMSNorm requires a single-channel one-shot runtime and at most 36 rows")
        self.runtime, self.residual, self.weight, self.output = runtime, residual, weight, output
        self.eps, self.zero_centered, self.reduction_dtype = eps, zero_centered, reduction_dtype
        self.plan = runtime.plan(source, operation="all_reduce_fused_add_rms_norm", residual=residual,
            weight=weight, out=output, residual_out=residual, zero_centered=zero_centered, reduction_dtype=reduction_dtype)
        ranks = tuple(dist.get_process_group_ranks(runtime.exchange_group))
        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: _prepare_fused_call(state, inp=source, residual=residual, weight=weight,
                out=output, residual_out=residual, epsilon=eps, owners=(runtime, source, residual, weight, output)),
            collective=CollectiveRequirement(key=name, ranks=ranks)),), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def __call__(self, source, residual):
        _all_reduce_rmsnorm(source, residual, self.weight, self.output, self.handle)
        return self.output


def _gather_columns(source, output, handle):
    owner = _OWNERS[handle]
    runtime = owner.runtime
    if owner.roce:
        runtime.all_gather(source, dim=-1, out=output, plan=owner.plan)
        return
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

        self.runtime, self.output = runtime, output
        # Traced forwards test this flag, never the runtime: compile workers see a runtime that
        # owns RDMA or IPC resources only as a placeholder.
        self.single = runtime is None
        if runtime is None:
            if source.shape != output.shape or source.dtype != output.dtype:
                raise ValueError("a single-rank gather copies its source into a matching output")
            return
        self.roce = _roce(runtime)
        if self.roce:
            if (source.ndim != 2 or source.dtype != torch.bfloat16 or output.dtype != source.dtype
                    or not source.is_contiguous() or not output.is_contiguous()
                    or output.shape != (source.shape[0], source.shape[1] * runtime.world_size)
                    or source.shape[1] * 2 % 16 or not 0 < source.numel() * 2 <= runtime.max_gather_bytes):
                raise ValueError("RoCE gathers require contiguous BF16 column partitions within the runtime's capacity")
            self.plan = _roce_plan(runtime, coordinator, "AllReduce.all_gather")
            ranks = tuple(dist.get_process_group_ranks(runtime._group))
            # Priming runs before the plan is published, so it launches the materialized gather directly.
            declare(session, (self.plan.request(name=name,
                prepare_call=lambda state: PreparedCall(
                    run=lambda: runtime._run_prepared_all_gather(source, prepared=state, dim=-1, out=output),
                    output=output, owners=(runtime, source, output)),
                collective=CollectiveRequirement(key=name, ranks=ranks)),), coordinator=coordinator)
            self.handle = next(_HANDLES)
            _OWNERS[self.handle] = self
            return
        if (runtime.world_size < 2 or source.ndim != 2
                or source.dtype != torch.bfloat16 or output.dtype != source.dtype
                or source.device != runtime.device or output.device != source.device
                or not source.is_contiguous() or not output.is_contiguous()
                or not 0 < source.shape[0] <= runtime.max_batch_size
                or source.shape[1] != runtime.heads_per_rank * runtime.query_head_dim
                or output.shape != (source.shape[0], source.shape[1] * runtime.world_size)):
            raise ValueError("all-gather requires contiguous BF16 column partitions matching its collective channel")
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
        if self.single:
            self.output.copy_(source)
            return self.output
        _all_gather(source, self.output, self.handle)
        return self.output

    def run_eager(self, source):
        """Gather on the current stream from inside another eager operation."""
        if self.single:
            self.output.copy_(source)
            return self.output
        _gather_columns(source, self.output, self.handle)
        return self.output


@torch.library.custom_op("flare::roce_all_gather_varlen", mutates_args=("output",))
def _all_gather_varlen(counts: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    owner = _OWNERS[handle]
    owner.runtime.all_gather_varlen(owner.source, counts, out=output, plan=owner.plan)


@_all_gather_varlen.register_fake
def _fake_all_gather_varlen(counts, output, handle):
    pass


class PreparedVarlenGather:
    """Gather byte prefixes of every rank's row of `output` [world, capacity] to every rank over RoCEnante.

    Each rank writes its contribution into its own row in place; `counts` is a [world] int64 device
    tensor of each rank's bytes, a multiple of 16, holding the same values on every rank and free to
    change between replays. Row r then holds rank r's first counts[r] bytes on every rank; bytes
    past them keep their values. Operations sharing the runtime must execute serially on every rank.
    """

    def __init__(self, output, counts, *, runtime, session, coordinator, name):
        import torch.distributed as dist

        if not _roce(runtime):
            raise ValueError("variable-length gathers run over RoCEnante")
        ranks = tuple(dist.get_process_group_ranks(runtime._group))
        if (output.shape[0] != len(ranks) or not output.is_contiguous() or counts.shape != (len(ranks),)
                or counts.dtype != torch.int64 or output[0].numel() * output.element_size() % 16):
            raise ValueError("variable-length gathers require one contiguous 16-byte-multiple row per rank")
        self.runtime, self.output, self.counts = runtime, output, counts
        # A rank's own row is its source, so the gather leaves it in place.
        self.source = output[ranks.index(dist.get_rank())]
        self.plan = _roce_plan(runtime, coordinator, "AllReduce.all_gather_varlen")
        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: PreparedCall(
                run=lambda: runtime._run_prepared_all_gather_varlen(self.source, counts, prepared=state, out=output),
                output=output, owners=(runtime, self.source, counts, output)),
            collective=CollectiveRequirement(key=name, ranks=ranks)),), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def __call__(self):
        _all_gather_varlen(self.counts, self.output, self.handle)
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
    channel must execute serially on every rank. Over RoCEnante, ranks all-gather
    the shards each peer keeps through `staging` (send [rows, width] and received
    [rows, W * width] BF16, from head_exchange.pack_width) and merge locally;
    `rank` is this rank as a one-element device tensor.
    """

    def __init__(self, partial, lse, output, *, runtime, session, coordinator, name, staging=None, rank=None):
        import torch.distributed as dist
        from b12x.comm import pcie

        rows = partial.shape[0]
        self.roce = _roce(runtime)
        if self.roce:
            from .head_exchange import pack_width

            heads, dim = partial.shape[1:]
            world = runtime.world_size
            width = pack_width(world, heads, dim)
            send, received = (None, None) if staging is None else staging
            if (rank is None or send is None or heads % world or partial.dtype != torch.bfloat16
                    or lse.shape != (rows, heads) or lse.dtype != torch.float32
                    or output.shape != (rows, heads // world, dim) or output.dtype != torch.bfloat16
                    or send.shape[1:] != (width,) or received.shape[1:] != (world * width,)
                    or min(send.shape[0], received.shape[0]) < rows):
                raise ValueError("RoCE head combination requires BF16 partials, FP32 LSE, staging and the rank")
            self.output, self.rank, self.world = output, rank, world
            self.send, self.received = send[:rows], received[:rows]
            self.gather = PreparedAllGather(self.send, self.received, runtime=runtime, session=session,
                                            coordinator=coordinator, name=f"{name}.gather")
            return
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
        if self.roce:
            from .head_exchange import merge, pack

            pack(partial, lse, self.send, self.rank, self.world)
            self.gather(self.send)
            merge(self.received, partial, lse, self.output, self.rank, self.world)
            return self.output
        _lse_reduce_scatter(partial, lse, self.output, self.handle)
        return self.output
