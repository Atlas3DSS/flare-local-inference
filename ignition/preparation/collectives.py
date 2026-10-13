"""Bootstrap coordination of b12x collectives over an established Gloo group."""

from datetime import timedelta
from pathlib import Path

import torch.distributed as dist


def host_identity():
    """This boot of this host; ranks sharing it can map each other's memory through CUDA IPC."""
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def create_roce_runtime(group, device, *, max_tokens, hidden_size, gather_bytes):
    """Allocate RoCEnante for ranks on separate DGX Sparks: BF16 hidden reductions and logit gathers.

    One runtime serves every reduction and gather in rank-serial order. Its pinned
    region holds one slot per peer and transfer, sized for the largest operation.
    """
    from b12x.comm import roce

    if any(type(value) is not int or value <= 0 for value in (max_tokens, hidden_size, gather_bytes)):
        raise ValueError("RoCE capacities must be positive integers")
    nbytes = max_tokens * hidden_size * 2
    if nbytes % 16 or gather_bytes % 16:
        raise ValueError("RoCE capacities must occupy complete 16-byte packs")
    return roce.AllReduce(exchange_group=group, device=device, max_size=nbytes, max_gather_bytes=gather_bytes)


def create_all_reduce_runtime(group, device, *, max_tokens, hidden_size, ipc=None):
    """Allocate one b12x IPC channel for the largest BF16 hidden activation.

    Every rank supplies identical capacities before preparing graph variants.
    The bootstrap retains the runtime until every native rank worker drains.
    """
    from b12x.comm.pcie import AllReduce

    if any(type(value) is not int or value <= 0 for value in (max_tokens, hidden_size)):
        raise ValueError("all-reduce capacities must be positive integers")
    nbytes = max_tokens * hidden_size * 2
    if nbytes % 16:
        raise ValueError("BF16 all-reduce capacity must occupy complete 16-byte packs")
    return AllReduce.from_exchange_group(exchange_group=group, device=device, single_channel=True,
                                         max_size=nbytes, eager_buffer_bytes=nbytes, ipc=ipc)


def create_all_gather_runtime(group, device, *, max_rows, columns, ipc=None):
    """Allocate b12x staging for serial BF16 column gathers in prepared graphs."""
    from b12x.comm.pcie import DcpAllToAll

    world_size = dist.get_world_size(group)
    if (world_size not in (2, 4) or type(max_rows) is not int or max_rows <= 0
            or type(columns) is not int or columns <= 0 or columns % (world_size * 64)):
        raise ValueError("column gathers require positive rows and 64-column-aligned TP2 or TP4 partitions")
    return DcpAllToAll.from_exchange_group(exchange_group=group, device=device,
        max_batch_size=max_rows, total_heads=columns // 64, head_dim=64, stream_affine=False, ipc=ipc)


def create_dma_all_reduce_runtime(group, device, *, max_tokens, hidden_size, ipc=None):
    """Allocate a b12x copy-engine ring for prefill-sized BF16 reductions in caller-owned shared storage."""
    from b12x.comm.pcie import DmaAllReduce

    if any(type(value) is not int or value <= 0 for value in (max_tokens, hidden_size)):
        raise ValueError("all-reduce capacities must be positive integers")
    return DmaAllReduce(exchange_group=group, device=device, max_bytes=max_tokens * hidden_size * 2, ipc=ipc)


def create_head_exchange(group, device, *, max_rows, heads, head_dim, ipc=None):
    """Allocate b12x staging for all-gathering head shards and LSE-combining partial attention."""
    from b12x.comm.pcie import DcpAllToAll

    world_size = dist.get_world_size(group)
    if world_size not in (2, 4) or max_rows <= 0 or heads % world_size or head_dim % 8:
        raise ValueError("head exchanges require TP2 or TP4 head shards and positive capacity")
    return DcpAllToAll.from_exchange_group(exchange_group=group, device=device,
        max_batch_size=max_rows, total_heads=heads, head_dim=head_dim, stream_affine=False, ipc=ipc)


class CollectivePreparation:
    """Authorize matching collective declarations and synchronize kernel entry.

    Local compilation may complete at different times on each rank. This
    coordinator releases a collective only after all group members report the
    same declaration. It supports whole-group operations without distributed
    candidate sharding and is unused after graph handoff.
    """

    def __init__(self, group, *, timeout=timedelta(minutes=2)):
        if dist.get_backend(group) != "gloo":
            raise ValueError("collective preparation requires a Gloo control group")
        self.group = group
        self.ranks = tuple(dist.get_process_group_ranks(group))
        self.timeout = timeout
        hosts = [None] * len(self.ranks)
        dist.all_gather_object(hosts, host_identity(), group=group)
        # Each rank's host, in group order; ranks on one host share its CPUs and CUDA IPC.
        self.hosts = tuple(hosts)
        self.local_ranks = self.hosts.count(host_identity())

    @property
    def spans_hosts(self):
        return len(set(self.hosts)) > 1

    def __call__(self, progress):
        if progress.ready_cache is not None or progress.ready_tuning:
            raise ValueError("collective preparation does not shard tuning candidates")
        pending = tuple((item.key, item.ranks) for item in progress.ready_collectives)
        states = [None] * len(self.ranks)
        dist.all_gather_object(states, pending, group=self.group)
        if not all(states):
            return None
        if any(state != states[0] for state in states[1:]):
            raise RuntimeError(f"collective preparation declarations differ across ranks: {states}")
        key, ranks = states[0][0]
        if tuple(ranks) != self.ranks:
            raise ValueError("collective preparation requires the complete control group")
        return key

    def prepare(self, session, requests, *, progress=None, autotune=None):
        """Drive a mixed local/collective job until every rank has finished.

        A finished rank must still exchange readiness while its peers compile
        or finish local work. Returning on local completion can strand peers in
        their next control-group exchange.
        """
        job = None
        state = None
        error = None
        key = None
        try:
            while True:
                try:
                    if job is None:
                        job = session.begin(requests, autotune=autotune)
                    if state is None or not state.done:
                        state = job.advance(collective_key=key)
                        if progress is not None:
                            progress(state)
                    if state.ready_cache is not None or state.ready_tuning:
                        raise ValueError("collective preparation does not shard tuning candidates")
                except Exception as exc:  # noqa: BLE001 - Every rank must receive preparation failures.
                    error = exc
                pending = () if state is None else tuple((item.key, item.ranks) for item in state.ready_collectives)
                local = (state is not None and state.done, pending,
                         None if error is None else f"{type(error).__name__}: {error}")
                states = [None] * len(self.ranks)
                dist.all_gather_object(states, local, group=self.group)
                failures = [(rank, item[2]) for rank, item in zip(self.ranks, states, strict=True) if item[2] is not None]
                if failures:
                    raise RuntimeError(f"rank preparation failed: {failures}") from error
                if all(item[0] for item in states):
                    return job.result()
                if any(item[0] for item in states) and any(item[1] for item in states):
                    raise RuntimeError("a rank finished before its peer's collective declaration")
                ready = [item[1] for item in states]
                key = None
                if all(ready):
                    if any(item != ready[0] for item in ready[1:]):
                        raise RuntimeError(f"collective preparation declarations differ across ranks: {ready}")
                    key, ranks = ready[0][0]
                    if tuple(ranks) != self.ranks:
                        raise ValueError("collective preparation requires the complete control group")
        except BaseException as failure:
            try:
                if job is not None:
                    if state is not None and state.done:
                        job.result().close()
                    else:
                        job.close()
            except BaseException as cleanup:  # noqa: BLE001 - Preserve the distributed failure.
                failure.add_note(f"preparation cleanup failed: {cleanup!r}")
            raise

    def barrier(self, key, ranks):
        if tuple(ranks) != self.ranks:
            raise ValueError(f"collective {key!r} does not span its control group")
        dist.monitored_barrier(group=self.group, timeout=self.timeout, wait_all_ranks=True)
