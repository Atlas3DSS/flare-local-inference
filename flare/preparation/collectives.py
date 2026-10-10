"""Bootstrap coordination of b12x collectives over an established Gloo group."""

from datetime import timedelta

import torch.distributed as dist


def create_all_reduce_runtime(group, device, *, max_tokens, hidden_size):
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
                                         max_size=nbytes, eager_buffer_bytes=nbytes)


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
