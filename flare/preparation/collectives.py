"""Bootstrap coordination of b12x collectives over an established Gloo group."""

from datetime import timedelta

import torch.distributed as dist


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

    def barrier(self, key, ranks):
        if tuple(ranks) != self.ranks:
            raise ValueError(f"collective {key!r} does not span its control group")
        dist.monitored_barrier(group=self.group, timeout=self.timeout, wait_all_ranks=True)
