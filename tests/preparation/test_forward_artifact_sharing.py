"""Ranks compile each forward variant once, and ranks on separate hosts receive the others by path."""

import multiprocessing
import os
from pathlib import Path


def _rank(rank, world, store, home, owned, outcome):
    import torch.distributed as dist

    from ignition.preparation.aot import _share_artifacts

    # Each rank's working directory stands in for its host's disk; artifact paths are identical.
    os.chdir(home)
    try:
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
        _share_artifacts(owned, dist.group.WORLD, tuple(range(world)))
        outcome.put((rank, {path: Path(path).read_bytes() for path in ("forwards/a.bin", "forwards/b.bin")}))
        dist.destroy_process_group()
    except Exception as error:  # noqa: BLE001 - The parent reports every rank's failure.
        outcome.put((rank, repr(error)))


def test_each_variant_compiles_on_the_least_loaded_rank_that_binds_it():
    from ignition.preparation.aot import _variant_compilers

    shared = [{name: "shared" for name in "abcdef"}] * 4
    assert list(_variant_compilers(shared).values()) == [0, 1, 2, 3, 0, 1]
    # Rank zero samples; its peers consume, so neither variant compiles on a rank that cannot bind it.
    sampled = [{"decode": "sample", "resume": "sample"}] + [{"decode": "consume", "resume": "consume"}] * 3
    assert _variant_compilers(sampled) == {("decode", "sample"): 0, ("decode", "consume"): 1,
                                           ("resume", "sample"): 0, ("resume", "consume"): 2}


def test_every_rank_receives_what_it_lacks_from_the_rank_that_compiled_it(tmp_path):
    context = multiprocessing.get_context("fork")
    homes = [tmp_path / f"host{rank}" for rank in range(3)]
    for home in homes:
        (home / "forwards").mkdir(parents=True)
    (homes[0] / "forwards/a.bin").write_bytes(b"compiled a")
    (homes[1] / "forwards/b.bin").write_bytes(b"compiled b")
    # Rank two compiled nothing, already holds a, and holds a stale b.
    (homes[2] / "forwards/a.bin").write_bytes(b"compiled a")
    (homes[2] / "forwards/b.bin").write_bytes(b"stale b")
    owned = [["forwards/a.bin"], ["forwards/b.bin"], []]
    outcome = context.Queue()
    processes = [context.Process(target=_rank, args=(rank, 3, tmp_path / "store", homes[rank], owned, outcome))
                 for rank in range(3)]
    for process in processes:
        process.start()
    results = dict(outcome.get(timeout=60) for _ in processes)
    for process in processes:
        process.join(timeout=60)
    expected = {"forwards/a.bin": b"compiled a", "forwards/b.bin": b"compiled b"}
    assert results == {rank: expected for rank in range(3)}
