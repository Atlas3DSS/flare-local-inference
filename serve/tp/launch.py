"""Multi-process TP launch.

Uses mp.Process (not mp.spawn) for direct process lifecycle control.
Parent process manages worker lifetimes and handles cleanup.
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime
import os
import signal
import sys
import tempfile
from typing import Optional

from serve.runtime_warnings import import_torch_safely

torch = import_torch_safely()
import torch.distributed as dist
import torch.multiprocessing as mp

from serve.tp.group import TPGroup

_ACTIVE_TP_MONITOR = None


def _cleanup_followers(followers: list[mp.Process]) -> None:
    for p in followers:
        if p.is_alive():
            p.kill()
    for p in followers:
        p.join(timeout=5)


@dataclass
class TPFollowerMonitor:
    world_size: int
    gpu_ids: list[int]
    followers: list[mp.Process]

    def health(self) -> dict:
        workers = []
        fatal = False
        details = []
        for rank, proc in enumerate(self.followers, start=1):
            exitcode = proc.exitcode
            alive = proc.is_alive()
            worker = {
                "rank": rank,
                "gpu_id": self.gpu_ids[rank],
                "pid": proc.pid,
                "alive": alive,
                "exitcode": exitcode,
                "log_path": _rank_log_path(rank),
            }
            if exitcode is not None:
                worker["error"] = f"rank {rank} exited unexpectedly with code {exitcode}"
                details.append(worker["error"])
                fatal = True
            workers.append(worker)
        return {
            "world_size": self.world_size,
            "healthy": not fatal,
            "fatal": fatal,
            "summary": "; ".join(details) if details else None,
            "workers": workers,
        }


def _find_free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _rank_log_path(rank: int) -> str:
    log_dir = os.environ.get("B12X_TP_LOG_DIR", tempfile.gettempdir())
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"b12x-tp-rank{rank}.log")


def _set_active_tp_monitor(monitor) -> None:
    global _ACTIVE_TP_MONITOR
    _ACTIVE_TP_MONITOR = monitor


def _clear_active_tp_monitor() -> None:
    global _ACTIVE_TP_MONITOR
    _ACTIVE_TP_MONITOR = None


def get_active_tp_health() -> dict | None:
    if _ACTIVE_TP_MONITOR is None:
        return None
    return _ACTIVE_TP_MONITOR.health()


def launch_tp(
    fn,
    world_size: int,
    args: tuple = (),
    gpu_ids: list[int] | None = None,
):
    """Launch *world_size* processes and run *fn(tp_group, *args)* in each.

    If world_size == 1, runs in the current process (no distributed).
    Rank 0 runs in the current process to preserve stdin (interactive mode).
    Followers are spawned as daemon processes.
    """
    if gpu_ids is None:
        gpu_ids = list(range(world_size))
    assert len(gpu_ids) == world_size

    if world_size == 1:
        torch.cuda.set_device(gpu_ids[0])
        fn(None, *args)
        return

    port = _find_free_port()
    mp.set_start_method("spawn", force=True)

    # Spawn followers (ranks 1..N-1).
    followers = []
    for rank in range(1, world_size):
        p = mp.Process(
            target=_worker,
            args=(rank, world_size, gpu_ids, port, fn, args),
            daemon=True,
        )
        p.start()
        followers.append(p)
    _set_active_tp_monitor(TPFollowerMonitor(world_size=world_size, gpu_ids=gpu_ids, followers=followers))

    previous_handlers: dict[int, object] = {}

    def _handle_parent_signal(signum, _frame):
        _cleanup_followers(followers)
        _clear_active_tp_monitor()
        raise KeyboardInterrupt(f"launch_tp interrupted by signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _handle_parent_signal)

    # Run rank 0 in the current process (preserves stdin for interactive mode).
    try:
        _worker(0, world_size, gpu_ids, port, fn, args)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        _clear_active_tp_monitor()
        _cleanup_followers(followers)


def _worker(rank, world_size, gpu_ids, port, fn, args):
    """Per-process worker."""
    device_id = gpu_ids[rank]

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    # Redirect follower logs to rank-specific files so failures are debuggable.
    if rank != 0:
        import warnings, logging
        log_path = _rank_log_path(rank)
        log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        os.close(log_fd)
        sys.stdout = open(log_path, "a")
        sys.stderr = open(log_path, "a")
        warnings.filterwarnings("ignore")
        logging.disable(logging.CRITICAL)

    torch.cuda.set_device(device_id)
    torch.set_grad_enabled(False)
    # Long timeout for first-pass kernel compilation (Triton JIT + CuTe DSL).
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=30),
    )

    tp_group = TPGroup(
        rank=rank,
        world_size=world_size,
        device=torch.device("cuda", device_id),
        process_group=dist.group.WORLD,
    )

    try:
        fn(tp_group, *args)
    finally:
        if dist.is_initialized():
            # Flush any pending NCCL work and let all ranks reach teardown
            # before destroying the shared process group.
            try:
                torch.cuda.synchronize(device_id)
            except Exception:
                pass
            try:
                dist.barrier(device_ids=[device_id])
            except Exception:
                pass
        TPGroup.destroy()
