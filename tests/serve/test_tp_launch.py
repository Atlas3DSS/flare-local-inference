"""Tests for TP launch-side health monitoring helpers."""

from __future__ import annotations

from serve.tp.launch import TPFollowerMonitor, _cleanup_followers


class _FakeProcess:
    def __init__(self, *, pid: int, alive: bool, exitcode: int | None):
        self.pid = pid
        self._alive = alive
        self.exitcode = exitcode
        self.kill_calls = 0
        self.join_calls = 0

    def is_alive(self) -> bool:
        return self._alive

    def kill(self) -> None:
        self.kill_calls += 1
        self._alive = False

    def join(self, timeout: float | None = None) -> None:
        del timeout
        self.join_calls += 1


def test_tp_health_marks_clean_early_exit_as_fatal():
    monitor = TPFollowerMonitor(
        world_size=2,
        gpu_ids=[0, 1],
        followers=[_FakeProcess(pid=1234, alive=False, exitcode=0)],
    )

    health = monitor.health()
    assert health["fatal"] is True
    assert "exited unexpectedly" in health["summary"]


def test_cleanup_followers_kills_alive_processes_and_joins_all():
    alive = _FakeProcess(pid=1234, alive=True, exitcode=None)
    dead = _FakeProcess(pid=5678, alive=False, exitcode=0)

    _cleanup_followers([alive, dead])

    assert alive.kill_calls == 1
    assert alive.join_calls == 1
    assert dead.kill_calls == 0
    assert dead.join_calls == 1
