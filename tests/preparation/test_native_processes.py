"""Process-level checks for native rank failure containment and parent lifetime."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("flare_helios")

DRIVER = r'''
import json, os, subprocess, sys, time
import flare_helios
child = "import flare_helios,sys; flare_helios.bind_parent_lifetime(int(sys.argv[1])); print('ready',flush=True); sys.stdin.read(1)"
peers = [subprocess.Popen([sys.executable, '-c', child, str(os.getpid())], stdin=subprocess.PIPE, stdout=subprocess.PIPE) for _ in range(2)]
for peer in peers:
    assert peer.stdout.readline() == b'ready\n'
supervisor = flare_helios.RankSupervisor([peer.pid for peer in peers])
print(json.dumps([peer.pid for peer in peers]), flush=True)
mode = sys.stdin.readline().strip()
if mode == 'normal':
    supervisor.begin_shutdown(5000)
    for peer in peers:
        peer.stdin.write(b'q'); peer.stdin.flush()
    supervisor.finish_shutdown()
    assert all(peer.wait() == 0 for peer in peers)
elif mode == 'crash':
    peers[0].kill()
    time.sleep(60)
elif mode == 'deadline':
    supervisor.begin_shutdown(100)
    supervisor.finish_shutdown()
elif mode == 'orphan':
    os._exit(0)
'''


def terminated(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].startswith("Z ")
    except (FileNotFoundError, ProcessLookupError):
        return True


@pytest.mark.parametrize("mode", ["normal", "crash", "deadline", "orphan"])
def test_rank_group_shutdown_and_failure_do_not_leave_live_peers(mode):
    driver = subprocess.Popen([sys.executable, "-u", "-c", DRIVER], text=True,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    peers = []
    try:
        peers = json.loads(driver.stdout.readline())
        _, error = driver.communicate(mode + "\n", timeout=10)
        assert driver.returncode == (0 if mode in ("normal", "orphan") else 1), error
        if mode == "crash":
            assert "exited before coordinated shutdown" in error
        if mode == "deadline":
            assert "shutdown deadline expired" in error
        deadline = time.monotonic() + 3
        while not all(terminated(pid) for pid in peers) and time.monotonic() < deadline:
            time.sleep(.01)
        assert all(terminated(pid) for pid in peers)
    finally:
        if driver.poll() is None or not all(terminated(pid) for pid in peers):
            os.killpg(driver.pid, signal.SIGKILL)
        driver.wait(timeout=5)


def test_supervisor_rejects_unowned_process():
    import flare_helios

    with pytest.raises(ValueError):
        flare_helios.RankSupervisor([os.getpid()])
