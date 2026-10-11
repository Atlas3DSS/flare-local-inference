"""Process-level checks for native rank failure containment and parent lifetime."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("flare_heliopause")

DRIVER = r'''
import json, os, subprocess, sys, time
import flare_heliopause
child = "import flare_heliopause,sys; flare_heliopause.bind_parent_lifetime(int(sys.argv[1])); print('ready',flush=True); sys.stdin.read(1)"
peers = [subprocess.Popen([sys.executable, '-c', child, str(os.getpid())], stdin=subprocess.PIPE, stdout=subprocess.PIPE) for _ in range(2)]
for peer in peers:
    assert peer.stdout.readline() == b'ready\n'
supervisor = flare_heliopause.RankSupervisor([peer.pid for peer in peers])
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
    import flare_heliopause

    with pytest.raises(ValueError):
        flare_heliopause.RankSupervisor([os.getpid()])


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_cli_signals_exit_during_native_gil_hold_without_python_finalization(signum):
    code = r'''
import atexit, ctypes, time
import flare_heliopause
atexit.register(lambda: print("finalizer ran", flush=True))
with flare_heliopause.CliSignals():
    print("ready", flush=True)
    ctypes.PyDLL(None).sleep(60)
'''
    process = subprocess.Popen([sys.executable, "-u", "-c", code],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        assert process.stdout.readline() == b"ready\n"
        started = time.monotonic()
        process.send_signal(signum)
        output, error = process.communicate(timeout=2)
        assert process.returncode == 128 + signum
        assert time.monotonic() - started < 1
        assert output == error == b""
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def test_cli_signal_scope_restores_embedded_python_handlers():
    import flare_heliopause

    observed = []
    previous = signal.signal(signal.SIGTERM, lambda signum, frame: observed.append(signum))
    try:
        with flare_heliopause.CliSignals():
            with pytest.raises(RuntimeError, match="already installed"):
                flare_heliopause.CliSignals()
        os.kill(os.getpid(), signal.SIGTERM)
        assert observed == [signal.SIGTERM]
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_cli_interrupt_leaves_no_compiler_processes():
    code = r'''
import json, os, time
from pathlib import Path
import flare_heliopause
from torch._inductor.compile_worker.subproc_pool import SubprocPool
with flare_heliopause.CliSignals():
    pool = SubprocPool(1)
    pending = pool.submit(time.sleep, 60)
    coordinator = pool.process.pid
    children = Path(f"/proc/{coordinator}/task/{coordinator}/children")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        workers = [int(pid) for pid in children.read_text().split()]
        if workers:
            break
        time.sleep(.01)
    else:
        raise RuntimeError("compiler worker did not start")
    print(json.dumps([coordinator, *workers]), flush=True)
    pending.result()
'''
    process = subprocess.Popen([sys.executable, "-u", "-c", code], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    peers = []
    try:
        line = process.stdout.readline()
        assert line, process.stderr.read()
        peers = json.loads(line)
        process.send_signal(signal.SIGINT)
        process.communicate(timeout=3)
        assert process.returncode == 130
        deadline = time.monotonic() + 5
        while not all(terminated(pid) for pid in peers) and time.monotonic() < deadline:
            time.sleep(.01)
        assert all(terminated(pid) for pid in peers)
    finally:
        if process.poll() is None or not all(terminated(pid) for pid in peers):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
