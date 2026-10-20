"""Sample an experiment unit, whole-host interfaces, memory, CPU, and NVMe counters."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

unit, seconds, interval = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
root = Path(sys.argv[4])
interfaces = [p for p in Path('/sys/class/net').iterdir() if p.name.startswith('en')]
started = time.monotonic()
while time.monotonic() - started <= seconds:
    state = dict(line.split('=', 1) for line in subprocess.check_output(
        ['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '-p', 'MainPID',
         '-p', 'ControlGroup', '-p', 'ExecMainStatus'], text=True).splitlines())
    row = {'time': time.time(), 'elapsed': time.monotonic() - started, 'unit': state,
           'clock_ticks': os.sysconf('SC_CLK_TCK'), 'page_size': os.sysconf('SC_PAGE_SIZE')}
    row['network'] = {p.name: {k: int((p/'statistics'/k).read_text())
                      for k in ('rx_bytes', 'tx_bytes', 'rx_dropped', 'tx_dropped')}
                      for p in interfaces}
    row['cpu'] = [int(n) for n in Path('/proc/stat').read_text().splitlines()[0].split()[1:]]
    row['nvme'] = [int(n) for n in Path('/sys/block/nvme0n1/stat').read_text().split()]
    row['memory_kib'] = {line.split(':')[0]: int(line.split()[1])
                         for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.split(':')[0] in ('MemAvailable', 'Cached', 'Dirty', 'Writeback')}
    disk = os.statvfs(root)
    row['disk_available_bytes'] = disk.f_bavail * disk.f_frsize
    row['processes'] = {}
    try:
        pids = Path('/sys/fs/cgroup', state['ControlGroup'].lstrip('/'), 'cgroup.procs').read_text().split()
    except FileNotFoundError:
        pids = []
    for pid in pids:
        try:
            fields = Path('/proc', pid, 'stat').read_text().split(') ', 1)[1].split()
            process = dict(user_ticks=int(fields[11]), system_ticks=int(fields[12]),
                           threads=int(fields[17]), rss_pages=int(fields[21]),
                           start_ticks=int(fields[19]),
                           descriptors=len(list(Path('/proc', pid, 'fd').iterdir())))
            io = {line.split(':')[0]: int(line.split()[1])
                  for line in Path('/proc', pid, 'io').read_text().splitlines()}
            row['processes'][pid] = {**process, 'io': io}
            if pid == state['MainPID']:
                row['process'], row['io'] = process, io
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    print(json.dumps(row), flush=True)
    if state['ActiveState'] not in ('active', 'activating'):
        break
    time.sleep(interval)
