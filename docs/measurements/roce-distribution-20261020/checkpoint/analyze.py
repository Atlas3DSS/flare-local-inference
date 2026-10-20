"""Summarize retained interface and NVMe counters around checkpoint loading."""

import gzip
import json
from pathlib import Path
import re


ROOT = Path(__file__).parent
FABRIC = ("enp1s0f0np0", "enp1s0f1np1", "enP2p1s0f0np0", "enP2p1s0f1np1")


for experiment in sorted(ROOT.iterdir()):
    ranks = sorted(experiment.glob("rank*")) if experiment.is_dir() else []
    if not ranks:
        continue
    loaded, loading = {}, {}
    for rank in ranks:
        if (rank / "prepare.log").exists():
            match = re.search(r"([\d.]+)s rank \d+: after weights:", (rank / "prepare.log").read_text())
            if match:
                loaded[rank.name] = float(match[1])
        elif (rank / "load.log").exists():
            for line in (rank / "load.log").read_text().splitlines():
                if line.startswith('{'):
                    result = json.loads(line)
                    for item in result.get('results', [result]):
                        loading[f"rank{item['rank']}"] = item
    # Include the next five-second sample after completed loading. These are
    # whole-interface counters, including admission and transport initialization.
    end = max(loaded.values()) + 5 if loaded else float("inf")
    for rank in ranks:
        samples = next(rank.glob("*resources.jsonl.gz"), None)
        if (rank / "resources.jsonl.gz").exists():
            samples = rank / "resources.jsonl.gz"
        with gzip.open(samples, "rt") as stream:
            rows = [json.loads(line) for line in stream]
        first = rows[0]
        last = min(rows, key=lambda row: abs(row["elapsed"] - end)) if loaded else rows[-1]
        def network(interfaces, field):
            return sum(last["network"][name][field] - first["network"][name][field] for name in interfaces)
        agent_descriptors = None
        if loading:
            with gzip.open(rank / 'agent-resources.jsonl.gz', 'rt') as stream:
                agent_rows = [json.loads(line) for line in stream]
            agent_descriptors = max(p['descriptors'] for row in agent_rows for p in row['processes'].values())
        print(json.dumps({
            "experiment": experiment.name, "rank": rank.name,
            "after_weights_seconds": loaded.get(rank.name),
            "loading_only": loading.get(rank.name),
            "peak_agent_descriptors": agent_descriptors,
            "counter_window_seconds": last["time"] - first["time"],
            "fabric_GiB": {field: network(FABRIC, field) / 2**30 for field in ("rx_bytes", "tx_bytes")},
            "management_MiB": {field: network(("enP7s7",), field) / 2**20 for field in ("rx_bytes", "tx_bytes")},
            "NVMe_read_GiB": (last["nvme"][2] - first["nvme"][2]) * 512 / 2**30,
            "NVMe_write_GiB": (last["nvme"][6] - first["nvme"][6]) * 512 / 2**30,
        }, sort_keys=True))
        if (rank / 'export-before.json').exists() and (rank / 'export-after.json').exists():
            first = json.loads((rank / 'export-before.json').read_text())
            last = json.loads((rank / 'export-after.json').read_text())
            with gzip.open(rank / 'agent-resources.jsonl.gz', 'rt') as stream:
                samples = [json.loads(line) for line in stream]
            active = {}
            for field in ('rx_bytes', 'tx_bytes'):
                windows = [(a, b, sum(b['network'][n][field] - a['network'][n][field] for n in FABRIC))
                           for a, b in zip(samples, samples[1:]) if first['time'] <= a['time'] < last['time']]
                # Bound the active transfer by samples moving more than 64 MiB;
                # preserve the enclosing snapshots separately, including idle time.
                busy = [(a, b, size) for a, b, size in windows if size > 64 << 20]
                if busy:
                    a, b = busy[0][0], busy[-1][1]
                    seconds = b['time'] - a['time']
                    size = sum(b['network'][n][field] - a['network'][n][field] for n in FABRIC)
                    active[field] = dict(seconds=seconds, mean_Gbps=size * 8 / seconds / 1e9,
                        peak_Gbps=max(size * 8 / (b['time'] - a['time']) / 1e9 for a, b, size in busy))
            print(json.dumps({
                'experiment': experiment.name, 'rank': rank.name, 'phase': 'export',
                # Snapshots bracket the transfer and can include idle time on either side.
                'counter_window_seconds': last['time'] - first['time'],
                'fabric_GiB': {field: network(FABRIC, field) / 2**30 for field in ('rx_bytes', 'tx_bytes')},
                'management_MiB': {field: network(('enP7s7',), field) / 2**20
                                   for field in ('rx_bytes', 'tx_bytes')},
                'active_fabric': active,
            }, sort_keys=True))
