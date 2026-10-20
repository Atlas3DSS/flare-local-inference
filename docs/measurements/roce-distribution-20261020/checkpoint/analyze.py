"""Summarize retained checkpoint-loading windows; later RDMA tuning is excluded."""

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
    loaded = {}
    for rank in ranks:
        match = re.search(r"([\d.]+)s rank \d+: after weights:", (rank / "prepare.log").read_text())
        if match:
            loaded[rank.name] = float(match[1])
    # The next five-second sample includes each peer's completed loading. Catalog
    # construction then takes about 20 seconds before collective tuning begins.
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
        print(json.dumps({
            "experiment": experiment.name, "rank": rank.name,
            "after_weights_seconds": loaded.get(rank.name),
            "counter_window_seconds": last["time"] - first["time"],
            "fabric_GiB": {field: network(FABRIC, field) / 2**30 for field in ("rx_bytes", "tx_bytes")},
            "management_MiB": {field: network(("enP7s7",), field) / 2**20 for field in ("rx_bytes", "tx_bytes")},
            "NVMe_read_GiB": (last["nvme"][2] - first["nvme"][2]) * 512 / 2**30,
            "NVMe_write_GiB": (last["nvme"][6] - first["nvme"][6]) * 512 / 2**30,
        }, sort_keys=True))
