"""Summarize complete preparation counters and native request measurements."""
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).parent
FABRIC = ('enp1s0f0np0', 'enp1s0f1np1', 'enP2p1s0f0np0', 'enP2p1s0f1np1')

for rank in sorted(ROOT.glob('rank*')):
    run = rank / 'latest-prepare2'
    with gzip.open(run / 'resources.jsonl.gz', 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    first, last = rows[0], rows[-1]

    def transferred(interfaces, field):
        return sum(last['network'][name][field] - first['network'][name][field]
                   for name in interfaces)

    print(json.dumps(dict(
        rank=rank.name,
        exit=json.loads((run / 'exit.json').read_text()),
        counter_seconds=last['time'] - first['time'],
        fabric_GiB={field: transferred(FABRIC, field) / 2**30
                    for field in ('rx_bytes', 'tx_bytes')},
        management_MiB={field: transferred(('enP7s7',), field) / 2**20
                        for field in ('rx_bytes', 'tx_bytes')},
        NVMe_read_GiB=(last['nvme'][2] - first['nvme'][2]) * 512 / 2**30,
    ), sort_keys=True))

results = ROOT / 'rank0/latest-inference/results.json'
if results.exists():
    for phase in json.loads(results.read_text()):
        print(json.dumps(dict(phase=phase['phase'], wall_seconds=phase['wall_seconds'],
                              requests=phase['requests']), sort_keys=True))
