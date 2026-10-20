"""Summarize the post-merge loading counters and native request timings."""
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).parent
FABRIC = ('enp1s0f0np0', 'enp1s0f1np1', 'enP2p1s0f0np0', 'enP2p1s0f1np1')

for rank in sorted(ROOT.glob('rank*')):
    for run in sorted(rank.glob('merged-load*')):
        with gzip.open(run/'resources.jsonl.gz', 'rt') as stream:
            rows = [json.loads(line) for line in stream]
        first, last = rows[0], rows[-1]
        with gzip.open(run/'agent-resources.jsonl.gz', 'rt') as stream:
            agents = [json.loads(line) for line in stream]
        def transferred(interfaces, field):
            return sum(last['network'][name][field] - first['network'][name][field] for name in interfaces)
        records = [json.loads(line) for line in (run/'load.log').read_text().splitlines() if line.startswith('{')]
        record = records[-1]
        result = record['results'][0] if 'results' in record else record
        print(json.dumps(dict(
            rank=rank.name, run=run.name, load=result,
            exit=json.loads((run/'exit.json').read_text()),
            counter_seconds=last['time'] - first['time'],
            fabric_GiB={field: transferred(FABRIC, field) / 2**30 for field in ('rx_bytes', 'tx_bytes')},
            management_MiB={field: transferred(('enP7s7',), field) / 2**20 for field in ('rx_bytes', 'tx_bytes')},
            NVMe_read_GiB=(last['nvme'][2] - first['nvme'][2]) * 512 / 2**30,
            NVMe_write_GiB=(last['nvme'][6] - first['nvme'][6]) * 512 / 2**30,
            peak_agent_descriptors=max(p['descriptors'] for row in agents for p in row['processes'].values()),
        ), sort_keys=True))

results = ROOT/'rank0/merged-inference/results.json'
if results.exists():
    for phase in json.loads(results.read_text()):
        print(json.dumps(dict(phase=phase['phase'], wall_seconds=phase['wall_seconds'],
                              requests=phase['requests']), sort_keys=True))
