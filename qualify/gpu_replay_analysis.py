"""Compare GPU graph envelopes from the gpu_replay Nsight captures.

CUDA API correlation IDs associate GPU activities with NVTX submission ranges.
vLLM decode metadata verifies actual batches and context positions. The
forward_sample_gpu metric includes GPU work submitted inside the forward and
sampling calls, including their work outside the model graph. Other worker
callbacks and their GPU work are excluded.
Mixed prefill steps and shrinking batch tails are excluded. Request lanes may
be offset by one token when vLLM staggers prompt admission.
"""

import argparse
import bisect
from collections import defaultdict, Counter
import json
from pathlib import Path
import re
import sqlite3
import statistics

def union_ns(rows):
    total, tail = 0, 0
    for start, end in sorted((r['start'], r['end']) for r in rows):
        total += max(0, end - max(tail, start))
        tail = max(tail, end)
    return total

def summarize(values):
    return {'count': len(values), 'mean_ms': statistics.mean(values) / 1e6,
            'median_ms': statistics.median(values) / 1e6,
            'p90_ms': sorted(values)[int(.9 * (len(values)-1))] / 1e6}

def analyze(root, engine, batches=(1, 2, 4)):
    completed = json.loads((root / engine / 'results.json').read_text())
    groups = [g for g in completed['groups'] if g['batch'] in batches]
    expected = {(b, r, r > 0) for b in batches for r in range(4)}
    assert len(groups) == len(expected) and {(g['batch'], g['repeat'], g['measured']) for g in groups} == expected, \
        'require all selected warmup and measured groups to complete'
    db = sqlite3.connect(f'file:{root / (engine + "-trace.sqlite")}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    def rows(table):
        return [dict(r) for r in db.execute('select * from ' + table)] if table in tables else []
    strings = {r['id']: r['value'] for r in rows('StringIds')}
    scopes = defaultdict(list)
    for r in rows('NVTX_EVENTS'):
        name = r.get('text') or strings.get(r.get('textId'), '')
        if name.startswith(engine + '.') and r.get('end') is not None:
            r['name'] = name
            scopes[r['globalTid']].append(r)
    for rs in scopes.values():
        rs.sort(key=lambda r: r['start'])
    starts = {tid: [r['start'] for r in rs] for tid, rs in scopes.items()}
    corr = {}
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
        for api in rows(table):
            tid = api['globalTid']
            if tid not in scopes:
                continue
            i = bisect.bisect_right(starts[tid], api['start']) - 1
            if i >= 0 and api['start'] < scopes[tid][i]['end']:
                key = (tid >> 24, api['correlationId'])
                name = scopes[tid][i]['name']
                assert key not in corr or corr[key] == name
                corr[key] = name
    by_scope = defaultdict(lambda: defaultdict(list))
    unmatched = Counter()
    for table in ('GRAPH_TRACE', 'KERNEL', 'MEMCPY', 'MEMSET'):
        for gpu in rows('CUPTI_ACTIVITY_KIND_' + table):
            name = corr.get((gpu['globalPid'] >> 24, gpu['correlationId']))
            if name:
                by_scope[name][table].append(gpu)
            else:
                unmatched[table] += 1
    print(engine, 'scopes', sum(map(len, scopes.values())), 'matched', len(by_scope), 'unmatched GPU records', dict(unmatched))
    samples = defaultdict(list)
    if engine == 'flare':
        graphs = json.loads((root / 'flare/graphs.json').read_text())
        decode = {g['graph_exec_id']: g['batch_capacity'] for g in graphs['graphs'] if g['name'] in ('decode1', 'decode2', 'decode4')}
        for name, types in by_scope.items():
            m = re.fullmatch(r'flare.decode.b([124]).r([123])', name)
            if not m:
                continue
            b, repeat = map(int, m.groups())
            if b not in batches:
                continue
            gs = sorted(types['GRAPH_TRACE'], key=lambda g: g['start'])
            assert len(gs) == 255, (name, len(gs))
            assert all(decode[g['graphExecId']] == b for g in gs)
            for pos, g in enumerate(gs, 512):
                samples[b].append({'group': repeat, 'position': pos, 'graph_ns': g['end']-g['start'], 'total_ns': g['end']-g['start'], 'graphs': 1})
    else:
        steps = json.loads((root / 'vllm/steps.json').read_text())
        request_sets = {frozenset(step['request_ids']) for step in steps}
        cohorts = {ids for ids in request_sets if len(ids) in batches and not any(ids < other for other in request_sets)}
        assert Counter(map(len, cohorts)) == {b: 3 for b in batches}, cohorts
        groups = defaultdict(list)
        for step in steps:
            if step['has_prefill'] or frozenset(step['request_ids']) not in cohorts:
                continue
            b, pos = step['batch'], step['positions']
            assert b in (1,2,4) and step['tokens'] == b and step['padded_tokens'] == b, step
            assert step['scheduled'] == [1] * b and max(pos) - min(pos) <= 1, step
            assert all(512 <= p <= 766 for p in pos), step
            group = tuple(sorted(step['request_ids']))
            groups[(b, group)].append(pos)
            forward = by_scope[f"vllm.forward.{step['index']}"]
            sample = by_scope[f"vllm.sample.{step['index']}"]
            gs = forward['GRAPH_TRACE']
            assert gs, step
            all_gpu = [g for types in (forward, sample) for rs in types.values() for g in rs]
            samples[b].append({'group': group, 'position': pos[0], 'graph_ns': union_ns(gs), 'total_ns': union_ns(all_gpu), 'graphs': len(gs)})
        assert Counter(b for b,g in groups) == {b: 3 for b in batches}, Counter(b for b,g in groups)
        for (b, group), positions in groups.items():
            assert len(positions) in (254, 255), (group, len(positions))
            assert max(positions[-1]) == 766 and min(positions[0]) == 512
            for lane in range(b):
                values = [p[lane] for p in positions]
                assert values == list(range(values[0], values[-1] + 1)), group
    report = {}
    for b, rs in sorted(samples.items()):
        assert 762 <= len(rs) <= 765, (b, len(rs))
        groups = defaultdict(list)
        for r in rs:
            groups[str(r['group'])].append(r['graph_ns'])
        report[b] = {'graph': summarize([r['graph_ns'] for r in rs]), 'forward_sample_gpu': summarize([r['total_ns'] for r in rs]),
                     'graphs_per_step': dict(Counter(r['graphs'] for r in rs)), 'repeat_mean_ms': [statistics.mean(v)/1e6 for v in groups.values()]}
    assert set(report) == set(batches), 'require all selected measured batches'
    db.close()
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--batches', type=int, nargs='+', choices=(1, 2, 4), default=(1, 2, 4))
    parser.add_argument('--engines', nargs='+', choices=('flare', 'vllm'), default=('flare', 'vllm'))
    args = parser.parse_args()
    report = {e: analyze(args.directory.resolve(), e, args.batches) for e in args.engines}
    text = json.dumps(report, indent=2) + '\n'
    (args.directory / 'timings.json').write_text(text)
    print(text, end='')
