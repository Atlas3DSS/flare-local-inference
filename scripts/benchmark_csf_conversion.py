"""Measure checkpoint-to-CUDA CSF loading and compressed layout conversion.

Times one complete Qwen expert layer at TP=1 with the production b12x loader
and conversion entrypoints. Header discovery and first-use kernel compilation
are excluded. Timed loads include allocation, routing, I/O, and conversion.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from b12x.loader import DirectWeightSession
from b12x.loader._nvfp4_csf import queue_nvfp4_csf
from flare.preparation.checkpoint import inspect_indexed_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--trials', type=int, default=7)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    manifest = inspect_indexed_checkpoint(args.snapshot)
    config = json.loads((args.snapshot / 'config.json').read_text())['text_config']
    h, m, e = config['hidden_size'], config['moe_intermediate_size'], config['num_experts']
    prefix = f'model.language_model.layers.{args.layer}.mlp.experts.'
    names = {n for n in manifest.tensors if n.startswith(prefix) and 'nvfp4_csf_' in n}
    paths = sorted({manifest.snapshot / manifest.tensors[n].shard for n in names})
    with DirectWeightSession(device=0) as session:
        sources = dict(session.weights(paths, skip=lambda name: name not in names))
        groups = []
        for projections, rows, columns in ((('up_proj', 'gate_proj'), m, h // 16),
                                            (('down_proj',), h, m // 16)):
            names_in_order = [f'{prefix}{expert}.{proj}.weight_scale.nvfp4_csf_'
                              for expert in range(e) for proj in projections]
            groups.append(([sources[n + 'fixed'] for n in names_in_order],
                           [sources[n + 'exceptions'] for n in names_in_order],
                           rows, columns, len(projections)))

        def load():
            transfers = [queue_nvfp4_csf(session, fixed, exceptions, rows=r, columns=c, join=join)
                         for fixed, exceptions, r, c, join in groups]
            session.flush()
            return transfers

        def convert(transfers):
            return tuple(t.convert() for t in transfers)

        transfers = load()
        expected = convert(transfers)
        torch.cuda.synchronize()
        timings = {'checkpoint+conversion': [], 'resident-conversion': []}
        for _ in range(args.trials):
            for name in timings:
                torch.cuda.synchronize()
                started = perf_counter()
                result = convert(load() if name == 'checkpoint+conversion' else transfers)
                torch.cuda.synchronize()
                timings[name].append((perf_counter() - started) * 1000)
                for a, b in zip(result, expected, strict=True):
                    for field in ('fixed', 'exceptions', 'task_offsets'):
                        assert torch.equal(getattr(a, field), getattr(b, field)), field
                del result
        print(f'Layer {args.layer}: {e} experts, hidden={h}, intermediate={m}, TP=1, reader={session.read_mode}')
        print(f'Checkpoint CSF: {sum(manifest.tensors[n].nbytes for n in names) / 2**20:.3f} MiB')
        for name, values in timings.items():
            print(f'{name} wall ms: {[round(v, 3) for v in values]}, median={statistics.median(values):.3f}')


if __name__ == '__main__':
    main()
