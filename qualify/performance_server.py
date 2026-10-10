"""Prepare the production model, optionally measure replay, and bind native HTTP."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .provenance import file_record, utc, write_json

ROOT = Path(__file__).resolve().parents[1]


def source_record(root):
    paths = subprocess.check_output(
        ['git', '-C', str(root), 'ls-files', '-c', '-o', '--exclude-standard', '-z']
    ).decode().split('\0')
    return {
        'path': str(root),
        'head': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        'status': subprocess.check_output(['git', '-C', str(root), 'status', '--short'], text=True),
        'sources': {path: file_record(root / path) for path in sorted(set(paths))
                    if path and (root / path).is_file()
                    and Path(path).suffix in ('.py', '.rs', '.cu', '.cuh', '.h', '.cpp', '.toml', '.lock')},
    }


def main():
    started = time.perf_counter_ns()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bind', default='127.0.0.1:38761')
    parser.add_argument('--prefill-chunk-size', type=int, choices=(512, 4096), required=True)
    parser.add_argument('--replay-corpus', type=Path)
    parser.add_argument('--trace-graph-ids', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {'started_at': utc(), 'pid': os.getpid(), 'argv': sys.argv,
               'cwd': str(Path.cwd()), 'python': sys.executable,
               'gpu_uuid': os.environ.get('CUDA_VISIBLE_DEVICES'),
               'cuda_home': os.environ.get('CUDA_HOME'),
               'prefill_chunk_size': args.prefill_chunk_size,
               'snapshot': str(args.snapshot.resolve()), 'stages': [],
               'replay_corpus': file_record(args.replay_corpus) if args.replay_corpus else None}
    write_json(args.output / 'launch.json', receipt)
    import flashinfer
    import flare_helios
    import torch
    from flare.recipes.qwen38.bootstrap import PreparationConfig, prepare_model

    def progress(message):
        record = {'elapsed_ns': time.perf_counter_ns() - started, 'message': message}
        receipt['stages'].append(record)
        print(json.dumps(record), flush=True)
        write_json(args.output / 'launch.json', receipt)

    receipt.update(
        sources={'flare': source_record(ROOT),
                 'flashinfer': source_record(Path(flashinfer.__file__).resolve().parents[1])},
        native_extension=file_record(Path(flare_helios.__file__).parent /
                                     'flare_helios.cpython-312-x86_64-linux-gnu.so'),
        versions={name: importlib.metadata.version(name) for name in
                  ('torch', 'transformers', 'flashinfer-python', 'nvidia-cutlass-dsl', 'triton')})
    receipt['source_recording_finished_ns'] = time.perf_counter_ns() - started
    config = PreparationConfig(prefill_chunk_size=args.prefill_chunk_size)
    with prepare_model(args.snapshot, config, progress=progress) as model:
        write_json(args.output / 'memory.json', model.memory_report)
        receipt['preparation_finished_ns'] = time.perf_counter_ns() - started
        if args.replay_corpus:
            from .performance import validate_corpus
            from .replay_floor import measure
            corpus = json.loads(args.replay_corpus.read_text())
            validate_corpus(corpus)
            receipt['replay_started_ns'] = time.perf_counter_ns() - started
            measure(model, corpus, args.output / 'replay')
            receipt['replay_finished_ns'] = time.perf_counter_ns() - started
        if args.trace_graph_ids:
            from .performance_trace import prepared_graph_map
            write_json(args.output / 'trace-graphs.json', prepared_graph_map(model.manifest()))
        bind_started = time.perf_counter_ns()
        server = model.start_server(bind=args.bind)
        receipt.update(ready_at=utc(), ready_ns=time.perf_counter_ns() - started,
                       server_binding_ns=time.perf_counter_ns() - bind_started,
                       address=server.address, device_name=torch.cuda.get_device_name(model.device))
        write_json(args.output / 'launch.json', receipt)
        print(f'Listening on {server.address}', flush=True)

        def stop(signum, frame):
            raise KeyboardInterrupt

        previous = signal.signal(signal.SIGTERM, stop)
        try:
            while server.wait(1000):
                pass
            raise RuntimeError('native HTTP server stopped unexpectedly')
        except KeyboardInterrupt:
            return 0
        finally:
            signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
