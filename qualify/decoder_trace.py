"""Supervise the complete native decoder trace on the assigned checkpoint GPU."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import signal
import subprocess
import sys
import traceback

from .provenance import digest, file_record, utc, write_json


ROOT = Path(__file__).resolve().parents[1]
GPU = 'GPU-afd4b1ad-8a64-7057-4bde-241822724c7f'
CHECKPOINT = Path('/data/checkpoints-masters/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD')


def checkpoint_identity():
    expected = json.loads((ROOT / 'docs/validation/exact-serving-reference.json').read_text())['checkpoint']
    if str(CHECKPOINT) != expected['served_path']:
        raise ValueError('decoder trace checkpoint differs from the serving reference')
    assets = []
    for entry in expected['assets']:
        actual = file_record(CHECKPOINT / entry['name'])
        if actual['sha256'] != entry['master_sha256']:
            raise ValueError(f'checkpoint asset changed: {actual["path"]}')
        assets.append(actual)
    shards = []
    for entry in expected['shards']:
        path = CHECKPOINT / entry['relative_path']
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if identity != (entry['master_device'], entry['master_inode'], entry['bytes'], entry['mtime_ns']):
            raise ValueError(f'checkpoint shard identity changed: {path}')
        shards.append({'path': str(path), 'device_inode_size_mtime_ns': identity})
    return {'assets': assets, 'shards': shards, 'revision': expected['declared_revision']}


def source_paths():
    paths = {Path(__file__), ROOT / 'qualify/provenance.py',
             ROOT / 'qualify/vllm_observation.py', ROOT / 'scripts/qualify_decoder.py'}
    for directory in ('sparkgap/preparation', 'sparkgap/recipes', 'native/src'):
        paths.update(path for path in (ROOT / directory).rglob('*')
                     if path.suffix in ('.py', '.rs'))
    paths.update(ROOT / name for name in ('pyproject.toml', 'uv.lock',
        'native/Cargo.toml', 'native/Cargo.lock'))
    for name in ('b12x', 'flashinfer', 'sparkgap_native'):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f'required preparation package is absent: {name}')
        package = Path(spec.origin).parent
        paths.update(package.rglob('*.py'))
        if name == 'sparkgap_native':
            paths.update(package.glob('*.so'))
    spec = importlib.util.find_spec('transformers')
    paths.update((Path(spec.origin).parent / 'models/qwen4_exp').glob('*.py'))
    return sorted(paths)


def worker(output):
    from .vllm_observation import _libraries

    started = utc()
    sys.argv = [str(ROOT / 'scripts/qualify_decoder.py'), '--tune', '--layer-trace',
                '--snapshot', str(CHECKPOINT),
                '--prompt', 'The capital of France is', '--steps', '4',
                '--prefill-capacity', '512', '--output', str(output / 'decoder.json'),
                '--tensor-output', str(output / 'decoder.pt')]
    try:
        runpy.run_path(sys.argv[0], run_name='__main__')
    except BaseException:
        write_json(output / 'failure.json', {'utc': utc(), 'traceback': traceback.format_exc()})
        raise
    finally:
        write_json(output / 'worker-libraries.json', _libraries())
        write_json(output / 'worker-exit.json', {'started_utc': started,
            'finished_utc': utc(), 'pid': os.getpid(), 'python': sys.executable,
            'command': sys.argv})


def parent(output):
    checkpoint = checkpoint_identity()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'checkpoint-before.json', checkpoint)
    sources = {}
    for path in source_paths():
        relative = str(path).lstrip('/')
        destination = output / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        source = file_record(path)
        if digest(destination) != source['sha256']:
            raise RuntimeError(f'source changed during snapshot: {path}')
        sources[str(path)] = source
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    environment.update(CUDA_VISIBLE_DEVICES=GPU, CUDA_HOME='/opt/cuda')
    command = [str(ROOT / '.venv/bin/python'), '-u', '-m', 'qualify.decoder_trace',
               '--worker', '--output', str(output)]
    write_json(output / 'source-freeze.json', {'utc': utc(), 'files': sources})
    process = None
    handlers = {}
    try:
        with (output / 'worker.log').open('xb', buffering=0) as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            def forward(signum, frame):
                if process.poll() is None:
                    os.killpg(process.pid, signum)
            for signum in (signal.SIGINT, signal.SIGTERM):
                handlers[signum] = signal.signal(signum, forward)
            write_json(output / 'launch.json', {'utc': utc(), 'pid': process.pid,
                'command': command, 'cwd': str(ROOT), 'environment': {
                    name: environment.get(name) for name in (
                        'CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'PYTHONPATH',
                        'B12X_COMPILE_WORKERS', 'PYTORCH_ALLOC_CONF',
                        'PYTORCH_CUDA_ALLOC_CONF')}})
            print(json.dumps({'pid': process.pid, 'log': str(output / 'worker.log')}), flush=True)
            process.wait()
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        changed = [path for path, entry in sources.items()
                   if not Path(path).is_file() or digest(path) != entry['sha256']]
        checkpoint_after = checkpoint_identity()
        write_json(output / 'checkpoint-after.json', checkpoint_after)
        gpu = subprocess.run(['nvidia-smi', '-i', GPU, '--query-gpu=memory.used',
                              '--format=csv,noheader'], capture_output=True, text=True)
        record = {'utc': utc(), 'pid': None if process is None else process.pid,
            'exit_code': None if process is None else process.returncode,
            'changed_sources': changed,
            'checkpoint_unchanged': checkpoint_after == checkpoint,
            'process_absent': process is None or not Path(f'/proc/{process.pid}').exists(),
            'gpu_memory_after_exit': gpu.stdout.strip(), 'gpu_query_exit_code': gpu.returncode}
        write_json(output / 'exit.json', record)
    if process is None or process.returncode or changed:
        raise RuntimeError(f'decoder trace failed; inspect {output / "exit.json"}')
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    (worker if args.worker else parent)(args.output.resolve())


if __name__ == '__main__':
    main()
