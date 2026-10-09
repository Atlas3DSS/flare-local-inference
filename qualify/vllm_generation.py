"""Record cold 64-token greedy continuations through the pinned vLLM engine."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from .provenance import checkout_record, digest, file_record, plain, utc, write_json
from .vllm_text import GPU, MODEL_ID, REVISION, ROOT, checkpoint_identity, parse_engine_args


def validate_completion(output, prompt, *, tokens=64):
    """Reject truncated, stopped, or differently conditioned observations."""
    completion, = output.outputs
    generated = list(completion.token_ids)
    if list(output.prompt_token_ids) != prompt['prompt_tokens']:
        raise ValueError(f'{prompt["id"]}: vLLM consumed different prompt IDs')
    if len(generated) != tokens or completion.finish_reason != 'length':
        raise ValueError(f'{prompt["id"]}: expected {tokens} generated tokens and length termination')
    if any(type(token) is not int or not 0 <= token < 248320 for token in generated):
        raise ValueError(f'{prompt["id"]}: invalid generated token ID')
    return {'id': prompt['id'], 'category': prompt['category'],
            'prompt_tokens': list(output.prompt_token_ids), 'generated_tokens': generated,
            'finish_reason': completion.finish_reason, 'stop_reason': completion.stop_reason}


def worker(args):
    import torch
    from .generation_corpus import load_corpus

    plan = json.loads(args.plan.read_text())
    corpus = load_corpus(args.output / 'corpus.json')
    if (corpus['model_id'], corpus['revision'], corpus['max_new_tokens'], corpus['ignore_eos']) != (
            MODEL_ID, REVISION, 64, True):
        raise ValueError('generation corpus differs from the checkpoint or 64-token contract')
    sources = {path: checkout_record(path, expected) for path, expected in plan['sources'].items()}
    roots = {'vllm': Path(plan['cwd']),
             'flashinfer': Path(next(path for path in sources if 'flashinfer' in path))}
    roots['b12x'] = roots['flashinfer']
    locations = {}
    for name, root in roots.items():
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None or not Path(spec.origin).resolve().is_relative_to(root):
            raise ValueError(f'{name} escaped recorded source isolation')
        locations[name] = spec.origin
    engine_args = parse_engine_args(plan, check_only=args.check)
    engine_args.enable_trace_replay = False
    engine_args.max_logprobs = 20
    record = {'schema_version': 1, 'started_utc': utc(), 'pid': os.getpid(),
              'python': sys.executable, 'sources': sources, 'module_origins': locations,
              'engine_args': plain(engine_args), 'corpus': file_record(args.output / 'corpus.json'),
              'generation_contract': {'tokens': 64, 'temperature': 0, 'ignore_eos': True,
                                      'teacher_forcing': False, 'prefix_reuse': False},
              'repetitions': args.repetitions}
    write_json(args.output / 'worker-start.json', record)
    if args.check:
        if torch.cuda.is_initialized():
            raise RuntimeError('generation preflight initialized CUDA')
        write_json(args.output / 'preflight.json', record | {'passed': True, 'cuda_initialized': False})
        return
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU or torch.cuda.device_count() != 1:
        raise ValueError('generation qualification requires only its assigned GPU')

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from .vllm_observation import observe_worker

    llm = None
    observations = []
    try:
        started = time.monotonic()
        llm = LLM.from_engine_args(engine_args)
        record['startup_seconds'] = time.monotonic() - started
        write_json(args.output / 'effective-config.json', llm.llm_engine.vllm_config)
        write_json(args.output / 'worker-before.json', llm.collective_rpc(observe_worker))
        params = SamplingParams(temperature=0, top_p=1, top_k=-1, min_p=0,
            repetition_penalty=1, presence_penalty=0, frequency_penalty=0,
            max_tokens=64, ignore_eos=True, detokenize=False, seed=0,
            output_kind=RequestOutputKind.FINAL_ONLY)
        write_json(args.output / 'sampling-params.json', params)
        for prompt in corpus['prompts']:
            for repetition in range(args.repetitions):
                reset = llm.reset_prefix_cache(reset_running_requests=False, reset_connector=False)
                if reset is not True:
                    raise RuntimeError(f'{prompt["id"]}: completed-request prefix reset failed')
                started = time.monotonic()
                output, = llm.generate([{'prompt_token_ids': prompt['prompt_tokens']}], params, use_tqdm=False)
                observation = validate_completion(output, prompt)
                observation.update(repetition=repetition, prefix_cache_reset=True,
                    generation_seconds=time.monotonic() - started, finished_utc=utc())
                observations.append(observation)
                with (args.output / 'observations.jsonl').open('a') as handle:
                    handle.write(json.dumps(observation, allow_nan=False) + '\n')
                    handle.flush()
                print(json.dumps({'completed': len(observations),
                                  'total': len(corpus['prompts']) * args.repetitions,
                                  'id': prompt['id'], 'repetition': repetition}), flush=True)
        write_json(args.output / 'worker-after.json', llm.collective_rpc(observe_worker))
        record.update(finished_utc=utc(), completed_observations=len(observations),
                      observations=file_record(args.output / 'observations.jsonl'),
                      sources_after={path: checkout_record(path, plan['sources'][path]) for path in sources})
        write_json(args.output / 'capture.json', record)
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown(timeout=60)


def parent(args):
    from .generation_corpus import load_corpus

    corpus = load_corpus(args.corpus)
    if len(corpus['prompts']) < 50:
        raise ValueError('generation agreement requires at least 50 fixed prompts')
    plan = json.loads(args.plan.read_text())
    if plan['gpu_uuid'] != GPU or plan['declared_revision'] != REVISION:
        raise ValueError('reference plan does not identify the assigned checkpoint and GPU')
    for path, expected_hash in plan['sha256'].items():
        if digest(path) != expected_hash:
            raise ValueError(f'reference launch dependency changed: {path}')
    for path, pin in plan['sources'].items():
        checkout_record(path, pin)
    checkpoint = checkpoint_identity(plan)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'corpus.json', corpus)
    write_json(args.output / 'corpus-source.json', file_record(args.corpus))
    write_json(args.output / 'checkpoint-before.json', checkpoint)
    frozen = {str(path): digest(path) for path in [*sorted((ROOT / 'qualify').glob('*.py')), args.plan]}
    for name in frozen:
        path = Path(name)
        destination = args.output / 'source' / (str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    env = dict(os.environ)
    env.update({key: value for key, value in plan['environment'].items()
                if key not in ('HOME', 'USER', 'LOGNAME', 'PATH')})
    env['PYTHONPATH'] += ':' + str(ROOT)
    env['VLLM_B12X_DUMP_QUERIES'] = str(args.output / 'declarations.jsonl')
    env['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
    if args.check:
        env['CUDA_VISIBLE_DEVICES'] = ''
    command = [plan['python'], '-u', '-m', 'qualify.vllm_generation', '--worker',
               '--plan', str(args.plan), '--output', str(args.output),
               '--repetitions', str(args.repetitions)]
    if args.check:
        command.append('--check')
    started = utc()
    process = None
    failure = None
    previous = {}
    try:
        with (args.output / 'worker.log').open('xb', buffering=0) as log:
            process = subprocess.Popen(command, cwd=plan['cwd'], env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            def forward(signum, frame):
                if process.poll() is None:
                    os.killpg(process.pid, signum)
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous[signum] = signal.signal(signum, forward)
            write_json(args.output / 'launch.json', {'command': command, 'pid': process.pid,
                'started_utc': started, 'cwd': plan['cwd'], 'source_sha256': frozen,
                'environment': {key: env[key] for key in (*plan['environment'],
                    'VLLM_B12X_DUMP_QUERIES', 'VLLM_ALLOW_INSECURE_SERIALIZATION')}})
            print(json.dumps({'pid': process.pid, 'log': str(args.output / 'worker.log')}), flush=True)
            process.wait()
    except BaseException as error:
        failure = repr(error)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        changed = {path: digest(path) for path, value in frozen.items() if digest(path) != value}
        checkpoint_after = checkpoint_identity(plan)
        write_json(args.output / 'checkpoint-after.json', checkpoint_after)
        gpu = subprocess.run(['nvidia-smi', '-i', GPU, '--query-gpu=memory.used', '--format=csv,noheader'],
                             capture_output=True, text=True)
        write_json(args.output / 'exit.json', {'started_utc': started, 'finished_utc': utc(),
            'pid': None if process is None else process.pid,
            'exit_code': None if process is None else process.returncode,
            'supervisor_error': failure, 'changed_sources': changed,
            'checkpoint_unchanged': checkpoint_after == checkpoint,
            'process_absent': process is None or not Path(f'/proc/{process.pid}').exists(),
            'gpu_memory_after_exit': gpu.stdout.strip(), 'gpu_query_exit_code': gpu.returncode,
            'log': file_record(args.output / 'worker.log')})
    if failure or process is None or process.returncode or changed:
        raise RuntimeError(f'generation process failed; inspect {args.output / "exit.json"}')
    print(json.dumps({'exit_code': 0, 'output': str(args.output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path('/tmp/sparkgap-vllm-tp1-reference-64fc1eec/plan.json'))
    parser.add_argument('--corpus', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, choices=(1, 2), default=2)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker and args.corpus is None:
        parser.error('--corpus is required')
    args.plan, args.output = args.plan.resolve(), args.output.resolve()
    if args.corpus is not None:
        args.corpus = args.corpus.resolve()
    try:
        (worker if args.worker else parent)(args)
    except BaseException:
        if args.worker and args.output.is_dir():
            write_json(args.output / 'failure.json', {'utc': utc(), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
