"""Capture full-vocabulary teacher-forced logits through the pinned vLLM engine."""

from __future__ import annotations

import argparse
import importlib.metadata
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


ROOT = Path(__file__).resolve().parents[1]
GPU = 'GPU-afd4b1ad-8a64-7057-4bde-241822724c7f'
MODEL_ID = 'local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD'
REVISION = 'dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2'
VOCABULARY = 248320


def checkpoint_identity(plan, reference_path):
    reference = json.loads(reference_path.read_text())['checkpoint']
    if plan['checkpoint'] != reference['served_path'] or reference['declared_revision'] != REVISION:
        raise ValueError('oracle checkpoint differs from the recorded model artifact')
    root = Path(plan['checkpoint'])
    assets = []
    for entry in reference['assets']:
        actual = file_record(root / entry['name'])
        if actual['sha256'] != entry['master_sha256']:
            raise ValueError(f'checkpoint asset changed: {actual["path"]}')
        assets.append(actual)
    shards = []
    for entry in reference['shards']:
        path = root / entry['relative_path']
        stat = path.stat()
        actual = {'path': str(path), 'device': stat.st_dev, 'inode': stat.st_ino,
                  'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != (
                entry['master_device'], entry['master_inode'], entry['bytes'], entry['mtime_ns']):
            raise ValueError(f'checkpoint shard identity changed: {actual}')
        shards.append(actual)
    return {'repository': MODEL_ID, 'revision': REVISION, 'assets': assets, 'shards': shards}


def parse_engine_args(plan, *, check_only):
    import torch

    if check_only:
        def forbidden(*args, **kwargs):
            raise RuntimeError('CUDA initialization is forbidden during oracle preflight')
        torch.cuda._lazy_init = forbidden
        torch.cuda.is_available = lambda: False
        torch.cuda.device_count = lambda: 0
        import vllm.platforms
        from vllm.platforms.cpu import CpuPlatform
        vllm.platforms._current_platform = CpuPlatform()

    from vllm.engine.arg_utils import EngineArgs
    from vllm.entrypoints.launchers.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parsed = make_arg_parser(FlexibleArgumentParser()).parse_args(plan['serve_args'])
    parsed.model = parsed.model_tag
    args = EngineArgs.from_cli_args(parsed)
    expected = {'model': plan['checkpoint'], 'tensor_parallel_size': 1,
        'pipeline_parallel_size': 1, 'data_parallel_size': 1,
        'decode_context_parallel_size': 1, 'dtype': 'bfloat16',
        'kv_cache_dtype': 'bfloat16', 'mamba_cache_dtype': 'bfloat16',
        'mamba_ssm_cache_dtype': 'float32', 'speculative_config': None,
        'linear_backend': 'b12x', 'moe_backend': 'b12x',
        'gdn_prefill_backend': 'b12x', 'gdn_decode_kernel': 'b12x',
        'max_model_len': 262144, 'max_num_seqs': 4, 'max_num_batched_tokens': 512,
        'additional_config': {'ple_table_memory': 'ram'}, 'enforce_eager': False}
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise ValueError(f'oracle engine setting differs: {name}={getattr(args, name)!r}')
    args.enable_trace_replay = True
    args.logprobs_mode = 'raw_logits'
    args.max_logprobs = -1
    return args


def reconstruct_logits(flat, *, steps, vocabulary):
    import torch

    if len(flat.start_indices) != steps or len(flat.end_indices) != steps:
        raise ValueError('vLLM returned a different number of scored positions')
    result = []
    for start, end in zip(flat.start_indices, flat.end_indices, strict=True):
        ids = torch.tensor(flat.token_ids[start:end], dtype=torch.int64)
        scores = torch.tensor(flat.logprobs[start:end], dtype=torch.float32)
        if len(ids) != vocabulary + 1 or not torch.isfinite(scores).all():
            raise ValueError('vLLM must return finite full-vocabulary logits plus the sampled token')
        sorted_ids, order = ids.sort(stable=True)
        if not torch.equal(sorted_ids.unique(), torch.arange(vocabulary)):
            raise ValueError('vLLM logit token IDs do not cover the exact vocabulary')
        sorted_scores = scores[order]
        duplicates = sorted_ids[1:] == sorted_ids[:-1]
        if duplicates.sum().item() != 1 or not torch.equal(
                sorted_scores[1:][duplicates], sorted_scores[:-1][duplicates]):
            raise ValueError('vLLM returned inconsistent scores for the duplicated sampled token')
        row = torch.empty(vocabulary, dtype=torch.float32)
        row[ids] = scores
        result.append(row)
    return result



def capture_cold_repetitions(llm, *, count, output, capture):
    """Complete each request before invalidating prefix reuse for its successor."""
    if count not in (1, 2):
        raise ValueError('oracle repetitions must be one or two')
    records = []
    for index in range(count):
        destination = output if index == 0 else output / f'repetition-{index:02d}'
        if index:
            destination.mkdir(exist_ok=False)
            success = llm.reset_prefix_cache(reset_running_requests=False, reset_connector=False)
            write_json(destination / 'prefix-cache-reset.json', {'performed': True,
                'success': success, 'completed_requests': index, 'utc': utc(),
                'reset_running_requests': False, 'reset_connector': False})
            if success is not True:
                raise RuntimeError('completed-request prefix-cache reset failed')
        records.append(capture(index, destination))
    return records


def worker(args):
    import torch

    plan = json.loads(args.plan.read_text())
    inputs = json.loads((args.output / 'inputs.json').read_text())
    sources = {path: checkout_record(path, expected) for path, expected in plan['sources'].items()}
    locations = {}
    roots = {'vllm': Path(plan['cwd']), 'flashinfer': Path(next(path for path in sources if 'flashinfer' in path))}
    roots['b12x'] = roots['flashinfer']
    for name, root in roots.items():
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None or not Path(spec.origin).resolve().is_relative_to(root):
            raise ValueError(f'{name} import escaped the recorded source isolation: {spec}')
        locations[name] = spec.origin
    engine_args = parse_engine_args(plan, check_only=args.check)
    if args.layer_trace:
        engine_args.worker_cls = 'qualify.vllm_trace_worker.LayerTraceWorker'
    record = {'started_utc': utc(), 'pid': os.getpid(), 'python': sys.executable,
        'sources': sources, 'module_origins': locations, 'engine_args': plain(engine_args),
        'numerical_instrumentation': {'enable_trace_replay': True, 'logprobs_mode': 'raw_logits',
                                     'max_logprobs': -1},
        'versions': {name: importlib.metadata.version(name) for name in
            ('torch', 'vllm', 'flashinfer-python', 'transformers', 'tokenizers', 'nvidia-cutlass-dsl')},
        'inputs': inputs}
    write_json(args.output / 'worker-start.json', record)
    if args.check:
        if torch.cuda.is_initialized():
            raise RuntimeError('oracle preflight initialized CUDA')
        write_json(args.output / 'preflight.json', record | {'passed': True, 'cuda_initialized': False})
        return
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU or torch.cuda.device_count() != 1:
        raise ValueError('the oracle requires only its assigned GPU')
    from vllm import LLM, SamplingParams
    from vllm.logprobs import FlatLogprobs
    from vllm.sampling_params import RequestOutputKind
    from .vllm_observation import observe_worker

    llm = None
    try:
        started = time.monotonic()
        llm = LLM.from_engine_args(engine_args)
        record['startup_seconds'] = time.monotonic() - started
        config = llm.llm_engine.vllm_config
        write_json(args.output / 'effective-config.json', config)
        if config.speculative_config is not None or config.cache_config.cache_dtype != 'bfloat16':
            raise ValueError('constructed engine changed the target arithmetic configuration')
        write_json(args.output / 'worker-before.json', llm.collective_rpc(observe_worker))
        def capture(index, destination):
            if args.layer_trace:
                from .vllm_layer_trace import reset_worker_trace, save_worker_trace
                write_json(destination / 'layer-trace-reset.json', llm.collective_rpc(reset_worker_trace))
            params = SamplingParams(temperature=0, max_tokens=len(inputs['trace_tokens']),
                trace_decode_token_ids=inputs['trace_tokens'], logprobs=-1,
                flat_logprobs=True, detokenize=False, ignore_eos=True,
                repetition_penalty=1, presence_penalty=0, frequency_penalty=0,
                output_kind=RequestOutputKind.FINAL_ONLY)
            write_json(destination / 'request.json', {'repetition': index,
                'prompt_tokens': inputs['prompt_tokens'], 'trace_tokens': inputs['trace_tokens'],
                'sampling_params': plain(params), 'started_utc': utc()})
            started = time.monotonic()
            outputs = llm.generate([{'prompt_token_ids': inputs['prompt_tokens']}], params, use_tqdm=False)
            seconds = time.monotonic() - started
            output, = outputs
            completion, = output.outputs
            if list(output.prompt_token_ids) != inputs['prompt_tokens'] or list(completion.token_ids) != inputs['trace_tokens']:
                raise ValueError('vLLM did not execute the exact recorded prompt and forced continuation')
            if completion.finish_reason != 'length':
                raise ValueError(f'forced continuation ended unexpectedly: {completion.finish_reason}')
            if not isinstance(completion.logprobs, FlatLogprobs):
                raise TypeError('vLLM did not return the requested flat full-vocabulary scores')
            logits = reconstruct_logits(completion.logprobs, steps=len(inputs['trace_tokens']), vocabulary=VOCABULARY)
            torch.save({'schema_version': 1, 'prompt_tokens': inputs['prompt_tokens'],
                'teacher_tokens': inputs['trace_tokens'][:-1], 'logits': logits,
                'model_id': MODEL_ID, 'revision': REVISION}, destination / 'vllm.pt')
            report = None
            if args.layer_trace:
                traces = llm.collective_rpc(save_worker_trace,
                    args=(str(destination / 'vllm-layer-tensors.pt'),))
                raw_layers = torch.load(destination / 'vllm-layer-tensors.pt', map_location='cpu', weights_only=True)
                torch.save({'schema_version': 1, 'prompt_tokens': inputs['prompt_tokens'],
                    'teacher_tokens': inputs['trace_tokens'][:-1], 'model_id': MODEL_ID, 'revision': REVISION,
                    'logits': logits, 'layer_outputs': raw_layers['layer_outputs'],
                    'stage_outputs': raw_layers['stage_outputs'],
                    **{key: raw_layers[key] for key in ('stage_storage_dtype', 'layer_source_dtypes',
                                                       'stage_source_dtypes')}}, destination / 'vllm-layers.pt')
                baseline = torch.load(args.output / 'vllm-baseline.pt', map_location='cpu', weights_only=True)
                equal = [torch.equal(actual, reference.float())
                         for actual, reference in zip(logits, baseline['logits'], strict=True)]
                report = {'workers': traces, 'logits_bitwise_equal_per_step': equal,
                    'baseline': file_record(args.output / 'vllm-baseline.pt'),
                    'artifact': file_record(destination / 'vllm-layers.pt'),
                    'passed': len(traces) == 1 and traces[0]['passed'] and all(equal)}
                write_json(destination / 'layer-trace.json', report)
            result = {'repetition': index, 'generation_seconds': seconds,
                'finished_utc': utc(), 'trace': file_record(destination / 'vllm.pt'),
                'generated_tokens': list(completion.token_ids), 'finish_reason': completion.finish_reason,
                'stop_reason': completion.stop_reason, 'steps': len(logits), 'vocabulary': VOCABULARY,
                'layer_trace_passed': None if report is None else report['passed']}
            write_json(destination / 'request-capture.json', result)
            return result

        repetitions = capture_cold_repetitions(llm, count=args.repetitions, output=args.output, capture=capture)
        write_json(args.output / 'worker-after.json', llm.collective_rpc(observe_worker))
        record.update(repetitions[0], repetitions=repetitions, finished_utc=utc(),
            sources_after={path: checkout_record(path, plan['sources'][path]) for path in sources})
        if args.layer_trace:
            record['layer_trace_passed'] = all(item['layer_trace_passed'] for item in repetitions)
        write_json(args.output / 'capture.json', record)
        if args.layer_trace and not record['layer_trace_passed']:
            raise ValueError('layer trace failed coverage/context or unchanged-logit validation')
        print(json.dumps({'captured_steps': repetitions[0]['steps'], 'vocabulary': VOCABULARY,
                          'repetitions': len(repetitions)}), flush=True)

    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown(timeout=60)


def parent(args):
    from .text_oracle import load_trace
    import torch

    plan = json.loads(args.plan.read_text())
    receipt = json.loads(args.decoder_reference.read_text())
    expected = {'flare': receipt['artifacts']['tensors/native-sequential-k1.pt']['sha256'],
                'canonical': receipt['artifacts']['tensors/canonical-reference.pt']['sha256']}
    paths = {'flare': args.flare_trace, 'canonical': args.canonical_trace}
    for name, path in paths.items():
        if digest(path) != expected[name]:
            raise ValueError(f'{name} trace differs from the recorded comparison input')
    native, canonical = (load_trace(paths[name]) for name in ('flare', 'canonical'))
    if (native.prompt_tokens, native.teacher_tokens, len(native.logits)) != (
            canonical.prompt_tokens, canonical.teacher_tokens, len(canonical.logits)):
        raise ValueError('recorded Flare and canonical contexts differ')
    trace = torch.load(args.flare_trace, weights_only=True, map_location='cpu')['tokens']
    if list(trace[:-1]) != list(native.teacher_tokens) or len(trace) != len(native.logits):
        raise ValueError('recorded continuation does not match its logit contexts')
    if plan['gpu_uuid'] != GPU or plan['declared_revision'] != REVISION:
        raise ValueError('reference plan does not identify the assigned checkpoint and GPU')
    if args.layer_trace:
        if args.baseline_vllm_trace is None:
            raise ValueError('--layer-trace requires --baseline-vllm-trace')
        baseline_receipt = args.baseline_vllm_trace.parent / 'capture.json'
        baseline_record = json.loads(baseline_receipt.read_text())
        if digest(args.baseline_vllm_trace) != baseline_record['trace']['sha256']:
            raise ValueError('baseline vLLM logits differ from their completed capture receipt')
        baseline = load_trace(args.baseline_vllm_trace)
        if (baseline.prompt_tokens, baseline.teacher_tokens, len(baseline.logits)) != (
                native.prompt_tokens, native.teacher_tokens, len(native.logits)):
            raise ValueError('baseline vLLM contexts differ from the trace request')
    for path, expected_hash in plan['sha256'].items():
        if digest(path) != expected_hash:
            raise ValueError(f'reference launch dependency changed: {path}')
    for path, pin in plan['sources'].items():
        checkout_record(path, pin)
    checkpoint = checkpoint_identity(plan, args.checkpoint_reference)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, path in paths.items():
        shutil.copyfile(path, args.output / f'{name}.pt')
    if args.layer_trace:
        shutil.copyfile(args.baseline_vllm_trace, args.output / 'vllm-baseline.pt')
        shutil.copyfile(baseline_receipt, args.output / 'vllm-baseline-capture.json')
    inputs = {'prompt_tokens': list(native.prompt_tokens), 'trace_tokens': list(trace),
        'teacher_tokens': list(native.teacher_tokens), 'sources': {name: file_record(path) for name, path in paths.items()}}
    write_json(args.output / 'inputs.json', inputs)
    write_json(args.output / 'checkpoint-before.json', checkpoint)
    source_files = [*sorted((ROOT / 'qualify').glob('*.py')), args.plan]
    frozen = {str(path): digest(path) for path in source_files}
    for path in source_files:
        destination = args.output / 'source' / (str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    env = dict(os.environ)
    env.update({key: value for key, value in plan['environment'].items() if key not in ('HOME', 'USER', 'LOGNAME', 'PATH')})
    env['PYTHONPATH'] += ':' + str(ROOT)
    env['VLLM_B12X_DUMP_QUERIES'] = str(args.output / 'declarations.jsonl')
    # Worker provenance uses a trusted callable over the offline engine's local RPC.
    env['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
    if args.layer_trace:
        env['FLARE_VLLM_LAYER_TRACE_INPUTS'] = str(args.output / 'inputs.json')
    if args.check:
        env['CUDA_VISIBLE_DEVICES'] = ''
    command = [plan['python'], '-u', '-m', 'qualify.vllm_text', '--worker', '--plan', str(args.plan), '--output', str(args.output)]
    if args.check:
        command.append('--check')
    if args.layer_trace:
        command.append('--layer-trace')
    command.extend(('--repetitions', str(args.repetitions)))
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
                    'VLLM_B12X_DUMP_QUERIES', 'VLLM_ALLOW_INSECURE_SERIALIZATION',
                    *(('FLARE_VLLM_LAYER_TRACE_INPUTS',) if args.layer_trace else ()))}})
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
        checkpoint_after = checkpoint_identity(plan, args.checkpoint_reference)
        write_json(args.output / 'checkpoint-after.json', checkpoint_after)
        gpu = subprocess.run(['nvidia-smi', '-i', GPU, '--query-gpu=memory.used', '--format=csv,noheader'], capture_output=True, text=True)
        write_json(args.output / 'exit.json', {'started_utc': started, 'finished_utc': utc(),
            'pid': None if process is None else process.pid,
            'exit_code': None if process is None else process.returncode,
            'supervisor_error': failure, 'changed_sources': changed,
            'checkpoint_unchanged': checkpoint_after == checkpoint,
            'process_absent': process is None or not Path(f'/proc/{process.pid}').exists(),
            'gpu_memory_after_exit': gpu.stdout.strip(), 'gpu_query_exit_code': gpu.returncode,
            'log': file_record(args.output / 'worker.log')})
    if failure or process is None or process.returncode or changed:
        raise RuntimeError(f'oracle process failed; inspect {args.output / "exit.json"}')
    print(json.dumps({'exit_code': 0, 'output': str(args.output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path('/tmp/sparkgap-vllm-tp1-reference-64fc1eec/plan.json'))
    parser.add_argument('--flare-trace', type=Path, default=Path('/tmp/sparkgap-decoder-sequential-k1.pt'))
    parser.add_argument('--canonical-trace', type=Path, default=Path('/tmp/sparkgap-canonical-text-short.pt'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-reference', type=Path)
    parser.add_argument('--decoder-reference', type=Path)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--repetitions', type=int, choices=(1, 2), default=1,
                        help='Cold requests in one engine, resetting prefix reuse between requests')
    parser.add_argument('--layer-trace', action='store_true',
                        help='Capture actual layer tensors and require unchanged baseline logits')
    parser.add_argument('--baseline-vllm-trace', type=Path,
                        help='Completed full-vocabulary vLLM artifact required by --layer-trace')
    args = parser.parse_args()
    if not args.worker and (args.checkpoint_reference is None or args.decoder_reference is None):
        parser.error('--checkpoint-reference and --decoder-reference are required')
    args.plan, args.output = args.plan.resolve(), args.output.resolve()
    try:
        (worker if args.worker else parent)(args)
    except BaseException:
        if args.worker and args.output.is_dir():
            write_json(args.output / 'failure.json', {'utc': utc(), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
