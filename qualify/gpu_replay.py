"""Capture whole-graph GPU decode time with identical prompt token workloads.

Generate a corpus with ``prepare`` and reuse it for both engine commands.
The vLLM plan supplies its existing launch surface as a ``serve_args`` list.
Run each engine under Nsight with ``--trace=cuda-sw,nvtx``,
``--cuda-graph-trace=graph``, ``--capture-range=cudaProfilerApi``,
``--capture-range-end=stop`` and ``--export=sqlite``. Name the reports
``flare-trace`` and ``vllm-trace`` beside their respective output directories,
then run ``python -m qualify.gpu_replay_analysis DIRECTORY``.
"""

import argparse
import json
import os
from pathlib import Path


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(args):
    from transformers import AutoTokenizer
    from qualify.performance import make_prompt

    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    groups = []
    for repeat in range(4):
        for batch in (1, 2, 4):
            cases = []
            for lane in range(batch):
                request, tokens, _ = make_prompt(tokenizer, f'gpu-b{batch}-r{repeat}-lane{lane}', 512, 256)
                request.update(temperature=1, top_k=20, top_p=.95)
                case = {'request': request, 'prompt_tokens': tokens, 'output_tokens': 256}
                cases.append(case)
            groups.append({'batch': batch, 'repeat': repeat, 'measured': repeat > 0, 'cases': cases})
    save(args.output, {'prompt_tokens': 512, 'output_tokens': 256, 'groups': groups})


def flare(args, corpus):
    import torch
    from flare.recipes.qwen38.bootstrap import PreparationConfig, prepare_model
    from qualify.performance_trace import prepared_graph_map
    from qualify.replay_floor import ReplayCase

    result = {'engine': 'flare', 'groups': []}
    with prepare_model(args.snapshot, PreparationConfig(prefill_chunk_size=4096),
                       progress=lambda message: print(message, flush=True)) as model:
        save(args.output / 'graphs.json', prepared_graph_map(model.manifest()))
        runner = ReplayCase(model)
        profiling = False
        try:
            for group in corpus['groups']:
                batch, repeat = group['batch'], group['repeat']
                initial = runner.prepare(group['cases'])
                if group['measured'] and not profiling:
                    torch.cuda.profiler.start()
                    profiling = True
                graph = model.graphs[f'decode{batch}']
                before = torch.cuda.memory_stats()
                with torch.cuda.stream(runner.stream):
                    with torch.cuda.nvtx.range(f'flare.decode.b{batch}.r{repeat}'):
                        for _ in range(255):
                            graph.replay()
                runner.stream.synchronize()
                after = torch.cuda.memory_stats()
                _, records = runner.snapshot(group['cases'])
                keys = ('allocation.all.allocated', 'allocation.all.freed',
                        'segment.all.allocated', 'segment.all.freed')
                stable = all(before[key] == after[key] for key in keys)
                if not stable:
                    raise RuntimeError('decode replay allocated or freed CUDA storage')
                result['groups'].append({'batch': batch, 'repeat': repeat,
                    'measured': group['measured'], 'initial_tokens': initial,
                    'records': records, 'allocator_stable': stable})
                save(args.output / 'results.json', result)
                print(f'flare batch={batch} repeat={repeat} completed 256 tokens per request', flush=True)
        finally:
            if profiling:
                torch.cuda.profiler.stop()
            runner.stream.synchronize()


def vllm(args, corpus):
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.entrypoints.launchers.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.config import ProfilerConfig

    plan = json.loads(args.plan.read_text())
    parsed = make_arg_parser(FlexibleArgumentParser()).parse_args(plan['serve_args'])
    parsed.model = parsed.model_tag
    engine = EngineArgs.from_cli_args(parsed)
    if Path(engine.model).resolve() != args.snapshot.resolve():
        raise ValueError('vLLM launch plan must use the requested checkpoint')
    engine.worker_cls = 'qualify.vllm_graph_worker.GraphTimingWorker'
    engine.profiler_config = ProfilerConfig(profiler='cuda')
    if engine.speculative_config is not None or engine.enforce_eager:
        raise ValueError('require ordinary graph-enabled vLLM')
    os.environ['FLARE_GPU_REPLAY_OUTPUT'] = str(args.output.resolve())
    llm = LLM.from_engine_args(engine)
    profiling = False
    result = {'engine': 'vllm', 'groups': []}
    try:
        for group in corpus['groups']:
            if group['measured'] and not profiling:
                llm.start_profile()
                profiling = True
            cases = group['cases']
            params = [SamplingParams(temperature=case['request']['temperature'],
                      top_k=case['request']['top_k'], top_p=case['request']['top_p'],
                      seed=case['request']['seed'], max_tokens=256) for case in cases]
            outputs = llm.generate([{'prompt_token_ids': case['prompt_tokens']} for case in cases],
                                   params, use_tqdm=False)
            rows = []
            for case, output in zip(cases, outputs, strict=True):
                completion, = output.outputs
                if (list(output.prompt_token_ids) != case['prompt_tokens']
                        or len(completion.token_ids) != 256 or completion.finish_reason != 'length'):
                    raise RuntimeError('vLLM did not complete the identical 256-token workload')
                rows.append({'tokens': list(completion.token_ids), 'finish_reason': completion.finish_reason})
            result['groups'].append({'batch': group['batch'], 'repeat': group['repeat'],
                                    'measured': group['measured'], 'outputs': rows})
            save(args.output / 'results.json', result)
            print(f"vllm batch={group['batch']} repeat={group['repeat']} completed 256 tokens per request", flush=True)
    finally:
        if profiling:
            llm.stop_profile()
        llm.llm_engine.engine_core.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('engine', choices=('prepare', 'flare', 'vllm'))
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--corpus', type=Path)
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be fresh')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.engine == 'prepare':
        return prepare(args)
    if not args.corpus or (args.engine == 'vllm' and not args.plan):
        parser.error('require --corpus and vLLM --plan')
    args.output.mkdir()
    corpus = json.loads(args.corpus.read_text())
    (flare if args.engine == 'flare' else vllm)(args, corpus)


if __name__ == '__main__':
    main()
