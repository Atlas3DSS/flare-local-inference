"""Teacher-forced exact-checkpoint MTP component comparison.

Each backend consumes the same stored target residuals and shifted token IDs.
The pinned vLLM owner runs its ordinary weight loader, preparation coordinator,
QSA owner, decoder, and shared BF16 vocabulary head without a target model.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch

from qualify.provenance import file_record, plain, utc, write_json

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path('/data/checkpoints-masters/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD')
TRACE = Path('/data/sparkgap-artifacts/text-layer-trace/vllm-moe-inprocess-repeat/vllm-layers.pt')
VLLM = Path('/home/luke/projects/vllm-lunar-leviathan')
FI = Path('/tmp/lunar-rebase-20261008/flashinfer-pinned')
PLAN = Path('/tmp/sparkgap-vllm-tp1-reference-64fc1eec/plan.json')
MODEL_ID = 'local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD'
REVISION = 'dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2'


class _ObservedCall:
    def __init__(self, operation, before=None, after=None):
        self.operation, self.before, self.after = operation, before, after

    def __getattr__(self, name):
        return getattr(self.operation, name)

    def __call__(self, *args, **kwargs):
        if self.before is not None:
            self.before(args, kwargs)
        result = self.operation(*args, **kwargs)
        if self.after is not None:
            self.after(args, kwargs, result)
        return result


class StageTrace:
    """Copies existing operands into separate storage; never substitutes math."""

    def __init__(self):
        self.buffers, self.handles, self.originals = {}, [], []

    def record(self, name, value):
        if not isinstance(value, torch.Tensor) or value.ndim < 2:
            raise TypeError(f'{name}: stage requires row tensor')
        value = value.reshape(value.shape[0], -1)
        prior = self.buffers.get(name)
        if prior is None or prior.shape != value.shape or prior.dtype != value.dtype:
            if value.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError('stage storage must be allocated by eager priming')
            prior = torch.empty_like(value)
            self.buffers[name] = prior
        prior.copy_(value)

    def wrap(self, owner, name, *, before=None, after=None):
        original = getattr(owner, name)
        self.originals.append((owner, name, original))
        setattr(owner, name, _ObservedCall(original, before, after))

    def cpu(self, rows):
        return {name: (value if name.startswith('history_') else value[:rows]).cpu().clone()
                for name, value in self.buffers.items()}

    def close(self):
        for handle in self.handles:
            handle.remove()
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)

    def install_flare(self, op):
        self.wrap(op, 'feedback',
            before=lambda args, kwargs: self.record('embedding', op.scratch.embedding),
            after=lambda args, kwargs, result: self.record('feedback', op.scratch.residual))
        for prefix, hc in (('attn', op.attn_hc), ('mlp', op.mlp_hc), ('final', op.final_mixer)):
            def mix(args, kwargs, result, prefix=prefix, hc=hc):
                for field in ('normalized', 'projected_down', 'bottleneck', 'gate_logits'):
                    self.record(f'{prefix}_{field}', getattr(hc.scratch, field))
                if hc.injection is not None:
                    self.record(f'{prefix}_injection_logits', hc.scratch.injection_logits)
                self.record(f'{prefix}_mix', result)
            self.wrap(hc, 'mix', after=mix)
            if hc.injection is not None:
                for method in ('combine', 'combine_norm'):
                    self.wrap(hc, method, after=lambda args, kwargs, result, prefix=prefix:
                              self.record(f'{prefix}_residual', result))
        def attention(args, kwargs, result):
            self.record('attn_output', result)
            for name, value in (('qsa_query', op.scratch.qsa.query), ('qsa_key', op.scratch.qsa.key),
                                ('qsa_attended', op.scratch.qsa.attended), ('qsa_gated', op.scratch.qsa.gated),
                                ('qsa_selected', op.scratch.qsa.selected_positions),
                                ('history_keys', op.attention.state.main_k_cache[1, :8]),
                                ('history_values', op.attention.state.main_v_cache[1, :8])):
                self.record(name, value)
            for key, value in (('q_projected', op.scratch.qsa.projected_q),
                               ('k_projected', op.scratch.qsa.projected_k),
                               ('v_projected', op.scratch.qsa.projected_v),
                               ('index_projected', op.scratch.qsa.projected_index)):
                self.record(key, value)
        self.wrap(op, 'attention', after=attention)
        def moe(args, kwargs, result):
            self.record('moe_router_logits', op.moe.router_logits)
            self.record('moe_ids', op.moe.router.ids)
            self.record('moe_weights', op.moe.router.weights)
            self.record('moe_shared', op.moe.shared.output)
            self.record('moe_routed', op.moe.routed_output)
            self.record('mlp_output', result)
        self.wrap(op, 'moe', after=moe)

    def install_vllm(self, model):
        from qualify.vllm_layer_trace import LayerTrace
        language, layer = model.model, model.model.layers[0]
        self.wrap(language, '_prepare_feedback',
            before=lambda args, kwargs: self.record('embedding', args[0]),
            after=lambda args, kwargs, result: self.record('feedback', result))
        for prefix, hc in (('attn', layer.attn_hyper_connection), ('mlp', layer.mlp_hyper_connection),
                           ('final', language.hyper_connection_mixer)):
            self.wrap(hc, '_mix_normalized', before=lambda args, kwargs, prefix=prefix:
                      self.record(f'{prefix}_normalized', args[0]))
            down = hc.input_mix_weight_down_block_inject if hc.use_combine else hc.input_mix_weight_down
            def projected(module, args, result, prefix=prefix, hc=hc):
                value = result[0] if isinstance(result, tuple) else result
                self.record(f'{prefix}_projected_down', value[:, :320])
                if hc.use_combine:
                    self.record(f'{prefix}_injection_logits', value[:, 320:324])
            self.handles.append(down.register_forward_hook(projected))
            def up(module, args, result, prefix=prefix):
                self.record(f'{prefix}_bottleneck', args[0])
                self.record(f'{prefix}_gate_logits', result[0] if isinstance(result, tuple) else result)
            self.handles.append(hc.input_mix_weight_up.register_forward_hook(up))
            def mixed(args, kwargs, result, prefix=prefix):
                self.record(f'{prefix}_mix', result[1])
                if prefix != 'attn':
                    self.record('attn_residual' if prefix == 'mlp' else 'mlp_residual', result[0])
            self.wrap(hc, 'mix' if prefix == 'attn' else 'combine_and_mix', after=mixed)
        for name, module in (('attn_output', layer.self_attn), ('mlp_output', layer.mlp)):
            self.handles.append(module.register_forward_hook(lambda module, args, result, name=name:
                self.record(name, result[0] if isinstance(result, tuple) else result)))
        def qkv(module, args, result):
            value = result[0] if isinstance(result, tuple) else result
            for name, tensor in zip(('q_projected', 'k_projected', 'v_projected'), value.split((12288, 512, 512), dim=-1)):
                self.record(name, tensor)
        self.handles.append(layer.self_attn.qkv_proj.register_forward_hook(qkv))
        self.handles.append(layer.self_attn.indexer.index_qk_proj.register_forward_hook(
            lambda module, args, result: self.record('index_projected', result[0] if isinstance(result, tuple) else result)))
        def qsa_before(args, kwargs):
            self.record('qsa_query', kwargs['query'])
            self.record('qsa_key', kwargs['key'])
        def qsa_after(args, kwargs, result):
            self.record('qsa_attended', kwargs['output'])
            self.record('qsa_selected', layer.self_attn._selected_positions[:kwargs['rows']])
            keys, values = layer.self_attn.impl._kv_cache_views(layer.self_attn.kv_cache)
            self.record('history_keys', keys[1, :8])
            self.record('history_values', values[1, :8])
        self.wrap(layer.self_attn, '_run_b12x_qsa', before=qsa_before, after=qsa_after)
        self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
            lambda module, args: self.record('qsa_gated', args[0])))
        LayerTrace.install_moe(self, layer.mlp.experts)


def linear_inventory(model):
    result = {}
    for name, module in model.named_modules():
        weight = getattr(module, 'weight', None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.dtype != torch.bfloat16:
            continue
        method = getattr(module, 'quant_method', None)
        implementation = getattr(method, '_gemm_impl', None)
        plans = getattr(module, 'b12x_bf16_gemv_plans', None)
        result[name] = {'weight_shape': list(weight.shape), 'weight_dtype': str(weight.dtype),
            'module_class': f'{type(module).__module__}.{type(module).__qualname__}',
            'quant_method': None if method is None else f'{type(method).__module__}.{type(method).__qualname__}',
            'gemm_implementation': None if implementation is None else
                f'{implementation.__module__}.{implementation.__qualname__}',
            'prepared_plans': None if plans is None else
                {str(rows): plain(plan.selection) for rows, plan in plans.items()}}
    return result


@contextmanager
def observe_functional_linear(model):
    """Record actual functional.linear invocations without changing their operands."""
    import torch.nn.functional as functional
    names = {}
    for name, module in model.named_modules():
        weight = getattr(module, 'weight', None)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            names.setdefault(weight.data_ptr(), []).append(name)
    original, calls = functional.linear, []
    def observed(source, weight, bias=None):
        result = original(source, weight, bias)
        calls.append({'modules': names.get(weight.data_ptr(), []), 'source_shape': list(source.shape),
                      'weight_shape': list(weight.shape), 'source_dtype': str(source.dtype),
                      'weight_dtype': str(weight.dtype), 'output_dtype': str(result.dtype)})
        return result
    functional.linear = observed
    try:
        yield calls
    finally:
        functional.linear = original


def fixture_from_trace(trace):
    if trace['model_id'] != MODEL_ID or trace['revision'] != REVISION:
        raise ValueError('MTP feedback source must identify the exact checkpoint')
    prompt, teachers = trace['prompt_tokens'], trace['teacher_tokens']
    layers = trace['layer_outputs']
    if len(layers) != len(teachers) + 1 or not prompt or not teachers:
        raise ValueError('MTP source requires one target context per teacher token plus the final context')
    cases = []
    for step, teacher in enumerate(teachers):
        residual = layers[step][-1]
        expected_rows = len(prompt) if step == 0 else 1
        if residual.shape != (expected_rows, 10240) or not torch.isfinite(residual).all():
            raise ValueError('MTP feedback requires finite full four-stream target residuals')
        if residual.device.type != 'cpu' or not torch.equal(residual.bfloat16().float(), residual.float()):
            raise ValueError('MTP target residual source must preserve exact BF16 values on CPU')
        ids = prompt[1:] + [teacher] if step == 0 else [teacher]
        start = 0 if step == 0 else len(prompt) + step - 1
        cases.append({'step': step, 'target_context': prompt + teachers[:step],
            'hidden': residual.bfloat16().contiguous(),
            'token_ids': torch.tensor(ids, dtype=torch.int64),
            'positions': torch.arange(start, start + expected_rows, dtype=torch.int64),
            'is_prefilling': step == 0})
    return {'schema_version': 1, 'model_id': MODEL_ID, 'revision': REVISION,
            'cases': cases, 'unconsumed_target_contexts': 1}


def fixture(args):
    trace = torch.load(args.trace, map_location='cpu', weights_only=True)
    data = fixture_from_trace(trace)
    torch.save(data, args.output / 'fixture.pt')
    return {'source': file_record(args.trace), 'fixture': file_record(args.output / 'fixture.pt'),
        'cases': [{k: plain(v) if not isinstance(v, torch.Tensor) else
                   {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in c.items()} for c in data['cases']],
        'feedback': 'target residual h[i], token x[i+1], unshifted position i',
        'final_target_context_excluded': 'no following teacher token is stored for that context'}


def device():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != 'GPU-afd4b1ad-8a64-7057-4bde-241822724c7f':
        raise RuntimeError('MTP qualification requires the assigned GPU4 UUID')
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError('MTP qualification requires one SM120 device')
    return torch.device('cuda:0')


def inputs(args):
    value = torch.load(args.output / 'fixture.pt', map_location='cpu', weights_only=True)
    if value['model_id'] != MODEL_ID or value['revision'] != REVISION:
        raise ValueError('fixture checkpoint identity mismatch')
    return value['cases']


def source_records():
    paths = sorted((ROOT / 'flare/preparation').rglob('*.py'))
    paths += sorted((ROOT / 'flare/recipes/qwen38').glob('*.py'))
    return [file_record(path) for path in [Path(__file__), Path(__file__).with_name('mtp_operators.py'), *paths]]


@torch.no_grad()
def flare(args):
    import b12x
    from b12x.preparation import PreparationSession
    from flare.preparation import inspect_qwen38_checkpoint
    from flare.preparation.qwen38 import TEXT_PREFIX
    from flare.preparation.scratch import ScratchArena
    from flare.recipes.qwen38.attention import QSAMetadata, QSAState, RotaryTables
    from flare.recipes.qwen38.mtp import MTPScratch, PreparedMTP
    from flare.recipes.qwen38.mtp_weights import MTPWeights
    from flare.recipes.qwen38.weights import QwenWeightLoader

    dev = device()
    manifest = inspect_qwen38_checkpoint(args.snapshot)
    cases, sources = inputs(args), source_records()
    with QwenWeightLoader(manifest, dev, prefixes=(f'{TEXT_PREFIX}.embed_tokens.', 'lm_head.')) as loader:
        embedding = loader.load_tensor(f'{TEXT_PREFIX}.embed_tokens.weight')
        head = loader.load_tensor('lm_head.weight')
    weights = MTPWeights.load(manifest, dev, embedding_weight=embedding, lm_head_weight=head)
    cache = QSAState.allocate(pages=2, slots=1, device=dev, draft_tokens=7)
    rotary = RotaryTables.allocate(weights.module.config, 128, dev)
    arena = ScratchArena(dev)
    graphs, owners, outputs, allocation_checks = [], [], [], []
    with PreparationSession(device=dev, autotune=False, compile_workers=0) as session:
        try:
            for capacity in (8, 1):
                scratch = MTPScratch.allocate(capacity, 1, dev)
                hidden = torch.zeros((capacity, 10240), dtype=torch.bfloat16, device=dev)
                tokens = torch.zeros(capacity, dtype=torch.int64, device=dev)
                selected = torch.zeros(1, dtype=torch.int64, device=dev)
                metadata = QSAMetadata.allocate(tokens=capacity, batch=1, context=128, device=dev)
                op = PreparedMTP(weights, scratch, tokens, hidden, cache, metadata, rotary, arena,
                    output_indices=selected, session=session, name=f'mtp.{capacity}', max_cache_pages=2)
                trace = StageTrace() if args.trace_stages else None
                if trace is not None:
                    trace.install_flare(op)
                owners.append((op, tokens, hidden, selected, metadata, trace))
            arena.finalize(); arena.freeze()
            stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
            for op, *_ in owners:
                graph = torch.cuda.CUDAGraph(); graphs.append(graph)
                with torch.cuda.stream(stream):
                    op()
                stream.synchronize()
                with session.capture(), torch.cuda.graph(graph, stream=stream):
                    op()
            session.freeze()
            for name in cache.__dataclass_fields__:
                value = getattr(cache, name)
                value.fill_(-1 if value.dtype == torch.int64 else 0)
            for case in cases:
                index = 0 if case['is_prefilling'] else 1
                op, tokens, hidden, selected, metadata, trace = owners[index]
                count, start = len(case['token_ids']), int(case['positions'][0])
                tokens.zero_(); hidden.zero_()
                tokens[:count].copy_(case['token_ids']); hidden[:count].copy_(case['hidden'])
                selected.fill_(count - 1)
                metadata.main_block_table[0].copy_(torch.tensor([1, 0], dtype=torch.int32, device=dev))
                metadata.compressed_block_table.copy_(metadata.main_block_table)
                metadata.raw_state_slot_ids.fill_(0)
                metadata.request_ids.fill_(-1); metadata.request_ids[:count].fill_(0)
                metadata.query_positions.fill_(-1); metadata.query_positions[:count].copy_(case['positions'])
                metadata.rope_positions.fill_(-1)
                metadata.rope_positions[:count].copy_(case['positions'][:, None].expand(-1, 3))
                metadata.sequence_lengths.fill_(start + count)
                metadata.query_start_loc[-1] = count
                metadata.num_accepted_tokens.fill_(1)
                metadata.is_prefilling.fill_(case['is_prefilling'])
                metadata.slot_mapping.fill_(-1)
                metadata.slot_mapping[:count].copy_(case['positions'] + 64)
                arena.tensor.fill_(0xA5)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()['allocation.all.allocated']
                graphs[index].replay(); torch.cuda.synchronize()
                after = torch.cuda.memory_stats()['allocation.all.allocated']
                allocation_checks.append({'step': case['step'], 'before': before, 'after': after, 'equal': before == after})
                outputs.append({'step': case['step'], 'residual': op.scratch.residual[:count].cpu().clone(),
                    'hidden': op.scratch.hidden[:count].cpu().clone(), 'logits': op.scratch.logits.cpu().clone(),
                    'stages': {} if trace is None else trace.cpu(count),
                    'attention_constants': {'q_norm': weights.module.layers[0].self_attn.q_norm.weight.cpu().clone(),
                        'k_norm': weights.module.layers[0].self_attn.k_norm.weight.cpu().clone(),
                        'rope_cos': rotary.cos[:8].cpu().clone(), 'rope_sin': rotary.sin[:8].cpu().clone()}})
            if any(not check['equal'] for check in allocation_checks):
                raise RuntimeError('Flare MTP graph replay allocated')
            torch.save(outputs, args.output / 'flare.pt')
            operator_probe = None
            if args.operator_inputs is not None:
                from qualify.mtp_operators import run_flare
                operator_probe = run_flare(args, owners, cases)
            return {'sources': sources, 'sources_unchanged': sources == source_records(),
                'operator_probe': operator_probe,
                'b12x_origin': b12x.__file__, 'stored_mtp_bytes': weights.stored_bytes,
                'shared_embedding_head': True, 'arena_bytes': arena.nbytes,
                'capacities': [8, 1], 'kv_dtype': 'bfloat16', 'cache_page_size': 64,
                'allocation_checks': allocation_checks, 'result': file_record(args.output / 'flare.pt'),
                'preparation': 'public default plans; BF16Projection explicit recipe override retained'}
        finally:
            torch.cuda.synchronize()
            for graph in graphs:
                graph.reset()


@torch.no_grad()
def vllm(args):
    import b12x
    import vllm as package
    from safetensors import safe_open
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import initialize_model_parallel, init_distributed_environment, cleanup_dist_env_and_memory
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.model_loader.utils import configure_quant_config, process_weights_after_loading
    from vllm.model_executor.warmup.b12x_prepare import begin_b12x_preparation
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForConditionalGeneration
    from vllm.models.qwen4_exp.nvidia.mtp import Qwen4ExpMTP
    from vllm.models.qwen4_exp.nvidia.b12x_qsa import Qwen4ExpQSAMetadata
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager
    from qualify.provenance import checkout_record
    from qualify.vllm_text import checkpoint_identity

    plan = json.loads(PLAN.read_text())
    sources = [checkout_record(path, expected) for path, expected in plan['sources'].items()]
    checkpoint = checkpoint_identity(plan)
    if not Path(package.__file__).resolve().is_relative_to(VLLM) or not Path(b12x.__file__).resolve().is_relative_to(FI):
        raise RuntimeError('reference imports escaped the pinned vLLM/FlashInfer checkouts')
    if os.environ.get('VLLM_MTP_NVFP4_LM_HEAD') != '0':
        raise RuntimeError('the checkpoint shared BF16 vocabulary head must remain uncompressed')
    dev, cases = device(), inputs(args)
    config = EngineArgs(model=str(args.snapshot), dtype='bfloat16', quantization='modelopt_mixed',
        tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1,
        max_model_len=128, max_num_seqs=1, max_num_batched_tokens=8,
        enable_chunked_prefill=True, kv_cache_dtype='bfloat16', block_size=48,
        mamba_cache_mode='align', linear_backend='b12x', moe_backend='b12x',
        gdn_prefill_backend='b12x', gdn_decode_kernel='b12x', enable_flashinfer_autotune=False,
        enforce_eager=True, load_format='instanttensor',
        speculative_config={'method': 'mtp', 'num_speculative_tokens': 7}).create_engine_config()
    outputs, preparation, executed_linear_calls = [], [], []
    with set_current_vllm_config(config), tempfile.TemporaryDirectory(prefix='flare-mtp-dist-') as temporary, ExitStack() as cleanup:
        init_distributed_environment(world_size=1, rank=0, local_rank=0,
            distributed_init_method=f'file://{temporary}/rendezvous', backend='nccl')
        cleanup.callback(cleanup_dist_env_and_memory)
        initialize_model_parallel(1, 1)
        init_workspace_manager(dev); cleanup.callback(reset_workspace_manager)
        torch.set_default_dtype(torch.bfloat16)
        configure_quant_config(config.quant_config, Qwen4ExpForConditionalGeneration)
        with torch.device(dev):
            model = Qwen4ExpMTP(vllm_config=config)
        model.eval().requires_grad_(False)
        index = json.loads((args.snapshot / 'model.safetensors.index.json').read_text())['weight_map']
        names = sorted(name for name in index if name.startswith('mtp.') or
                       name in ('model.language_model.embed_tokens.weight', 'lm_head.weight'))
        if len(names) != 4639:
            raise RuntimeError(f'exact MTP + shared embedding/head tensor count differs: {len(names)}')
        loaded_bytes = 0
        def weights():
            nonlocal loaded_bytes
            for shard in sorted({index[name] for name in names}):
                with safe_open(args.snapshot / shard, framework='pt', device='cpu') as handle:
                    for name in names:
                        if index[name] == shard:
                            value = handle.get_tensor(name)
                            loaded_bytes += value.nbytes
                            yield name, value
        loaded = model.load_weights(weights())
        process_weights_after_loading(model, config.model_config, dev)
        attention = model.model.layers[0].self_attn
        spec = attention.get_kv_cache_spec(config)
        storage = torch.zeros(3 * spec.page_size_bytes // 2, dtype=torch.bfloat16, device=dev)
        cache = storage.as_strided((3, 2, 48, 512), (spec.page_size_bytes // 2, 48 * 512, 512, 1))
        attention.bind_kv_cache(cache)
        worker = SimpleNamespace(vllm_config=config, model_config=config.model_config,
            scheduler_config=config.scheduler_config, device=dev, rank=0, use_v2_model_runner=False,
            model_runner=SimpleNamespace(mm_registry=None, decode_query_len=8, _draft_workspace_lane=0),
            get_model=lambda: model, get_draft_model=lambda: None)
        for stage in ('weights', 'state'):
            coordinator = begin_b12x_preparation(worker, stage=stage)
            if stage == 'weights':
                cleanup.callback(worker._b12x_session.close)
            outcome = coordinator.status()
            while not outcome['done']:
                outcome = coordinator.advance()
            preparation.append({'stage': stage, 'outcome': plain(outcome)})
            if outcome['error'] is not None:
                raise RuntimeError(f'reference preparation failed: {outcome["error"]}')
        trace = StageTrace() if args.trace_stages else None
        if trace is not None:
            trace.install_vllm(model); cleanup.callback(trace.close)
        storage.zero_()
        for tensor in attention.get_recurrent_checkpoint_tensors():
            tensor.fill_(-1 if tensor.dtype == torch.int64 else 0)
        for case in cases:
            count, start = len(case['token_ids']), int(case['positions'][0])
            positions = case['positions'].to(dev)
            selected_rows = torch.tensor([count - 1], dtype=torch.int64, device=dev)
            model.model.set_prefill_output_indices(selected_rows)
            metadata = Qwen4ExpQSAMetadata(num_actual_tokens=count, max_query_len=count,
                query_start_loc=torch.tensor([0, count], dtype=torch.int32, device=dev),
                max_seq_len=start + count, seq_lens=torch.tensor([start + count], dtype=torch.int32, device=dev),
                block_table=torch.tensor([[1, 0, 2]], dtype=torch.int32, device=dev),
                slot_mapping=positions + 48, request_ids=torch.zeros(count, dtype=torch.int32, device=dev),
                is_prefilling=torch.tensor([case['is_prefilling']], dtype=torch.bool, device=dev),
                qsa_state_slot_ids=torch.zeros(1, dtype=torch.int32, device=dev),
                qsa_state_is_fresh=torch.tensor([case['step'] == 0], dtype=torch.bool, device=dev),
                qsa_num_accepted_tokens=torch.ones(1, dtype=torch.int32, device=dev))
            with observe_functional_linear(model) as linear_calls, set_forward_context(
                    {attention.layer_name: metadata}, vllm_config=config, num_tokens=count):
                hidden, residual = model(case['token_ids'].to(dev), positions[None].expand(3, -1).contiguous(),
                    hidden_states=case['hidden'].to(dev))
                logits = model.compute_logits(hidden[-1:])
            torch.cuda.synchronize()
            executed_linear_calls.append({'step': case['step'], 'calls': linear_calls})
            outputs.append({'step': case['step'], 'selected_rows': selected_rows.cpu().tolist(),
                'residual': residual.cpu().clone(), 'hidden': hidden.cpu().clone(), 'logits': logits.cpu().clone(),
                'stages': {} if trace is None else trace.cpu(count),
                'attention_constants': {'q_norm': attention.q_norm.weight.cpu().clone(),
                    'k_norm': attention.k_norm.weight.cpu().clone(),
                    'rope_cos': attention.rotary_emb.cos_sin_cache[:8, :32].cpu().clone(),
                    'rope_sin': attention.rotary_emb.cos_sin_cache[:8, 32:].cpu().clone()}})
        torch.save(outputs, args.output / 'vllm.pt')
        operator_probe = None
        if args.operator_inputs is not None:
            from qualify.mtp_operators import run_vllm
            operator_probe = run_vllm(args, model, cases)
        return {'sources': sources, 'checkpoint': checkpoint, 'config': plain(config),
            'operator_probe': operator_probe,
            'preparation': preparation, 'linear_inventory': linear_inventory(model),
            'executed_functional_linear_calls': executed_linear_calls,
            'inherited_bf16_gemv_environment': os.environ.get('VLLM_B12X_BF16_GEMV'),
            'resolved_bf16_gemv_enabled': __import__('vllm.envs', fromlist=['VLLM_B12X_BF16_GEMV']).VLLM_B12X_BF16_GEMV,
            'loaded_checkpoint_tensors': len(names), 'loaded_bytes': loaded_bytes,
            'loaded_parameter_names': sorted(loaded), 'b12x_origin': b12x.__file__, 'vllm_origin': package.__file__,
            'result': file_record(args.output / 'vllm.pt'), 'kv_dtype': str(cache.dtype),
            'cache_page_size': 48, 'cache_page_bytes': spec.page_size_bytes,
            'raw_ring_capacity': attention.raw_ring_capacity, 'has_own_lm_head': model.has_own_lm_head,
            'scope': 'module-only Qwen4ExpMTP serving operators, eager execution; no target model or scheduler',
            'qsa_plans': [plain(c.prepared_plan.prepared.selection) for c in
                (*attention._qsa_prefill_bindings, attention._qsa_decode_context)]}


def compare_output(actual, reference):
    from qualify.layer_comparison import _measurement
    from qualify.text_oracle import _metrics

    rows = reference['selected_rows']
    if rows != [actual['residual'].shape[0] - 1]:
        raise ValueError('MTP reference must select the exact request-tail row')
    result = {}
    for name in ('residual', 'hidden', 'logits'):
        value = actual[name] if name == 'logits' else actual[name][rows]
        if not torch.isfinite(value).all() or not torch.isfinite(reference[name]).all():
            raise ValueError('MTP outputs must be finite')
        result[name] = _measurement(value, reference[name])
    result['vocabulary'] = _metrics(actual['logits'][0].float(), reference['logits'][0].float())
    a_stages, r_stages = actual.get('stages', {}), reference.get('stages', {})
    result['stage_coverage'] = {'missing_in_flare': sorted(r_stages.keys() - a_stages.keys()),
                                'missing_in_vllm': sorted(a_stages.keys() - r_stages.keys())}
    stages = {}
    for name in sorted(a_stages.keys() & r_stages.keys()):
        value, expected = a_stages[name], r_stages[name]
        selected = None
        if value.shape[0] != expected.shape[0]:
            if expected.shape[0] != 1 or value.shape[0] != actual['residual'].shape[0]:
                raise ValueError(f'{name}: unexplained MTP stage row geometry')
            selected, value = rows, value[rows]
        if not torch.isfinite(value).all() or not torch.isfinite(expected).all():
            raise ValueError(f'{name}: MTP stage is nonfinite')
        stages[name] = {'selected_actual_rows': selected, **_measurement(value, expected)}
    result['stages'] = stages
    return result


def compare(args):
    from qualify.layer_comparison import _measurement
    a = torch.load(args.output / 'flare.pt', map_location='cpu', weights_only=True)
    b = torch.load(args.output / 'vllm.pt', map_location='cpu', weights_only=True)
    if len(a) != len(b):
        raise ValueError('MTP step count differs')
    records = []
    for actual, reference in zip(a, b, strict=True):
        if actual['step'] != reference['step']:
            raise ValueError('MTP steps differ')
        records.append({'step': actual['step'], 'reference_selected_rows': reference['selected_rows'],
            **compare_output(actual, reference)})
    return {'reference': 'pinned vLLM Qwen4ExpMTP; identical target-derived feedback',
        'acceptance_threshold': None, 'cases': records,
        'artifacts': [file_record(args.output / name) for name in ('fixture.pt', 'flare.pt', 'vllm.pt')]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backend', choices=('fixture', 'flare', 'vllm', 'compare'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, default=SNAPSHOT)
    parser.add_argument('--trace', type=Path, default=TRACE)
    parser.add_argument('--trace-stages', action='store_true')
    parser.add_argument('--operator-inputs', type=Path,
                        help='immutable recorded MTP fixture and stage tensors for independent identical-input probes')
    args = parser.parse_args()
    if args.operator_inputs is not None and args.backend not in ('flare', 'vllm'):
        parser.error('--operator-inputs requires a model backend')
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f'{args.backend}.json'
    if destination.exists() or (args.backend in ('flare', 'vllm') and (args.output / f'{args.backend}.pt').exists()):
        raise FileExistsError(f'qualification output already exists: {destination}')
    report = {'started_at': utc(), 'command': sys.argv, 'script': file_record(__file__), 'backend': args.backend,
        'environment': {key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'PYTHONPATH',
            'B12X_AUTOTUNE', 'VLLM_MTP_NVFP4_LM_HEAD', 'VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT',
            'VLLM_QWEN3_8_FLASH_NEXT_OVERLAP')},
        'fixture': file_record(args.output / 'fixture.pt') if args.backend != 'fixture' else None}
    try:
        report.update(globals()[args.backend](args)); report['completed'] = True
    except BaseException as error:
        report.update(completed=False, error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['finished_at'] = utc()
        write_json(destination, report)


if __name__ == '__main__':
    main()
