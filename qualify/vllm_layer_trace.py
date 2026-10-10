"""Capture actual Qwen decoder boundaries into graph-owned diagnostic buffers.

The four teacher-forced steps share one uninterrupted vLLM request. Tensor copies
and a device step counter are captured alongside the existing model operations;
no residual, projection, or normalization arithmetic is reconstructed here.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import torch

from .provenance import file_record


STAGE_WIDTHS = {
    'embedding': 2560, 'residual_in': 10240,
    **{f'{block}_{field}': width for block in ('attn', 'mlp') for field, width in (
        ('normalized', 10240), ('projected_down', 320), ('bottleneck', 320),
        ('gate_logits', 10240), ('injection_logits', 4), ('mix', 2560),
        ('output', 2560), ('residual', 10240))},
    'gdn_projected_qkvz': 16384, 'gdn_projected_ba': 96, 'gdn_gated': 6144,
    'moe_router_logits': 512, 'moe_ids': 10, 'moe_weights': 10,
    'moe_shared': 2560, 'moe_routed': 2560,
    'hidden': 2560,
}
FLOAT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
INTEGER_DTYPES = (torch.int32, torch.int64)
SOURCE_DTYPES = FLOAT_DTYPES + INTEGER_DTYPES


def _tensor(value):
    return value[0] if isinstance(value, tuple) else value


class LayerTrace:
    """Fixed buffers whose writes observe, but never modify, model operands."""

    def __init__(self, *, prompt_tokens, trace_tokens, device):
        if len(prompt_tokens) != 5 or len(trace_tokens) != 4:
            raise ValueError('layer trace requires the recorded five-token/four-step case')
        self.prompt_tokens = list(prompt_tokens)
        self.trace_tokens = list(trace_tokens)
        self.counter = torch.zeros(1, dtype=torch.int64, device=device)
        self.index = torch.zeros_like(self.counter)
        self.one = torch.ones_like(self.counter)
        self.names = [*(f'layer.{i:02d}' for i in range(48)), *STAGE_WIDTHS]
        self.banks = {name: torch.full((4, 5, 10240 if name.startswith('layer.') else STAGE_WIDTHS[name]),
                        -1 if name == 'moe_ids' else float('nan'),
                        dtype=torch.int64 if name == 'moe_ids' else torch.float32,
                        device=device) for name in self.names}
        self.hits = torch.zeros((len(self.names), 4), dtype=torch.int64, device=device)
        self.capacities = torch.zeros_like(self.hits)
        self.source_dtype_codes = torch.full_like(self.hits, -1)
        self.hit_views = {name: self.hits[i] for i, name in enumerate(self.names)}
        self.capacity_views = {name: self.capacities[i] for i, name in enumerate(self.names)}
        self.dtype_views = {name: self.source_dtype_codes[i] for i, name in enumerate(self.names)}
        self.input_ids = torch.full((4, 5), -1, dtype=torch.int64, device=device)
        self.positions = torch.full((4, 3, 5), -1, dtype=torch.int64, device=device)
        self.position_axes = torch.zeros(4, dtype=torch.int64, device=device)
        self.handles = []
        self.method_owners = []

    def record(self, name, value):
        if not isinstance(value, torch.Tensor) or value.ndim < 2 or value.shape[0] < 1:
            raise TypeError(f'{name}: expected a nonempty row tensor')
        rows = min(value.shape[0], 5)
        flat = value[:rows].reshape(rows, -1)
        bank = self.banks[name]
        if flat.shape[1] != bank.shape[-1]:
            raise ValueError(f'{name}: unexpected tensor width')
        allowed = INTEGER_DTYPES if name == 'moe_ids' else FLOAT_DTYPES
        if value.dtype not in allowed:
            raise TypeError(f'{name}: unsupported diagnostic source dtype')
        # FP32 storage preserves every BF16/FP16/FP32 source value exactly.
        bank[:, :rows].index_copy_(0, self.index, flat.to(bank.dtype).unsqueeze(0))
        self.hit_views[name].index_add_(0, self.index, self.one)
        self.capacity_views[name].index_fill_(0, self.index, value.shape[0])
        self.dtype_views[name].index_fill_(0, self.index, SOURCE_DTYPES.index(value.dtype))

    def begin(self, module, args, kwargs):
        torch.clamp(self.counter, min=0, max=3, out=self.index)
        residual, ids, positions = kwargs['hidden_states'], kwargs['input_ids'], kwargs['positions']
        rows = min(residual.shape[0], 5)
        if ids.dtype not in (torch.int32, torch.int64) or positions.dtype not in (torch.int32, torch.int64):
            raise TypeError('trace input IDs and positions must be int32 or int64')
        if ids.ndim != 1 or ids.shape[0] < rows or positions.ndim not in (1, 2):
            raise ValueError('trace input IDs and positions do not cover the observed rows')
        self.input_ids[:, :rows].index_copy_(0, self.index, ids[:rows].to(torch.int64).unsqueeze(0))
        position_rows = positions.unsqueeze(0) if positions.ndim == 1 else positions
        if position_rows.shape[0] not in (1, 3) or position_rows.shape[1] < rows:
            raise ValueError('unexpected Qwen position axes')
        self.positions[:, :position_rows.shape[0], :rows].index_copy_(
            0, self.index, position_rows[:, :rows].to(torch.int64).unsqueeze(0))
        self.position_axes.index_fill_(0, self.index, position_rows.shape[0])
        self.record('residual_in', residual)
        self.record('embedding', residual[:, :2560])

    def wrap(self, owner, method_name, *, before=None, after=None):
        original = getattr(owner, method_name)

        @functools.wraps(original)
        def observed(*args, **kwargs):
            if before is not None:
                before(args, kwargs)
            result = original(*args, **kwargs)
            if after is not None:
                after(args, kwargs, result)
            return result

        setattr(owner, method_name, observed)
        self.method_owners.append((owner, method_name, original))

    def output_hook(self, module, name):
        def observe(owner, args, result):
            self.record(name, _tensor(result))
        self.handles.append(module.register_forward_hook(observe))

    def install(self, model):
        language = model.language_model.model
        if len(language.layers) != 48 or language.start_layer != 0 or language.end_layer != 48:
            raise ValueError('layer trace requires the complete TP1 Qwen decoder')
        self.handles.append(language.layers[0].register_forward_pre_hook(self.begin, with_kwargs=True))
        for i, layer in enumerate(language.layers):
            if i:
                name = f'layer.{i - 1:02d}'

                def boundary(args, kwargs, result, name=name):
                    self.record(name, result[0])

                self.wrap(layer.attn_hyper_connection, 'combine_and_mix', after=boundary)
                if layer.ple is not None:
                    # PLE receives an already-materialized previous MLP residual.
                    def before_ple(args, kwargs, result, name=name):
                        self.record(name, result)
                        if name == 'layer.00':
                            self.record('mlp_residual', result)
                    self.wrap(layer.attn_hyper_connection, 'combine', after=before_ple)
        first = language.layers[0]
        for prefix, hc in (('attn', first.attn_hyper_connection), ('mlp', first.mlp_hyper_connection)):
            if hc.tp_size != 1 or hc.workspace is None or not hc.use_combine:
                raise ValueError('layer stage trace requires the prepared TP1 hyperconnection path')
            def normalized(args, kwargs, prefix=prefix):
                self.record(f'{prefix}_normalized', args[0])
            self.wrap(hc, '_mix_normalized', before=normalized)

            def down_hook(owner, args, result, prefix=prefix):
                projected = _tensor(result)
                self.record(f'{prefix}_projected_down', projected[:, :320])
                self.record(f'{prefix}_injection_logits', projected[:, 320:324])
            self.handles.append(hc.input_mix_weight_down_block_inject.register_forward_hook(down_hook))

            def up_hook(owner, args, result, prefix=prefix):
                self.record(f'{prefix}_bottleneck', args[0])
                self.record(f'{prefix}_gate_logits', _tensor(result))
            self.handles.append(hc.input_mix_weight_up.register_forward_hook(up_hook))

            def mixed(args, kwargs, result, prefix=prefix):
                self.record(f'{prefix}_mix', result[1])
                if prefix == 'mlp':
                    self.record('attn_residual', result[0])
            self.wrap(hc, 'mix' if prefix == 'attn' else 'combine_and_mix', after=mixed)

        self.output_hook(first.linear_attn, 'attn_output')
        self.output_hook(first.mlp, 'mlp_output')
        self.install_moe(first.mlp.experts)
        gdn = first.linear_attn
        # The projection custom op invokes these modules on its original
        # streams and joins them before the recurrent operation.
        self.output_hook(gdn.in_proj_qkvz, 'gdn_projected_qkvz')
        self.output_hook(gdn.in_proj_ba, 'gdn_projected_ba')

        def gdn_out(owner, args):
            self.record('gdn_gated', args[0])
        self.handles.append(gdn.out_proj.register_forward_pre_hook(gdn_out))

        def finish(args, kwargs, result):
            self.record('layer.47', result[0])
            self.record('hidden', result[1])
            self.counter.add_(1)
        self.wrap(language.hyper_connection_mixer, 'combine_and_mix', after=finish)

    def install_moe(self, runner):
        def routes(args, kwargs, result):
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError('MoE router must return (weights, IDs)')
            weights, ids = result
            logits = kwargs['router_logits']
            if (not isinstance(logits, torch.Tensor) or logits.ndim != 2
                    or not isinstance(weights, torch.Tensor) or weights.ndim != 2
                    or not isinstance(ids, torch.Tensor) or ids.shape != weights.shape
                    or weights.shape[0] != logits.shape[0]):
                raise ValueError('MoE router tensor rows or ranks differ')
            self.record('moe_router_logits', logits)
            self.record('moe_weights', weights)
            self.record('moe_ids', ids)

        def outputs(args, kwargs, result):
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError('MoE execution must return (shared, routed)')
            shared, routed = result
            if (not isinstance(shared, torch.Tensor) or shared.ndim != 2
                    or not isinstance(routed, torch.Tensor) or routed.shape != shared.shape):
                raise TypeError('MoE trace requires finalized shared/routed tensors of equal shape')
            if shared.shape[0] != kwargs['hidden_states'].shape[0]:
                raise ValueError('MoE output rows differ from input rows')
            # The runner has joined its shared-expert stream before this return.
            self.record('moe_shared', shared)
            self.record('moe_routed', routed)

        self.wrap(runner.router, 'select_experts', after=routes)
        self.wrap(runner, '_apply_quant_method', after=outputs)

    def reset(self):
        torch.cuda.synchronize(self.counter.device)
        self.counter.zero_()
        self.index.zero_()
        self.hits.zero_()
        self.capacities.zero_()
        self.source_dtype_codes.fill_(-1)
        self.input_ids.fill_(-1)
        self.positions.fill_(-1)
        self.position_axes.zero_()
        for name, tensor in self.banks.items():
            tensor.fill_(-1 if name == 'moe_ids' else float('nan'))
        torch.cuda.synchronize(self.counter.device)

    def snapshot(self):
        torch.cuda.synchronize(self.counter.device)
        counter = int(self.counter.cpu()[0])
        ids, positions = self.input_ids.cpu(), self.positions.cpu()
        axes, hits, capacities = self.position_axes.cpu(), self.hits.cpu(), self.capacities.cpu()
        dtype_codes = self.source_dtype_codes.cpu()
        source_dtypes = {name: [str(SOURCE_DTYPES[code]) if 0 <= code < len(SOURCE_DTYPES) else None
                                for code in dtype_codes[i].tolist()] for i, name in enumerate(self.names)}
        banks = {name: value.cpu() for name, value in self.banks.items()}
        checks = {'four_forwards': counter == 4, 'each_stage_once_per_step': bool((hits == 1).all()),
                  'all_source_dtypes_recorded': bool(((dtype_codes >= 0) & (dtype_codes < len(SOURCE_DTYPES))).all())}
        for step in range(4):
            expected_ids = self.prompt_tokens if step == 0 else self.trace_tokens[step - 1:step]
            expected_positions = list(range(5)) if step == 0 else [4 + step]
            rows = len(expected_ids)
            checks[f'step{step}_input_ids'] = ids[step, :rows].tolist() == expected_ids
            checks[f'step{step}_positions'] = axes[step].item() in (1, 3) and all(
                positions[step, axis, :rows].tolist() == expected_positions for axis in range(int(axes[step])))
            checks[f'step{step}_finite'] = all(bool(torch.isfinite(bank[step, :rows]).all()) for bank in banks.values())
        return {'schema_version': 1, 'counter': counter, 'checks': checks, 'passed': all(checks.values()),
            'prompt_tokens': self.prompt_tokens, 'teacher_tokens': self.trace_tokens[:-1],
            'input_ids': ids, 'positions': positions, 'position_axes': axes,
            'stage_names': self.names, 'stage_hits': hits, 'stage_capacities': capacities,
            'stage_storage_dtype': {name: str(self.banks[name].dtype) for name in STAGE_WIDTHS},
            'layer_source_dtypes': [[source_dtypes[f'layer.{i:02d}'][step] for i in range(48)] for step in range(4)],
            'stage_source_dtypes': [{name: source_dtypes[name][step] for name in STAGE_WIDTHS} for step in range(4)],
            'layer_outputs': [torch.stack([banks[f'layer.{i:02d}'][step, :5 if step == 0 else 1]
                                          for i in range(48)]) for step in range(4)],
            'stage_outputs': [{name: banks[name][step, :5 if step == 0 else 1]
                               for name in STAGE_WIDTHS} for step in range(4)]}


def reset_worker_trace(worker):
    worker._flare_layer_trace.reset()
    return {'reset': True, 'capture_before_profile': True}


def save_worker_trace(worker, path):
    trace = worker._flare_layer_trace.snapshot()
    target = Path(path)
    with target.open('xb') as handle:
        torch.save(trace, handle)
    return {'passed': trace['passed'], 'counter': trace['counter'], 'checks': trace['checks'],
            'artifact': file_record(target), 'stage_names': trace['stage_names']}
