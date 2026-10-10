"""Annotate actual vLLM forward and sampling submissions for GPU timelines."""

from contextlib import nullcontext
import json
import os
from pathlib import Path

import torch
from vllm.v1.worker.gpu_worker import Worker


class GraphTimingWorker(Worker):
    def load_model(self, *, load_dummy_weights=False):
        super().load_model(load_dummy_weights=load_dummy_weights)
        self._timing_active = False
        self._timing_steps = []
        runner = self.model_runner
        execute, sample = runner.execute_model, runner.sample
        prepare_inputs = runner.prepare_inputs
        profile_nodes = os.environ.get('FLARE_GPU_REPLAY_NCU') == '1'
        profiled_batches = set()
        kernel_scope_open = False
        sampling_batch = None

        def measured_prepare(*args, **kwargs):
            nonlocal kernel_scope_open, sampling_batch
            batch = prepare_inputs(*args, **kwargs)
            if (profile_nodes and self._timing_active and not batch.has_prefill
                    and batch.num_reqs not in profiled_batches
                    and batch.num_reqs in (1, 2, 4)
                    and batch.num_tokens == batch.num_reqs
                    and max(batch.num_computed_tokens_np) == 575):
                profiled_batches.add(batch.num_reqs)
                sampling_batch = batch.num_reqs
                torch.cuda.nvtx.range_push(f'kernel.b{batch.num_reqs}')
                kernel_scope_open = True
            return batch

        def measured_execute(*args, **kwargs):
            nonlocal kernel_scope_open
            index = len(self._timing_steps)
            scope = torch.cuda.nvtx.range(f'vllm.forward.{index}') if self._timing_active else nullcontext()
            with scope:
                try:
                    output = execute(*args, **kwargs)
                finally:
                    if kernel_scope_open:
                        torch.cuda.nvtx.range_pop()
                        kernel_scope_open = False
            state = runner.execute_model_state
            if self._timing_active and state is not None:
                batch = state.input_batch
                self._timing_steps.append({'index': index, 'batch': batch.num_reqs,
                    'tokens': batch.num_tokens, 'padded_tokens': batch.num_tokens_after_padding,
                    'has_prefill': batch.has_prefill, 'request_ids': list(batch.req_ids),
                    'positions': batch.num_computed_tokens_np.tolist(),
                    'scheduled': batch.num_scheduled_tokens.tolist()})
            return output

        def measured_sample(*args, **kwargs):
            nonlocal sampling_batch
            index = len(self._timing_steps) - 1
            scope = torch.cuda.nvtx.range(f'vllm.sample.{index}') if self._timing_active else nullcontext()
            with scope:
                kernels = torch.cuda.nvtx.range(f'kernel.b{sampling_batch}') if sampling_batch is not None else nullcontext()
                try:
                    with kernels:
                        return sample(*args, **kwargs)
                finally:
                    sampling_batch = None

        runner.execute_model, runner.sample = measured_execute, measured_sample
        runner.prepare_inputs = measured_prepare

    def profile(self, is_start=True, *args, **kwargs):
        if not is_start:
            torch.cuda.synchronize()
            self._timing_active = False
            filename = 'steps.json' if self.rank == 0 else f'steps-rank{self.rank}.json'
            path = Path(os.environ['FLARE_GPU_REPLAY_OUTPUT']) / filename
            path.write_text(json.dumps(self._timing_steps, indent=2) + '\n')
        result = super().profile(is_start, *args, **kwargs)
        if is_start:
            self._timing_active = True
        return result
