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

        def measured_execute(*args, **kwargs):
            index = len(self._timing_steps)
            scope = torch.cuda.nvtx.range(f'vllm.forward.{index}') if self._timing_active else nullcontext()
            with scope:
                output = execute(*args, **kwargs)
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
            index = len(self._timing_steps) - 1
            scope = torch.cuda.nvtx.range(f'vllm.sample.{index}') if self._timing_active else nullcontext()
            with scope:
                return sample(*args, **kwargs)

        runner.execute_model, runner.sample = measured_execute, measured_sample

    def profile(self, is_start=True, *args, **kwargs):
        if not is_start:
            torch.cuda.synchronize()
            self._timing_active = False
            path = Path(os.environ['FLARE_GPU_REPLAY_OUTPUT']) / 'steps.json'
            path.write_text(json.dumps(self._timing_steps, indent=2) + '\n')
        result = super().profile(is_start, *args, **kwargs)
        if is_start:
            self._timing_active = True
        return result
