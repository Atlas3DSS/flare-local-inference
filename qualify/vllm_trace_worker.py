"""Worker extension installing tensor observation before profiling and capture."""

import json
import os
from pathlib import Path

from vllm.v1.worker.gpu_worker import Worker

from .vllm_layer_trace import LayerTrace


class LayerTraceWorker(Worker):
    def load_model(self, *, load_dummy_weights=False):
        super().load_model(load_dummy_weights=load_dummy_weights)
        inputs = json.loads(Path(os.environ['FLARE_VLLM_LAYER_TRACE_INPUTS']).read_text())
        trace = LayerTrace(prompt_tokens=inputs['prompt_tokens'], trace_tokens=inputs['trace_tokens'],
                           device=self.device)
        trace.install(self.get_model())
        self._flare_layer_trace = trace
