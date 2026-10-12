"""Compact b12x Engram lookup behind a captured external readiness event."""

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from b12x.preparation import PreparedCall
from b12x.sequence import engram
from b12x.sequence.engram._kernels import lookup_op

from ignition.preparation.tuning import declare


@torch.library.custom_op("flare::staged_engram", mutates_args=("out",))
def _lookup(source: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor,
            hashes: torch.Tensor, count: torch.Tensor, out: torch.Tensor,
            plan: int, event: int) -> None:
    from cuda.bindings import driver

    stream = torch.cuda.current_stream(source.device)
    flags = int(torch.cuda.is_current_stream_capturing())
    status, = driver.cuStreamWaitEvent(driver.CUstream(stream.cuda_stream), driver.CUevent(event), flags)
    if int(status):
        raise RuntimeError(f"Engram readiness wait failed: {status}")
    lookup_op(plan, weight.view(torch.float8_e4m3fn), scales, hashes, count, out, hashes.shape[0], True)


@_lookup.register_fake
def _lookup_fake(source, weight, scales, hashes, count, out, plan, event):
    pass


@dataclass(frozen=True)
class EngramInputs:
    weights: torch.Tensor
    scales: torch.Tensor
    hashes: torch.Tensor
    ready: object

    @classmethod
    def allocate(cls, tokens, device):
        ready = torch.cuda.Event(external=True)
        ready.record(torch.cuda.current_stream(device))
        return cls(torch.zeros((tokens * 24, 256), dtype=torch.uint8, device=device),
                   torch.zeros((tokens * 24, 8), dtype=torch.uint8, device=device),
                   torch.full((tokens, 24), -1, dtype=torch.int64, device=device), ready)

    def buffers(self, layer):
        return {f"engram_{layer}_{name}": getattr(self, name) for name in ("weights", "scales", "hashes")}


class PreparedEngramLookup:
    def __init__(self, inputs, output, count, *, layer, token_map, geometry, session, name):
        tokens = output.shape[0]
        self.weights = inputs.weights[:tokens * 24]
        self.scales = inputs.scales[:tokens * 24]
        self.hashes = inputs.hashes[:tokens]
        self.count, self.output = count, output
        self.handle = inputs.ready.cuda_event
        self.plan = engram.plan(engram.Caps(device=output.device, max_tokens=tokens,
            max_seqs=tokens, max_requests=tokens, vocab_size=len(token_map), layer_id=layer,
            tp_size=1, tp_rank=0), token_map=token_map, geometry=geometry,
            invocation={"operation": "lookup", "compact_rows": True})
        binding = SimpleNamespace(weight=self.weights.view(torch.float8_e4m3fn),
            scale_bytes=self.scales, hash_ids=self.hashes, num_tokens=count, out=output)
        declare(session, (self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
            run=lambda: state.run_lookup(binding, tokens, clear_tail=True), owners=(inputs, binding))),))

    def __call__(self, source):
        _lookup(source, self.weights, self.scales, self.hashes, self.count,
                self.output, self.plan.handle, self.handle)
        return self.output
