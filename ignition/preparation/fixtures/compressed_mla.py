"""Finite mixed-format latent records for compressed-attention tuning."""

import torch
from b12x.preparation import PreparedCall
from b12x.attention._shared.mla.compressed_reference import pack_deepseek_v41_cache_reference

from ignition.preparation.scratch import planned_scratch
from ignition.preparation.tuning import benchmark_resource


class Inputs:
    def __init__(self, query, window, indexed, binding):
        device = query.device
        generator = torch.Generator(device=device).manual_seed(410528)
        self.binding = {name: None if value is None else torch.empty_like(value) for name, value in binding.items()}
        q = self.binding["q"]
        q.normal_(std=.2, generator=generator)
        self.query_seed = q.clone()
        self.out = torch.empty_like(q)
        self.sink = torch.zeros(q.shape[1], device=device)
        page_size = window.shape[1] // 528
        self.window = torch.empty_strided((1, window.shape[1]), window.stride(), dtype=torch.uint8, device=device)
        values = torch.randn((page_size, 512), dtype=torch.bfloat16, device=device, generator=generator) * .2
        self.window.copy_(pack_deepseek_v41_cache_reference(values, page_size=page_size, cache_kind="swa"))
        width = binding["swa_indices"].shape[1]
        self.binding["swa_indices"].copy_(torch.arange(width, dtype=torch.int32, device=device)[None] % page_size)
        self.binding["swa_lengths"].fill_(width)
        self.indexed = None
        if indexed is not None:
            page_size = indexed.shape[1] // 288
            width = binding["indexed_indices"].shape[1]
            pages = max(1, (width + page_size - 1) // page_size)
            values = torch.randn((pages * page_size, 512), dtype=torch.bfloat16, device=device, generator=generator) * .2
            self.indexed = pack_deepseek_v41_cache_reference(values, page_size=page_size, cache_kind="indexed")
            self.binding["indexed_indices"].copy_(torch.arange(width, dtype=torch.int32, device=device)[None])
            self.binding["indexed_lengths"].fill_(width)
            table = self.binding["indexed_page_table"]
            if table is not None:
                # Repeated physical pages also cover small prepared logical contexts.
                table.copy_(torch.arange(table.shape[1], dtype=torch.int32, device=device)[None] % pages)
                self.binding["indexed_indices"].remainder_(table.shape[1] * page_size)

    @property
    def nbytes(self):
        return sum(value.nbytes for value in (*self.binding.values(), self.window, self.indexed, self.out, self.sink, self.query_seed)
                   if value is not None)

    def produce(self):
        self.binding["q"].copy_(self.query_seed)


def benchmark_call(state, *, query, window, indexed, arguments, session, workspace):
    key = (query.shape, query.device, window.shape[1:], window.stride(),
           None if indexed is None else indexed.shape[1:],
           tuple((name, None if value is None else (value.shape, value.stride())) for name, value in arguments.items()))
    inputs = benchmark_resource(session, ("compressed_mla", key), lambda: Inputs(query, window, indexed, arguments))
    scratch = planned_scratch(state.scratch_specs(), workspace)
    binding = state.bind_for_preparation(scratch=scratch, **inputs.binding)
    return PreparedCall(run=lambda: state.run(binding, swa_k_cache=inputs.window,
        indexed_k_cache=inputs.indexed, attn_sink=inputs.sink, sm_scale=512 ** -.5, out=inputs.out),
        produce=inputs.produce, output=inputs.out, owners=(inputs, scratch, binding))
