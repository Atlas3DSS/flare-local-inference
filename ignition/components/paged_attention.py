"""Prepared b12x GQA over explicit cache pages, metadata and shared scratch."""

import torch
from b12x.attention import paged, paged_decode, paged_kv_write
from b12x.preparation import PreparedCall, plan_from_handle

from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import benchmark_resource, declare, register_finalizer


def _scratch(specs, workspace):
    specs = tuple(specs)
    if not specs:
        return []
    values = planned_scratch(specs, workspace)
    return list(values) if isinstance(values, tuple) else [values]


@torch.library.custom_op("flare::paged_attention", mutates_args=("output", "scratch"))
def _attend(q: torch.Tensor, key: torch.Tensor, value: torch.Tensor, pages: torch.Tensor,
            lengths: torch.Tensor, cu: torch.Tensor, sink: torch.Tensor | None,
            output: torch.Tensor, scratch: list[torch.Tensor], handle: int, decode: bool, window: int) -> None:
    plan = plan_from_handle(handle)
    arguments = dict(q=q, k_cache=key, v_cache=value, page_table=pages, cache_seqlens=lengths,
                     cu_seqlens_q=cu, attention_sink_bias=sink, output=output, scratch=scratch)
    if decode:
        paged_decode.run(paged_decode.bind(plan, **arguments))
    else:
        paged.run(binding=paged.bind(plan, **arguments, window_left=window), plan=plan)


@_attend.register_fake
def _fake_attend(q, key, value, pages, lengths, cu, sink, output, scratch, handle, decode, window):
    pass


class PreparedPagedAttention:
    def __init__(self, q, key, value, pages, lengths, cu, output, *, sink=None, mode,
                 window=-1, width=1, causal=True, session, workspace, name):
        self.decode = mode in ("decode", "verify") and paged_decode.supports(
            num_q_heads=q.shape[1], num_kv_heads=key.shape[2], head_dim_qk=q.shape[2],
            head_dim_vo=value.shape[3], page_size=key.shape[1], kv_dtype=key.dtype,
            max_q_per_req=width, window_left=window, causal=causal, has_sinks=sink is not None)
        self.window, self.workspace = window, workspace
        self.arguments = dict(q=q, k_cache=key, v_cache=value, page_table=pages,
            cache_seqlens=lengths, cu_seqlens_q=cu, output=output, attention_sink_bias=sink)
        if self.decode:
            self.plan = paged_decode.plan(paged_decode.Caps(device=q.device, num_q_heads=q.shape[1],
                num_kv_heads=key.shape[2], head_dim_qk=q.shape[2], head_dim_vo=value.shape[3],
                page_size=key.shape[1], max_batch=pages.shape[0], max_q_per_req=width,
                window_left=window, has_sinks=sink is not None, causal=causal))
        else:
            geometry = dict(device=q.device, q_dtype=q.dtype, kv_dtype=key.dtype,
                num_q_heads=q.shape[1], num_kv_heads=key.shape[2], head_dim_qk=q.shape[2],
                head_dim_vo=value.shape[3], page_size=key.shape[1], batch=pages.shape[0],
                max_cache_page_count=pages.shape[1], window_left=window)
            if mode == "decode":
                capacity = paged.decode_graph_capacity(**geometry)
            elif mode == "verify":
                capacity = paged.verify_graph_capacity(**geometry, query_len=width)
            else:
                capacity = paged.extend_graph_capacity(**geometry, total_q_capacity=q.shape[0])
            caps = paged.Caps(device=q.device, mode=mode if mode in ("decode", "verify") else "extend",
                dtype=q.dtype, kv_dtype=key.dtype,
                num_q_heads=q.shape[1], num_kv_heads=key.shape[2], head_dim_qk=q.shape[2],
                head_dim_vo=value.shape[3], page_size=key.shape[1], max_total_q=q.shape[0],
                max_batch=pages.shape[0], max_page_table_width=pages.shape[1], num_cache_pages=key.shape[0],
                max_work_items=capacity.max_work_items, max_partial_rows=getattr(capacity, "max_partial_rows", 0),
                use_cuda_graph=True)
            invocation = dict(paged.invocation_from_tensors(caps, **self.arguments), window_left=window)
            self.plan = paged.plan(caps, invocation=invocation)

        def prepared_call(state, arguments):
            from b12x.attention.paged_decode._preparation import scratch_specs

            specs = scratch_specs(state.query, state.config, q.device) if self.decode else state.scratch_plan.scratch_specs()
            scratch = _scratch(specs, workspace)
            kwargs = {"plan": self.plan} if self.decode else {"window_left": window}
            binding = state.bind(scratch=scratch, **arguments, **kwargs)
            return PreparedCall(run=lambda: state.run(binding), owners=prime_owners(workspace, scratch, binding))

        def benchmark(state):
            def inputs():
                arguments = dict(self.arguments)
                for field in ("q", "page_table", "cache_seqlens", "cu_seqlens_q", "output"):
                    value = arguments[field]
                    arguments[field] = torch.empty_strided(value.shape, value.stride(),
                                                         dtype=value.dtype, device=value.device)
                arguments["q"].normal_(std=.2)
                arguments["page_table"].zero_()
                arguments["cache_seqlens"].fill_(min(max(4096, q.shape[0]), pages.shape[1] * key.shape[1]))
                arguments["cu_seqlens_q"].copy_(torch.arange(cu.numel(), device=cu.device, dtype=cu.dtype)
                                                   * (q.shape[0] // pages.shape[0]))
                return arguments, arguments["q"].clone()

            operands, seed = benchmark_resource(session, ("paged_attention", name), inputs)
            call = prepared_call(state, operands)
            return PreparedCall(run=call.run, produce=lambda: operands["q"].copy_(seed),
                                owners=(*call.owners, operands, seed), output=operands["output"])

        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: prepared_call(state, self.arguments), benchmark_call=benchmark),))
        register_finalizer(session, self.rebind, workspace=workspace)

    def rebind(self):
        self.scratch = _scratch(self.plan.scratch_specs(), self.workspace)
        if not self.decode:
            paged.bind(self.plan, scratch=self.scratch, **self.arguments, window_left=self.window)

    def __call__(self):
        a = self.arguments
        _attend(a["q"], a["k_cache"], a["v_cache"], a["page_table"], a["cache_seqlens"],
                a["cu_seqlens_q"], a["attention_sink_bias"], a["output"], self.scratch,
                self.plan.handle, self.decode, self.window)
        return a["output"]


class PreparedPagedKVWriter:
    def __init__(self, key, value, key_cache, value_cache, slots, *, session, name):
        self.arguments = dict(key=key, value=value, k_cache=key_cache, v_cache=value_cache, slot_mapping=slots)
        self.plan = paged_kv_write.plan(paged_kv_write.Caps(device=key.device, num_kv_heads=key.shape[1],
            head_dim_k=key.shape[2], head_dim_v=value.shape[2], page_size=key_cache.shape[1]))
        declare(session, (self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(**self.arguments))),))

    def __call__(self):
        paged_kv_write.run(self.plan, **self.arguments)
