"""Publish graph-produced row requests before an external table consumer waits."""

import torch
import triton
import triton.language as tl
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared
from b12x.sequence import ple_embedding

from ignition.preparation.tuning import declare


@triton.jit
def _copy_requests(ids, mailbox, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(mailbox + row, tl.load(ids + row, row < N, other=-1), row < N)


@triton.jit
def _publish(mailbox, sequence, FLAG: tl.constexpr):
    value = tl.load(sequence)
    tl.atomic_xchg(mailbox + FLAG, value, sem="release", scope="sys")


@torch.library.custom_op("flare::ple_requests", mutates_args=("ids", "requests", "mailbox"))
def _requests(token_ids: torch.Tensor, query_start: torch.Tensor, history: torch.Tensor,
              num_seqs: torch.Tensor, num_tokens: torch.Tensor, multipliers: torch.Tensor,
              primes: torch.Tensor, offsets: torch.Tensor, ids: torch.Tensor,
              requests: torch.Tensor, mailbox: torch.Tensor, sequence: torch.Tensor,
              plan: int) -> None:
    state = require_prepared(plan_from_handle(plan), "sequence.ple_embedding", ids.device)
    state.hash_state.run_tensors(token_ids, query_start, history, num_seqs, num_tokens,
        multipliers, primes, offsets, ids, requests, token_count=ids.shape[0])
    _copy_requests[(triton.cdiv(ids.numel(), 256),)](ids, mailbox, ids.numel(), 256)
    _publish[(1,)](mailbox, sequence, mailbox.numel() - 1)


@_requests.register_fake
def _requests_fake(token_ids, query_start, history, num_seqs, num_tokens, multipliers,
                   primes, offsets, ids, requests, mailbox, sequence, plan):
    pass


@torch.library.custom_op("flare::staged_ple", mutates_args=("out",))
def _lookup(source: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor,
            scale: torch.Tensor, ids: torch.Tensor, count: torch.Tensor, out: torch.Tensor,
            plan: int, event: int) -> None:
    from cuda.bindings import driver

    stream = torch.cuda.current_stream(source.device)
    status, = driver.cuStreamWaitEvent(driver.CUstream(stream.cuda_stream), driver.CUevent(event),
                                      int(torch.cuda.is_current_stream_capturing()))
    if int(status):
        raise RuntimeError(f"PLE readiness wait failed: {status}")
    state = require_prepared(plan_from_handle(plan), "sequence.ple_embedding", out.device)
    state.run_lookup(weight, scales.view(torch.float8_e4m3fn), scale, ids, count, out,
                     token_count=out.shape[0])


@_lookup.register_fake
def _lookup_fake(source, weight, scales, scale, ids, count, out, plan, event):
    pass


class PreparedPLELookup:
    def __init__(self, table, metadata, output, caps, *, session, name):
        self.metadata, self.output = metadata, output
        self.geometry, self.scale = table.geometry, table.scale
        self.mailbox, self.sequence = table.mailbox.device_view, table.sequence
        tokens, heads = output.shape[0], table.layout.caps.head_count
        self.ids = table.hashes[:tokens]
        self.requests = torch.empty(tokens, dtype=torch.int32, device=output.device)
        self.weights, self.scales = table.weights[:tokens * heads], table.scales[:tokens * heads]
        self.handle = table.ready.cuda_event
        geometry = table.geometry
        self.plan = ple_embedding.plan(caps, geometry=geometry.geometry,
            prime_sizes=geometry.prime_sizes, table_offsets=geometry.table_offsets,
            multipliers=geometry.multipliers,
            invocation=ple_embedding.invocation_from_tensors(weight=self.weights,
                weight_scale=self.scales.view(torch.float8_e4m3fn), weight_scale_2=table.scale,
                token_ids=metadata.token_ids, query_start_loc=metadata.query_start_loc,
                committed_history=metadata.committed_history, num_seqs=metadata.num_seqs,
                num_tokens=metadata.num_tokens, prime_sizes=geometry.prime_sizes,
                table_offsets=geometry.table_offsets, multipliers=geometry.multipliers, out=output))
        declare(session, (self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
            run=lambda: state.run_lookup(self.weights, self.scales.view(torch.float8_e4m3fn),
                table.scale, self.ids, metadata.num_tokens, output, token_count=tokens),
            owners=(table, self.requests))),))

    def publish(self):
        m, g = self.metadata, self.geometry
        _requests(m.token_ids, m.query_start_loc, m.committed_history, m.num_seqs, m.num_tokens,
            g.multipliers, g.prime_sizes, g.table_offsets,
            self.ids, self.requests, self.mailbox, self.sequence, self.plan.handle)

    def __call__(self, source):
        _lookup(source, self.weights, self.scales, self.scale, self.ids,
                self.metadata.num_tokens, self.output, self.plan.handle, self.handle)
