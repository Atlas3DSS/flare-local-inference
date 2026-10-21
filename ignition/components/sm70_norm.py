"""Prepared row-independent zero-centered Qwen RMSNorm on SM70."""

from dataclasses import dataclass
import math

import torch
import triton
import triton.language as tl
from b12x._lib.compile_plan import attach_programs, compile_only_launches, launch_triton, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.tuning import declare
from .native_operator import native_contract, native_plan
from .sm70_gdn import _bf16


@triton.jit(do_not_specialize=["row_stride"])
def _norm(X, Weight, Y, row_stride, WIDTH: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    column = tl.arange(0, BLOCK)
    value = tl.load(X + row * row_stride + column, column < WIDTH, 0.)
    weight = tl.load(Weight + column, column < WIDTH, 0.)
    mean = tl.sum(value * value, 0) / WIDTH
    normalized = value * tl.rsqrt(mean + EPS)
    result = _bf16(normalized * (1. + weight))
    tl.store(Y + row * WIDTH + column, result, column < WIDTH)


@dataclass(frozen=True)
class Query:
    rows: int
    width: int
    epsilon: float


CONTRACT = native_contract(component_id="flare.sm70.row_rms_norm", query_type=Query,
                           tactics=lambda q: ((4,),))


def _options(query):
    return dict(WIDTH=query.width, EPS=query.epsilon, BLOCK=triton.next_power_of_2(query.width),
                num_warps=4, enable_fp_fusion=False)


@program_cache(scope="preparation")
def compile_norm(payload, ordinal):
    query = Query(**dict(payload))
    source = torch.empty((query.rows, query.width), device="meta", dtype=torch.float32)
    weight = torch.empty(query.width, device="meta", dtype=torch.float32)
    with torch.cuda.device(ordinal), compile_only_launches():
        return launch_triton(_norm, (query.rows, 1, 1), source, weight, source, query.width, **_options(query))


@dataclass(frozen=True)
class State:
    query: Query
    program: object

    def run(self, source, weight, output):
        self.program[(self.query.rows, 1, 1)](source, weight, output, source.stride(0))


@torch.library.custom_op("flare::sm70_row_rms_norm", mutates_args=("output",))
def _execute(source: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, source.device).run(source, weight, output)


@_execute.register_fake
def _fake(source, weight, output, handle):
    pass


class PreparedRMSNorm:
    """One fixed reduction per row; loaded before capture, with caller-owned output."""
    def __init__(self, source, weight, output, epsilon, *, session, name):
        if (source.device.type != "cuda" or torch.cuda.get_device_capability(source.device) != (7, 0)
                or source.ndim < 2 or source.shape != output.shape or source.stride(-1) != 1
                or not output.is_contiguous() or not weight.is_contiguous()
                or weight.shape != (source.shape[-1],) or not math.isfinite(epsilon) or epsilon <= 0
                or any(t.dtype != torch.float32 or t.device != source.device for t in (source, weight, output))
                or torch._C._overlaps(output, source) or torch._C._overlaps(output, weight)):
            raise ValueError("row RMSNorm requires distinct FP32 SM70 operands and contiguous output")
        self.weight, self.output = weight, output
        self.width = source.shape[-1]
        inputs, result = source.view(-1, self.width), output.view(-1, self.width)
        query = Query(inputs.shape[0], self.width, epsilon)
        payload = CONTRACT.encode_query(query)
        def materialize(tactic):
            program = compile_norm(payload, source.device.index)
            load_programs(program)
            return attach_programs(State(query, program), program)
        self.plan = native_plan(contract=CONTRACT, query=query, device=source.device,
            scratch_bytes=lambda _: 0, materialize=materialize,
            compile_jobs=lambda t, d: (CompileJob.create(
                "ignition.components.sm70_norm:compile_norm", payload, d.ordinal),))
        declare(session, (self.plan.request(name=name, prepare_call=lambda state:
            PreparedCall(run=lambda: state.run(inputs, weight, result))),))

    def __call__(self, source):
        _execute(source.view(-1, self.width), self.weight, self.output.view(-1, self.width), self.plan.handle)
        return self.output
