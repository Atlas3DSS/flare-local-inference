"""SM103 block-FP8 GEMMs retaining arbitrary FP32 checkpoint scales."""

from dataclasses import dataclass

import torch

from .native_operator import native_contract, native_plan

COMPONENT = "flare.sm103.block_fp8_projection"


@dataclass(frozen=True)
class BlockFP8Query:
    rows: int
    in_features: int
    out_features: int


CONTRACT = native_contract(component_id=COMPONENT, query_type=BlockFP8Query,
                           tactics=lambda query: (1, 2))


class PreparedBlockFP8:
    def __init__(self, mma_sm):
        self.mma_sm = mma_sm

    def run_serialized(self, values, activation_scales, weight, scales, global_scale, **options):
        from flashinfer.gemm import gemm_fp8_nt_groupwise

        if global_scale is not None:
            raise ValueError("block-FP8 projections carry scales per weight block, without a global scale")
        return gemm_fp8_nt_groupwise(values, weight, activation_scales.contiguous(), scales,
            scale_major_mode="K", mma_sm=self.mma_sm, out_dtype=torch.bfloat16, backend="cutlass")


def plan(rows, in_features, out_features, device):
    return native_plan(contract=CONTRACT, query=BlockFP8Query(rows, in_features, out_features),
                       device=device, scratch_bytes=lambda tactic: 0, materialize=PreparedBlockFP8)
