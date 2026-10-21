"""Vocabulary-sharded projection followed by native PCIe logit gathering."""

import torch

from .collectives import PreparedAllGather
from .linear import QuantizedProjection, UnquantizedProjection


class PreparedVocabularyProjection:
    """Project replicated hidden rows with a contiguous vocabulary weight shard.

    Local and full-vocabulary outputs are caller-owned. Every rank participates
    in the gather; the recipe chooses which rank samples from the full logits.
    A quantized shard, such as an NVFP4 draft head, runs as a b12x A16 projection.
    Optional row indices expand gathered physical rows into caller-owned sample
    slots after communication, avoiding repeated projection and transfer.
    """

    def __init__(self, source, weight, local_output, output, *, runtime, session,
                 coordinator, name, activation_mode="a16", buffers=None, row_indices=None, workspace=None, row_invariant=False):
        rows = source.shape[0]
        if row_indices is None and output.shape[0] != rows:
            raise ValueError("vocabulary output rows must match projected rows without row indices")
        if row_indices is not None and (row_indices.dtype != torch.int64
                or row_indices.device != source.device or row_indices.shape != (output.shape[0],)
                or output.shape[0] < rows):
            raise ValueError("vocabulary row indices require one device int64 index per expanded output row")
        self.row_indices = row_indices
        if isinstance(weight, torch.Tensor):
            self.projection = UnquantizedProjection(source, weight, local_output,
                session=session, name=f"{name}.projection", workspace=workspace, row_invariant=row_invariant)
        else:
            self.projection = QuantizedProjection(source, weight, local_output,
                session=session, name=f"{name}.projection", activation_mode=activation_mode, workspace=workspace, row_invariant=row_invariant)
        world = 1 if runtime is None else runtime.world_size
        columns = local_output.shape[1] * world
        if columns < output.shape[1]:
            raise ValueError("vocabulary shards must cover every output token")
        self.gathered = output[:rows] if columns == output.shape[1] else (
            torch.empty((rows, columns), dtype=output.dtype, device=output.device) if buffers is None
            else buffers.zeros((rows, columns), output.dtype))
        self.gather = PreparedAllGather(local_output, self.gathered, runtime=runtime,
            session=session, coordinator=coordinator, name=f"{name}.gather")
        self.output = output
        self.padded = columns != output.shape[1]

    def __call__(self, source):
        gathered = self.gather(self.projection(source))
        if self.row_indices is not None:
            # Indexed reads materialize before the overlapping destination write.
            self.output.copy_(gathered[self.row_indices, :self.output.shape[1]])
        elif self.padded:
            self.output.copy_(gathered[:, :self.output.shape[1]])
        return self.output


_E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


@torch.no_grad()
def nvfp4_draft_head(weight: torch.Tensor, *, rows_per_chunk: int = 8192):
    """Quantize a BF16 [vocab, hidden] head or vocabulary shard to NVFP4 for draft proposals.

    E4M3 scales cover 16 values and one amax-derived FP32 gain covers the shard. A
    proposal distribution only has to be the q that generated it, so verification
    keeps the BF16 head and stays exact.
    """
    from b12x._lib.intrinsics.reference import swizzle_block_scale
    from b12x.gemm.blockscaled import pack_weight

    rows, columns = weight.shape
    # Swizzled scales pad rows to 128; a TP4 vocabulary shard holds 38,720.
    if weight.dtype != torch.bfloat16 or columns % 128 or rows % 64 or not weight.is_contiguous():
        raise ValueError("NVFP4 draft head requires a contiguous BF16 head of 64-aligned rows and 128-aligned columns")
    global_scale = (6.0 * 448.0) / weight.abs().amax().float().clamp_min(1e-8)
    values = torch.empty((rows, columns // 2), dtype=torch.uint8, device=weight.device)
    scales = torch.empty((rows, columns // 16), dtype=torch.float8_e4m3fn, device=weight.device)
    bounds = torch.tensor(_E2M1_BOUNDS, device=weight.device)
    for start in range(0, rows, rows_per_chunk):
        block = weight[start:start + rows_per_chunk].float().view(-1, columns // 16, 16)
        scale = (block.abs().amax(-1) / 6.0 * global_scale).clamp(max=448.0).to(torch.float8_e4m3fn)
        effective = scale.float() / global_scale
        quantized = block / torch.where(effective > 0, effective, torch.ones_like(effective)).unsqueeze(-1)
        codes = (torch.bucketize(quantized.abs(), bounds) | ((quantized < 0).long() << 3)).to(torch.uint8)
        codes = codes.view(-1, columns)
        values[start:start + rows_per_chunk] = codes[:, 0::2] | (codes[:, 1::2] << 4)
        scales[start:start + rows_per_chunk] = scale.view(-1, columns // 16)
    return pack_weight(values, swizzle_block_scale(scales), recipe="nvfp4",
                       global_scale=(1.0 / global_scale).reshape(1).float(), global_scale_kind="multiplier")
