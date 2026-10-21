"""Compile and capture BF16 checkpoint lookups with FP32 V100 graph buffers."""

import os

import pytest
import torch

from ignition.components.embedding import PreparedEmbedding
from ignition.preparation.compilation import compile_forward


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned V100 device")


@pytest.mark.parametrize("shape,ids_dtype,count_shape", [
    ((1,), torch.int32, None), ((17,), torch.int64, (1,)), ((3, 5), torch.int32, (1, 1)),
])
@torch.no_grad()
def test_embedding_compiles_packed_bfloat_weights_and_masks_inactive_rows(shape, ids_dtype, count_shape):
    from b12x.preparation import PreparationSession

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    cpu_weight = torch.arange(256 * 128).reshape(256, 128).float().remainder(251).bfloat16()
    cpu_weight[0] = float("nan")
    cpu_weight[1, :4] = torch.tensor([131072., -131072., 1.17549435e-38, -1.17549435e-38]).bfloat16()
    weight = cpu_weight.cuda()
    ids = torch.ones(shape, dtype=ids_dtype, device="cuda")
    output = torch.empty((*shape, weight.shape[1]), dtype=torch.float32, device="cuda")
    count = None if count_shape is None else torch.zeros(count_shape, dtype=torch.int32, device="cuda")
    with PreparationSession(device=weight.device, autotune=False, compile_workers=0) as session:
        operation = PreparedEmbedding(weight, ids, output, session=session, name="embedding", num_rows=count)
        assert operation.weight_bytes.dtype == torch.uint8
        assert operation.weight_bytes.data_ptr() == weight.data_ptr()
        assert operation.weight_bytes.untyped_storage().nbytes() == cpu_weight.numel() * 2
        compiled = compile_forward(operation)
        compiled()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        for live in ((ids.numel(),) if count is None else (0, 1, ids.numel(), ids.numel() - 1)):
            cpu_ids = torch.arange(1, ids.numel() + 1, dtype=ids_dtype)
            cpu_ids[live:] = -2147483648  # Inactive IDs must never access the table.
            ids.copy_(cpu_ids.reshape(shape))
            if count is not None:
                count.fill_(live)
            output.fill_(float("nan"))
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            expected = torch.zeros((ids.numel(), weight.shape[1]), dtype=torch.float32)
            expected[:live] = cpu_weight[cpu_ids[:live].long()].float()
            torch.testing.assert_close(output.cpu(), expected.reshape(output.shape), rtol=0, atol=0)
