"""Checkpoint CSF partition loading and SiTU expert execution against MXFP8 arithmetic."""

import os
from pathlib import Path

import pytest

CHECKPOINT = Path("/data/models/Kimi-K3-MXFP4-CSF")
pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires CUDA checkpoint tests")


def _decode_scales(values, rows, columns):
    import numpy as np

    bases, codes, page_bases, offsets, exceptions = (x.numpy() for x in values)
    result = np.empty((rows, columns), dtype=np.uint8)
    block = group = code_offset = 0
    for tile_r in range(0, rows, 128):
        for tile_c in range(0, columns, 4):
            for c in range(tile_c, min(tile_c + 4, columns), 2):
                for r in range(tile_r, min(tile_r + 128, rows), 32):
                    width = min(2, columns - c)
                    count = 32 * width // 8
                    fields = np.unpackbits(codes[code_offset:code_offset + count], bitorder="little")
                    tile = bases[r:r + 32, None] + fields.reshape(32, width)
                    page = group // 32
                    begin, end = map(int, offsets[block + page:block + page + 2])
                    origin = int(page_bases[page]) // 2
                    for record in exceptions[origin + begin:origin + end]:
                        row, column = divmod(int(record) & 255, 2)
                        tile[row, column] = int(record) >> 8
                    result[r:r + 32, c:c + width] = tile
                    block += 1
                    code_offset += count
            group += 1
    assert code_offset == codes.size
    return result


@pytest.mark.parametrize("partition", [(0, 192), (2880, 3072)])
def test_checkpoint_experts(partition):
    import torch
    from safetensors import safe_open
    from b12x.moe._shared.kernels.reference import moe_reference_w4a8_mx
    from ignition.components.moe import PreparedRoutedExperts
    from ignition.components.weights import DirectWeightLoader
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    if not CHECKPOINT.is_dir():
        pytest.skip("Kimi K3 checkpoint is not installed")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    manifest = inspect_indexed_checkpoint(CHECKPOINT)
    prefixes = tuple(f"language_model.model.layers.1.block_sparse_moe.experts.{i}" for i in (0, 17, 433, 895))
    scales = (torch.empty((4, 512, 112), dtype=torch.uint8, device=device),
              torch.empty((4, 3584, 8), dtype=torch.uint8, device=device))
    with DirectWeightLoader(manifest, device, prefixes=tuple(p + "." for p in prefixes)) as loader:
        experts = loader.load_mxfp4_csf_experts(prefixes, hidden_size=3584, intermediate_size=3072,
            intermediate_range=partition, projection_names=("w1", "w3", "w2"), weight_name="weight_packed",
            activation="situ", scale_scratch=scales)

    def read(name):
        with safe_open(CHECKPOINT / manifest.tensors[name].shard, framework="pt") as file:
            return file.get_tensor(name)

    packed, expanded = [], []
    for prefix in prefixes:
        weights, scale_values = {}, {}
        for projection in ("w1", "w3", "w2"):
            name = f"{prefix}.{projection}"
            weights[projection] = read(name + ".weight_packed")
            rows, columns = (3584, 96) if projection == "w2" else (3072, 112)
            fields = [read(name + ".weight_scale.csf32_" + field)
                      for field in ("bases", "codes", "page_bases", "offsets", "exceptions")]
            scale_values[projection] = torch.from_numpy(_decode_scales(fields, rows, columns))
        first, last = partition
        packed.append((torch.cat((weights["w3"][first:last], weights["w1"][first:last])),
                       weights["w2"][:, first // 2:last // 2]))
        expanded.append((torch.cat((scale_values["w3"][first:last], scale_values["w1"][first:last])),
                         scale_values["w2"][:, first // 32:last // 32]))
    w13, w2 = (torch.stack(x).to(device) for x in zip(*packed))
    s13, s2 = (torch.stack(x).to(device) for x in zip(*expanded))
    torch.manual_seed(41)
    source = torch.randn((17, 3584), device=device, dtype=torch.bfloat16)
    ids = torch.randint(0, 4, (17, 1), dtype=torch.int32, device=device)
    ids = torch.cat((ids, (ids + 1) % 4), -1)
    probabilities = torch.softmax(torch.randn(17, 2, device=device), -1)
    output = torch.empty_like(source)
    with RecordedPreparationSession(device=device, checkpoint=str(CHECKPOINT),
            workload_schema="kimi3-experts", numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        op = PreparedRoutedExperts(source, experts, ids, probabilities, output,
                                   session=session, name="experts", workspace=arena)
        session.tune(show_progress=False)
        expected = moe_reference_w4a8_mx(source, w13, s13, None, torch.ones(4, device=device),
            w2, s2, None, torch.ones(4, device=device), ids, probabilities, 4, 3584, 192, activation="situ")
        actual = op().clone()
        error = (actual.float() - expected).square().mean().sqrt() / expected.square().mean().sqrt()
        assert error.item() < .02
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op()
        for scale in scales:
            scale.fill_(255)
        allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize(device)
        assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
        torch.testing.assert_close(output, actual, rtol=0, atol=0)
