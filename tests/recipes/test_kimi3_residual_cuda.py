"""Checkpoint residual mixing without a hidden-sized concatenation per decoder layer."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an assigned CUDA device")
CHECKPOINT = Path("/data/models/Kimi-K3-MXFP4-CSF")


@torch.library.custom_op("flare_test::residual_branch", mutates_args=("out",))
def _branch(source: torch.Tensor, out: torch.Tensor) -> None:
    torch.tanh(source, out=out)


@_branch.register_fake
def _fake_branch(source, out):
    pass


@torch.no_grad()
def test_checkpoint_residual_reductions_and_compiled_memory():
    from ignition.components.weights import DirectWeightLoader
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.recipes.kimi3.model import attention_residual

    if not CHECKPOINT.is_dir():
        pytest.skip("Kimi K3 checkpoint is unavailable")
    path = CHECKPOINT / "modeling_kimi_linear.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_apply_attn_res")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    reference = namespace["_apply_attn_res"]
    names = [f"language_model.model.layers.24.{field}.weight"
             for field in ("self_attention_res_norm", "self_attention_res_proj")]
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    with DirectWeightLoader(inspect_indexed_checkpoint(CHECKPOINT), device, prefixes=names) as loader:
        norm, projection = loader.load_tensors(names)
    reference_norm = SimpleNamespace(weight=norm, variance_epsilon=1e-5)
    reference_projection = SimpleNamespace(weight=projection)
    generator = torch.Generator(device=device).manual_seed(6131)
    for rows in (1, 4, 4096):
        for count in (2, 9):
            values = torch.randn((count, rows, 7168), dtype=torch.bfloat16,
                                 device=device, generator=generator).unbind(0)
            prefix, blocks = values[-1], values[:-1]
            for dtype in (torch.float32, torch.bfloat16):
                source = prefix.to(dtype)
                history = tuple(value.to(dtype) for value in blocks)
                expected = reference(source, torch.stack(history, 1), reference_projection, reference_norm)
                actual = attention_residual(source, history, norm, projection, 1e-5)
                torch.testing.assert_close(actual, expected, atol=1e-6,
                    rtol=1e-5 if dtype == torch.float32 else torch.finfo(dtype).eps)
            del source, history, expected, actual

    initial = prefix.clone()
    del prefix, blocks, values
    output, branch = torch.empty_like(initial), torch.empty_like(initial)

    def sequence(mix):
        prefix, blocks = initial, ()
        for index in range(93):
            hidden = mix(prefix, blocks) if blocks else prefix
            boundary = index % 12 == 0
            if boundary:
                blocks = (*blocks, prefix)
            _branch(hidden, branch)
            prefix = branch.clone() if boundary else prefix + branch
            _branch(mix(prefix, blocks), branch)
            prefix = prefix + branch
        return mix(prefix, blocks)

    def forward():
        output.copy_(sequence(lambda prefix, blocks:
            attention_residual(prefix, blocks, norm, projection, 1e-5)))

    expected = sequence(lambda prefix, blocks:
        reference(prefix, torch.stack(blocks, 1), reference_projection, reference_norm))
    compiled = compile_forward(forward)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    compiled()
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - baseline
    print(f"4096-row residual mixing peak intermediate storage: {extra / (1 << 20):.1f} MiB")
    assert extra < 1 << 30
    torch.testing.assert_close(output, expected, atol=1e-5, rtol=.02)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    allocated = torch.cuda.memory_allocated()
    for _ in range(3):
        output.fill_(float("nan"))
        graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    torch.testing.assert_close(output, expected, atol=1e-5, rtol=.02)
