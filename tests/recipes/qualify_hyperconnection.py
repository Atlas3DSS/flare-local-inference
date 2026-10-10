"""Qualify prepared hyperconnection against exact-checkpoint canonical weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from b12x.preparation import PreparationSession

from flare.preparation import inspect_qwen38_checkpoint
from flare.preparation.qwen38 import TRANSFORMERS_REVISION
from flare.preparation.graphs import PreparedGraphBundle
from flare.recipes.qwen38 import create_model_tree
from flare.recipes.qwen38.hyperconnection import HyperconnectionScratch, PreparedHyperconnection
from flare.recipes.qwen38.weights import QwenWeightLoader


def qualify(snapshot: Path) -> list[dict]:
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
        raise RuntimeError("hyperconnection qualification requires the assigned SM120/121 GPU")
    manifest = inspect_qwen38_checkpoint(snapshot)
    tree = create_model_tree(manifest)
    prefix = "model.language_model.layers.0.attn_hyper_connection"
    module = tree.get_submodule(prefix)
    with QwenWeightLoader(manifest, device, prefixes=(prefix,)) as loader:
        for name in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
            child = module.get_submodule(name)
            child.weight = torch.nn.Parameter(loader.load_tensor(f"{prefix}.{name}.weight"), requires_grad=False)
    generator = torch.Generator().manual_seed(1607)
    reports = []
    with torch.inference_mode(), PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        for tokens in (1, 4, 16, 64):
            scratch = HyperconnectionScratch.allocate(tokens, device)
            source = torch.zeros((tokens, 10240), dtype=torch.bfloat16, device=device)
            prepared = PreparedHyperconnection(module, scratch, priming_state=source,
                session=session, name=f"hyperconnection_{tokens}")
            block = torch.zeros((tokens, 2560), dtype=torch.bfloat16, device=device)
            mixed = torch.empty_like(block)
            combined = torch.empty_like(source)

            def run():
                mixed.copy_(prepared.mix(source))
                combined.copy_(prepared.combine(source, block))

            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                run()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                run()
            bundle = PreparedGraphBundle(device=0, graphs={"hyperconnection": graph},
                buffers={"source": source, "block": block, "mixed": mixed, "combined": combined},
                resources=(module, prepared, session, stream))
            worker = bundle.start_worker()
            try:
                for iteration in range(2):
                    source.copy_(torch.randn(source.shape, dtype=torch.bfloat16, generator=generator).to(device) * .25)
                    block.copy_(torch.randn(block.shape, dtype=torch.bfloat16, generator=generator).to(device) * .25)
                    expected_mixed, residual, injection = module(source)
                    expected_combined = residual + (block[:, None, :] * injection[:, :, None]).flatten(1)
                    torch.cuda.synchronize(device)
                    allocated = torch.cuda.memory_allocated(device)
                    ticket = worker.submit(0)
                    completion = worker.poll(timeout_ms=30000)
                    assert completion is not None and completion.ticket == ticket and completion.error is None
                    assert torch.cuda.memory_allocated(device) == allocated
                    for name, actual, expected in (("mix", mixed, expected_mixed), ("combine", combined, expected_combined)):
                        torch.testing.assert_close(actual, expected, rtol=.02, atol=.02)
                        difference = (actual.float() - expected.float())
                        reports.append({"tokens": tokens, "iteration": iteration, "operation": name,
                            "max_absolute_error": difference.abs().max().item(),
                            "relative_l2_error": (difference.norm() / expected.float().norm().clamp_min(1e-12)).item()})
            finally:
                worker.close()
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    comparisons = qualify(args.snapshot)
    report = {
        "component": "Qwen layer 0 attention hyperconnection",
        "snapshot": str(args.snapshot.resolve()),
        "transformers_revision": TRANSFORMERS_REVISION,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "seed": 1607,
        "rtol": .02,
        "atol": .02,
        "comparisons": comparisons,
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
