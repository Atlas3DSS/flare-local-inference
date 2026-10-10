"""Compare exact-checkpoint vision modules across isolated Python environments.

Run fixtures and flare with Flare's preparation environment. Run vllm
with the source-isolated serving environment recorded in
docs/validation/exact-serving-reference.json. Only the vision tower is loaded;
the vLLM encoder retains its own weight hooks, preparation policy and forward.
The complementary serving comparison does not replace the canonical 3% gate.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import fields, is_dataclass
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import MethodType, SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
VLLM_SOURCE = Path("/home/luke/projects/vllm-lunar-leviathan")
FLASHINFER_SOURCE = Path("/tmp/lunar-rebase-20261008/flashinfer-pinned")
GPU_UUID = "GPU-afd4b1ad-8a64-7057-4bde-241822724c7f"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if is_dataclass(value):
        return {field.name: plain(getattr(value, field.name)) for field in fields(value)}
    if hasattr(value, "items"):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    return str(value)


def save_json(path, value):
    path.write_text(json.dumps(plain(value), indent=2) + "\n")


def helpers():
    path = ROOT / "tests/recipes/test_qwen38_vision.py"
    spec = importlib.util.spec_from_file_location("vision_qualification", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def fixtures(args):
    from flare.preparation import inspect_qwen38_checkpoint
    from PIL import Image

    manifest = inspect_qwen38_checkpoint(args.snapshot)
    helper = helpers()
    cases = []
    for variant in (0, 1):
        item = helper._input(manifest, variant)
        assert item.pixel_values.shape == (256, 1536)
        assert item.image_grid_thw.tolist() == [[1, 16, 16]]
        rgb = bytes(component for y in range(256) for x in range(256)
                    for component in ((x * 7 + variant * 31) % 256, (y * 11) % 256, (x + y * 3) % 256))
        Image.frombytes("RGB", (256, 256), rgb).save(args.output / f"image-{variant}.png")
        cases.append({"variant": variant, "pixels": item.pixel_values, "grid": item.image_grid_thw})
    torch.save(cases, args.output / "fixtures.pt")
    save_json(args.output / "fixtures.json", {
        "checkpoint": str(args.snapshot), "config_sha256": sha256(args.snapshot / "config.json"),
        "helper_sha256": sha256(ROOT / "tests/recipes/test_qwen38_vision.py"),
        "fixtures_sha256": sha256(args.output / "fixtures.pt"),
        "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
        "cases": [{"variant": x["variant"], "pixels": plain(x["pixels"]), "grid": x["grid"].tolist(),
                   "image_sha256": sha256(args.output / f"image-{x['variant']}.png"),
                   "pixels_sha256": tensor_hash(x["pixels"])} for x in cases],
    })


def assigned_device():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID or torch.cuda.device_count() != 1:
        raise RuntimeError(f"this qualification requires only assigned GPU {GPU_UUID}")
    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability(0) not in ((12, 0), (12, 1)):
        raise RuntimeError("SM120/121 required")
    return torch.device("cuda", 0)


def provenance(args, label):
    files = {Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name is not None:
            path = Path(name).resolve()
            if path.is_file() and path.suffix == ".py":
                if any(path.is_relative_to(root) for root in (VLLM_SOURCE, FLASHINFER_SOURCE, ROOT)):
                    files.add(path)
    return {
        "implementation": label, "command": sys.argv, "python": sys.executable,
        "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
        "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "gpu": torch.cuda.get_device_name(0),
        "fixtures_sha256": sha256(args.output / "fixtures.pt"),
        "config_sha256": sha256(args.snapshot / "config.json"),
        "index_sha256": sha256(args.snapshot / "model.safetensors.index.json"),
        "source_sha256": {str(path): sha256(path) for path in sorted(files)},
    }


def metrics(actual, expected):
    a, b = actual.float(), expected.float()
    return {"relative_l2": float((a - b).norm() / b.norm().clamp_min(1e-12)),
            "max_absolute": float((a - b).abs().max()), "reference_maximum": float(b.abs().max()),
            "changed_elements": int((a != b).sum()), "elements": a.numel()}


@torch.no_grad()
def flare(args):
    from b12x.preparation import PreparationSession
    from flare.preparation import inspect_qwen38_checkpoint
    from flare.recipes.qwen38.vision import PreparedVision, VisionMetadata, VisionScratch, VisionWeights
    from flare.recipes.qwen38.weights import QwenWeightLoader
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids

    device = assigned_device()
    torch.backends.cuda.matmul.allow_tf32 = False
    helper = helpers()
    manifest = inspect_qwen38_checkpoint(args.snapshot)
    module = helper._module(manifest)
    with QwenWeightLoader(manifest, device, prefixes=("model.visual.",)) as loader:
        weights = VisionWeights.load(module, loader)
    canonical = helper._reference(manifest, weights, device)
    scratch = VisionScratch.allocate(256, device)
    metadata = VisionMetadata.allocate(256, 2, device)
    outputs, cases = [], []
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        prepared = PreparedVision(module, scratch, metadata, weights, session=session, attention_math_mode="flash")
        session.freeze()
        for case in torch.load(args.output / "fixtures.pt", weights_only=True):
            grid, pixels = case["grid"], case["pixels"].to(device)
            indices, interpolation = get_vision_interpolation_indices_and_weights(
                grid, num_grid_per_side=48, mode="bilinear", align_corners=True, spatial_merge_size=2)
            metadata.interpolation_indices.copy_(indices)
            metadata.interpolation_weights.copy_(interpolation)
            metadata.position_ids.copy_(get_vision_position_ids(grid, 2))
            metadata.cu_seqlens.copy_(torch.tensor([0, 256, 256], dtype=torch.int32))
            metadata.num_patches.fill_(256)
            prepared(pixels)
            actual = scratch.output.clone()
            expected = canonical(pixels, grid.to(device)).pooler_output
            prepared(pixels)
            torch.testing.assert_close(actual, scratch.output, rtol=0, atol=0)
            outputs.append({"variant": case["variant"], "flare_flash": actual.cpu(), "canonical": expected.cpu()})
            error = metrics(actual, expected)
            cases.append({"variant": case["variant"], "canonical_comparison": error,
                          "canonical_relative_l2_limit": .03,
                          "canonical_relative_l2_passed": error["relative_l2"] < .03})
        torch.cuda.synchronize()
    torch.save(outputs, args.output / "flare.pt")
    save_json(args.output / "flare.json", provenance(args, "Flare PreparedVision flash; canonical eager adapter") |
              {"cases": cases, "output_sha256": sha256(args.output / "flare.pt")})


def source_identity(path, expected):
    head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True)
    if head != expected or status:
        raise RuntimeError(f"serving source identity mismatch: {path}: {head}: {status}")
    return {"path": str(path), "revision": head, "tracked_status": status}


@torch.no_grad()
def vllm(args):
    import b12x
    import vllm as vllm_package
    from safetensors import safe_open
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import initialize_model_parallel, init_distributed_environment, cleanup_dist_env_and_memory
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.layers.fusion.mm_input_norm import build_mm_input_norm
    from vllm.model_executor.model_loader.utils import configure_quant_config, process_weights_after_loading
    from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForConditionalGeneration
    from vllm.model_executor.warmup.b12x_prepare import begin_b12x_preparation
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.transformers_utils.processor import cached_image_processor_from_config
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager
    from PIL import Image

    sources = [source_identity(VLLM_SOURCE, "64fc1eec593397cbd765cc082e9ec90fbeb5ea66"),
               source_identity(FLASHINFER_SOURCE, "98fb87a461183f98d6a7049ca947fb136cc6fb59")]
    if not Path(vllm_package.__file__).resolve().is_relative_to(VLLM_SOURCE):
        raise RuntimeError("vLLM import escaped the recorded source isolation")
    if not Path(b12x.__file__).resolve().is_relative_to(FLASHINFER_SOURCE):
        raise RuntimeError("b12x import escaped the recorded source isolation")
    device = assigned_device()
    config = EngineArgs(model=str(args.snapshot), dtype="bfloat16", quantization="modelopt_mixed",
        tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1,
        max_model_len=262144, max_num_seqs=4, max_num_batched_tokens=4096,
        mm_encoder_tp_mode="data", mm_processor_cache_gb=4, enable_prefix_caching=True,
        enable_chunked_prefill=True, kv_cache_dtype="fp8", block_size=16,
        mamba_cache_mode="align", linear_backend="b12x", moe_backend="b12x",
        gdn_prefill_backend="b12x", gdn_decode_kernel="b12x", enable_flashinfer_autotune=False,
        gpu_memory_utilization=.94, load_format="instanttensor").create_engine_config()
    fixture_cases = torch.load(args.output / "fixtures.pt", weights_only=True)
    trace, outputs, input_comparisons = [], [], []
    with set_current_vllm_config(config), tempfile.TemporaryDirectory(prefix="flare-vision-dist-") as temporary, ExitStack() as cleanup:
        init_distributed_environment(world_size=1, rank=0, local_rank=0,
            distributed_init_method=f"file://{temporary}/rendezvous", backend="nccl")
        cleanup.callback(cleanup_dist_env_and_memory)
        initialize_model_parallel(1, 1)
        init_workspace_manager(device)
        cleanup.callback(reset_workspace_manager)
        torch.set_default_dtype(torch.bfloat16)
        configure_quant_config(config.quant_config, Qwen4ExpForConditionalGeneration)
        hf = config.model_config.hf_config
        with torch.device(device):
            visual = Qwen3_VisionTransformer(hf.vision_config, norm_eps=hf.text_config.rms_norm_eps,
                quant_config=config.quant_config, input_norm=build_mm_input_norm(config.model_config), prefix="visual")
        visual.eval().requires_grad_(False)
        index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        names = sorted(name for name in index if name.startswith("model.visual."))
        loaded_bytes = 0

        def weights():
            nonlocal loaded_bytes
            for shard in sorted({index[name] for name in names}):
                with safe_open(args.snapshot / shard, framework="pt", device="cpu") as handle:
                    for name in names:
                        if index[name] == shard:
                            value = handle.get_tensor(name)
                            loaded_bytes += value.numel() * value.element_size()
                            yield name.removeprefix("model.visual."), value

        loaded = visual.load_weights(weights())
        expected = {name.removeprefix("model.visual.") for name in names}
        if loaded != expected:
            raise RuntimeError(f"vision load coverage mismatch: missing={expected-loaded}, extra={loaded-expected}")
        process_weights_after_loading(visual, config.model_config, device)
        owner = torch.nn.Module()
        owner.visual, owner.config = visual, hf
        for name in ("get_mm_lora_token_counts", "get_mm_mapping"):
            setattr(owner, name, MethodType(getattr(Qwen4ExpForConditionalGeneration, name), owner))
        worker = SimpleNamespace(vllm_config=config, model_config=config.model_config,
            scheduler_config=config.scheduler_config, device=device, rank=0, use_v2_model_runner=True,
            model_runner=SimpleNamespace(mm_registry=MULTIMODAL_REGISTRY, decode_query_len=1),
            get_model=lambda: owner, get_draft_model=lambda: None)
        coordinator = begin_b12x_preparation(worker, stage="weights")
        cleanup.callback(worker._b12x_session.close)
        outcome = coordinator.status()
        while not outcome["done"]:
            outcome = coordinator.advance()
        if outcome["error"] is not None:
            raise RuntimeError(f"vLLM vision preparation failed: {outcome['error']}")

        def observe(name):
            def hook(module, arguments):
                entry = {"module": name, "class": f"{type(module).__module__}.{type(module).__qualname__}"}
                if hasattr(module, "attn_backend"):
                    entry.update(backend=str(module.attn_backend), fa_version=getattr(module, "_fa_version", None),
                                 fp8_enabled=getattr(module, "fp8_enabled", None))
                holder = getattr(module, "b12x_linear", None)
                if holder is not None:
                    state = holder.plan.prepared.state.resolve(arguments[0])
                    entry.update(recipe=holder.recipe, query=plain(state.query), config=plain(state.config),
                                 rows=arguments[0].numel() // holder.in_features)
                trace.append(entry)
            return hook

        handles = []
        for name, module in visual.named_modules():
            if hasattr(module, "b12x_linear") or type(module).__name__ == "MMEncoderAttention":
                handles.append(module.register_forward_pre_hook(observe(name)))
        for case in fixture_cases:
            first = len(trace)
            processor = cached_image_processor_from_config(config.model_config)
            with Image.open(args.output / f"image-{case['variant']}.png") as image:
                processed = processor(images=[image], return_tensors="pt")
            torch.testing.assert_close(processed.image_grid_thw, case["grid"], atol=0, rtol=0)
            pixels = processed.pixel_values.to(device)
            grid = processed.image_grid_thw.tolist()
            normalized = visual.input_norm(pixels, visual.dtype)
            input_comparisons.append({"variant": case["variant"], "processor": type(processor).__qualname__,
                "processor_input_dtype": str(pixels.dtype), "processor_pixels_sha256": tensor_hash(pixels),
                "normalized_vs_canonical_bf16": metrics(normalized, case["pixels"].to(device).bfloat16())})
            output = visual(pixels, grid).clone()
            torch.cuda.synchronize()
            outputs.append({"variant": case["variant"], "vllm": output.cpu()})
            for event in trace[first:]:
                event["variant"] = case["variant"]
        for handle in handles:
            handle.remove()
        report = provenance(args, "Exact vLLM Qwen3_VisionTransformer used by Qwen4Exp")
        report.update(sources=sources, config=plain(config), loaded_tensors=len(loaded), loaded_bytes=loaded_bytes,
            input_comparisons=input_comparisons,
            executed_modules=trace, vision_tp_size=visual.tp_size, input_normalization=type(visual.input_norm).__qualname__,
            b12x_profile_rows={name: getattr(module, "b12x_eager_token_counts", None)
                              for name, module in visual.named_modules() if hasattr(module, "b12x_linear")},
            scope={"world_ranks": 1, "recorded_serving_world_ranks": 2, "vision_data_parallel": True,
                   "language_model_loaded": False, "mtp_loaded": False,
                   "weight_transfer": "bounded safetensors CPU iteration through unchanged vLLM load_weights and post-load hooks",
                   "forward": "unchanged Qwen3_VisionTransformer.forward, eager as in recorded serving configuration",
                   "autotuning": "unchanged vLLM weights preparation coordinator, single-rank tuning and executed plans recorded"})
        torch.save(outputs, args.output / "vllm.pt")
        report["output_sha256"] = sha256(args.output / "vllm.pt")
        save_json(args.output / "vllm.json", report)


def compare(args):
    native = torch.load(args.output / "flare.pt", weights_only=True)
    serving = torch.load(args.output / "vllm.pt", weights_only=True)
    cases = []
    for a, b in zip(native, serving, strict=True):
        assert a["variant"] == b["variant"]
        cases.append({"variant": a["variant"], "flare_flash_vs_vllm": metrics(a["flare_flash"], b["vllm"]),
                      "vllm_vs_canonical": metrics(b["vllm"], a["canonical"]),
                      "flare_flash_vs_canonical": metrics(a["flare_flash"], a["canonical"])})
    report = {"cases": cases, "canonical_relative_l2_limit": .03, "serving_reference_bound": None,
              "artifacts": {name: sha256(args.output / name) for name in
                            ("fixtures.pt", "fixtures.json", "vllm.pt", "vllm.json", "flare.pt", "flare.json")}}
    save_json(args.output / "comparison.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fixtures", "vllm", "flare", "compare"))
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    globals()[args.mode](args)
