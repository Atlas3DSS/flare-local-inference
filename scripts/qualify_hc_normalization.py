"""Replay recorded HC normalization plans on exact checkpoint trace operands."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qualify.provenance import file_record, plain, utc, write_json

TRACE_ROOT = Path("/data/sparkgap-artifacts/text-layer-trace")
WEIGHT_KEY = "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight"
MODULE_KEY = "language_model.model.layers.0.attn_hyper_connection"
PINNED_REVISION = "98fb87a461183f98d6a7049ca947fb136cc6fb59"


def query_record(query):
    payload = plain(query)
    if payload.get("limit") == float("inf"):
        payload["limit"] = "+inf"
    json.dumps(payload, allow_nan=False)
    return payload


def tensor_record(value):
    import torch

    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return {"shape": list(value.shape), "stride": list(value.stride()),
            "dtype": str(value.dtype), "sha256": hashlib.sha256(raw).hexdigest()}


def compare(actual, expected):
    import torch

    if actual.shape != expected.shape:
        raise ValueError("normalization comparison shape mismatch")
    a, b = actual.float(), expected.float()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    difference = a - b
    equal_bytes = actual.dtype == expected.dtype and torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
    return {"finite": finite, "bitwise_equal": equal_bytes,
            "different_elements": int((actual != expected).sum()),
            "elements": actual.numel(),
            "max_absolute_error": float(difference.abs().max()) if finite else None,
            "relative_l2_error": float(difference.norm() / b.norm().clamp_min(1e-30)) if finite else None}


def _selection(receipt, name):
    found = [entry for entry in receipt["tuning"]["selections"] if entry["name"] == name]
    if len(found) != 1 or found[0]["component"] != "norm.hyperconnection":
        raise ValueError(f"expected one recorded HC selection: {name}")
    return found[0]


@dataclass
class Inputs:
    sg: dict
    vllm: dict
    weight: object
    eps: float
    sg_selections: dict
    vllm_plan: dict
    provenance: dict
    tracked: list


def load_inputs(args):
    import torch
    from safetensors import safe_open
    from b12x.norm import hyperconnection as hc

    sg = torch.load(args.sparkgap_trace, map_location="cpu", weights_only=True)
    vllm = torch.load(args.vllm_trace, map_location="cpu", weights_only=True)
    receipt = json.loads(args.sparkgap_receipt.read_text())
    worker = json.loads(args.vllm_worker.read_text())
    if len(worker) != 1:
        raise ValueError("normalization probe requires a TP1 worker record")
    worker = worker[0]
    plan_id = worker["modules"][MODULE_KEY]["plans"]["_plans"]["grouped_rmsnorm"]
    vplan = worker["plans"][str(plan_id)]
    query = vplan["query"]
    expected_query = {"operation": "grouped_rmsnorm", "max_tokens": 512,
                      "hidden_size": 2560, "streams": 4, "lowrank": 320,
                      "dtype": "bfloat16", "zero_centered": True, "weight_dtype": "bfloat16"}
    if any(query.get(key) != value for key, value in expected_query.items()):
        raise ValueError("recorded vLLM normalization query does not match the trace contract")
    if receipt["prefill_capacity"] != 512 or not vllm["passed"]:
        raise ValueError("expected a 512-row Sparkgap trace and complete vLLM stage coverage")
    if (receipt["prompt_tokens"] != vllm["prompt_tokens"]
            or receipt["generated_tokens"][:3] != vllm["teacher_tokens"]):
        raise ValueError("trace prompt or teacher-forced continuation differs")
    if len(sg["stage_outputs"]) != 4 or len(vllm["stage_outputs"]) != 4:
        raise ValueError("normalization probe requires all four recorded steps")
    capacity_index = vllm["stage_names"].index("residual_in")
    norm_index = vllm["stage_names"].index("attn_normalized")
    for step in range(4):
        live, execution = (5, 8) if step == 0 else (1, 1)
        for index in (capacity_index, norm_index):
            if int(vllm["stage_capacities"][index, step]) != execution:
                raise ValueError("recorded vLLM execution rows differ from the 8/1 contract")
        for trace in (sg, vllm):
            for name in ("residual_in", "attn_normalized"):
                value = trace["stage_outputs"][step][name]
                if value.shape != (live, 10240) or not torch.isfinite(value).all():
                    raise ValueError("trace normalization operand has invalid geometry or values")
                if not torch.equal(value.float(), value.bfloat16().float()):
                    raise ValueError("trace operand is not losslessly representable as BF16")
        if not torch.equal(sg["stage_outputs"][step]["residual_in"].float(),
                           vllm["stage_outputs"][step]["residual_in"].float()):
            raise ValueError("common-input probe requires identical recorded layer-zero residuals")

    index_path = args.snapshot / "model.safetensors.index.json"
    config_path = args.snapshot / "config.json"
    weight_index = json.loads(index_path.read_text())["weight_map"]
    shard = args.snapshot / weight_index[WEIGHT_KEY]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(WEIGHT_KEY).clone()
    if weight.dtype != torch.float32 or weight.shape != (10240,):
        raise ValueError("checkpoint HC normalization weight must be FP32[10240]")
    config = json.loads(config_path.read_text())
    eps = float(config["text_config"]["rms_norm_eps"])
    if eps != query["eps"]:
        raise ValueError("checkpoint and recorded normalization eps differ")
    selections = {mode: _selection(receipt, f"{mode}.layer0.attn_hc.grouped_rmsnorm")
                  for mode in ("prefill", "decode")}

    installed = Path(hc.__file__).resolve().parent
    pinned = args.pinned_flashinfer / "flashinfer/experimental/b12x/norm/hyperconnection"
    source_pairs = []
    installed_paths = sorted(installed.glob("*.py"))
    if len(installed_paths) != 9 or {p.name for p in pinned.glob("*.py")} != {p.name for p in installed_paths}:
        raise ValueError("expected the nine recorded HC Python source files")
    for path in installed_paths:
        pair = {"installed": file_record(path), "pinned": file_record(pinned / path.name)}
        pair["identical"] = pair["installed"]["sha256"] == pair["pinned"]["sha256"]
        if not pair["identical"]:
            raise ValueError(f"installed and pinned HC sources differ: {path.name}")
        source_pairs.append(pair)
    tracked_paths = [Path(__file__), ROOT / "qualify/provenance.py", args.sparkgap_trace,
                     args.sparkgap_receipt, args.vllm_trace, args.vllm_worker,
                     index_path, config_path, shard]
    tracked = [file_record(path) for path in tracked_paths]
    if tracked[2]["sha256"] != receipt["tensor_sha256"]:
        raise ValueError("Sparkgap tensor artifact differs from its recorded digest")
    tracked += [pair[side] for pair in source_pairs for side in ("installed", "pinned")]
    stage_report = args.vllm_trace.parent / "layer-trace.json"
    trace_validation = json.loads(stage_report.read_text()) if stage_report.is_file() else None
    if stage_report.is_file():
        tracked.append(file_record(stage_report))
        if trace_validation["workers"][0]["artifact"]["sha256"] != tracked[4]["sha256"]:
            raise ValueError("vLLM tensor artifact differs from its recorded digest")
    provenance = {
        "inputs_and_sources": tracked, "hc_sources": source_pairs,
        "pinned_flashinfer_revision": PINNED_REVISION,
        "checkpoint_weight_key": WEIGHT_KEY, "checkpoint_weight": tensor_record(weight),
        "recorded_sparkgap_selections": selections, "recorded_vllm_plan": vplan,
        "vllm_trace_qualification": trace_validation,
        "prompt_tokens": receipt["prompt_tokens"], "teacher_tokens": vllm["teacher_tokens"],
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "padding": "Trace files retain active rows only. Unobserved rows are zero-filled at the recorded execution capacity; grouped normalization has no cross-row reduction.",
        "replay_engine": "torch.cuda.CUDAGraph",
    }
    return Inputs(sg, vllm, weight, eps, selections, vplan, provenance, tracked)


def qualify(inputs, args, report, raw):
    import torch
    from b12x.norm import hyperconnection as hc
    from b12x.preparation import PreparationSession, PreparedCall

    if not os.environ.get("CUDA_VISIBLE_DEVICES") or torch.cuda.device_count() != 1:
        raise ValueError("assign exactly one CUDA device through CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
        raise ValueError("HC normalization qualification requires SM120/121")
    report["device"] = {"visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                        "name": torch.cuda.get_device_name(device),
                        "capability": torch.cuda.get_device_capability(device)}
    stream = torch.cuda.Stream(device=device)
    original_gain = (1 + inputs.weight.float()).contiguous()
    rounded_weight = inputs.weight.bfloat16()
    rounded_gain = (1 + rounded_weight.float()).contiguous()
    weights = {name: value.to(device) for name, value in
               (("original_fp32_gain", original_gain), ("rounded_fp32_gain", rounded_gain),
                ("bf16_zero_centered", rounded_weight))}
    raw["weights"] = {"checkpoint": inputs.weight, "original_fp32_gain": original_gain,
                      "rounded_fp32_gain": rounded_gain, "bf16_zero_centered": rounded_weight}
    prepared = []
    graphs = []

    with torch.inference_mode(), PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        def prepare_case(name, step, geometry, arm, source_cpu, expected=None):
            rows = (512 if geometry == "sparkgap" else 8) if step == 0 else 1
            mode = "prefill" if step == 0 else "decode"
            zero_centered = arm == "bf16_zero_centered"
            max_tokens = 512 if zero_centered or geometry == "vllm" else rows
            config = (inputs.vllm_plan["selection"]["config"] if zero_centered
                      else inputs.sg_selections[mode]["config"])
            source = torch.zeros((rows, 10240), dtype=torch.bfloat16, device=device)
            source[:source_cpu.shape[0]].copy_(source_cpu)
            output = torch.full((max_tokens, 10240), float("nan"), dtype=torch.bfloat16, device=device)
            bottleneck = torch.empty((max_tokens, 320), dtype=torch.bfloat16, device=device)
            block_input = torch.empty((max_tokens, 2560), dtype=torch.bfloat16, device=device)
            weight = weights[arm]
            caps = hc.Caps(device=device, max_tokens=max_tokens, hidden_size=2560, streams=4, lowrank=320)
            invocation = {"operation": "grouped_rmsnorm", "eps": inputs.eps,
                          "zero_centered": zero_centered,
                          "weight_dtype": "bfloat16" if zero_centered else "float32"}
            plan = hc.plan(caps, invocation=invocation, override=hc.HyperConnectionConfig(**config))
            session.prepare((plan.request(name=name, prepare_call=lambda state: PreparedCall(
                run=lambda: state.launch(source, weight, output, eps=inputs.eps),
                owners=(source, weight, output, bottleneck, block_input))),))
            binding = hc.bind(plan, tokens=rows, normalized=output,
                              bottleneck=bottleneck, block_input=block_input)

            def run():
                hc.run_grouped_rmsnorm(source, weight, eps=inputs.eps,
                                      binding=binding, zero_centered=zero_centered)

            prepared.append({"name": name, "step": step, "geometry": geometry, "arm": arm,
                "rows": rows, "live": source_cpu.shape[0], "source": source, "output": output,
                "expected": expected, "run": run, "owners": (binding, plan, bottleneck, block_input),
                "query": query_record(plan.query), "config": config})

        for step in range(4):
            for geometry, trace, arm in (("sparkgap", inputs.sg, "original_fp32_gain"),
                                        ("vllm", inputs.vllm, "bf16_zero_centered")):
                stages = trace["stage_outputs"][step]
                prepare_case(f"own.{geometry}.{step}", step, geometry, arm,
                             stages["residual_in"].bfloat16(), stages["attn_normalized"].bfloat16())
        for step in range(4):
            common = inputs.sg["stage_outputs"][step]["residual_in"].bfloat16()
            for geometry in ("sparkgap", "vllm"):
                for arm in weights:
                    prepare_case(f"common.{geometry}.{step}.{arm}", step, geometry, arm, common)

        try:
            stream.wait_stream(torch.cuda.current_stream(device))
            for case in prepared:
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        case["run"]()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                graphs.append(graph)
                with session.capture(), torch.cuda.graph(graph, stream=stream):
                    case["run"]()
            session.freeze()
            torch.cuda.synchronize(device)
            for case, graph in zip(prepared, graphs):
                before = torch.cuda.memory_stats(device)
                with torch.cuda.stream(stream):
                    for _ in range(args.replays):
                        graph.replay()
                stream.synchronize()
                after = torch.cuda.memory_stats(device)
                allocation_count = after["allocation.all.allocated"] - before["allocation.all.allocated"]
                byte_delta = after["allocated_bytes.all.current"] - before["allocated_bytes.all.current"]
                output = case["output"][:case["live"]].cpu()
                record = {key: case[key] for key in ("name", "step", "geometry", "arm", "rows", "live", "query", "config")}
                record.update(replays=args.replays, replay_allocations=allocation_count, replay_byte_delta=byte_delta,
                              output=tensor_record(output))
                report["checks"][f"{case['name']}.allocation_stable"] = allocation_count == byte_delta == 0
                report["checks"][f"{case['name']}.finite"] = bool(torch.isfinite(output).all())
                if case["expected"] is not None:
                    record["own_input_reproduction"] = compare(output, case["expected"])
                    report["checks"][f"{case['name']}.reproduces_recorded"] = record["own_input_reproduction"]["bitwise_equal"]
                report["cases"].append(record)
                raw["cases"][case["name"]] = {"source": case["source"].cpu(), "output": output,
                                              "expected": case["expected"]}
        finally:
            stream.synchronize()
            for graph in graphs:
                graph.reset()

    own_gates = [value for name, value in report["checks"].items() if name.endswith(".reproduces_recorded")]
    report["own_input_reproduction_passed"] = len(own_gates) == 8 and all(own_gates)
    report["same_input_comparisons"] = []
    for step in range(4):
        for geometry in ("sparkgap", "vllm"):
            get = lambda arm: raw["cases"][f"common.{geometry}.{step}.{arm}"]["output"]
            report["same_input_comparisons"].append({"step": step, "geometry": geometry,
                "attribution_ready": report["own_input_reproduction_passed"],
                "affine_weight_rounding": compare(get("original_fp32_gain"), get("rounded_fp32_gain")),
                "norm_kernel_with_equivalent_gain": compare(get("rounded_fp32_gain"), get("bf16_zero_centered")),
                "full_norm_difference": compare(get("original_fp32_gain"), get("bf16_zero_centered"))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparkgap-trace", type=Path, default=TRACE_ROOT / "sparkgap-b12x-combine/decoder.pt")
    parser.add_argument("--sparkgap-receipt", type=Path, default=TRACE_ROOT / "sparkgap-b12x-combine/decoder.json")
    parser.add_argument("--vllm-trace", type=Path, default=TRACE_ROOT / "vllm-quantized-mxfp8-int32/vllm-layer-tensors.pt")
    parser.add_argument("--vllm-worker", type=Path, default=TRACE_ROOT / "vllm-quantized-mxfp8-int32/worker-before.json")
    parser.add_argument("--snapshot", type=Path, default=Path("/data/checkpoints-masters/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD"))
    parser.add_argument("--pinned-flashinfer", type=Path, default=Path("/tmp/lunar-rebase-20261008/flashinfer-pinned"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=5)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.replays < 2:
        parser.error("at least two graph replays are required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "started_utc": utc(), "argv": sys.argv,
              "passed": None, "gpu_executed": False, "checks": {}, "cases": []}
    raw = {"cases": {}}
    inputs = None
    exit_code = 1
    try:
        inputs = load_inputs(args)
        report["provenance"] = inputs.provenance
        import torch
        report["cuda_initialized_after_cpu_validation"] = torch.cuda.is_initialized()
        if report["cuda_initialized_after_cpu_validation"]:
            raise RuntimeError("CPU input validation unexpectedly initialized CUDA")
        if args.validate_only:
            report["status"] = "cpu_inputs_validated"
            exit_code = 0
        else:
            report["gpu_executed"] = True
            qualify(inputs, args, report, raw)
            report["status"] = "completed"
            exit_code = 0
    except Exception:
        report["failure"] = traceback.format_exc()
        report["status"] = "failed"
    finally:
        if inputs is not None:
            changed = [item["path"] for item in inputs.tracked if file_record(item["path"])["sha256"] != item["sha256"]]
            report["changed_inputs_or_sources"] = changed
            report["checks"]["inputs_and_sources_unchanged"] = not changed
        if raw["cases"]:
            import torch
            target = args.output_dir / "tensors.pt"
            with target.open("xb") as handle:
                torch.save(raw, handle)
            report["tensor_artifact"] = file_record(target)
        if report["gpu_executed"]:
            report["passed"] = exit_code == 0 and all(report["checks"].values())
            exit_code = 0 if report["passed"] else 1
        elif report.get("changed_inputs_or_sources"):
            exit_code = 1
        report["finished_utc"] = utc()
        write_json(args.output_dir / "report.json", report)
        print(json.dumps({"status": report["status"], "passed": report["passed"],
                          "checks": len(report["checks"]),
                          "failed_checks": [key for key, value in report["checks"].items() if not value],
                          "output": str(args.output_dir.resolve())}, indent=2), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
