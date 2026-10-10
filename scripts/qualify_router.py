"""Qualify prepared b12x routing against recorded Qwen vLLM routing operands."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qualify.provenance import file_record, plain, utc, write_json

TRACE_ROOT = Path("/data/sparkgap-artifacts/text-layer-trace/vllm-moe-inprocess-repeat")


def routing_kernel_record(path):
    source = path.read_text()
    node = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "_route_topk_kernel")
    first = min([node.lineno] + [item.lineno for item in node.decorator_list])
    kernel = "".join(source.splitlines(keepends=True)[first - 1:node.end_lineno])
    return {"path": str(path), "first_line": first, "last_line": node.end_lineno,
            "sha256": hashlib.sha256(kernel.encode()).hexdigest()}


def byte_equal(left, right):
    import torch

    return left.shape == right.shape and left.dtype == right.dtype and torch.equal(
        left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def comparisons(ids, weights, expected_ids, expected_weights):
    import torch

    expected_ids = expected_ids.to(torch.int32)
    actual_order, expected_order = ids.argsort(-1), expected_ids.argsort(-1)
    same_set = torch.equal(ids.gather(1, actual_order), expected_ids.gather(1, expected_order))
    record = {"ids_bitwise_equal": byte_equal(ids, expected_ids),
              "weights_bitwise_equal": byte_equal(weights, expected_weights),
              "expert_sets_equal": same_set,
              "finite_weights": bool(torch.isfinite(weights).all())}
    if same_set:
        a, b = weights.gather(1, actual_order), expected_weights.gather(1, expected_order)
        delta = a - b
        record.update(weights_per_expert_bitwise_equal=byte_equal(a, b),
                      max_absolute_weight_error=float(delta.abs().max()) if record["finite_weights"] else None,
                      relative_l2_weight_error=float(delta.norm() / b.norm().clamp_min(1e-30)) if record["finite_weights"] else None)
    return record


def load_traces(args):
    import torch
    import b12x

    traces, records = [], []
    for path in args.trace:
        path = path.resolve()
        trace = torch.load(path, map_location="cpu", weights_only=True)
        record = file_record(path)
        receipt_path = path.parent / "layer-trace.json"
        receipt = json.loads(receipt_path.read_text())
        if receipt["workers"][0]["artifact"]["sha256"] != record["sha256"]:
            raise ValueError("saved routing tensor artifact does not match its trace receipt")
        if not trace["passed"] or len(trace["stage_outputs"]) != 4:
            raise ValueError("routing requires complete four-step trace coverage")
        for step, stages in enumerate(trace["stage_outputs"]):
            live, capacity = (5, 8) if step == 0 else (1, 1)
            for name, width, dtype in (("moe_router_logits", 512, "torch.bfloat16"),
                                       ("moe_ids", 10, "torch.int32"),
                                       ("moe_weights", 10, "torch.float32")):
                value = stages[name]
                if value.shape != (live, width) or not torch.isfinite(value).all():
                    raise ValueError("recorded routing operands must be finite and have exact Qwen geometry")
                index = trace["stage_names"].index(name)
                if (int(trace["stage_capacities"][index, step]) != capacity
                        or trace["stage_source_dtypes"][step][name] != dtype):
                    raise ValueError("recorded routing dtype or execution capacity differs")
            logits = stages["moe_router_logits"]
            ids = stages["moe_ids"]
            if not torch.equal(logits, logits.bfloat16().float()):
                raise ValueError("recorded BF16 router logits are not losslessly represented")
            if not bool(((ids >= 0) & (ids < 512)).all()) or any(row.unique().numel() != 10 for row in ids):
                raise ValueError("recorded routes contain invalid or repeated experts")
        traces.append(trace)
        records.extend((record, file_record(receipt_path)))
    if any((trace["prompt_tokens"], trace["teacher_tokens"]) !=
           (traces[0]["prompt_tokens"], traces[0]["teacher_tokens"]) for trace in traces):
        raise ValueError("routing traces use different prompts or teacher continuations")
    library_root = Path(b12x.__file__).resolve().parent
    library = []
    for relative in ("moe/fused_moe/api.py", "moe/fused_moe/_preparation.py", "moe/_shared/routing.py"):
        pair = {"installed": file_record(library_root / relative),
                "serving_checkout": file_record(args.serving_flashinfer / "flashinfer/experimental/b12x" / relative),
                "pinned_reference": file_record(args.pinned_flashinfer / "flashinfer/experimental/b12x" / relative)}
        if pair["installed"]["sha256"] != pair["serving_checkout"]["sha256"]:
            raise ValueError(f"loaded routing source differs from the declared serving checkout: {relative}")
        records.extend(pair.values())
        if relative == "moe/_shared/routing.py":
            kernels = {name: routing_kernel_record(Path(record["path"])) for name, record in pair.items()}
            if len({item["sha256"] for item in kernels.values()}) != 1:
                raise ValueError("routing kernel arithmetic or specialization differs from the pinned reference")
            pair["kernel_source"] = kernels
        elif pair["installed"]["sha256"] != pair["pinned_reference"]["sha256"]:
            raise ValueError(f"routing API/preparation differs from the pinned reference: {relative}")
        library.append(pair)
    records += [file_record(Path(__file__)), file_record(ROOT / "flare/recipes/qwen38/moe.py"),
                file_record(ROOT / "qualify/provenance.py")]
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU input validation initialized CUDA")
    return traces, records, library


def run_gpu(args, traces, report, raw):
    import torch
    from b12x.preparation import PreparationSession
    from flare.components.moe import PreparedRouter, RouterScratch

    if not os.environ.get("CUDA_VISIBLE_DEVICES") or torch.cuda.device_count() != 1:
        raise ValueError("assign exactly one CUDA device")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
        raise ValueError("Qwen routing qualification requires SM120/121")
    report["device"] = {"name": torch.cuda.get_device_name(device),
                        "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]}
    capacities = sorted({1, 8, 512, args.prefill_capacity})
    stream = torch.cuda.Stream(device=device)
    operators, graphs = {}, []
    with torch.inference_mode(), PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        try:
            for capacity in capacities:
                logits = torch.zeros((capacity, 512), device=device, dtype=torch.bfloat16)
                scratch = RouterScratch.allocate(capacity, device, experts=512, top_k=10)
                operator = PreparedRouter(logits, scratch, session=session, name=f"router.{capacity}")
                operators[capacity] = (logits, scratch, operator)
            stream.wait_stream(torch.cuda.current_stream(device))
            for capacity in capacities:
                logits, _, operator = operators[capacity]
                with torch.cuda.stream(stream):
                    operator(logits)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                graphs.append(graph)
                with session.capture(), torch.cuda.graph(graph, stream=stream):
                    operator(logits)
            session.freeze()

            def observe(name, capacity, values, expected_ids, expected_weights, *, exact):
                logits, scratch, operator = operators[capacity]
                logits.zero_()
                logits[:len(values)].copy_(values)
                scratch.ids.fill_(-1)
                scratch.weights.fill_(float("nan"))
                scratch.selected_logits.fill_(float("nan"))
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)
                with torch.cuda.stream(stream):
                    for _ in range(args.replays):
                        graphs[capacities.index(capacity)].replay()
                stream.synchronize()
                after = torch.cuda.memory_stats(device)
                allocations = after["allocation.all.allocated"] - before["allocation.all.allocated"]
                byte_delta = after["allocated_bytes.all.current"] - before["allocated_bytes.all.current"]
                ids = scratch.ids[:len(values)].cpu()
                weights = scratch.weights[:len(values)].cpu()
                selected = scratch.selected_logits[:len(values)].cpu()
                metrics = comparisons(ids, weights, expected_ids, expected_weights)
                report["checks"][name + ".allocation_stable"] = allocations == byte_delta == 0
                report["checks"][name + ".finite_weights"] = metrics["finite_weights"]
                report["checks"][name + ".ids_exact"] = metrics["ids_bitwise_equal"]
                report["checks"][name + ".weights_exact" if exact else name + ".weights_close"] = (
                    metrics["weights_bitwise_equal"] if exact else
                    bool(torch.isclose(weights, expected_weights, rtol=2e-6, atol=1e-7).all()))
                valid_ids = bool(((ids >= 0) & (ids < 512)).all())
                report["checks"][name + ".selected_logits"] = valid_ids and byte_equal(
                    selected, values.gather(1, ids.long()).float())
                report["cases"].append({"name": name, "capacity": capacity, "live_rows": len(values),
                    "query": plain(operator.plan.query), "replay_allocations": allocations,
                    "replay_byte_delta": byte_delta, **metrics})
                raw[name] = {"logits": values, "selected_logits": selected, "ids": ids, "weights": weights,
                             "expected_ids": expected_ids, "expected_weights": expected_weights}

            for repetition, trace in enumerate(traces):
                for step, stages in enumerate(trace["stage_outputs"]):
                    for capacity in (capacities[1:] if step == 0 else (1,)):
                        observe(f"recorded.{repetition}.{step}.capacity{capacity}", capacity,
                            stages["moe_router_logits"].bfloat16(), stages["moe_ids"].int(),
                            stages["moe_weights"], exact=True)
            for capacity in capacities:
                values = torch.zeros((1, 512), dtype=torch.bfloat16)
                for fixture in ("all_tied", "boundary_tied", "finite_extremes"):
                    if fixture == "boundary_tied":
                        values[:, :12] = 3
                    elif fixture == "finite_extremes":
                        values.copy_(torch.linspace(-1000, 1000, 512).bfloat16())
                    ids = 511 - torch.argsort(values.flip(1).float(), descending=True, stable=True)[:, :10]
                    expected = values.gather(1, ids).float().softmax(-1)
                    observe(f"semantic.{fixture}.capacity{capacity}", capacity, values.clone(), ids.int(), expected, exact=False)
        finally:
            stream.synchronize()
            for graph in graphs:
                graph.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, action="append")
    parser.add_argument("--pinned-flashinfer", type=Path, default=Path("/tmp/lunar-rebase-20261008/flashinfer-pinned"))
    parser.add_argument("--serving-flashinfer", type=Path, default=Path("/home/luke/projects/flashinfer"))
    parser.add_argument("--prefill-capacity", type=int, default=4096)
    parser.add_argument("--replays", type=int, default=3)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.trace = args.trace or [TRACE_ROOT / "vllm-layer-tensors.pt", TRACE_ROOT / "repetition-01/vllm-layer-tensors.pt"]
    if args.prefill_capacity < 8 or args.replays < 2:
        parser.error("prefill capacity must be at least 8 and at least 2 replays are required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "started_utc": utc(), "argv": sys.argv,
              "passed": None, "gpu_executed": False, "checks": {}, "cases": [],
              "policy": "b12x selected-logit softmax, FP32 weights, larger expert IDs first on ties; no BF16 weight roundtrip",
              "padding": "Only active logits are recorded; other rows are zero-filled at each preserved execution capacity. Routing is independent across rows."}
    records, raw, code = [], {}, 1
    try:
        traces, records, library = load_traces(args)
        report.update(inputs_and_sources=records, routing_sources=library,
                      source_contract="Loaded source equals the declared serving checkout; public API, preparation, and routing kernel source equal the pinned reference. The host launcher may differ and is hashed in full.",
                      prompt_tokens=traces[0]["prompt_tokens"], teacher_tokens=traces[0]["teacher_tokens"])
        if args.validate_only:
            report["status"] = "cpu_inputs_validated"
        else:
            report["gpu_executed"] = True
            run_gpu(args, traces, report, raw)
            report["status"] = "completed"
        code = 0
    except Exception:
        report.update(status="failed", failure=traceback.format_exc())
    finally:
        report["checks"]["sources_and_inputs_unchanged"] = all(
            file_record(item["path"])["sha256"] == item["sha256"] for item in records)
        if raw:
            import torch
            target = args.output_dir / "tensors.pt"
            with target.open("xb") as handle:
                torch.save(raw, handle)
            report["tensor_artifact"] = file_record(target)
        if report["gpu_executed"]:
            report["passed"] = code == 0 and all(report["checks"].values())
            code = 0 if report["passed"] else 1
        elif not report["checks"]["sources_and_inputs_unchanged"]:
            code = 1
        report["finished_utc"] = utc()
        write_json(args.output_dir / "report.json", report)
        print(json.dumps({"status": report["status"], "passed": report["passed"],
                          "cases": len(report["cases"]), "checks": len(report["checks"]),
                          "failed_checks": [name for name, value in report["checks"].items() if not value]}, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
