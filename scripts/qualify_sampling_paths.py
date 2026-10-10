"""Measure captured production samplers on fingerprinted checkpoint logits."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.machinery
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL = "local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD"
REVISION = "dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2"
SLOTS = (2, 0, 3, 1)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(tensor):
    if tensor.device.type != "cpu":
        raise ValueError("hashing requires an explicit CPU readback")
    return hashlib.sha256(tensor.contiguous().reshape(-1).view(__import__("torch").uint8).numpy().tobytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_inputs(receipt_path, artifact_path):
    import torch

    receipt = json.loads(receipt_path.read_text())
    if receipt.get("checkpoint") != {"repository": MODEL, "revision": REVISION}:
        raise ValueError("decoder receipt does not identify the pinned checkpoint")
    artifact_path = (artifact_path or Path(receipt["tensor_output"])).resolve()
    actual = digest(artifact_path)
    if actual != receipt.get("tensor_sha256"):
        raise ValueError(f"decoder artifact SHA-256 mismatch: expected {receipt.get('tensor_sha256')}, observed {actual}")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    logits = artifact["logits"]
    if artifact["prompt"] != receipt["prompt_tokens"] or artifact["tokens"] != receipt["generated_tokens"]:
        raise ValueError("decoder artifact prompt/generated IDs differ from its receipt")
    if len(logits) != 4 or any(t.dtype != torch.bfloat16 or t.shape != (248320,)
                              or not bool(t.isfinite().all()) for t in logits):
        raise ValueError("require four finite BF16 full-vocabulary decoder logits rows")
    if [int(t.argmax()) for t in logits] != artifact["tokens"]:
        raise ValueError("recorded greedy tokens differ from the saved logits")
    return artifact, {"receipt": str(receipt_path), "receipt_sha256": digest(receipt_path),
        "artifact": str(artifact_path), "artifact_sha256": actual, "checkpoint": receipt["checkpoint"],
        "prompt_tokens": artifact["prompt"], "generated_tokens": artifact["tokens"],
        "logits": [{"row": i, "shape": list(t.shape), "dtype": str(t.dtype), "sha256": tensor_hash(t),
            "top_k_20_survivors": int((t >= t.topk(20).values[-1]).sum())} for i, t in enumerate(logits)],
        "model_numerical_reference": receipt.get("numerical_reference")}


def cpu_case(artifact, rows, block_steps, policy, *, one_sample=False):
    import torch
    from flare.recipes.qwen38.sampling import DeviceRequestState, RequestAdmission, philox4x32_reference

    steps = 1 if one_sample else block_steps
    initial = DeviceRequestState.allocate("cpu")
    parameters = []
    for row, slot in enumerate(SLOTS[:rows]):
        prefix = artifact["prompt"] + artifact["tokens"][:row]
        parameter = RequestAdmission(generation=31 + slot, seed=0xF123000000000001 + slot * 0x100000001,
            max_new_tokens=block_steps + 1, temperature=0. if policy == "temperature_zero" else 1.,
            top_k=20, top_p=.95)
        initial.admit(slot, parameter, prefix, consumed_tokens=len(prefix), ple_history=tuple(prefix[-2:]))
        initial.rng_counter[slot] = row * 7
        parameters.append(parameter)
    state = {name: getattr(initial, name).clone() for name in initial.__dataclass_fields__}
    expected = {name: value.clone() for name, value in state.items()}
    trace = torch.full((steps, 4), -1, dtype=torch.int64)
    records = torch.full((2, 4, 8), -99, dtype=torch.int64)
    records[0].zero_()
    records[0, :, 4] = -1
    sampled_rows = [0] * steps
    consumed_rows = [0] * steps
    terminals = []
    for row, slot in enumerate(SLOTS[:rows]):
        scores = artifact["logits"][row].float()
        values, ids = torch.sort(scores, descending=True, stable=True)
        values[values < values[19].clone()] = -torch.inf
        probabilities = values.softmax(0)
        cumulative = probabilities.cumsum(0)
        starts = torch.searchsorted(-values, -values, right=False)
        probabilities[(cumulative - probabilities)[starts] >= torch.tensor(.95, dtype=torch.float32)] = 0
        probabilities /= probabilities.sum()
        probabilities = torch.zeros_like(probabilities).scatter_(0, ids, probabilities)
        cdf = probabilities.cumsum(0).double()
        for step in range(steps):
            if int(expected["status"][slot]) != 1:
                records[0, slot, 4] = -1
                continue
            counter = int(expected["rng_counter"][slot].item())
            if policy == "temperature_zero":
                token = int(scores.argmax())
            else:
                word = philox4x32_reference(parameters[row].seed, counter)[0]
                uniform = float(torch.tensor(word, dtype=torch.float32) * 2**-32 + 2**-33)
                index = int(torch.searchsorted(cdf, torch.tensor(uniform, dtype=torch.float64), right=False))
                token = index if index < len(cdf) else int(torch.nonzero(probabilities > 0)[-1])
            trace[step, slot] = token
            sampled_rows[step] += 1
            consumed = int(expected["consumed_tokens"][slot])
            expected["generated_tokens"][slot] += 1
            expected["rng_counter"][slot] = counter + 1
            expected["histogram"][slot, token] += 1
            terminal = token in parameters[row].eos_token_ids
            records[0, slot] = torch.tensor([parameters[row].generation, 1, consumed,
                step + 1, token, 2 if terminal else 1, 0, 0], dtype=torch.int64)
            if terminal:
                expected["status"][slot] = 2
                expected["pending_token"][slot] = token
                terminals.append({"row": row, "slot": slot, "step": step, "token": token,
                    "seed": parameters[row].seed, "rng_counter_before": counter})
                continue
            consumed_rows[step] += 1
            previous = int(expected["ple_history"][slot, 1])
            expected["ple_history"][slot] = torch.tensor([previous, token])
            expected["consumed_tokens"][slot] = consumed + 1
    expected = {"state." + name: value for name, value in expected.items()}
    expected.update(sampled_ids=trace, completion_records=records)
    return {"name": f"{policy}-b{rows}-n{block_steps}" + ("-one-sample" if one_sample else ""),
        "rows": rows, "steps": steps, "budget_steps": block_steps, "policy": policy,
        "timing_scope": "one_active_sample" if one_sample else "block_with_terminal_semantics",
        "sampled_rows_per_step": sampled_rows, "consumed_rows_per_step": consumed_rows,
        "sampled_lane_tokens": sum(sampled_rows), "consumed_lane_tokens": sum(consumed_rows), "terminals": terminals,
        "initial": state, "expected": expected, "parameters": [asdict(p) for p in parameters]}


class CapturedCase:
    def __init__(self, case, artifact, stream):
        import torch
        from flare.recipes.qwen38.sampling import DeviceRequestState, PreparedSampling, PreparedTokenState

        self.case = case
        self.name = case["name"]
        rows, steps = case["rows"], case["steps"]
        self.state = DeviceRequestState.allocate("cuda")
        self.logits = torch.stack(artifact["logits"][:rows]).cuda()
        self.slots = torch.tensor(SLOTS[:rows], dtype=torch.int32, device="cuda")
        self.epochs = torch.tensor([31 + slot for slot in SLOTS[:rows]], dtype=torch.int64, device="cuda")
        self.live = torch.tensor([rows], dtype=torch.int32, device="cuda")
        self.records = torch.full((2, 4, 8), -99, dtype=torch.int64, device="cuda")
        self.frame = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.sequence = torch.ones(1, dtype=torch.int64, device="cuda")
        self.trace = torch.full((steps, 4), -1, dtype=torch.int64, device="cuda")
        self.tokens = torch.zeros(rows, dtype=torch.int64, device="cuda")
        self.spans = torch.arange(rows + 1, dtype=torch.int32, device="cuda")
        self.sampler = PreparedSampling(self.logits, self.state, self.slots, self.epochs, self.live,
            self.records, self.frame, self.sequence)
        self.token_state = PreparedTokenState(self.state, self.tokens, self.spans, self.slots, self.epochs, self.live)
        for name, value in case["initial"].items():
            getattr(self.state, name).copy_(value)
        self.sampler.prepare()
        self.token_state.prepare()
        self.buffers = {"state." + name: getattr(self.state, name) for name in self.state.__dataclass_fields__}
        self.buffers.update(logits=self.logits, slots=self.slots, epochs=self.epochs, live=self.live,
            completion_records=self.records, frame=self.frame, sequence=self.sequence, sampled_ids=self.trace,
            tokens=self.tokens, spans=self.spans, sample_enabled=self.sampler.sample_enabled)
        for name, tensor in self.sampler.scratch_tensors.items():
            self.buffers["sampler." + name] = tensor
        for name in ("pending_tokens", "positions", "ple_history", "active"):
            self.buffers["token_state." + name] = getattr(self.token_state, name)
        self.initial = {name: value.clone() for name, value in self.buffers.items()}
        self.start = torch.cuda.Event(enable_timing=True, external=True)
        self.end = torch.cuda.Event(enable_timing=True, external=True)
        self.start.record()
        self.end.record()
        self.step(0)
        self.reset()
        torch.cuda.synchronize()
        self.reset_graph, self.graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.reset_graph, stream=stream):
            self.reset()
        with torch.cuda.graph(self.graph, stream=stream):
            self.start.record(stream)
            for step in range(steps):
                self.step(step)
            self.end.record(stream)
        torch.cuda.synchronize()

    def reset(self):
        for name, value in self.buffers.items():
            value.copy_(self.initial[name])

    def step(self, step):
        self.sampler()
        self.trace[step].copy_(self.records[0, :, 4])
        self.token_state.gather()
        self.tokens.copy_(self.token_state.pending_tokens)
        self.token_state.consume()

    def readback(self):
        return {name: self.buffers[name].cpu() for name in self.case["expected"]}


def quantiles(values):
    ordered = sorted(values)

    def quantile(q):
        position = (len(ordered) - 1) * q
        low = math.floor(position)
        high = math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {"count": len(values), "minimum": ordered[0], "median": quantile(.5),
        "p90": quantile(.9), "p95": quantile(.95), "maximum": ordered[-1], "quantile_method": "linear"}


def check_outputs(expected, observed, artifacts, label):
    import torch

    hashes, failures = {}, []
    for name, reference in expected.items():
        actual = observed[name]
        equal = actual.shape == reference.shape and actual.dtype == reference.dtype and tensor_hash(actual) == tensor_hash(reference)
        hashes[name] = tensor_hash(actual)
        if not equal:
            path = artifacts / f"{label}-{name}.pt"
            torch.save({"expected": reference, "actual": actual}, path)
            failures.append({"tensor": name, "expected_sha256": tensor_hash(reference),
                "actual_sha256": hashes[name], "artifact": str(path), "artifact_sha256": digest(path)})
    return hashes, failures


def measure(args, artifact, cases, receipt):
    import torch
    import flare_native
    from flare.preparation.graphs import PreparedGraphBundle

    if not args.gpu_uuid or os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise ValueError("GPU execution requires --gpu-uuid matching the single assigned CUDA_VISIBLE_DEVICES UUID")
    if not args.gpu_uuid.startswith("GPU-"):
        raise ValueError("GPU selection must use its explicit UUID")
    package = Path(flare_native.__file__).resolve()
    extensions = [p for p in package.parent.iterdir() if any(p.name.endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES)]
    if len(extensions) != 1:
        raise ValueError("cannot identify one native extension")
    receipt["native_extension"] = {"path": str(extensions[0]), "sha256": digest(extensions[0])}
    receipt["device"] = {"uuid": args.gpu_uuid, "name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)), "torch_cuda_version": torch.version.cuda}
    query = ["nvidia-smi", "--id=" + args.gpu_uuid,
        "--query-gpu=uuid,name,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw", "--format=csv,noheader"]
    receipt["device"]["before"] = subprocess.run(query, check=True, text=True, capture_output=True).stdout.strip()
    stream = torch.cuda.Stream()
    owners = []
    for case in cases:
        print("CAPTURE", case["name"], flush=True)
        owners.append(CapturedCase(case, artifact, stream))
    graphs = {owner.name + suffix: graph for owner in owners for suffix, graph in (("-reset", owner.reset_graph), ("-sample", owner.graph))}
    buffers = {owner.name + "." + name: tensor for owner in owners for name, tensor in owner.buffers.items()}
    bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers=buffers, resources=(*owners, stream))
    receipt["graph_handles"] = {name: graph.raw_cuda_graph_exec() for name, graph in graphs.items()}
    receipt["capture_allocated_bytes"] = torch.cuda.memory_allocated()
    receipt["observations"] = []
    receipt["timings"] = {owner.name: [] for owner in owners}
    save(args.output, receipt)

    def replay(name):
        ticket = worker.submit(bundle.graph_indices[name])
        result = worker.poll(timeout_ms=60000)
        if result is None or result.ticket != ticket or result.error is not None:
            raise RuntimeError(f"native replay did not complete {name}: {result}")

    baselines = {}
    worker = None
    try:
        worker = bundle.start_worker()
        for iteration in range(args.warmup + args.repetitions):
            # Reverse case order between repetitions.
            order = owners if iteration % 2 == 0 else list(reversed(owners))
            for owner in order:
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                replay(owner.name + "-reset")
                replay(owner.name + "-sample")
                after = torch.cuda.memory_stats()["allocation.all.allocated"]
                elapsed = owner.start.elapsed_time(owner.end)
                if not math.isfinite(elapsed) or elapsed <= 0:
                    raise RuntimeError("CUDA timing events did not report a positive finite duration")
                observed = owner.readback()
                label = f"{owner.name}-iteration{iteration}"
                hashes, failures = check_outputs(owner.case["expected"], observed, args.artifacts, label)
                paired_key = owner.case["name"]
                if owner.case["policy"] == "temperature_zero":
                    if paired_key in baselines and baselines[paired_key] != hashes:
                        failures.append({"tensor": "repeated_temperature_zero_state", "expected": baselines[paired_key], "actual": hashes})
                    baselines.setdefault(paired_key, hashes)
                record = {"name": owner.name, "iteration": iteration, "warmup": iteration < args.warmup,
                    "allocation_count_before": before, "allocation_count_after": after,
                    "elapsed_block_ms": elapsed, "sampled_lane_tokens": owner.case["sampled_lane_tokens"],
                    "state_sha256": hashes, "sampled_ids": observed["sampled_ids"].tolist(), "failures": failures}
                if owner.case["policy"] == "temperature_zero" or owner.case["steps"] == 1:
                    record["elapsed_per_active_batch_step_ms"] = elapsed / owner.case["steps"]
                receipt["observations"].append(record)
                if iteration >= args.warmup:
                    receipt["timings"][owner.name].append(elapsed)
                save(args.output, receipt)
                if failures or after != before:
                    raise AssertionError(f"sampler correctness/allocation failure in {label}; receipt preserves details")
            print(f"CHECKED block repetition {iteration + 1}/{args.warmup + args.repetitions}", flush=True)
        receipt["summary_block_ms"] = {name: quantiles(values) for name, values in receipt["timings"].items()}
        receipt["summary_active_batch_step_ms"] = {owner.name:
            quantiles([value / owner.case["steps"] for value in receipt["timings"][owner.name]])
            for owner in owners if owner.case["policy"] == "temperature_zero" or owner.case["steps"] == 1}
        receipt["passed"] = True
    finally:
        try:
            if worker is not None:
                worker.close()
                receipt["native_worker_closed"] = True
        except Exception as error:
            receipt["native_worker_closed"] = False
            receipt["close_error"] = repr(error)
            receipt["passed"] = False
            raise
        finally:
            save(args.output, receipt)
    receipt["device"]["after"] = subprocess.run(query, check=True, text=True, capture_output=True).stdout.strip()
    receipt["native_extension_unchanged"] = digest(extensions[0]) == receipt["native_extension"]["sha256"]
    if not receipt["native_extension_unchanged"]:
        raise AssertionError("native extension changed during qualification")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=ROOT / "docs/validation/decoder-execution.json")
    parser.add_argument("--logits-artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true", help="Verify provenance and build exact CPU state oracles without initializing CUDA.")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--blocks", nargs="+", type=int, choices=(32, 64), default=(32, 64))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists() or (args.artifacts.exists() and any(args.artifacts.iterdir())):
        parser.error("output and artifact destinations must not contain an earlier run")
    if args.warmup < 1 or args.repetitions < 5 or len(set(args.blocks)) != len(args.blocks):
        parser.error("require at least one warmup, five repetitions and unique block sizes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    import torch
    import triton

    torch.set_num_threads(2)
    import importlib.metadata
    import importlib.util
    flashinfer_root = Path(importlib.util.find_spec("flashinfer").origin).resolve().parent
    source_paths = [Path(__file__).resolve(), ROOT / "flare/recipes/qwen38/sampling.py", ROOT / "flare/preparation/graphs.py", *(
        flashinfer_root / relative for relative in ("sampling.py", "jit/sampling.py", "utils.py", "data/csrc/sampling.cu",
        "data/include/flashinfer/sampling.cuh", "data/include/flashinfer/air_top_p.cuh"))]
    receipt = {"schema": "flare-sampling-path-qualification-v2", "sampling_policy": "flashinfer_top_k_top_p_min_p_philox_v1", "flashinfer": importlib.metadata.version("flashinfer-python"), "historical_baseline": "sampling-paths-original.json", "started_at": datetime.now(timezone.utc).isoformat(),
        "passed": False, "mode": "cpu_validation" if args.validate_only else "native_cuda_measurement",
        "command": sys.argv, "python": sys.version, "torch": torch.__version__, "triton": triton.__version__,
        "source_sha256": {str(path): digest(path) for path in source_paths}, "warmup_blocks": args.warmup,
        "measured_blocks": args.repetitions, "limitations": [
            "Operator measurement on recorded checkpoint logits; excludes vocabulary projection, transformer computation and HTTP/runtime scheduling.",
            "Timed blocks include production sampler, sampled-ID trace copy, pending-token gather/copy/consume and CUDA graph event scope.",
            "Each block reuses fixed logits with neutral penalties; captured copies reset independent request state outside the timed span.",
            "Temperature-zero cases use one-hot probabilities through the same FlashInfer sampling path as checkpoint-default cases.",
            "Stochastic block timings retain EOS and inactive iterations and are not divided by block length as fully active latency; one-sample captures measure active default sampling separately.",
            "Model arithmetic acceptance remains unresolved; these logits are fingerprinted execution artifacts, not an independent model oracle."]}
    started = time.monotonic()
    try:
        artifact, provenance = load_inputs(args.receipt.resolve(), args.logits_artifact)
        receipt["inputs"] = provenance
        cases = [cpu_case(artifact, rows, steps, policy) for rows in (1, 2, 4) for steps in args.blocks
                 for policy in ("checkpoint_defaults", "temperature_zero")]
        cases.extend(cpu_case(artifact, rows, steps, "checkpoint_defaults", one_sample=True)
                     for rows in (1, 2, 4) for steps in args.blocks)
        receipt["cases"] = [{key: value for key, value in case.items() if key not in ("initial", "expected")}
            | {"expected_sha256": {name: tensor_hash(value) for name, value in case["expected"].items()},
               "expected_sampled_ids": case["expected"]["sampled_ids"].tolist()} for case in cases]
        if args.validate_only:
            if torch.cuda.is_initialized():
                raise AssertionError("CPU-only validation initialized CUDA")
            receipt["cuda_initialized"] = False
            receipt["passed"] = True
        else:
            measure(args, artifact, cases, receipt)
        receipt["sources_unchanged"] = all(digest(path) == value for path, value in receipt["source_sha256"].items())
        if not receipt["sources_unchanged"]:
            raise AssertionError("measurement sources changed during execution")
    except Exception:
        receipt["passed"] = False
        receipt["failure"] = traceback.format_exc()
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        save(args.output, receipt)
        print(json.dumps({"output": str(args.output), "passed": receipt["passed"], "mode": receipt["mode"]}), flush=True)


if __name__ == "__main__":
    main()
