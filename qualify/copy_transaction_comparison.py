"""Compare actual-checkpoint W1 and copy-verification state at equal positions.

Singleton grammar admits every copied token through the public FlashInfer chain
sampler. Integer state and identical-shape snapshot continuations are exact gates.
Cross-width floating state is measured without an uncalibrated numerical gate.
The ordinary, non-speculative decoder and rejected-prefix cases remain separate
qualification requirements.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import os
from pathlib import Path
import time

from qualify.provenance import digest, file_record, write_json
from qualify.speculative_state_parity import Commands, LEGAL_SHAPES, MutableState, _allocated, command_plan

ROOT = Path(__file__).resolve().parents[1]


def parse_shape(value):
    try:
        shape = tuple(int(x) for x in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must be BxW") from error
    if shape not in LEGAL_SHAPES:
        raise argparse.ArgumentTypeError("shape must have B in {1,2,4}, W in [1,8], B*W <= 8")
    return shape


def case_plan(batch, width):
    """Equal-position prefixes and identical-shape restored continuations."""
    if (batch, width) not in LEGAL_SHAPES:
        raise ValueError("invalid verification shape")
    slots = [3, 0, 2, 1][:batch]

    def round_(w, **extra):
        return {"kind": "speculative", "batch": batch, "width": w, "slots": slots, **extra}

    common = [{"kind": "init", "slots": slots, "generation": 11,
               "lengths": [63] * batch, "budget": 512}]
    common += [{"kind": "prefill", "slots": [slot], "length": 63} for slot in slots]
    common += [round_(1, tag="export_boundary64")]
    common += [{"kind": "snapshot", "slots": [slot], "snapshot": index}
               for index, slot in enumerate(slots)]
    common[-1]["observe"] = "boundary64"
    baseline = [round_(1) for _ in range(width)]
    baseline[-1]["observe"] = "transaction"
    baseline.append(round_(1, observe="following_w1"))
    wide = [round_(width, observe="transaction"), round_(1, observe="following_w1")]
    restore = [{"kind": "restore", "slots": [slot], "generation": 12,
                "snapshot": index, "budget": 512} for index, slot in enumerate(slots)]
    restore += [{"kind": "resume", "slots": slots, "observe": "boundary64"}]
    restored_slots = [0, 2, 1, 3][:batch]
    selected_restore = {
        "slots": restored_slots,
        "snapshot_ids": list(range(batch, 2 * batch)),
        "prepare": common + [wide[0]],
        "snapshots": [{"kind": "snapshot", "slots": [slot], "snapshot": batch + index}
                      for index, slot in enumerate(slots)],
        "restore": [{"kind": "restore", "slots": [slot], "generation": 13,
                     "snapshot": batch + index, "budget": 512}
                    for index, slot in enumerate(restored_slots)],
        "continuation": [{"kind": "resume", "slots": restored_slots, "observe": "transaction"},
                         {**round_(1, observe="following_w1"), "slots": restored_slots}],
    }
    return {"batch": batch, "width": width, "slots": slots, "common": common,
            "w1": baseline, "wide": wide, "restore": restore, "selected_restore": selected_restore,
            "positions": {"boundary64": 64, "transaction": 64 + width, "following_w1": 65 + width},
            "accepted_drafts_per_lane": width - 1,
            "unexercised_rejection_prefixes": list(range(width - 1)),
            "unexercised_clipped_emission_counts": list(range(1, width))}


def state_column(slot, accepted, extra_base):
    if type(slot) is not int or not 0 <= slot < 4 or type(accepted) is not int or not 1 <= accepted <= 8:
        raise ValueError("physical slot and accepted-column count are out of range")
    return slot + 1 if accepted == 1 else extra_base + slot * 7 + accepted - 2


def byte_equal(a, b):
    import torch

    return (a.shape == b.shape and a.dtype == b.dtype
            and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                            b.contiguous().reshape(-1).view(torch.uint8)))


def tensor_digest(value):
    import torch

    return hashlib.sha256(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def compare_states(expected, actual, *, exact):
    """Integer and finite checks are gates; cross-width floating errors are data."""
    import torch

    checks = {"same_regions": expected.keys() == actual.keys()}
    measurements = {}
    for name in sorted(expected.keys() & actual.keys()):
        reference, value = expected[name], actual[name]
        shape = reference.shape == value.shape and reference.dtype == value.dtype
        checks[name + ".layout"] = shape
        if not shape:
            continue
        equal = byte_equal(reference, value)
        record = {"shape": list(value.shape), "dtype": str(value.dtype), "bitwise_equal": equal,
                  "expected_sha256": tensor_digest(reference), "actual_sha256": tensor_digest(value)}
        if value.is_floating_point():
            finite = bool(torch.isfinite(reference).all() and torch.isfinite(value).all())
            checks[name + ".finite"] = finite
            if finite:
                ref, got = reference.float(), value.float()
                diff = got - ref
                norm = float(torch.linalg.vector_norm(ref))
                error = float(torch.linalg.vector_norm(diff))
                record.update(max_abs=float(diff.abs().max()) if diff.numel() else 0,
                              l2_error=error, reference_l2=norm,
                              relative_l2=error / norm if norm else (0.0 if not error else None))
        if exact or not value.is_floating_point():
            checks[name + ".bitwise"] = equal
        measurements[name] = record
    return {"passed": all(checks.values()), "checks": checks, "measurements": measurements,
            "floating_gate": "bitwise" if exact else "finite_only; numerical calibration pending"}


def logical_pages(tensor, pages, count):
    """Read physical pages individually, then normalize logical order on CPU."""
    import torch

    size = tensor.shape[1]
    if count < 0 or count > len(pages) * size:
        raise ValueError("logical cache read exceeds the request page table")
    if not count:
        return torch.empty((0, *tensor.shape[2:]), dtype=tensor.dtype)
    return torch.cat([tensor[page].detach().cpu() for page in pages[:(count + size - 1) // size]])[:count].clone()


def committed_state(owner, builder, slots):
    """Read the model state that the next accepted transaction will consume."""
    import torch

    pool = owner.pool
    accepted = owner.previous_accepted.cpu().tolist()
    result, metadata, checks = {}, {}, {}
    for slot in slots:
        request = builder.requests[slot]
        count, position = accepted[slot], request["consumed"]
        recurrent_row = state_column(slot, count, owner.extra_base)
        prefix = f"slot{slot}."
        metadata[str(slot)] = {"consumed": position, "accepted_column": count,
                               "recurrent_row": recurrent_row, "pages": request["pages"], "raw_anchors": {}}
        for layer, state in pool.gdn.items():
            result[prefix + f"gdn.{layer}.recurrent"] = state.recurrent[recurrent_row].detach().cpu()
            result[prefix + f"gdn.{layer}.convolution"] = state.convolution[slot + 1, :, count - 1:count + 2].detach().cpu()
        result[prefix + "ple_convolution"] = pool.ple_convolution[slot + 1, :, count - 1:count + 8].detach().cpu()
        for name in ("last_hidden", "consumed_tokens", "ple_history", "rope_delta", "target_residual", "target_residual_position"):
            result[prefix + name] = pool.region(name)[slot + 1].detach().cpu()
        for name in ("pending_token", "histogram", "status", "error_code"):
            result[prefix + "request." + name] = getattr(owner.requests, name)[slot].detach().cpu()
        for layer, cache in owner.cache.items():
            qprefix = prefix + f"qsa.{layer}."
            for name in ("main_k_cache", "main_v_cache"):
                result[qprefix + name] = logical_pages(getattr(cache, name), request["pages"], position)
            result[qprefix + "compressed_k_cache"] = logical_pages(cache.compressed_k_cache, request["pages"], position // 4)
            start = position // 4 * 4
            logical = torch.arange(start, position, dtype=torch.int64)
            indices = logical % cache.raw_k_ring.shape[1]
            tags = cache.raw_logical_positions[slot + 1].cpu()[indices]
            rope = cache.raw_rope_positions[slot + 1].cpu()[indices]
            delta = int(result[prefix + "rope_delta"].item())
            checks[qprefix + "unfinished_group_tags"] = torch.equal(tags, logical)
            checks[qprefix + "unfinished_group_rope"] = torch.equal(rope, (logical + delta)[:, None].expand(-1, 3))
            result[qprefix + "raw_unfinished_keys"] = cache.raw_k_ring[slot + 1].cpu()[indices]
            result[qprefix + "raw_unfinished_positions"] = tags
            result[qprefix + "raw_unfinished_rope"] = rope
            metadata[str(slot)]["raw_anchors"][str(layer)] = int(cache.raw_interval_start_positions[slot + 1].cpu())
        checks[prefix + "consumed"] = int(result[prefix + "consumed_tokens"].item()) == position
    return result, metadata, checks


def remap_state_slots(tensors, actual_slots, reference_slots):
    """Compare corresponding requests independently of physical slot placement."""
    if (len(actual_slots) != len(reference_slots) or len(set(actual_slots)) != len(actual_slots)
            or len(set(reference_slots)) != len(reference_slots)):
        raise ValueError("state comparison requires a bijective physical-slot mapping")
    mapping = {f"slot{actual}.": f"slot{reference}."
               for actual, reference in zip(actual_slots, reference_slots)}
    result = {}
    for name, tensor in tensors.items():
        prefix, suffix = name.split(".", 1)
        if prefix + "." not in mapping:
            raise ValueError("state region is outside the declared request mapping")
        result[mapping[prefix + "."] + suffix] = tensor
    return result


def storage_fingerprints(owner, *, rows=(), pages=(), raw_slots=()):
    """Read exact source bytes on CPU; no retained GPU clones or replay allocations."""
    values = {}
    for row in sorted(set(rows)):
        values[f"hybrid.{row}"] = tensor_digest(owner.pool.backing[row].detach().cpu())
    for layer, cache in owner.cache.items():
        for name in ("main_k_cache", "main_v_cache", "compressed_k_cache"):
            for page in sorted(set(pages)):
                values[f"qsa.{layer}.{name}.page{page}"] = tensor_digest(getattr(cache, name)[page].detach().cpu())
        for name in ("raw_k_ring", "raw_logical_positions", "raw_rope_positions", "raw_interval_start_positions"):
            for slot in sorted(set(raw_slots)):
                values[f"qsa.{layer}.{name}.slot{slot}"] = tensor_digest(getattr(cache, name)[slot].detach().cpu())
    return values


def private_restore_checks(source_pages, restored_pages, consumed):
    """A restored request shares complete pages and owns every continuation page."""
    if len(source_pages) != len(restored_pages) or not source_pages:
        raise ValueError("restore page checks require corresponding nonempty request lists")
    full = consumed // 64
    source_storage = {page for pages in source_pages for page in pages}
    private = [page for pages in restored_pages for page in pages[full:]]
    return {
        "full_pages_shared": all(a[:full] == b[:full] for a, b in zip(source_pages, restored_pages)),
        "private_tail_present": all(len(pages) > full for pages in restored_pages),
        "private_pages_distinct_from_sources": not source_storage.intersection(private),
        "private_pages_distinct_between_requests": len(private) == len(set(private)),
    }


def request_checks(owner, builder):
    import torch

    state = {f.name: getattr(owner.requests, f.name).detach().cpu() for f in fields(owner.requests)}
    checks = {}
    for slot, request in builder.requests.items():
        key = f"slot{slot}."
        for name, expected in (("generation", request["generation"]), ("consumed_tokens", request["consumed"]),
                               ("generated_tokens", request["generated"]), ("pending_token", request["pending"]),
                               ("rng_counter", request["generated"]), ("seed", request["seed"]), ("status", 1), ("error_code", 0)):
            checks[key + name] = state[name][slot].item() == expected
        histogram = state["histogram"][slot]
        checks[key + "histogram"] = {i: int(histogram[i]) for i in histogram.nonzero().flatten().tolist()} == request["histogram"]
        checks[key + "ple_history"] = state["ple_history"][slot].tolist() == ([248044] * 2 + request["tokens"])[-2:]
    checks["previous_accepted"] = owner.previous_accepted.cpu().tolist() == builder.previous
    checks["accepted_generation"] = owner.accepted_generation.cpu().tolist() == builder.accepted_generation
    checks["target_rng_generation"] = owner.rng.generation.cpu().tolist() == builder.target_generation
    checks["target_rng_offset"] = owner.rng.target_offset.cpu().tolist() == builder.target_offset
    checks["proposal_rng_untouched"] = not bool(torch.any(owner.draft_rng_generation.cpu())) and not bool(torch.any(owner.draft_rng_offset.cpu()))
    return checks


class Replay:
    def __init__(self, model, owner, builder, checks):
        import torch

        self.model, self.owner, self.builder, self.checks = model, owner, builder, checks
        self.stream = torch.cuda.Stream(device=model.device)
        self.commands = []

    def run(self, operation):
        import torch

        command = self.builder.prepare(operation)
        uploads = [(name, torch.frombuffer(bytearray(self.builder.payloads[sha]), dtype=torch.uint8).pin_memory())
                   for name, sha in command["uploads"]]
        frame = torch.tensor([0], dtype=torch.int32, pin_memory=True)
        sequence = torch.tensor([command["sequence"]], dtype=torch.int64, pin_memory=True)
        before = _allocated(self.model.device)
        with torch.cuda.stream(self.stream):
            for name, value in uploads:
                self.model.buffers[name].view(torch.uint8).reshape(-1).copy_(value, non_blocking=True)
            self.model.buffers["frame_index"].copy_(frame, non_blocking=True)
            self.model.buffers["submission_sequence"].copy_(sequence, non_blocking=True)
            self.model.graphs[command["graph"]].replay()
        self.stream.synchronize()
        after = _allocated(self.model.device)
        records = self.model.buffers["completion_records"][0].cpu().tolist()
        checks, _ = self.builder.complete(command, records)
        checks["replay_noalloc"] = before == after
        checks.update(request_checks(self.owner, self.builder))
        if operation["kind"] in ("snapshot", "restore"):
            from qualify.checkpoint_state import checkpoint_transfer_checks

            slot = operation["slots"][0]
            request = self.builder.requests[slot]
            checks.update({"transfer." + key: value for key, value in
                checkpoint_transfer_checks(self.owner.pool, self.owner.cache, slot=slot,
                    snapshot=operation["snapshot"], pages=request["pages"],
                    consumed=request["consumed"], accepted=self.builder.previous[slot],
                    extra_base=self.owner.extra_base).items()})
        self.commands.append({**command, "records": records, "checks": checks,
                              "allocations_before": before, "allocations_after": after})
        self.checks.update({f"command{len(self.commands) - 1}.{key}": value for key, value in checks.items()})
        if not all(checks.values()):
            raise AssertionError("transaction failed: " + ", ".join(k for k, v in checks.items() if not v))


def run_case(model, owner, reset, fixture, plan, artifacts, result):
    import torch

    observations = {}
    for arm in ("w1", "wide"):
        reset.reset()
        directory = artifacts / arm
        (directory / "uploads").mkdir(parents=True)
        builder = Commands(model, fixture, directory)
        arm_result = result["arms"][arm] = {"checks": {}, "commands": [], "observations": {}}
        replay = Replay(model, owner, builder, arm_result["checks"])
        arm_result["commands"] = replay.commands
        sequences = [(arm, plan["common"] + plan[arm])]
        if arm == "wide":
            sequences.append(("restored", plan["restore"] + plan["wide"]))
        for label, sequence in sequences:
            for operation in sequence:
                replay.run(operation)
                if "observe" not in operation:
                    continue
                stage = operation["observe"]
                tensors, metadata, checks = committed_state(owner, builder, plan["slots"])
                checks["expected_positions"] = all(x["consumed"] == plan["positions"][stage] for x in metadata.values())
                for slot, state in metadata.items():
                    anchor = state["consumed"] - operation.get("width", 1)
                    checks[f"slot{slot}.raw_interval_anchor"] = all(value == anchor for value in state["raw_anchors"].values())
                arm_result["checks"].update({f"{label}.{stage}.{k}": v for k, v in checks.items()})
                path = artifacts / f"{label}-{stage}.pt"
                torch.save(tensors, path)
                arm_result["observations"][f"{label}.{stage}"] = {"artifact": file_record(path), "metadata": metadata, "checks": checks}
                if label == "restored":
                    comparison = compare_states(observations["wide", stage], tensors, exact=True)
                    result["comparisons"]["restore." + stage] = comparison
                else:
                    observations[label, stage] = tensors
                    if label == "wide":
                        shared_prefix = stage == "boundary64"
                        comparison = compare_states(observations["w1", stage], tensors, exact=shared_prefix)
                        result["comparisons"][("shared_prefix." if shared_prefix else "cross_width.") + stage] = comparison
                if not all(checks.values()):
                    raise AssertionError(f"invalid committed state: {label}.{stage}")
    run_selected_restore(model, owner, reset, fixture, plan, artifacts, result, observations)
    result["passed"] = (all(all(a["checks"].values()) for a in result["arms"].values())
                        and all(c["passed"] for c in result["comparisons"].values()))


def run_selected_restore(model, owner, reset, fixture, plan, artifacts, result, observations):
    """Snapshot a Wn-selected column before any W1 normalizes the live state."""
    import torch

    reset.reset()
    selected = plan["selected_restore"]
    directory = artifacts / "selected_restore"
    (directory / "uploads").mkdir(parents=True)
    builder = Commands(model, fixture, directory)
    arm = result["arms"]["selected_restore"] = {"checks": {}, "commands": [], "observations": {}}
    replay = Replay(model, owner, builder, arm["checks"])
    arm["commands"] = replay.commands
    for operation in selected["prepare"]:
        replay.run(operation)
    consumed = plan["positions"]["transaction"]
    source_pages = [list(builder.requests[slot]["pages"]) for slot in plan["slots"]]
    used_pages = sorted({page for pages in source_pages for page in pages[:(consumed + 63) // 64]})
    source_rows = [row for slot in plan["slots"] for row in
                   (slot + 1, state_column(slot, plan["width"], owner.extra_base))]

    def observe(stage, slots, *, label, anchor):
        tensors, metadata, checks = committed_state(owner, builder, slots)
        checks["expected_positions"] = all(x["consumed"] == plan["positions"][stage] for x in metadata.values())
        checks["accepted_column"] = all(x["accepted_column"] == (plan["width"] if label == "source" else 1)
                                        for x in metadata.values())
        checks["raw_interval_anchor"] = all(value == anchor for state in metadata.values()
                                            for value in state["raw_anchors"].values())
        arm["checks"].update({f"{label}.{stage}.{key}": value for key, value in checks.items()})
        path = artifacts / f"selected_restore-{label}-{stage}.pt"
        torch.save(tensors, path)
        arm["observations"][f"{label}.{stage}"] = {"artifact": file_record(path), "metadata": metadata, "checks": checks}
        normalized = remap_state_slots(tensors, slots, plan["slots"])
        result["comparisons"][f"selected_restore.{label}.{stage}"] = compare_states(observations["wide", stage], normalized, exact=True)
        result["comparisons"][f"selected_restore.cross_width.{label}.{stage}"] = compare_states(observations["w1", stage], normalized, exact=False)
        if not all(checks.values()):
            raise AssertionError(f"invalid selected-column state: {label}.{stage}")

    observe("transaction", plan["slots"], label="source", anchor=64)
    source_guard = {"rows": source_rows + list(range(5, 5 + plan["batch"])), "pages": used_pages,
                    "raw_slots": [slot + 1 for slot in plan["slots"]]}
    source_before = storage_fingerprints(owner, **source_guard)

    def protected_run(operation, guard, expected):
        replay.run(operation)
        actual = storage_fingerprints(owner, **guard)
        checks = {"same_guard_regions": actual.keys() == expected.keys(),
                  **{name: actual.get(name) == sha for name, sha in expected.items()}}
        replay.commands[-1]["immutable_storage"] = {"before": expected, "after": actual, "checks": checks}
        arm["checks"].update({f"command{len(replay.commands) - 1}.immutable.{key}": value for key, value in checks.items()})
        if not all(checks.values()):
            raise AssertionError("selected-column transfer modified protected source storage")

    for operation in selected["snapshots"]:
        protected_run(operation, source_guard, source_before)
    immutable_guard = {"rows": list(range(5, 5 + 2 * plan["batch"])), "pages": used_pages}
    immutable_before = storage_fingerprints(owner, **immutable_guard)
    arm["snapshot_endpoints"] = [{"source_slot": source, "destination_slot": destination,
        "snapshot": index, "consumed": consumed, "selected_column": plan["width"]}
        for source, destination, index in zip(plan["slots"], selected["slots"], selected["snapshot_ids"])]
    for operation in selected["restore"]:
        protected_run(operation, immutable_guard, immutable_before)
    restored_pages = [list(builder.requests[slot]["pages"]) for slot in selected["slots"]]
    page_checks = private_restore_checks(source_pages, restored_pages, consumed)
    arm["page_ownership"] = {"source_pages": source_pages, "restored_pages": restored_pages,
                             "consumed": consumed, "checks": page_checks}
    arm["checks"].update({"pages." + key: value for key, value in page_checks.items()})
    if not all(page_checks.values()):
        raise AssertionError("selected-column restore violated shared/private page ownership")
    for operation in selected["continuation"]:
        protected_run(operation, immutable_guard, immutable_before)
        stage = operation["observe"]
        anchor = consumed - 1 if operation["kind"] == "resume" else consumed
        observe(stage, selected["slots"], label="restored", anchor=anchor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--prefill-chunk-size", type=int, default=4096)
    parser.add_argument("--shape", type=parse_shape, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() or args.artifacts.exists():
        parser.error("fresh output and artifact paths are required")
    if args.prefill_chunk_size < 128 or args.prefill_chunk_size % 64:
        parser.error("prefill chunk size must be 64-aligned and at least 128")
    shapes = args.shape or list(LEGAL_SHAPES)
    if len(shapes) != len(set(shapes)):
        parser.error("duplicate shapes are not allowed")
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    token, = tokenizer.encode(" apple", add_special_tokens=False)
    alternative, = tokenizer.encode(" orange", add_special_tokens=False)
    prompt = tokenizer.encode("Speculative state parity: " + "The capital of France is Paris. " * 20, add_special_tokens=False)[:65]
    fixture = command_plan(prompt, token, alternative)
    fixture = {name: fixture[name] for name in ("prompt", "token", "alternative")}
    plans = [case_plan(*shape) for shape in shapes]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.artifacts.mkdir(parents=True)
    write_json(args.artifacts / "fixture.json", {**fixture, "cases": plans})
    receipt = {"schema": "flare-copy-transaction-comparison-v1", "gpu_execution": False,
               "fixture": file_record(args.artifacts / "fixture.json"), "shapes": shapes,
               "full_shape_coverage": set(shapes) == set(LEGAL_SHAPES), "cases": {},
               "coverage": {"executor": "Python replay of production captured graphs",
                   "ordinary_non_speculative_decoder": "pending separate preparation and execution",
                   "cross_width_floating_numerical_qualification": "pending independent component calibration",
                   "rejection_prefixes": "pending; only the all-accepted prefix is exercised",
                   "budget_and_eos_clipping": "pending", "inactive_packed_lanes": "not exercised",
                   "selected_column_snapshot": "Wn at S=64+W; immutable snapshot, different physical slot and private tail, Resume and following W1",
                   "sampler": "temperature 1, singleton legal grammar, public FlashInfer chain verification",
                   "qsa_state": "logical main KV, completed compressed groups and unfinished raw group; physical pages and raw anchors recorded separately",
                   "hybrid_state": "selected recurrent column, convolution history windows, hidden/residual/position/PLE state; all regions bitwise for transfers"}}
    if args.validate_only:
        if torch.cuda.is_initialized():
            raise AssertionError("CPU validation initialized CUDA")
        receipt["command_contract_valid"] = True
        write_json(args.output, receipt)
        return 0
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("an explicitly assigned CUDA_VISIBLE_DEVICES is required")
    import b12x
    import flashinfer
    from flare.recipes.qwen38.bootstrap import PreparationConfig, prepare_model
    from flare.recipes.qwen38.speculative_model import SpeculativePreparation

    paths = {Path(__file__), ROOT / "qualify/speculative_state_parity.py", ROOT / "qualify/provenance.py", ROOT / "qualify/graph_execution.py", ROOT / "qualify/checkpoint_state.py"}
    for directory, pattern in ((ROOT / "flare", "*.py"), (Path(b12x.__file__).resolve().parent, "*.py")):
        paths.update(directory.rglob(pattern))
    fi = Path(flashinfer.__file__).resolve().parent
    paths.update(fi / name for name in ("sampling.py", "jit/sampling.py", "data/include/flashinfer/sampling.cuh"))
    receipt["sources"] = {str(path.resolve()): digest(path) for path in sorted(paths)}
    receipt["checkpoint"] = {"path": str(args.snapshot.resolve()), "assets": [file_record(args.snapshot / name)
        for name in ("config.json", "tokenizer.json", "model.safetensors.index.json")]}
    receipt["environment"] = {name: os.environ.get(name) for name in ("CUDA_VISIBLE_DEVICES", "CUDA_HOME", "PYTHONPATH")}
    started = time.monotonic()
    progress = lambda text: print(f"{time.monotonic() - started:.1f}s {text}", flush=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    try:
        config = PreparationConfig(prefill_chunk_size=args.prefill_chunk_size, mtp=True, speculative_proposal_source="copy")
        with prepare_model(args.snapshot, config, progress=progress) as model:
            owner = next(x for x in model._resources if isinstance(x, SpeculativePreparation))
            receipt["gpu_execution"] = True
            receipt["memory_report"] = model.memory_report
            receipt["manifest"] = model.manifest()
            mapping = [line for line in Path("/proc/self/maps").read_text().splitlines() if "/sampling.so" in line]
            receipt["sampler_mapping"] = mapping
            receipt["sampler_libraries"] = [file_record(path) for path in sorted({line.split(maxsplit=5)[5] for line in mapping})]
            if not mapping:
                raise AssertionError("sampling library mapping is missing")
            reset = MutableState(model, owner)
            for plan in plans:
                name = f"{plan['batch']}x{plan['width']}"
                case = receipt["cases"][name] = {"plan": plan, "arms": {}, "comparisons": {}}
                run_case(model, owner, reset, fixture, plan, args.artifacts / name, case)
                write_json(args.output, receipt)
                progress(f"{name}: transaction gates {'passed' if case['passed'] else 'failed'}; floating calibration pending")
        receipt["sources_unchanged"] = all(digest(path) == sha for path, sha in receipt["sources"].items())
    except BaseException as error:
        import traceback

        receipt["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["transaction_gates_passed"] = (not receipt.get("error") and receipt.get("sources_unchanged", False)
            and len(receipt["cases"]) == len(plans) and all(case.get("passed", False) for case in receipt["cases"].values()))
        receipt["numerical_qualification"] = "pending independent cross-width calibration and ordinary decoder arm"
        write_json(args.output, receipt)
    return 0 if receipt["transaction_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
