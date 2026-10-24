"""Exact per-request MTP tau from retained production graph completion records.

Call ``measure(prepared, root, depths=(2, 3))`` after strict validation and before
export/serving. This helper never prepares a model or launches a server. The
caller owns the GPU lease and must back enough KV pages for the selected full
coding prompts. Failed runs retain their partial records and cleanup result.
"""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
import time
import traceback
import statistics

import torch

from benchmark_mtp import load_prompts
from validate_replay import _Harness


class _MeasurementHarness(_Harness):
    def __init__(self, prepared, output_tokens):
        super().__init__(prepared)
        self.output_tokens = output_tokens
        self.page_counts = []
        self.seeds = []

    def reset(self):
        super().reset()
        needed = sum(count + 1 for count in self.page_counts)
        backed = int(self.pool.memory_capacity[1])
        if needed > backed:
            raise ValueError(f"full prompt tau measurement needs {needed} backed KV pages; only {backed} are backed")
        self.inputs.page_tables.fill_(-1)
        self.owner.draft_tables.fill_(-1)
        begin = 0
        for slot, count in enumerate(self.page_counts):
            pages = torch.arange(begin, begin + count, dtype=torch.int32, device=self.device)
            self.inputs.page_tables[slot, :count].copy_(pages)
            # Native page_offset=1: draft logical page zero uses the second
            # physical page assigned to the request, including prefix restores.
            self.owner.draft_tables[slot, :count].copy_(pages + 1)
            begin += count + 1

    def admission_row(self, row, slot, ids):
        from ignition.components.sampling import RequestAdmission

        self.admission.stage(row, slot, RequestAdmission(generation=self.epoch, seed=self.seeds[slot],
            max_new_tokens=self.output_tokens, temperature=.7, top_k=20, top_p=.95,
            repetition_penalty=1., eos_token_ids=()), ids)

    def run(self, prompts, seeds, depth, result, *, batch_capacity=None):
        active, width = len(prompts), depth + 1
        batch = active if batch_capacity is None else batch_capacity
        if active > batch:
            raise ValueError("active prompts exceed prepared batch capacity")
        self.seeds = seeds
        self.page_counts = [(len(ids) + self.output_tokens + self.page_tokens - 1) // self.page_tokens for ids in prompts]
        if max(self.page_counts) > self.inputs.page_tables.shape[1]:
            raise ValueError("full coding prompt plus requested output exceeds prepared context")
        result.update(required_backed_pages=sum(count + 1 for count in self.page_counts), page_counts=self.page_counts,
                      output_token_ids=[[] for _ in prompts], rounds=[])
        self.initialize(prompts, mtp=True)
        slots = list(range(active))
        padded = slots + [-1] * (batch - active)
        self.rows(padded, sample=True)
        records = self.replay(self.name("resume", batch, True), padded)
        for slot in slots:
            result["output_token_ids"][slot].append(int(records[slot, 4]))
        name = self.name("speculative", batch, True, width=width)
        while any(len(ids) < self.output_tokens for ids in result["output_token_ids"]):
            limits = [min(width, self.output_tokens - len(ids)) for ids in result["output_token_ids"]] + [0] * (batch - active)
            live = [slot if slot < active and limits[slot] else -1 for slot in range(batch)]
            self.rows(live, lengths=limits, sample=True)
            before = self.requests.consumed_tokens[:batch].cpu().tolist()
            records = self.replay(name, live)
            rows = []
            for slot in slots:
                if live[slot] < 0:
                    continue
                record = records[slot].tolist()
                emitted, offered, accepted = map(int, record[8:11])
                if not 1 <= emitted <= limits[slot] or not 0 <= accepted <= min(emitted, offered) or offered > depth:
                    raise RuntimeError(f"invalid speculative completion counts for slot {slot}: {record}")
                if int(self.requests.consumed_tokens[slot]) != before[slot] + emitted:
                    raise RuntimeError("emitted count and consumed input prefix disagree")
                token_ids = record[16:16 + emitted]
                if any(not 0 <= token < self.requests.vocab_size for token in token_ids):
                    raise RuntimeError("completion emitted a token outside the vocabulary")
                result["output_token_ids"][slot].extend(token_ids)
                rows.append({"slot": slot, "emitted": emitted, "offered": offered, "accepted": accepted,
                    "consumed_before": before[slot], "consumed_after": record[2],
                    "verification_width": record[15], "token_ids": token_ids, "completion_record": record})
            result["rounds"].append({"graph": name, "requests": rows})
        result["requests"] = []
        for slot in slots:
            rows = [row for step in result["rounds"] for row in step["requests"] if row["slot"] == slot]
            emitted, offered, accepted = (sum(row[field] for row in rows) for field in ("emitted", "offered", "accepted"))
            result["requests"].append({"slot": slot, "seed": seeds[slot], "prompt_tokens": len(prompts[slot]),
                "output_tokens": len(result["output_token_ids"][slot]), "initial_resume_tokens": 1,
                "verification_steps": len(rows), "emitted_from_verification": emitted,
                "offered_drafts": offered, "accepted_drafts": accepted,
                "tau": emitted / len(rows) if rows else None,
                "acceptance_rate": accepted / offered if offered else None})
        result["status"] = "complete"


def _microtimings(harness, prompts, seeds, depth, *, repetitions=3):
    """Isolated captured component timing with restored representative state."""
    h, batch, width = harness, len(prompts), depth + 1
    h.seeds = seeds
    h.page_counts = [(len(ids) + h.output_tokens + h.page_tokens - 1) // h.page_tokens for ids in prompts]
    h.initialize(prompts, mtp=True)
    slots = list(range(batch))
    h.rows(slots, sample=True)
    h.replay(h.name("resume", batch, True), slots)
    h.rows(slots, lengths=[width] * batch, sample=True)
    name = h.name("speculative", batch, True, width=width)
    operation = h.rounds[batch, width][1]
    first_name = h.name("speculative", batch, True, width=2)
    # Preserve every live/checkpoint row and every backed page touched by this
    # representative prompt. Copies happen before timed events, on the CPU.
    saved = h._save_mutable()
    rows = h.inputs.request_state_rows[:batch, :h.width].cpu().reshape(-1).unique().tolist()
    saved.extend((f"row{row}", h.pool.backing[row], h.pool.backing[row].cpu().clone()) for row in rows)
    pages = sum(count + 1 for count in h.page_counts)
    saved.extend((key, value[:pages], value[:pages].cpu().clone()) for key, value in h.pool.pages.items())
    report = {"depth": depth, "batch": batch, "prompt_tokens": [len(ids) for ids in prompts],
        "method": "CUDA events around isolated component CUDA-graph replay; excludes Python operator launch gaps; native combined graph measured separately",
        "fixture": "Fresh full-prompt state restored from CPU backups before every replay",
        "numerical_check": "Each component must match retained production graph hidden/logit bytes before capture and on every timed repeat",
        "components": {}}

    def compare(label, scratch, expected):
        for field, value, reference in zip(("hidden", "logits"),
                (scratch.hidden, scratch.logits), expected, strict=True):
            if not torch.equal(value.cpu().view(torch.uint8), reference.view(torch.uint8)):
                raise RuntimeError(f"{label}: eager {field} differs from retained-graph reference")

    def timed(label, component, expected):
        h._load(saved)
        component()
        torch.cuda.synchronize(h.device)
        compare(label, component.scratch, expected)
        times = []
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        graph = torch.cuda.CUDAGraph()
        h._load(saved)
        with torch.cuda.graph(graph):
            component()
        try:
            for _ in range(repetitions):
                h._load(saved)
                torch.cuda.synchronize(h.device)
                allocations = torch.cuda.memory_stats(h.device)["allocation.all.allocated"]
                begin.record()
                graph.replay()
                end.record()
                end.synchronize()
                if torch.cuda.memory_stats(h.device)["allocation.all.allocated"] != allocations:
                    raise RuntimeError(f"{label}: allocation occurred during isolated graph replay")
                compare(label, component.scratch, expected)
                times.append(begin.elapsed_time(end))
        finally:
            graph.reset()
        report["components"][label] = {"gpu_event_ms": times, "median_gpu_event_ms": statistics.median(times),
            "output_bytes_exact_vs_retained_graph": True}

    try:
        # The combined graph leaves the decoder's raw outputs intact. Restore
        # its original input continuation but keep the captured verification
        # metadata/candidates for an exact eager-vs-captured comparison.
        h.replay(name, slots)
        expected = [value.cpu().clone() for value in
                    (operation.decoder.scratch.hidden, operation.decoder.scratch.logits)]
        timed("target_verification_step", operation.decoder, expected)
        # A width-two retained graph exposes exactly its first learned draft.
        # Recreate those initial inputs for the selected depth's draft operator.
        h._load(saved)
        h.rows(slots, lengths=[2] * batch, sample=True)
        h.replay(first_name, slots)
        first = h.rounds[batch, 2][1].draft
        expected = [value.cpu().clone() for value in (first.scratch.hidden, first.scratch.logits)]
        h._load(saved)
        states = h.inputs.request_state_rows[:batch, 0]
        operation.draft_metadata.set(h.requests.consumed_tokens[:batch] - 1,
            torch.ones(batch, dtype=torch.int32, device=h.device), states,
            h.owner.draft_tables[:batch], h.requests.pending_token[:batch, None], h.inputs.live_rows)
        operation.draft_feedback.copy_(h.pool.last_hidden[states])
        timed("draft_step", operation.draft, expected)
    finally:
        h._load(saved)
        torch.cuda.synchronize(h.device)
    return report


def _category(prompt):
    return "tool" if prompt.get("tools") else "thinking" if prompt.get("thinking") else "code"


def _tokenize(snapshot, prompts):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    result = []
    for prompt in prompts:
        options = {"enable_thinking": bool(prompt.get("thinking", False))}
        if prompt.get("tools"):
            options["tools"] = prompt["tools"]
        ids = tokenizer.apply_chat_template(prompt["messages"], tokenize=True,
            add_generation_prompt=True, **options)
        if isinstance(ids, Mapping):
            ids = ids["input_ids"]
        if not ids or any(type(token) is not int for token in ids):
            raise ValueError("chat template did not produce native token IDs")
        result.append(ids)
    return result


def measure(prepared, root, *, checkpoint, prompts_path, depths=(2, 3), batches=(1, 4), prompt_ids=None,
            output_tokens=128, repetitions=1, seed=7035, output=None, progress=print,
            qualification=None, microtiming_repetitions=3):
    """Return exact per-request counts, preserving failures and pre-export state.

    Tau is emitted verification tokens / progressing request verification calls.
    The one token sampled by resume is reported separately and excluded from
    both numerator and denominator. Batch graph counts are never used as the
    request-round denominator, and output-budget clipping is counted directly.
    ``output``, when supplied, must be a new file; partial evidence is saved after
    every batch. Existing reports are never replaced.
    """
    if output_tokens < 2 or repetitions < 1 or any(depth not in (2, 3, 4) for depth in depths):
        raise ValueError("measurement needs depth 2/3/4, at least two output tokens and positive repetitions")
    root, output = Path(root), None if output is None else Path(output)
    if output is not None:
        with output.open("x"):
            pass
    source = Path(prompts_path)
    report = {"status": "running", "source": str(source), "prompt_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "measurement_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sampling": {"temperature": .7, "top_k": 20, "top_p": .95, "eos_disabled": True},
        "tau_definition": "verification emitted tokens / progressing request verification calls; initial resume token excluded",
        "timing_note": "This is retained-graph acceptance measurement, not native serving throughput or separate draft/target timing.",
        "runs": [], "summary": [], "errors": []}
    started, harness = time.monotonic(), None

    def save():
        if output is not None:
            output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        if not isinstance(qualification, Mapping) or qualification.get("passed") is not True:
            raise ValueError("exact numerical validation must pass before measurement; pass qualification=validate_mtp report")
        report["qualification_passed"] = True
        prompts = load_prompts(source, prompt_ids)
        tokens = _tokenize(Path(checkpoint), prompts)
        report["prompts"] = [{"id": prompt["id"], "category": _category(prompt), "prompt_tokens": len(ids), "token_ids": ids}
                             for prompt, ids in zip(prompts, tokens, strict=True)]
        if any(batch not in (1, 4) for batch in batches):
            raise ValueError("select batch capacities 1 and/or 4")
        report["required_backed_pages"] = {str(batch): max(sum(1 + (len(ids) + output_tokens + 127) // 128
            for ids in tokens[offset:offset + batch]) for offset in range(0, len(tokens), batch)) for batch in batches}
        with torch.inference_mode(), torch.cuda.device(prepared.device):
            harness = _MeasurementHarness(prepared, output_tokens)
            for depth in depths:
                for batch in batches:
                    for repetition in range(repetitions):
                        for offset in range(0, len(tokens), batch):
                            run = {"depth": depth, "batch": batch, "repetition": repetition,
                                "prompt_ids": [prompt["id"] for prompt in prompts[offset:offset + batch]],
                                "categories": [_category(prompt) for prompt in prompts[offset:offset + batch]],
                                "active_requests": len(tokens[offset:offset + batch]), "status": "running"}
                            report["runs"].append(run)
                            try:
                                harness.run(tokens[offset:offset + batch],
                                    [seed + repetition * len(tokens) + offset + slot
                                     for slot in range(len(tokens[offset:offset + batch]))],
                                    depth, run, batch_capacity=batch)
                                for index, request in enumerate(run["requests"]):
                                    request.update(prompt_id=run["prompt_ids"][index], category=run["categories"][index])
                            except Exception as error:
                                run.update(status="failed", error={"type": type(error).__name__, "message": str(error),
                                                                    "traceback": traceback.format_exc()})
                                report["errors"].append({"depth": depth, "batch": batch, "message": str(error)})
                                raise
                            save()
                            if progress:
                                progress(f"MTP tau depth={depth} batch={batch} repetition={repetition + 1} complete")
            report["microtimings"] = []
            if microtiming_repetitions > 0:
                for depth in depths:
                    for batch in batches:
                        if len(tokens) < batch:
                            continue
                        try:
                            report["microtimings"].append(_microtimings(harness, tokens[:batch],
                                [seed + slot for slot in range(batch)], depth, repetitions=microtiming_repetitions))
                        except Exception as error:
                            report["microtimings"].append({"depth": depth, "batch": batch, "status": "failed",
                                "error": str(error), "traceback": traceback.format_exc()})
                            report["errors"].append({"stage": "microtiming", "depth": depth, "batch": batch,
                                                     "message": str(error)})
                        save()
            report["graph_replays"] = harness.replays
        for depth in depths:
            for batch in batches:
                available = [request for run in report["runs"] if run["depth"] == depth and run["batch"] == batch
                            and run["status"] == "complete" for request in run["requests"]]
                for category in ("all", "code", "tool", "thinking"):
                    requests = [request for request in available if category == "all" or request["category"] == category]
                    if not requests:
                        continue
                    steps = sum(request["verification_steps"] for request in requests)
                    emitted = sum(request["emitted_from_verification"] for request in requests)
                    offered = sum(request["offered_drafts"] for request in requests)
                    accepted = sum(request["accepted_drafts"] for request in requests)
                    report["summary"].append({"depth": depth, "batch": batch, "category": category, "requests": len(requests),
                        "verification_steps": steps, "emitted_from_verification": emitted,
                        "offered_drafts": offered, "accepted_drafts": accepted,
                        "tau": emitted / steps if steps else None, "acceptance_rate": accepted / offered if offered else None})
        report["status"] = "failed" if report["errors"] else "complete"
    except Exception as error:
        report.update(status="failed", setup_error={"type": type(error).__name__, "message": str(error),
                                                    "traceback": traceback.format_exc()})
    finally:
        if harness is not None:
            try:
                with torch.inference_mode(), torch.cuda.device(prepared.device):
                    report["cleanup"] = harness.cleanup()
                    if not report["cleanup"]["passed"]:
                        report["status"] = "failed"
            except Exception as error:
                report["status"] = "failed"
                report["cleanup"] = {"passed": False, "error": str(error), "traceback": traceback.format_exc()}
        report["elapsed_seconds"] = time.monotonic() - started
        save()
    return report
