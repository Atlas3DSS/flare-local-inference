"""Isolated full MTP draft and absorb/save CUDA graphs from prepared operators.

Call measure() after strict qualification, before exporting or closing prepared.
This helper does not prepare a model, select tactics, change a source file, or
acquire a GPU. The caller runs the reviewed and legacy-dense preparations
separately. Measurements are not extracted intervals of the compiled serving
graph: the isolated graphs retain the prepared operators and eager pointwise
operations. Their outputs must match the retained compiled whole-round graph.
"""

from collections.abc import Mapping
from pathlib import Path
import statistics
import time
import traceback

import torch

from benchmark_mtp import load_prompts
from measure_mtp import _MeasurementHarness, _tokenize
from validate_replay import _compare


def _snapshot(named):
    return [(name, value, value.detach().cpu().clone()) for name, value in named
            if not any(size > 1 and stride == 0
                       for size, stride in zip(value.shape, value.stride(), strict=True))]


def _metadata(metadata):
    return [(name, value) for name, value in vars(metadata).items() if isinstance(value, torch.Tensor)]


def _outputs(h, operation, *, draft):
    rows, pages = map(int, h.pool.memory_capacity.cpu().tolist())
    component = operation.draft if draft else operation.absorb
    result = {"last_hidden": h.pool.last_hidden[:rows],
        **{name: value[:pages] for name, value in h.pool.pages.items() if name.startswith("draft.")},
        "residual": component.scratch.residual}
    if draft:
        result.update(hidden=component.scratch.hidden, logits=component.scratch.logits)
    else:
        result["feedback"] = operation.absorb_feedback
        result.update(("metadata." + name, value) for name, value in _metadata(operation.absorb_metadata))
    return result


def _poisoned_outputs(outputs, expected, slots, hidden_rows):
    """Keep unwritten bytes as canaries; poison only known written destinations."""
    saved = []
    for name, value in outputs.items():
        # B=1 token_page_table expands page_table over physical token rows.
        # Restore/poison its underlying page_table once; copying into the
        # expanded alias is invalid. The alias is still byte-compared.
        if any(size > 1 and stride == 0 for size, stride in zip(value.shape, value.stride(), strict=True)):
            continue
        baseline = expected[name].clone()
        if name.startswith("draft."):
            for slot in slots:
                if slot >= 0:
                    page, token = divmod(slot, 128)
                    baseline[page].view(torch.uint8).reshape(128, -1)[token].bitwise_xor_(1)
        elif name == "last_hidden":
            for row in hidden_rows:
                if row > 0:
                    baseline[row].view(torch.uint8).bitwise_xor_(1)
        elif name in ("hidden", "logits", "residual", "feedback"):
            baseline.fill_(float("nan"))
        elif name.startswith("metadata."):
            baseline.contiguous().view(torch.uint8).bitwise_xor_(1)
        saved.append((name, value, baseline))
    return saved


def _timed(h, label, invoke, restore, outputs, expected, *, repetitions, warmups):
    def reset():
        h._load(restore)
        torch.cuda.synchronize(h.device)

    def compare():
        result = _compare(expected, outputs)
        if not result["passed"]:
            raise RuntimeError(f"{label}: bytes differ from retained graph: {result['mismatches']}")
        return result

    reset()
    invoke()
    torch.cuda.synchronize(h.device)
    eager = compare()
    graph = torch.cuda.CUDAGraph()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    allocation_checks = []
    try:
        reset()
        with torch.cuda.graph(graph):
            invoke()
        for iteration in range(warmups + repetitions):
            reset()
            before = torch.cuda.memory_stats(h.device)["allocation.all.allocated"]
            if iteration >= warmups:
                begin.record()
            graph.replay()
            if iteration >= warmups:
                end.record()
                end.synchronize()
            else:
                torch.cuda.synchronize(h.device)
            after = torch.cuda.memory_stats(h.device)["allocation.all.allocated"]
            if after != before:
                raise RuntimeError(f"{label}: CUDA allocation during isolated graph replay: {before}->{after}")
            compare()
            allocation_checks.append(after - before)
            if iteration >= warmups:
                samples.append(begin.elapsed_time(end))
    finally:
        graph.reset()
    return {"gpu_event_ms": samples, "median_gpu_event_ms": statistics.median(samples),
        "min_gpu_event_ms": min(samples), "max_gpu_event_ms": max(samples),
        "warmups": warmups, "repetitions": repetitions,
        "output_bytes_exact_vs_retained_graph": True, "compared_fields": list(outputs),
        "compared_bytes_per_replay": eager["bytes"], "allocation_count_deltas": allocation_checks,
        "state_restore_and_output_poisoning_outside_events": True}


def _initialize(h, prompts, seeds, depth):
    batch, width = len(prompts), depth + 1
    h.seeds = list(seeds)
    h.page_counts = [(len(ids) + h.output_tokens + h.page_tokens - 1) // h.page_tokens for ids in prompts]
    if max(h.page_counts) > h.inputs.page_tables.shape[1]:
        raise ValueError("component fixture exceeds the prepared context")
    h.initialize(prompts, mtp=True)
    slots = list(range(batch))
    h.rows(slots, sample=True)
    h.replay(h.name("resume", batch, True), slots)
    h.rows(slots, lengths=[width] * batch, sample=True)
    return slots, h.rounds[batch, width][1]


def _absorb_and_save(operation):
    """The complete absorption tail of qwen38_dense.speculative.PreparedRound.

    The prefix of the round has already produced before/active/previous,
    candidates, decoder hidden rows and commit.emitted_count. State/page table
    selection is included here so the saved feedback has the same dataflow.
    Sampling, target execution and recurrent/convolution commit are excluded.
    """
    from ignition.components.state import save_hidden

    o, b, w = operation.owner, operation.batch, operation.width
    slots = o.inputs.row_slot_ids[:b].long()
    safe = slots.clamp(0, o.requests.capacity - 1)
    states = o.inputs.request_state_rows[safe, :w]
    states = torch.where(operation.active[:, None], states, 0)
    draft_pages = o.draft_tables[safe]
    counts = operation.operations[3].emitted_count
    hidden = operation.decoder.scratch.hidden.view(b, w, -1)
    feedback = torch.cat((operation.previous[:, None, :], hidden[:, :-1]), dim=1)
    operation.absorb_metadata.set((operation.before - 1).clamp_min(0), counts, states[:, 0], draft_pages,
                                  operation.candidates, o.inputs.live_rows)
    request = operation.absorb_metadata.requests
    starts = operation.absorb_metadata.query_start_loc[:-1]
    offsets = torch.arange(b * w, device=counts.device) - starts[request]
    indices = (request * w + offsets).clamp(0, b * w - 1)
    operation.absorb_feedback.copy_(feedback.reshape(b * w, -1)[indices])
    operation.absorb()
    save_hidden(hidden[operation.rows, (counts - 1).long().clamp_min(0)], o.pool.last_hidden,
                operation.metadata.state_rows, counts > 0)


def _absorption(h, prompts, seeds, depth, *, repetitions, warmups):
    slots, operation = _initialize(h, prompts, seeds, depth)
    name = h.name("speculative", len(prompts), True, width=depth + 1)
    records = h.replay(name, slots)
    # The retained whole round supplies both the accepted-count fixture and the
    # exact final cache bytes. Proposal-only KV beyond the committed prefix is
    # retained as a canary: absorption must not touch it.
    outputs = _outputs(h, operation, draft=False)
    expected = {name: value.detach().cpu().clone() for name, value in outputs.items()}
    counts = operation.operations[3].emitted_count.cpu().tolist()
    if any(count != int(records[slot, 8]) for slot, count in enumerate(counts)):
        raise RuntimeError("absorption counts differ from retained completion records")
    o = operation.owner
    sources = [("before", operation.before), ("active", operation.active), ("previous", operation.previous),
        ("candidates", operation.candidates), ("target_hidden", operation.decoder.scratch.hidden),
        ("counts", operation.operations[3].emitted_count), ("state_rows", operation.metadata.state_rows),
        ("row_slot_ids", o.inputs.row_slot_ids), ("request_state_rows", o.inputs.request_state_rows),
        ("draft_tables", o.draft_tables), ("live_rows", o.inputs.live_rows)]
    restore = _snapshot(sources)
    written = operation.absorb.attention.slots.cpu().tolist()
    rows = operation.metadata.state_rows.cpu().tolist()
    restore.extend(_poisoned_outputs(outputs, expected, written,
        [row for row, count in zip(rows, counts, strict=True) if count > 0]))
    result = _timed(h, "absorb_and_save", lambda: _absorb_and_save(operation), restore,
                    outputs, expected, repetitions=repetitions, warmups=warmups)
    result.update(reference_graph=name, physical_token_capacity=len(prompts) * (depth + 1),
        committed_pairs_per_request=counts, written_kv_slots=[slot for slot in written if slot >= 0],
        scope="State/page selection, accepted-target feedback packing, absorb metadata, complete cache-only MTP (embedding/norms/fc/K/V/rotary/cache writes), and save_hidden. Excludes target decoder, sampling/verification and GDN state commit.")
    return result


def _draft(h, prompts, seeds, depth, *, repetitions, warmups):
    slots, operation = _initialize(h, prompts, seeds, depth)
    batch = len(prompts)
    # Save the first proposal's inputs before the reference round commits.
    states = h.inputs.request_state_rows[:batch, 0].clone()
    before = h.requests.consumed_tokens[:batch].clone()
    tokens = h.requests.pending_token[:batch, None].clone()
    feedback = h.pool.last_hidden[states].clone()
    pages = h.owner.draft_tables[:batch].clone()
    h.rows(slots, lengths=[2] * batch, sample=True)
    name = h.name("speculative", batch, True, width=2)
    h.replay(name, slots)
    first = h.rounds[batch, 2][1].draft
    expected_scratch = {name: value.detach().cpu().clone() for name, value in (
        ("residual", first.scratch.residual), ("hidden", first.scratch.hidden), ("logits", first.scratch.logits))}
    operation.draft_metadata.set(before - 1, torch.ones(batch, dtype=torch.int32, device=h.device),
        states, pages, tokens, h.inputs.live_rows)
    operation.draft_feedback.copy_(feedback)
    outputs = _outputs(h, operation, draft=True)
    expected = {name: value.detach().cpu().clone() for name, value in outputs.items()}
    expected.update(expected_scratch)
    restore = _snapshot([("feedback", operation.draft_feedback),
                         *_metadata(operation.draft_metadata)])
    # The first pair in absorption uses the same true previous target hidden
    # as the first proposal. Other retained-graph writes stay as canaries.
    written = first.attention.slots.cpu().tolist()
    restore.extend(_poisoned_outputs(outputs, expected, written, []))
    result = _timed(h, "draft_step", operation.draft, restore, outputs, expected,
                    repetitions=repetitions, warmups=warmups)
    result.update(reference_graph=name, selected_depth=depth, physical_token_capacity=batch,
        written_kv_slots=[slot for slot in written if slot >= 0],
        scope="One complete learned MTP call: embedding, feedback norms/fc, attention/cache writes, MLP, final norm and vocabulary head. Excludes proposal sampling, metadata setup and feedback gather. Uses the selected depth's prepared draft operator against the width-two graph's first proposal.")
    return result


def measure(prepared, root, *, checkpoint, prompts_path, qualification, depths=(3, 4), batches=(1, 4),
            prompt_ids=("code-00", "code-01", "code-02", "code-03"), repetitions=25,
            warmups=3, output_tokens=128, seed=7035, progress=print):
    """Return JSON-compatible timing/equality evidence; leave cleanup to the harness.

    A failed setup or comparison is returned as status='failed' with traceback;
    the caller must reject that result. No files are created by this helper.
    """
    started, h = time.monotonic(), None
    report = {"status": "running", "runs": [],
        "method": "CUDA events around replay of separately captured prepared-component graphs; restoration, output poisoning and byte comparisons are outside events. These are not production-graph subgraph timings and retain eager pointwise kernel topology.",
        "comparison": "Each preparation is compared to its own retained whole-round graph, not bitwise between the SM70 dense kernels and legacy cuBLAS reductions.",
        "cache_note": "CPU restoration touches draft caches and source buffers before each replay; no L2 eviction is performed. Use identical settings for the before/after runs.",
        "repetitions": repetitions, "warmups": warmups}
    try:
        if not isinstance(qualification, Mapping) or qualification.get("passed") is not True:
            raise ValueError("strict validation must pass before component measurement")
        if (repetitions < 1 or warmups < 1 or output_tokens < 2
                or not depths or any(depth not in (3, 4) for depth in depths)
                or not batches or any(batch not in (1, 4) for batch in batches)):
            raise ValueError("select depths 3/4, batches 1/4, positive repetitions/warmups and output budget >=2")
        root = Path(root)
        prompts = load_prompts(Path(prompts_path), prompt_ids)
        tokens = _tokenize(Path(checkpoint), prompts)
        if max(batches) > len(tokens):
            raise ValueError("component measurements require enough prompts to fill every selected batch")
        report["prompts"] = [{"id": prompt["id"], "token_ids": ids, "prompt_tokens": len(ids)}
                             for prompt, ids in zip(prompts, tokens, strict=True)]
        with torch.inference_mode(), torch.cuda.device(prepared.device):
            h = _MeasurementHarness(prepared, output_tokens)
            for depth in depths:
                for batch in batches:
                    run = {"depth": depth, "batch": batch, "status": "running",
                        "prompt_ids": [prompt["id"] for prompt in prompts[:batch]],
                        "seeds": [seed + slot for slot in range(batch)], "components": {}}
                    report["runs"].append(run)
                    for name, invoke in (("absorb_and_save", _absorption), ("draft_step", _draft)):
                        run["components"][name] = invoke(h, tokens[:batch], run["seeds"], depth,
                            repetitions=repetitions, warmups=warmups)
                        if progress:
                            progress(f"MTP component B={batch} depth={depth} {name}: "
                                     f"{run['components'][name]['median_gpu_event_ms']:.6f} ms, exact retained-graph bytes")
                    run["status"] = "complete"
            report["graph_replays"] = h.replays
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error),
                                               "traceback": traceback.format_exc()})
        if report["runs"] and report["runs"][-1]["status"] == "running":
            report["runs"][-1]["status"] = "failed"
    finally:
        if h is not None:
            try:
                with torch.inference_mode(), torch.cuda.device(prepared.device):
                    report["cleanup"] = h.cleanup()
                if not report["cleanup"]["passed"]:
                    report["status"] = "failed"
            except Exception as error:
                report.update(status="failed", cleanup={"passed": False, "error": str(error)})
        report["elapsed_seconds"] = time.monotonic() - started
    return report
