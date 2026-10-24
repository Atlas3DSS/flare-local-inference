"""Full-target consumer equality of the reviewed decode projections, using identical fused/member reductions.

Call check(prepared, root, progress=print) inside the retained preparation. This
diagnostic uses existing checkpoint weights and kernels; it never prepares a
model, launches a server, or changes a production implementation.
"""

import ctypes as C
from dataclasses import replace
import json
from pathlib import Path
import sys

import torch


_FAMILIES = {
    ("nvfp4", (17408, 17408)): "gate_up",
    ("tensor_fp8", (10240, 6144)): "qkv_z",
    ("tensor_fp8", (12288, 1024, 1024)): "q_k_v",
}


def _key(query, selected):
    return (query.kind, query.rows, query.in_features, query.padded_in_features,
            query.out_features, tuple(query.member_widths), tuple(selected))


class _MemberScratch:
    def __init__(self, query, selected, device, tc):
        rows = min(query.rows, 16)
        self.staging = torch.empty((rows, query.in_features), dtype=torch.float32, device=device)
        self.half = torch.empty_like(self.staging, dtype=torch.float16)
        self.inverse = torch.empty(rows, dtype=torch.float32, device=device)
        self.output = torch.empty(rows * max(query.member_widths), dtype=torch.float32, device=device)
        self.goldens = self.golden_workspace = None
        if query.rows == 20:
            self.goldens = torch.empty((20, query.out_features), dtype=torch.float32, device=device)
            size = max(tc.workspace_bytes(replace(query, rows=m), selected) for m in (16, 4))
            self.golden_workspace = torch.empty(size, dtype=torch.uint8, device=device)

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in
                   (self.staging, self.half, self.inverse, self.output, self.goldens, self.golden_workspace)
                   if t is not None)


def _members(tc, scratch, query, selected, source, values, scales, output, bias, build_options):
    """One unchanged producer followed by independent contiguous member outputs."""
    widths, gains = tuple(query.member_widths), tuple(query.member_gains)
    if (selected[0] != "row_tc" or selected[4] != 60 or query.rows not in (1, 16, 20)
            or len(widths) != len(gains) or sum(widths) != query.out_features
            or any(width % 32 for width in widths) or not query.column_gains
            or bias is not None or query.has_bias or not values.is_contiguous()
            or source.dtype != torch.float32 or output.dtype != torch.float32
            or output.shape != (query.rows, query.out_features)):
        raise ValueError("consumer diagnostic requires admitted bias-free path-60 fused projections")
    lib = tc.library(build_options.get("kernel_cache"))
    stream = C.c_void_p(torch.cuda.current_stream(source.device).cuda_stream)
    fp4 = query.kind == "nvfp4"
    if fp4 != (scales is not None):
        raise ValueError("fused member scale storage does not match its weight format")
    bk, warps, split, path = selected[1:]
    for row in range(0, query.rows, 16):
        live = min(16, query.rows - row)
        x = source[row:row + live]
        if not x.is_contiguous():
            scratch.staging[:live].copy_(x)
            x = scratch.staging[:live]
        tc.checked(lib.atlas_produce(tc.ptr(x), tc.ptr(scratch.half), tc.ptr(scratch.inverse),
            live, query.in_features, not fp4, query.input_scale_value, stream))
        begin = 0
        for width, gain in zip(widths, gains, strict=True):
            # Repacked storage is N-block-major, and every member boundary is a
            # complete 32-column block. These are views of the original weight bytes.
            weight_address = values.data_ptr() + begin * query.padded_in_features // (2 if fp4 else 1)
            scale_address = None if scales is None else C.c_void_p(
                scales.data_ptr() + begin * (query.padded_in_features // 16))
            member_output = scratch.output[:live * width].view(live, width)
            tc.checked(lib.atlas_project(tc.ptr(scratch.half), C.c_void_p(weight_address), scale_address,
                tc.ptr(scratch.inverse), gain, query.global_scale_kind == "reciprocal", None,
                tc.ptr(member_output), None, live, width, query.in_features, query.padded_in_features,
                split, fp4, bk, warps, path, None, stream))
            output[row:row + live, begin:begin + width].copy_(member_output)
            begin += width
    return output


def _outputs(decoder):
    hidden = decoder.scratch.hidden.detach().cpu().clone()
    logits = decoder.scratch.logits.detach().cpu().clone()
    return {"hidden": hidden, "logits": logits, "greedy_token_ids": logits.argmax(dim=-1)}


def check(prepared, root, progress=print):
    """Return JSON-compatible evidence; failures prevent a passing result."""
    root = Path(root)
    from tokenizers import Tokenizer
    from measure_mtp import _MeasurementHarness
    from validate_replay import PROMPTS, _compare
    from ignition.components import sm70_projection_tc as tc

    report = {"passed": False, "cases": [],
        "method": "Retained full-target graph vs fused eager vs independent member projections with identical BK/warps/split/path, producer and original packed weights",
        "token_policy": "Greedy argmax over every target logit row; complete hidden and logits also compared byte for byte",
        "scope": "Consumer integration at M=1, M=16 and M=20 (16+4); historical unfused split choices are not substituted",
        "historical_matrix_golden_evidence": {"included_in_this_run": False,
            "coverage": "Historical M=4/16/20 row rotations against M=16 goldens; not asserted by this consumer check"}}
    original_run, harness = tc.run, None
    try:
        with torch.inference_mode(), torch.cuda.device(prepared.device):
            harness = _MeasurementHarness(prepared, output_tokens=8)
            tokenizer = Tokenizer.from_file(prepared.server_config["assets"]["tokenizer"]["path"])
            prompts = [tokenizer.encode(text, add_special_tokens=False).ids for text in PROMPTS]
            expected_counts = {"gate_up": len(harness.owner.weights.config.layers),
                "qkv_z": len(harness.linear), "q_k_v": len(harness.full)}
            for batch, width in ((1, 1), (4, 4), (4, 5)):
                h, mtp = harness, width > 1
                h.seeds = [1205 + slot for slot in range(batch)]
                h.page_counts = [(len(ids) + h.output_tokens + h.page_tokens - 1) // h.page_tokens
                                 for ids in prompts[:batch]]
                h.initialize(prompts[:batch], mtp=mtp)
                slots = list(range(batch))
                h.rows(slots, sample=True)
                h.replay(h.name("resume", batch, mtp), slots)
                h.rows(slots, lengths=[width] * batch, sample=True)
                name = h.name("speculative", batch, True, width=width) if mtp else h.name("decode", 1, False)
                decoder = h.catalog[name].decoder
                if decoder.scratch.hidden.shape[0] != batch * width:
                    raise ValueError("target decoder does not expose every requested output row")
                saved = h._save_mutable()
                rows = h.inputs.request_state_rows[:batch, :h.width].cpu().reshape(-1).unique().tolist()
                saved.extend((f"row{row}", h.pool.backing[row], h.pool.backing[row].cpu().clone()) for row in rows)
                pages = sum(count + 1 for count in h.page_counts)
                saved.extend((key, value[:pages], value[:pages].cpu().clone()) for key, value in h.pool.pages.items())
                case = {"M": batch * width, "batch": batch, "width": width, "retained_graph": name,
                        "prompt_tokens": [len(ids) for ids in prompts[:batch]], "passed": False}
                report["cases"].append(case)
                if progress is not None:
                    progress(f"full-target consumer equality: M={batch * width}")
                # Keep the decoder metadata/candidates left by the retained
                # graph, but restore its input continuation before each decoder.
                h.replay(name, slots)
                captured = _outputs(decoder)
                queries, fused_counts = {}, dict.fromkeys(expected_counts, 0)

                def record(query, selected, source, values, scales, output, workspace, bias, gain=None, **build_options):
                    if query.member_widths:
                        family = _FAMILIES[query.kind, tuple(query.member_widths)]
                        fused_counts[family] += 1
                        queries[_key(query, selected)] = query
                    return original_run(query, selected, source, values, scales, output, workspace, bias, gain, **build_options)

                h._load(saved)
                tc.run = record
                try:
                    decoder()
                finally:
                    tc.run = original_run
                fused = _outputs(decoder)
                case["retained_vs_fused"] = _compare(captured, fused)
                case["fused_calls"] = fused_counts
                if fused_counts != expected_counts:
                    raise RuntimeError(f"did not exercise all three fused consumer families: {fused_counts}")
                scratch = {key: _MemberScratch(query, key[-1], h.device, tc) for key, query in queries.items()}
                case["diagnostic_scratch_bytes"] = sum(value.nbytes for value in scratch.values())
                case["tuples"] = [{"kind": query.kind, "K": query.in_features, "N": query.out_features,
                    "members": list(query.member_widths), "BK_warps_split_path": list(key[-1][1:])}
                    for key, query in queries.items()]
                member_counts = dict.fromkeys(expected_counts, 0)
                projection_goldens = {"projection_calls": 0, "values_compared": 0,
                    "row_golden_capacities": [16, 4], "bit_mismatches": 0}
                if batch * width == 20:
                    case["live_projection_goldens"] = projection_goldens

                def separate(query, selected, source, values, scales, output, workspace, bias, gain=None, **build_options):
                    if not query.member_widths:
                        return original_run(query, selected, source, values, scales, output, workspace, bias, gain, **build_options)
                    family = _FAMILIES[query.kind, tuple(query.member_widths)]
                    member_counts[family] += 1
                    buffers = scratch[_key(query, selected)]
                    if query.rows == 20:
                        # Independent calls use the M16 and M4 workspace layouts
                        # on the exact live inputs of this full-model projection.
                        for begin, live in ((0, 16), (16, 4)):
                            original_run(replace(query, rows=live), selected, source[begin:begin + live],
                                values, scales, buffers.goldens[begin:begin + live], buffers.golden_workspace,
                                bias, gain, **build_options)
                    result = _members(tc, buffers, query, selected, source, values, scales, output, bias, build_options)
                    if query.rows == 20:
                        equal = torch.equal(output.view(torch.int32), buffers.goldens.view(torch.int32))
                        projection_goldens["projection_calls"] += 1
                        projection_goldens["values_compared"] += output.numel()
                        if not equal:
                            projection_goldens["bit_mismatches"] += int(
                                (output.view(torch.int32) != buffers.goldens.view(torch.int32)).sum())
                            raise RuntimeError(f"{family}: live M20 rows differ from independent M16/M4 goldens")
                    return result

                h._load(saved)
                tc.run = separate
                try:
                    decoder()
                finally:
                    tc.run = original_run
                members = _outputs(decoder)
                case["fused_vs_members"] = _compare(fused, members)
                case["retained_vs_members"] = _compare(captured, members)
                case["member_calls"] = member_counts
                case["greedy_token_ids"] = {label: value["greedy_token_ids"].tolist()
                    for label, value in (("retained", captured), ("fused", fused), ("members", members))}
                case["passed"] = (member_counts == expected_counts and all(case[key]["passed"]
                    for key in ("retained_vs_fused", "fused_vs_members", "retained_vs_members")))
                if batch * width == 20:
                    case["passed"] &= projection_goldens["projection_calls"] == sum(expected_counts.values())
                torch.cuda.synchronize(h.device)
                del scratch, saved, captured, fused, members
                if not case["passed"]:
                    raise RuntimeError(f"full-target consumer equality failed at M={batch * width}")
                if progress is not None:
                    progress(f"full-target consumer equality: M={batch * width} PASS")
            report["passed"] = True
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        tc.run = original_run
        if harness is not None:
            try:
                report["cleanup"] = harness.cleanup()
                report["passed"] &= report["cleanup"]["passed"]
            except Exception as error:
                report["passed"] = False
                report["cleanup_error"] = {"type": type(error).__name__, "message": str(error)}
    (root / 'consumer-identity.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['passed']:
        raise RuntimeError('consumer identity failed; see consumer-identity.json')
    return report
