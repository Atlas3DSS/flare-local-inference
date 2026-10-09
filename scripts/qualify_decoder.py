"""Capture and replay the complete pinned Qwen text decoder through Rust."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _prepare_decoder_graphs(session, workspace, build, capture, *, prefill_capacity, tune, progress):
    pending = []
    for mode, capacity in (("prefill", prefill_capacity), ("decode", 1)):
        model = build(mode, capacity)
        pending.append((mode, capacity, model))
    if not tune:
        workspace.finalize()
        workspace.freeze()
        for mode, capacity, model in pending:
            capture(mode, capacity, model)
        return None
    session.register_scratch(workspace)
    progress("tuning complete prefill and decode kernel candidates")
    last_report = 0.0

    def tuning_progress(status):
        nonlocal last_report
        now = time.monotonic()
        if status.done or now - last_report >= 10:
            progress(f"kernel preparation: {status.phase}; {status.completed_requests}/{status.total_requests} requests, "
                     f"{status.measured_candidates} candidates measured, {status.cache_hits} cache hits")
            last_report = now

    report = session.tune(progress=tuning_progress)
    progress(f"selected prepared kernels: {report['selection_counts']}")
    for mode, capacity, model in pending:
        capture(mode, capacity, model)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/validation/decoder-execution.json")
    parser.add_argument("--tensor-output", type=Path, default=Path("/tmp/sparkgap-decoder-execution.pt"))
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--prefill-capacity", type=int, choices=(1, 16, 64, 128, 256, 512), default=512)
    parser.add_argument("--layer-trace", action="store_true", help="retain all layer outputs and layer-0 operator intermediates")
    parser.add_argument("--tune", action="store_true",
                        help="tune both decoders together and refresh selected bindings before capture")
    args = parser.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("assign CUDA_VISIBLE_DEVICES explicitly")
    if args.steps < 1:
        parser.error("steps must be positive")
    if args.extension:
        spec = importlib.util.spec_from_file_location("sparkgap_native", args.extension.resolve())
        module = importlib.util.module_from_spec(spec)
        sys.modules["sparkgap_native"] = module
        spec.loader.exec_module(module)

    import torch
    from b12x.preparation import PreparationSession
    from transformers import AutoTokenizer
    from sparkgap.preparation import inspect_qwen38_checkpoint
    from sparkgap.preparation.execution import GraphSignature, PreparedExecution
    from sparkgap.preparation.qwen38 import MODEL_ID, REVISION
    from sparkgap.recipes.qwen38.attention import QSAState, RotaryTables
    from sparkgap.recipes.qwen38.decoder import ModelWeights, DecoderScratch, PreparedDecoder
    from sparkgap.preparation.scratch import ScratchArena
    from sparkgap.recipes.qwen38.decoder_state import DecoderInputs, PreparedDecoderState
    from sparkgap.recipes.qwen38.ple import MappedPLETable
    from sparkgap.recipes.qwen38.sampling import DeviceRequestState, RequestAdmission
    from sparkgap.recipes.qwen38.state import SNAPSHOT_BASE, HybridStateLayout, HybridStatePool

    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "sparkgap/recipes/qwen38").glob("*.py"))}
    preparation_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "sparkgap/preparation").glob("*.py"))}

    start = time.monotonic()
    def progress(message):
        print(f"{time.monotonic() - start:.1f}s {message}", flush=True)

    snapshot = args.snapshot or (Path.home() / ".cache/huggingface/hub" /
        f"models--{MODEL_ID.replace('/', '--')}" / "snapshots" / REVISION)
    manifest = inspect_qwen38_checkpoint(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    prompt = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not 0 < len(prompt) <= args.prefill_capacity:
        parser.error("prompt must fit the selected nonempty prefill capacity")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    weights = ModelWeights.load(manifest, device, progress=progress)
    progress(f"loaded {weights.allocated_bytes} GPU weight bytes")
    table = MappedPLETable.load(manifest, device)
    progress("loaded complete mapped-host PLE table")
    context, pages = 262144, 4096
    state_pool = HybridStatePool.allocate(SNAPSHOT_BASE + HybridStateLayout.build().snapshot_capacity(), device)
    qsa_states = {layer: QSAState.allocate(pages=pages, slots=5, device=device) for layer in range(3, 48, 4)}
    requests = DeviceRequestState.allocate(device,
        consumed_tokens=state_pool.consumed_tokens[1:5, 0], ple_history=state_pool.ple_history[1:5])
    inputs = DecoderInputs.allocate(device, context=context)
    scratch = DecoderScratch.allocate(args.prefill_capacity, device)
    workspace = ScratchArena(device)
    rotary = RotaryTables.allocate(weights.tree.config.text_config, context, device)
    completion = torch.zeros((2, 4, 8), device=device, dtype=torch.int64)
    frame = torch.zeros(1, device=device, dtype=torch.int32)
    sequence = torch.zeros(1, device=device, dtype=torch.int64)
    stream = torch.cuda.Stream(device=device)
    graph_pool = torch.cuda.graph_pool_handle()
    graphs, signatures, prepared = {}, {}, []
    traces = {}
    stage_traces = {}
    buffers = {**inputs.buffers(), "frame_index": frame, "submission_sequence": sequence,
               "completion_records": completion}
    if args.tune:
        from sparkgap.preparation.qwen38 import CONFIG_SHA256, HEADERS_SHA256
        from sparkgap.preparation.tuning import RecordedPreparationSession
        preparation = RecordedPreparationSession(device=device, checkpoint=REVISION,
            namespace={"config": CONFIG_SHA256, "headers": HEADERS_SHA256})
    else:
        preparation = PreparationSession(device=device, autotune=False, compile_workers=0)
    with preparation as session:
        def build(mode, capacity):
            if args.layer_trace:
                traces[mode] = torch.empty((48, capacity, 10240), device=device, dtype=torch.bfloat16)
                stage_traces[mode] = {key: torch.empty((capacity, width), device=device, dtype=dtype)
                    for key, width, dtype in (
                        *((key, 2560, torch.bfloat16) for key in ("embedding", "attn_mix", "attn_output",
                            "mlp_mix", "mlp_output", "moe_shared", "moe_routed")),
                        *((key, 10240, torch.bfloat16) for key in ("residual_in", "attn_residual", "mlp_residual",
                            "gdn_projected_qkv", "gdn_convolved_qkv")),
                        *((f"{branch}_{field}", width, torch.bfloat16)
                          for branch in ("attn", "mlp") for field, width in (
                              ("normalized", 10240), ("projected_down", 320), ("bottleneck", 320),
                              ("gate_logits", 10240), ("injection_logits", 4))),
                        ("gdn_recurrence", 6144, torch.bfloat16), ("gdn_gated", 6144, torch.bfloat16),
                        ("moe_router_logits", 512, torch.bfloat16), ("moe_ids", 10, torch.int32),
                        ("moe_weights", 10, torch.float32))}
            metadata = PreparedDecoderState(inputs, requests, state_pool, tokens=capacity, rows=1, mode=mode)
            model = PreparedDecoder(weights, scratch.rows(capacity, 1), metadata, qsa_states, rotary,
                table, workspace, completion, frame, sequence, session=session, name=mode,
                progress=progress, layer_outputs=traces.get(mode),
                stage_outputs=stage_traces.get(mode))
            prepared.append(model)
            return model

        def capture(mode, capacity, model):
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                model()
            stream.synchronize()
            progress(f"warmed complete {mode}")
            graph = torch.cuda.CUDAGraph()
            capture_context = session.capture() if args.tune else nullcontext()
            with capture_context, torch.cuda.graph(graph, pool=graph_pool, stream=stream):
                model()
            graphs[mode] = graph
            signatures[mode] = GraphSignature(mode, 1, capacity, tuple(inputs.buffers()))
            progress(f"captured complete {mode}")
        tuning = _prepare_decoder_graphs(session, workspace, build, capture,
            prefill_capacity=args.prefill_capacity, tune=args.tune, progress=progress)
        session.freeze()
        requests.admit(0, RequestAdmission(1, 1234, max_new_tokens=args.steps, temperature=0), prompt)
        inputs.row_slot_ids[0] = 0
        inputs.row_generations[0] = 1
        inputs.live_rows.fill_(1)
        inputs.query_lengths[0] = len(prompt)
        inputs.token_ids[:len(prompt)].copy_(torch.tensor(prompt, device=device))
        inputs.prefill_positions[:len(prompt)].copy_(torch.arange(len(prompt), device=device)[:, None].expand(-1, 3))
        inputs.page_tables[0].copy_(torch.arange(context // 64, dtype=torch.int32, device=device))
        inputs.sample_enabled[0] = 1
        owner = PreparedExecution(device=0, graphs=graphs, signatures=signatures, buffers=buffers,
            resources=(weights, table, state_pool, qsa_states, requests, inputs, scratch, workspace,
                       rotary, prepared, session, graph_pool, stream))
        executor = owner.start_executor()
        progress("handed both complete graphs to native executor")
        tokens, records, logits, hidden, layer_outputs, stage_outputs = [], [], [], [], [], []
        diagnostic_allocations = 0
        try:
            for step in range(args.steps):
                allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                fence = executor.submit(0 if step == 0 else 1, [])
                deadline = time.monotonic() + 120
                result = executor.poll()
                while result is None:
                    if time.monotonic() > deadline:
                        raise TimeoutError("native decoder submission did not complete in 120 seconds")
                    time.sleep(.001)
                    result = executor.poll()
                if result.fence != fence:
                    raise RuntimeError("native decoder fence mismatch")
                record = result.records[0]
                records.append(record)
                if record[5] == 3:
                    raise RuntimeError(f"device decoder error {record[6]}")
                if record[4] < 0:
                    raise RuntimeError("decoder did not produce a token")
                if torch.cuda.memory_stats()["allocation.all.allocated"] != allocations:
                    raise RuntimeError("native graph replay allocated CUDA storage")
                tokens.append(record[4])
                logits.append(scratch.logits[0].cpu().clone())
                hidden.append(scratch.sample_hidden[0].cpu().clone())
                if args.layer_trace:
                    layer_outputs.append(traces["prefill" if step == 0 else "decode"][:, :len(prompt) if step == 0 else 1].cpu().clone())
                    stage_outputs.append({key: value[:len(prompt) if step == 0 else 1].cpu().clone()
                        for key, value in stage_traces["prefill" if step == 0 else "decode"].items()})
                    diagnostic_allocations += torch.cuda.memory_stats()["allocation.all.allocated"] - allocations
                progress(f"native step {step}: token={record[4]} text={tokenizer.decode(tokens)!r}")
                if record[5] == 2:
                    break
        finally:
            executor.close()
        torch.save({"prompt": prompt, "tokens": tokens, "logits": logits, "hidden": hidden,
                    "layer_outputs": layer_outputs, "stage_outputs": stage_outputs}, args.tensor_output)
        result = {"component": "complete Qwen text decoder native graph execution",
            "checkpoint": {"repository": MODEL_ID, "revision": REVISION},
            "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"], "prefill_capacity": args.prefill_capacity,
            "context_capacity": context, "pages": pages, "hybrid_state_slots": state_pool.backing.shape[0],
            "weight_bytes": weights.allocated_bytes, "gpu_allocated_bytes": torch.cuda.memory_allocated(),
            "gpu_peak_bytes": torch.cuda.max_memory_allocated(), "mapped_host_table_bytes": 28800138240,
            "prompt": args.prompt, "prompt_tokens": prompt, "generated_tokens": tokens,
            "generated_text": tokenizer.decode(tokens), "records": records,
            "source_sha256": source_hashes, "diagnostic_readback_allocations": diagnostic_allocations,
            "source_root": str(ROOT), "script_path": str(Path(__file__).resolve()),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "preparation_source_sha256": preparation_hashes, "tuning": tuning,
            "replay_allocations": 0, "all_logits_finite": all(bool(torch.isfinite(x).all()) for x in logits),
            "numerical_reference": "not yet compared", "elapsed_seconds": time.monotonic() - start,
            "tensor_output": str(args.tensor_output),
            "tensor_sha256": hashlib.sha256(args.tensor_output.read_bytes()).hexdigest()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        for graph in graphs.values():
            graph.reset()
        table.close()


if __name__ == "__main__":
    main()
