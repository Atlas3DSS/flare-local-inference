"""Measure a prepared Qwen candidate on the reference's exact token contexts.

Call ``validate_candidate(prepared, reference_path, output_dir)`` before export,
inside the owner's GPU lease. Only the actual captured catalog graphs run;
request admission, batch validation, state updates and completion stay enabled.
The retained decoder owner exposes logits for this qualification diagnostic.
Initial mutable staging/request bytes are restored after the run, and the pool
is cleared through its preparation cleanup API before returning to export.
"""

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import time


MODEL_ID = "nvidia/Qwen3.8-27B-NVFP4"
REVISION = "482ca0f3832238542f8f5295dde86b5f22711d80"


def _sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def distribution_metrics(reference, candidate, *, chunk_rows=8):
    """Return per-position KL(reference || candidate) in FP64, without clipping."""
    import torch

    if (reference.ndim != 2 or candidate.shape != reference.shape or min(reference.shape) <= 0
            or not reference.is_floating_point() or not candidate.is_floating_point()
            or reference.device.type != "cpu" or candidate.device.type != "cpu"
            or type(chunk_rows) is not int or chunk_rows < 1):
        raise ValueError("logit comparisons require matching nonempty CPU floating matrices")
    if not bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all()):
        raise ValueError("cannot qualify nonfinite logits")
    divergences = []
    for start in range(0, reference.shape[0], chunk_rows):
        p = reference[start:start + chunk_rows].double().log_softmax(-1)
        q = candidate[start:start + chunk_rows].double().log_softmax(-1)
        divergences.append((p.exp() * (p - q)).sum(-1))
    kld = torch.cat(divergences)
    reference_top1, candidate_top1 = reference.argmax(-1), candidate.argmax(-1)
    agreement = reference_top1 == candidate_top1
    return {
        "positions": int(kld.numel()), "vocabulary": int(reference.shape[1]),
        "kld_mean": float(kld.mean()), "kld_p99": float(torch.quantile(kld, .99)),
        "kld_max": float(kld.max()), "top1_agreement": float(agreement.double().mean()),
        "top1_disagreements": int((~agreement).sum()),
    }, {"kld": kld, "reference_top1": reference_top1, "candidate_top1": candidate_top1}


def _reference(path):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = Path(path)
    with safe_open(path, framework="pt", device="cpu") as source:
        metadata = source.metadata() or {}
    if metadata.get("model_id") != MODEL_ID or metadata.get("revision") != REVISION:
        raise ValueError("reference must identify the exact pinned Qwen checkpoint")
    tensors = load_file(path, device="cpu")
    prefixes = sorted({key.split(".")[0] for key in tensors}, key=lambda key: int(key.removeprefix("prompt_")))
    if prefixes != [f"prompt_{index}" for index in range(len(prefixes))] or not prefixes:
        raise ValueError("reference prompt indices must be nonempty and contiguous")
    vocabulary, prompts = None, []
    for prefix in prefixes:
        names = {f"{prefix}.{field}" for field in ("prompt_ids", "generated_ids", "prefill_logits", "decode_logits")}
        if {key for key in tensors if key.startswith(prefix + ".")} != names:
            raise ValueError(f"{prefix}: unexpected or missing reference fields")
        ids, generated, prefill, decode = (tensors[f"{prefix}.{name}"]
            for name in ("prompt_ids", "generated_ids", "prefill_logits", "decode_logits"))
        if (ids.dtype != torch.int64 or generated.dtype != torch.int64 or ids.ndim != 1
                or generated.ndim != 1 or not ids.numel() or not generated.numel()
                or prefill.ndim != 2 or prefill.shape[0] != ids.numel()
                or decode.shape != (generated.numel() - 1, prefill.shape[1])
                or prefill.dtype != torch.float32 or decode.dtype != torch.float32):
            raise ValueError(f"{prefix}: reference logits and token contexts disagree")
        width = prefill.shape[1]
        if width <= 0 or (vocabulary is not None and vocabulary != width):
            raise ValueError("reference vocabularies must agree")
        vocabulary = width
        if any(bool(((value < 0) | (value >= width)).any()) for value in (ids, generated)):
            raise ValueError(f"{prefix}: reference token outside the vocabulary")
        if not bool(torch.isfinite(prefill).all() and torch.isfinite(decode).all()):
            raise ValueError(f"{prefix}: nonfinite reference logits")
        if not torch.equal(torch.cat((prefill[-1:], decode)).argmax(-1), generated):
            raise ValueError(f"{prefix}: reference greedy IDs disagree with their saved logits")
        prompts.append((prefix, ids, generated, prefill, decode))
    return prompts, metadata


def require_capability(actual, expected=(7, 0)):
    """Opt in to a specific supported architecture; existing SM70 callers stay strict."""
    if (not isinstance(expected, tuple) or len(expected) != 2
            or any(type(value) is not int for value in expected)
            or expected not in ((7, 0), (12, 0))):
        raise ValueError("expected capability must explicitly be (7, 0) or (12, 0)")
    if tuple(actual) != expected:
        raise ValueError(f"candidate qualification requires actual SM{expected[0]}{expected[1]} hardware")


def _ordinary_graph_names(signatures, *, mode):
    """Select only non-speculative variants, including in an MTP-capable catalog."""
    names = {}
    kinds = ("init", "prefill", "snapshot", "restore", "resume")
    if mode != "sequential":
        kinds += ("decode",)
    for kind in kinds:
        candidates = [(signature.token_capacity, signature.batch_capacity, name)
            for name, signature in signatures.items()
            if signature.kind == kind and signature.batch_capacity == 1
            and signature.speculation_enabled is not True]
        if not candidates:
            raise ValueError(f"prepared catalog lacks a one-request ordinary {kind} graph")
        if kind == "resume" and len(candidates) != 1:
            raise ValueError("qualification requires one one-request captured resume graph")
        names[kind] = (max(candidates) if kind == "prefill" and mode == "chunked" else min(candidates))[-1]
    return names


def _owners(prepared, *, mode="sequential", expected_capability=(7, 0)):
    if mode not in ("sequential", "ordinary", "chunked"):
        raise ValueError("unknown qualification execution mode")
    import torch
    from ignition.components.paged_generation import PreparedGeneration
    from ignition.components.paged_maintenance import PreparedMaintenance
    from ignition.preparation.model import PreparedModel

    if not isinstance(prepared, PreparedModel):
        raise TypeError("qualification requires the production PreparedModel")
    if prepared.closed or prepared.consumed:
        raise ValueError("qualification requires an open model before native ownership handoff")
    require_capability(torch.cuda.get_device_capability(prepared.device), expected_capability)
    names = _ordinary_graph_names(prepared.signatures, mode=mode)
    for kind in ("init", "prefill", "snapshot", "restore"):
        if not isinstance(prepared.graphs[names[kind]], torch.cuda.CUDAGraph):
            raise TypeError("qualification must replay actual retained CUDA graphs")
    catalogs = [resource for resource in prepared.resources if isinstance(resource, Mapping)
        and isinstance(resource.get(names["prefill"]), PreparedGeneration)
        and all(isinstance(resource.get(names[kind]), PreparedMaintenance)
                for kind in ("init", "snapshot", "restore"))]
    if len(catalogs) != 1:
        raise ValueError("expected one retained production graph-owner catalog for logits diagnostics")
    generation = catalogs[0][names["prefill"]]
    maintenances = {kind: catalogs[0][names[kind]] for kind in ("init", "snapshot", "restore")}
    maintenance = maintenances["init"]
    batch, decoder = generation.batch, generation.decoder
    if (batch.mode != "prefill" or batch.rows != 1 or decoder.pool is not maintenance.pool
            or batch.requests is not maintenance.requests or batch.inputs is not maintenance.inputs):
        raise ValueError("captured generation and maintenance owners do not share their request state")
    for owner in maintenances.values():
        if (owner.pool is not decoder.pool or owner.requests is not batch.requests
                or owner.inputs is not batch.inputs or owner.commands is not maintenance.commands):
            raise ValueError("captured maintenance graphs do not share their production state")
    for name, tensor in batch.inputs.buffers().items():
        if prepared.buffers[name].data_ptr() != tensor.data_ptr():
            raise ValueError(f"retained owner does not use the published {name} buffer")
    admission = maintenance.state.admission
    if admission is None:
        raise ValueError("init graph must include real request admission")
    for name in admission.buffers.__dataclass_fields__:
        if prepared.buffers[name].data_ptr() != getattr(admission.buffers, name).data_ptr():
            raise ValueError(f"retained admission does not use the published {name} buffer")
    if maintenances["restore"].state.admission.buffers is not admission.buffers:
        raise ValueError("restore graph does not use the shared admission buffers")
    # Exercise the existing graph/device/storage registration checks as well.
    prepared.manifest(all_variants=True)
    if mode != "sequential":
        decode = catalogs[0][names["decode"]]
        if (not isinstance(decode, PreparedGeneration) or decode.batch.mode != "decode"
                or decode.batch.requests is not batch.requests or decode.batch.inputs is not batch.inputs
                or decode.decoder.pool is not decoder.pool
                or not isinstance(prepared.graphs[names["decode"]], torch.cuda.CUDAGraph)):
            raise ValueError("ordinary decode must share production state and retain an actual graph")
    from ignition.components.paged_resume import PreparedResume
    resume = catalogs[0][names["resume"]]
    if (not isinstance(resume, PreparedResume) or resume.state.requests is not batch.requests
            or not isinstance(prepared.graphs[names["resume"]], torch.cuda.CUDAGraph)):
        raise ValueError("resume must share the actual captured production request state")
    return names, generation, maintenances


def _pool_storage_addresses(pool):
    """Identify actual model storage, including independently wrapped row aliases."""
    return {tensor.untyped_storage().data_ptr()
            for tensor in (pool.backing, pool.reserved_backing, *pool.pages.values())}


def _assert_pool_cleared(pool, *, chunk_bytes=8 << 20):
    """Check only physical preparation rows/pages, never reserved virtual tails."""
    import torch

    if type(chunk_bytes) is not int or chunk_bytes < 1:
        raise ValueError("pool validation chunk size must be positive")
    rows, pages = pool.memory_capacity.cpu().tolist()
    if not 0 < rows <= pool.backing.shape[0] or any(
            not 0 < pages <= tensor.shape[0] for tensor in pool.pages.values()):
        raise ValueError("pool preparation capacities exceed their registered extents")
    values = [("continuations", pool.backing[:rows])]
    values.extend((name, tensor[:pages]) for name, tensor in pool.pages.items())
    for name, tensor in values:
        if tensor.dtype != torch.uint8:
            raise TypeError(f"pool backing {name} must expose raw bytes")
        if not tensor.is_contiguous():
            raise ValueError(f"pool backing {name} must be contiguous; do not materialize reserved storage")
        # CUDA count_nonzero may promote its reduction input. Never submit the
        # whole GiB-scale pool and accidentally allocate a pool-sized temporary.
        # view/narrow alias only the physically backed prefix established above.
        flat = tensor.view(-1)
        for start in range(0, flat.numel(), chunk_bytes):
            if bool(torch.count_nonzero(flat.narrow(0, start, min(chunk_bytes, flat.numel() - start)))):
                raise RuntimeError(f"qualification requires exact zero preparation pool bytes: {name}")


def _snapshot_mutable(prepared, generation, maintenances):
    """Copy initial bytes, excluding the model pool cleared by its own API."""
    import torch

    maintenance = maintenances["init"]
    pool = generation.decoder.pool
    pool_storage = _pool_storage_addresses(pool)
    values = list(prepared.buffers.items())
    for prefix, owner in (("request", generation.batch.requests), ("command", maintenance.commands),
                          ("admission", maintenance.state.admission.buffers)):
        values.extend((f"{prefix}.{name}", getattr(owner, name)) for name in owner.__dataclass_fields__)
    # These validation outputs also persist in the retained maintenance graphs.
    for kind, owner in maintenances.items():
        values.extend((f"{kind}.{name}", getattr(owner.state, name))
                      for name in ("validation_input", "validation_sum", "admission_errors"))
    seen, snapshots = set(), []
    torch.cuda.synchronize(prepared.device)
    # The recipe clears physical state/pages before returning PreparedModel.
    # Its published cache planes also describe unmapped virtual tails; copying
    # those tensors in full is invalid. Their exact initial bytes are zero and
    # their owner restores those bytes through clear_for_preparation().
    _assert_pool_cleared(pool)
    for name, tensor in values:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"mutable field {name} is not a tensor")
        if tensor.untyped_storage().data_ptr() in pool_storage:
            continue
        key = (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
        if key not in seen:
            seen.add(key)
            try:
                initial = tensor.detach().to("cpu", copy=True)
            except Exception as error:
                raise RuntimeError(
                    f"cannot snapshot mutable buffer {name}: device={tensor.device}, "
                    f"dtype={tensor.dtype}, shape={tuple(tensor.shape)}, "
                    f"stride={tuple(tensor.stride())}, address={tensor.data_ptr():#x}, "
                    f"bytes={tensor.numel() * tensor.element_size()}: {error}"
                ) from error
            snapshots.append((name, tensor, initial))
    return snapshots


def _restore(prepared, pool, snapshots):
    import torch

    torch.cuda.synchronize(prepared.device)
    pool.clear_for_preparation()
    for _, tensor, initial in snapshots:
        tensor.copy_(initial)
    torch.cuda.synchronize(prepared.device)
    for name, tensor, initial in snapshots:
        actual = tensor.detach().cpu().contiguous().view(torch.uint8)
        expected = initial.contiguous().view(torch.uint8)
        if not torch.equal(actual, expected):
            raise RuntimeError(f"qualification cleanup did not restore exact initial bytes: {name}")
    _assert_pool_cleared(pool)


def validate_candidate(prepared, reference_path, output_dir, *, progress=print,
                       mode="sequential", artifact_stem="candidate", expected_capability=(7, 0)):
    """Replay published inputs, save measurements, and restore export startup state.

    This reports measured metrics, never an invented numerical pass threshold.
    Independent greedy generation is explicitly separate from teacher forcing.
    Any graph rejection, nonfinite logit or restoration failure raises after
    writing the diagnostic report; it cannot be reported as a successful run.
    """
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from ignition.components.sampling import INACTIVE, RUNNING, RequestAdmission

    output_dir, reference_path = Path(output_dir), Path(reference_path)
    output_dir.mkdir(exist_ok=True)
    if Path(artifact_stem).name != artifact_stem:
        raise ValueError("artifact_stem must be a filename stem")
    artifact, report_path = output_dir / f"{artifact_stem}.safetensors", output_dir / f"{artifact_stem}.json"
    if artifact.exists() or report_path.exists():
        raise FileExistsError("qualification artifacts must not overwrite prior evidence")
    started = time.monotonic()
    report = {"status": "started", "model_id": MODEL_ID, "revision": REVISION,
        "reference": str(reference_path), "artifact": str(artifact), "prompts": [],
        "execution": "actual captured init/snapshot/restore and sequential one-token prefill graphs; sample_enabled=0",
        "ordinary_decode_graph_tested": mode != "sequential", "mode": mode, "native_runtime_tested": False,
        "independent_greedy": {"status": "not_run", "stop_on_eos": False,
            "sampling": "host argmax on captured-graph full logits; candidate tokens feed the next replay",
            "branch": "captured snapshot at the prompt endpoint, then captured restore with a new epoch"},
        "numerical_pass": None, "numerical_bounds": "not supplied; metrics are measurements, not qualification approval",
        "state_restored": False}
    tensors, collected, failures, snapshots, pool = {}, {}, [], None, None
    try:
        prompts, reference_metadata = _reference(reference_path)
        report.update(reference_sha256=_sha256(reference_path), reference_metadata=reference_metadata)
        reference_report = reference_path.with_suffix(".json")
        if reference_report.exists():
            provenance = json.loads(reference_report.read_text())
            if provenance.get("model_id") != MODEL_ID or provenance.get("revision") != REVISION:
                raise ValueError("reference JSON provenance does not identify the pinned checkpoint")
            for index, (_, ids, generated, _, _) in enumerate(prompts):
                entry = provenance["prompts"][index]
                if entry["prompt_ids"] != ids.tolist() or entry["generated_ids"] != generated.tolist():
                    raise ValueError("reference JSON token contexts disagree with the safetensors artifact")
            report.update(reference_report_sha256=_sha256(reference_report),
                reference_config_sha256=provenance.get("config_sha256"),
                reference_index_sha256=provenance.get("index_sha256"),
                reference_hf_source_sha256=provenance.get("hf_model_source_sha256"))
        names, generation, maintenances = _owners(prepared, mode=mode, expected_capability=expected_capability)
        catalog = next(resource for resource in prepared.resources
                       if isinstance(resource, Mapping) and resource.get(names["prefill"]) is generation)
        report["execution"] = ("actual captured init/snapshot/restore; " +
            {"sequential": "sequential one-token prefill", "ordinary": "first-token prefill then ordinary decode",
             "chunked": "chunked prompt prefill then ordinary decode"}[mode] + "; sample_enabled=0")
        maintenance = maintenances["init"]
        batch, decoder = generation.batch, generation.decoder
        requests, inputs, pool = batch.requests, batch.inputs, decoder.pool
        admission, commands = maintenance.state.admission.buffers, maintenance.commands
        tokenizer = Tokenizer.from_file(prepared.server_config["assets"]["tokenizer"]["path"])
        if reference_report.exists():
            checkpoint = Path(prepared.server_config["assets"]["tokenizer"]["path"]).parent
            for filename, key in (("config.json", "config_sha256"), ("model.safetensors.index.json", "index_sha256")):
                if _sha256(checkpoint / filename) != provenance[key]:
                    raise ValueError(f"prepared checkpoint {filename} differs from the reference provenance")
        if (bool((requests.status != INACTIVE).any()) or bool((requests.generation != 0).any())
                or bool((requests.consumed_tokens != 0).any())):
            raise ValueError("qualification requires fresh startup request state")
        logits = decoder.scratch.logits
        if logits.shape != (1, prompts[0][3].shape[1]):
            raise ValueError("captured prefill logits do not expose the reference's full vocabulary")
        max_context = max(ids.numel() + generated.numel() - 1 for _, ids, generated, _, _ in prompts)
        page_tokens = int(prepared.server_config["page_tokens"])
        needed_pages = (max_context + page_tokens - 1) // page_tokens
        capacity = prepared.buffers["memory_capacity"].cpu().tolist()
        if capacity[0] < 3 or capacity[1] < needed_pages or inputs.page_tables.shape[1] < needed_pages:
            raise ValueError("prepared physical state/page capacities do not cover reference contexts")
        report.update(graphs=names, torch=torch.__version__, device=str(logits.device),
            compute_capability=list(torch.cuda.get_device_capability(prepared.device)),
            device_name=torch.cuda.get_device_name(prepared.device),
            prepared_identity=prepared.server_config.get("identity"),
            positions_expected=sum(generated.numel() if mode == "chunked" else ids.numel() + generated.numel() - 1
                                   for _, ids, generated, _, _ in prompts))
        with torch.inference_mode(), torch.cuda.device(prepared.device):
            snapshots = _snapshot_mutable(prepared, generation, maintenances)
            inactive_initial = pool.backing[0].cpu().clone()
            buffers = prepared.buffers
            sequence = int(buffers["submission_sequence"].item())
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            report["gpu_graph_ms"] = {}
            report["gpu_timing_method"] = "CUDA events around complete retained graph replay; no node tracing"

            def replay(kind, epoch, consumed, *, sampled=False):
                nonlocal sequence
                sequence += 1
                frame = sequence % 2
                buffers["frame_index"].fill_(frame)
                buffers["submission_sequence"].fill_(sequence)
                events[0].record()
                prepared.graphs[names[kind]].replay()
                events[1].record()
                torch.cuda.synchronize(prepared.device)
                report["gpu_graph_ms"].setdefault(kind, []).append(events[0].elapsed_time(events[1]))
                record = buffers["completion_records"][frame, 0].cpu().tolist()
                token = int(record[4]) if sampled else -1
                if sampled and not 0 <= token < logits.shape[1]:
                    raise RuntimeError("resume did not return a valid native sampled token")
                expected = [epoch, sequence, consumed, int(sampled), token, RUNNING, 0]
                if record[:7] != expected:
                    raise RuntimeError(f"{names[kind]} completion rejected: {record}; expected prefix {expected}")
                observed = [int(getattr(requests, name)[0].item()) for name in
                    ("generation", "consumed_tokens", "generated_tokens", "pending_token", "status", "error_code", "rng_counter")]
                if observed != [epoch, consumed, int(sampled), token, RUNNING, 0, int(sampled)]:
                    raise RuntimeError(f"captured request state disagrees with unsampled completion: {observed}")
                if kind in maintenances:
                    owner = maintenances[kind]
                    if bool(commands.errors.any()) or bool(owner.state.admission_errors.any()):
                        raise RuntimeError(f"captured {kind} state/admission validation reported an error")

            def stage_admission(epoch, token_ids, generated_count):
                admission.stage(0, 0, RequestAdmission(generation=epoch, seed=0,
                    max_new_tokens=generated_count, temperature=1., top_k=1, top_p=1.,
                    repetition_penalty=1., eos_token_ids=()), token_ids)
                admission.admission_count.fill_(1)

            def stage_transfer(source, destination, consumed, *, reset=False):
                commands.request_slots.fill_(-1)
                commands.source_slots.fill_(-1)
                commands.destination_slots.fill_(-1)
                commands.operation.zero_()
                commands.expected_consumed.fill_(-1)
                commands.request_slots[0] = 0
                commands.source_slots[0] = source
                commands.destination_slots[0] = destination
                commands.operation[0] = int(reset)
                commands.expected_consumed[0] = consumed
                commands.num_ops.fill_(1)

            def read_logits(prefix, stage, position):
                current = active_logits[0].detach().cpu().float()
                if not bool(torch.isfinite(current).all()):
                    raise RuntimeError(f"{prefix}: nonfinite candidate logits during {stage} at position {position}")
                return current

            active_logits = logits
            all_reference, all_candidate = [], []
            for index, (prefix, ids, generated, prefill_reference, decode_reference) in enumerate(prompts):
                epoch = 2 * index + 1
                context = torch.cat((ids, generated[:-1])).tolist()
                inputs.request_state_rows.zero_()
                inputs.request_state_rows[0, 0] = 1
                inputs.page_tables.fill_(-1)
                inputs.page_tables[0, :needed_pages].copy_(torch.arange(needed_pages, dtype=torch.int32))
                inputs.row_slot_ids.fill_(-1)
                inputs.row_slot_ids[0] = 0
                inputs.row_generations.zero_()
                inputs.row_generations[0] = epoch
                inputs.live_rows.fill_(1)
                inputs.query_lengths.zero_()
                inputs.query_lengths[0] = 1
                inputs.token_ids.zero_()
                inputs.sample_enabled.zero_()
                stage_admission(epoch, ids.tolist(), generated.numel())
                stage_transfer(-1, 1, 0, reset=True)
                replay("init", epoch, 0)
                admission.admission_count.zero_()
                commands.num_ops.zero_()
                rows = []
                collected[prefix] = (ids, generated, rows, logits.shape[1])
                item = {"index": index, "prompt_ids": ids.tolist(), "reference_generated_ids": generated.tolist(),
                        "teacher_forced_input_ids": context, "positions_completed": 0}
                report["prompts"].append(item)
                for position, token in enumerate(context):
                    if mode == "chunked" and position < ids.numel() - 1:
                        continue
                    kind = "prefill" if mode == "sequential" or position == 0 or (mode == "chunked" and position < ids.numel()) else "decode"
                    if kind == "decode":
                        # Normal device-pending decode validates this token and consumes it.
                        requests.pending_token[0] = token
                    if mode == "chunked" and position == ids.numel() - 1:
                        # Use the actual prepared chunk capacity, including a partial final chunk.
                        capacity = generation.batch.tokens
                        for start in range(0, ids.numel(), capacity):
                            stop = min(ids.numel(), start + capacity)
                            inputs.query_lengths[0] = stop - start
                            inputs.token_ids[:stop-start].copy_(ids[start:stop])
                            replay("prefill", epoch, stop)
                        inputs.query_lengths[0] = 1
                    else:
                        inputs.token_ids[0] = token
                        replay(kind, epoch, position + 1)
                    active_logits = catalog[names[kind]].decoder.scratch.logits
                    current = read_logits(prefix, "teacher forcing", position)
                    rows.append(current)
                    item["positions_completed"] = position + 1
                    if position + 1 == ids.numel():
                        stage_transfer(1, 2, ids.numel())
                        replay("snapshot", epoch, ids.numel())
                        commands.num_ops.zero_()
                    if progress is not None:
                        progress(f"candidate {index + 1}/{len(prompts)} context {position + 1}/{len(context)}")
                reference = torch.cat((prefill_reference, decode_reference))
                if mode == "chunked":
                    reference = reference[ids.numel() - 1:]
                candidate = torch.stack(rows)
                metrics, per_position = distribution_metrics(reference, candidate)
                tensors.update({f"{prefix}.{name}": tensor for name, tensor in per_position.items()})
                item.update(metrics=metrics,
                    candidate_top1_on_reference_continuation=per_position["candidate_top1"][0 if mode == "chunked" else ids.numel() - 1:].tolist(),
                    top1_disagreement_positions=(per_position["reference_top1"] != per_position["candidate_top1"]).nonzero().flatten().tolist())
                for label, start, stop in (() if mode == "chunked" else (("prefill", 0, ids.numel()), ("decode", ids.numel(), len(context)))):
                    if stop > start:
                        item[label] = distribution_metrics(reference[start:stop], candidate[start:stop])[0]
                all_reference.append(reference)
                all_candidate.append(candidate)

                # Branch from the saved prompt state. Every following token now
                # comes from the candidate itself, even after the first mismatch.
                epoch += 1
                inputs.row_generations[0] = epoch
                stage_admission(epoch, ids.tolist(), generated.numel())
                stage_transfer(2, 1, ids.numel())
                replay("restore", epoch, ids.numel())
                admission.admission_count.zero_()
                commands.num_ops.zero_()
                greedy_rows = [rows[0 if mode == "chunked" else ids.numel() - 1]]
                greedy = [int(greedy_rows[0].argmax())]
                report["independent_greedy"]["status"] = "running"
                item["independent_greedy"] = {"positions_completed": 1, "candidate_ids": greedy}
                for step in range(generated.numel() - 1):
                    inputs.token_ids[0] = greedy[-1]
                    kind = "prefill" if mode == "sequential" else "decode"
                    if kind == "decode":
                        requests.pending_token[0] = greedy[-1]
                    replay(kind, epoch, ids.numel() + step + 1)
                    active_logits = catalog[names[kind]].decoder.scratch.logits
                    current = read_logits(prefix, "independent greedy", step + 1)
                    greedy_rows.append(current)
                    greedy.append(int(current.argmax()))
                    item["independent_greedy"]["positions_completed"] = len(greedy)
                    if progress is not None:
                        progress(f"candidate {index + 1}/{len(prompts)} independent greedy {len(greedy)}/{generated.numel()}")
                tensors[f"{prefix}.greedy_ids"] = torch.tensor(greedy, dtype=torch.int64)
                tensors[f"{prefix}.greedy_logits"] = torch.stack(greedy_rows)
                reference_ids = generated.tolist()
                common_prefix = next((i for i, (left, right) in enumerate(zip(greedy, reference_ids))
                                      if left != right), len(greedy))
                item["independent_greedy"].update(reference_ids=reference_ids,
                    candidate_text=tokenizer.decode(greedy, skip_special_tokens=False),
                    reference_text=tokenizer.decode(reference_ids, skip_special_tokens=False),
                    common_prefix_tokens=common_prefix, exact_sequence_match=greedy == reference_ids)
            # Restore a real saved endpoint, resume its captured head/sampling,
            # and prove that resume neither consumes a token nor mutates model state.
            epoch = 100
            inputs.row_generations[0] = epoch
            stage_admission(epoch, ids.tolist(), generated.numel())
            stage_transfer(2, 1, ids.numel())
            replay("restore", epoch, ids.numel())
            admission.admission_count.zero_()
            commands.num_ops.zero_()
            endpoint = pool.backing[1].cpu().clone()
            replay("resume", epoch, ids.numel(), sampled=True)
            resumed_logits = catalog[names["resume"]].logits[0].cpu()
            endpoint_logits = rows[0 if mode == "chunked" else ids.numel() - 1]
            if not torch.equal(resumed_logits, endpoint_logits):
                raise RuntimeError("captured resume logits differ from the saved endpoint")
            if not torch.equal(pool.backing[1].cpu(), endpoint):
                raise RuntimeError("captured resume modified the continuation state")
            report["resume"] = {"graph": names["resume"], "logits_bit_exact": True,
                                "continuation_bit_exact": True, "consumed_tokens": ids.numel(),
                                "generated_tokens": 1, "rng_counter": 1}
            # A zero-live-row request must leave physical model and request bytes
            # untouched; never read reserved virtual cache tails for this check.
            physical_rows, physical_pages = pool.memory_capacity.cpu().tolist()
            inactive_values = [("continuations", pool.backing[:physical_rows])]
            inactive_values.extend((name, tensor[:physical_pages]) for name, tensor in pool.pages.items())
            inactive_values.extend(("request." + name, getattr(requests, name))
                                   for name in requests.__dataclass_fields__
                                   if isinstance(getattr(requests, name), torch.Tensor))
            inactive_bytes = [(name, tensor, tensor.cpu().clone()) for name, tensor in inactive_values]
            inputs.live_rows.zero_()
            sequence += 1
            buffers["frame_index"].fill_(sequence % 2)
            buffers["submission_sequence"].fill_(sequence)
            prepared.graphs[names["prefill"]].replay()
            torch.cuda.synchronize(prepared.device)
            for name, tensor, initial in inactive_bytes:
                if not torch.equal(tensor.cpu(), initial):
                    raise RuntimeError("inactive request mutated " + name)
            report["inactive_request"] = {"physical_model_and_request_tensors_bit_exact": len(inactive_bytes)}
            report["metrics"] = distribution_metrics(torch.cat(all_reference), torch.cat(all_candidate))[0]
            if sum(value.shape[0] for value in all_candidate) != report["positions_expected"]:
                raise RuntimeError("qualification did not compare every expected position")
            if not torch.equal(pool.backing[0].cpu(), inactive_initial):
                raise RuntimeError("inactive continuation row changed during graph transitions")
            report["inactive_null_row_preserved"] = True
            report["independent_greedy"]["status"] = "complete"
            report.update(status="measured", all_expected_positions_compared=True,
                          batch_validation="passed every captured init, snapshot, restore and generation completion")
    except BaseException as error:
        failures.append(error)
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    finally:
        if snapshots is not None:
            try:
                with torch.inference_mode(), torch.cuda.device(prepared.device):
                    _restore(prepared, pool, snapshots)
                report["state_restored"] = True
                report["restored_tensors"] = len(snapshots)
            except BaseException as error:
                failures.append(error)
                report.update(status="failed", restoration_error={"type": type(error).__name__, "message": str(error)})
        report["elapsed_seconds"] = time.monotonic() - started
        for prefix, (ids, generated, rows, vocabulary) in collected.items():
            tensors[f"{prefix}.prompt_ids"], tensors[f"{prefix}.generated_ids"] = ids, generated
            split = 1 if mode == "chunked" else ids.numel()
            tensors[f"{prefix}.prefill_logits"] = (torch.stack(rows[:split]) if rows
                else torch.empty((0, vocabulary), dtype=torch.float32))
            tensors[f"{prefix}.decode_logits"] = (torch.stack(rows[split:]) if len(rows) > split
                else torch.empty((0, vocabulary), dtype=torch.float32))
        if tensors:
            try:
                save_file(tensors, artifact, metadata={"model_id": MODEL_ID, "revision": REVISION,
                    "status": report["status"], "reference_sha256": report.get("reference_sha256", ""),
                    "execution": report["execution"],
                    "generated_ids_role": "reference teacher-forcing IDs, not candidate generation"})
                report["artifact_sha256"] = _sha256(artifact)
            except BaseException as error:
                failures.append(error)
                report.update(status="failed", artifact_error={"type": type(error).__name__, "message": str(error)})
        else:
            report["artifact"] = None
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise BaseExceptionGroup("candidate qualification and cleanup failed", failures)
    return report
