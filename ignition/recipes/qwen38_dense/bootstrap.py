"""Prepare a single-GPU dense Qwen text graph catalog with device-resident state."""

import contextlib
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from ignition.components.completion import CompletionOutputs

from ignition.components.capacity import batch_capacities, prefill_capacities, request_capacity
from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.components.paged_generation import PreparedGeneration
from ignition.components.paged_maintenance import PreparedMaintenance
from ignition.components.paged_resume import PreparedResume
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from ignition.components.state_transfer import StateCommands
from ignition.preparation.capture import capture_forwards
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.disposable import storage_ranges
from ignition.preparation.execution import GraphSignature
from ignition.preparation.frontend import prepare_frontend
from ignition.preparation.fixtures.recurrent import register_benchmark_state
from ignition.preparation.host_memory import release_unused
from ignition.preparation.model import PreparedModel
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.preparation.tuning_replicas import TuningReplica
from ignition.progress import phase

from .config import PAGE_TOKENS
from .contract import prepare as prepare_contract
from .decoder import PreparedDecoder, Scratch
from .pool import ContinuationPool, continuation_layout, page_bytes
from .weights import ModelWeights


@dataclass(frozen=True)
class PreparationConfig:
    compile_workers: int | None = None
    tuning_replica: TuningReplica | None = None
    device: int = 0
    context: int = 262144
    prefill_chunk_size: int = 4096
    max_active: int = 4
    snapshot_budget_bytes: int = 4 << 30
    headroom_bytes: int = 4 << 30
    compile_decoder: bool = True
    retain_graphs: bool = False
    kernel_cache: Path | None = None
    mtp: bool = False
    speculative_widths: tuple[int, ...] = (1, 2, 3, 4)
    speculative_fixed_width: int | None = None
    preparation_pages: int | None = None

    def __post_init__(self):
        request_capacity(self.max_active)
        if not PAGE_TOKENS <= self.context <= 262144 or self.context % PAGE_TOKENS:
            raise ValueError("Qwen dense context must be 128-aligned in [128,262144]")
        if not 128 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 128:
            raise ValueError("Qwen dense prefill chunk must be 128-aligned and fit the context")
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0:
            raise ValueError("snapshot budget must be positive and headroom nonnegative")
        if self.preparation_pages is not None and (type(self.preparation_pages) is not int or self.preparation_pages < 1):
            raise ValueError("preparation pages must be a positive integer")
        if (not self.speculative_widths or self.speculative_widths[0] != 1
                or tuple(sorted(set(self.speculative_widths))) != self.speculative_widths
                or any(type(w) is not int or not 1 <= w <= 5 for w in self.speculative_widths)):
            raise ValueError("dense MTP widths must increase from one through at most five")
        if self.speculative_fixed_width is not None and self.speculative_fixed_width not in self.speculative_widths:
            raise ValueError("fixed verification width must belong to the prepared catalog")
        if self.mtp and (self.max_active > 4 or self.speculative_widths[-1] < 2):
            raise ValueError("dense MTP requires one through four requests and positive draft depth")

    @property
    def draft_tokens(self):
        return self.speculative_widths[-1] - 1 if self.mtp else 0


@torch.no_grad()
def prepare(snapshot, config, *, progress=None, package_storage):
    from .recipe import RECIPE

    snapshot = Path(snapshot).resolve()
    device = torch.device("cuda", config.device)
    torch.cuda.set_device(device)
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
    transient = contextlib.ExitStack()
    transient.enter_context(package_storage.transient())
    weights = ModelWeights.load(manifest, device, progress=progress, row_invariant=config.mtp, package_storage=package_storage)
    mtp_weights = None
    if config.mtp:
        from .mtp_weights import MTPWeights
        mtp_weights = MTPWeights.load(manifest, device, config=weights.config,
            embedding_weight=weights.tree.embed_tokens.weight, lm_head_weight=weights.head, progress=progress, package_storage=package_storage)
    release_unused()
    if config.mtp:
        # Repacking leaves large dead checkpoint/inverse buffers in the private
        # pool. End that allocation scope before backing recurrent/KV state.
        transient.close()
        transient.enter_context(package_storage.transient())
    from ignition.components.sm70_projection import is_sm70
    dtype, element_bytes = (torch.float32, 2) if is_sm70(device) else (torch.bfloat16, 1)
    layout = continuation_layout(weights.config, dtype=dtype, element_bytes=element_bytes,
                                 draft_tokens=config.draft_tokens)
    bytes_per_page = page_bytes(weights.config, element_bytes=element_bytes, mtp=config.mtp)
    snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
    if snapshots < 1:
        raise ValueError("snapshot budget must hold a complete Qwen dense continuation")
    slots, blocks = request_capacity(config.max_active), config.context // PAGE_TOKENS
    execution_rows = slots * (config.draft_tokens + 1)
    state_rows = execution_rows + snapshots
    pages = config.prefill_chunk_size // PAGE_TOKENS + 2 * slots
    pages = max(pages, config.preparation_pages or 0)
    total_memory = torch.cuda.mem_get_info(device)[1]
    with phase(progress, "Allocate model state"):
        pool = ContinuationPool.for_model(weights.config, state_rows + 1,
            min(pages, config.prefill_chunk_size // PAGE_TOKENS + 2 * slots) if config.compile_decoder else pages, device,
            draft_tokens=config.draft_tokens,
            storage=package_storage, maximum_pages=total_memory // bytes_per_page,
            maximum_rows=total_memory // layout.row_bytes,
            backed_slots=min(state_rows + 1, 1 + execution_rows + slots))
        vocabulary = weights.config.vocabulary
        requests = DeviceRequestState.allocate(device, vocab_size=vocabulary, capacity=slots)
        inputs = BatchInputs.allocate(max(config.prefill_chunk_size, execution_rows), blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=vocabulary, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        completion = CompletionOutputs.allocate((2, slots, 24 if config.mtp else 8), device=device)
        coordinator = None
        identity = hashlib.sha256(json.dumps({
            "config": weights.config.text, "headers": [s.header_sha256 for s in manifest.shards],
            "contract": "qwen38-dense.graph-inputs.v2", "draft_tokens": config.draft_tokens},
            sort_keys=True).encode()).hexdigest()
        session = RecordedPreparationSession(
            compile_workers=config.compile_workers, tuning_replica=config.tuning_replica,
            device=device, checkpoint=identity,
            workload_schema="qwen38-dense", numerical_policy="default", coordinator=coordinator,
            headroom_bytes=config.headroom_bytes, benchmark_storage=package_storage)
        session.__enter__()
        register_benchmark_state(session, pool.reserved_backing, storage=pool.reserved_backing,
                                 restore=pool.clear_for_preparation)
        arena = ScratchArena(device)
        session.register_scratch(arena)
        prefill_storage = Scratch.allocate(config.prefill_chunk_size, 1, weights.config, device)
        decode_storage = Scratch.allocate(execution_rows, execution_rows, weights.config, device)
        buffers = dict(inputs.buffers(), **commands.buffers(),
            **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing, memory_capacity=pool.memory_capacity,
            request_pending_token=requests.pending_token,
            sampling_grammar_mask=requests.grammar_mask, frame_index=frame,
            submission_sequence=sequence, **completion.buffers())
        speculation = None
        if config.mtp:
            from .speculative import SpeculativePreparation
            speculation = SpeculativePreparation(weights, mtp_weights, pool, requests, inputs, decode_storage,
                width=config.draft_tokens + 1, session=session, workspace=arena,
                completion=completion, frame=frame, sequence=sequence)
            buffers["draft_page_tables"] = speculation.draft_tables
    forwards, signatures, metadata = {}, {}, {}
    prefill_sizes, batch_sizes = prefill_capacities(config.prefill_chunk_size), batch_capacities(config.max_active)
    with phase(progress, "Construct graph catalog") as stage:
        def add(name, forward, kind, rows, tokens, fields, enabled=False):
            forwards[name] = forward
            signatures[name] = GraphSignature(kind, rows, tokens, tuple(fields), speculation_enabled=enabled)
            stage.update(advance=1)

        for enabled in ((False, True) if config.mtp else (False,)):
            suffix = "-no-mtp" if config.mtp and not enabled else ""
            modes = (("prefill", reversed(prefill_sizes)),)
            if not enabled:
                modes += (("decode", reversed(batch_sizes)),)
            forward_fields = (*inputs.buffers(), "sampling_grammar_mask",
                              *(("draft_page_tables",) if enabled else ()))
            for mode, sizes in modes:
                for capacity in sizes:
                    rows = 1 if mode == "prefill" else capacity
                    name = f"{mode}{capacity}" + (suffix if mode == "prefill" else "")
                    m = metadata[name] = Metadata.allocate(capacity, rows, blocks, device)
                    scratch = (prefill_storage if mode == "prefill" else decode_storage).rows(capacity, rows)
                    decoder = PreparedDecoder(weights, scratch, m, pool, mode=mode,
                        vocabulary_runtime=None, coordinator=coordinator, session=session, name=name, workspace=arena,
                        row_invariant=config.mtp and mode == "decode")
                    batch = PreparedBatch(inputs, requests, m, mode=mode, shard=weights.shard, pool=pool)
                    forward = PreparedGeneration(decoder, batch, completion, frame, sequence, runtime=None,
                        coordinator=coordinator, session=session, name=name + ".generation", workspace=arena)
                    if enabled:
                        from .speculative import PreparedPrefill
                        forward = PreparedPrefill(speculation, forward, name)
                    add(name, forward, mode, rows, capacity if mode == "prefill" else 1,
                        forward_fields, enabled)
            row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
            for capacity in reversed(batch_sizes):
                name = f"resume{capacity}" + suffix
                forward = PreparedResume(weights, requests, pool, inputs, completion, frame, sequence,
                    rows=capacity, runtime=None, vocabulary_runtime=None, coordinator=coordinator,
                    session=session, name=name, activation_mode="auto", workspace=arena, row_invariant=config.mtp)
                add(name, forward, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"), enabled)
                for mode in ("init", "restore", "snapshot"):
                    name = f"{mode}{capacity}" + suffix
                    maintenance, options = PreparedMaintenance, {}
                    if config.mtp:
                        from .speculative import DenseMaintenance
                        maintenance = DenseMaintenance
                        options["draft_tables"] = speculation.draft_tables if enabled else None
                    forward = maintenance(mode, requests, pool, commands, inputs, completion, frame, sequence,
                        runtime=None, coordinator=coordinator, session=session, name=name, capacity=capacity,
                        admission_buffers=admission, draft_state=enabled, **options)
                    fields = row_inputs if mode == "snapshot" else ("request_state_rows", *admission.__dataclass_fields__)
                    add(name, forward, mode, capacity, 1,
                        (*fields, "page_tables", *commands.buffers(), *(("draft_page_tables",) if enabled else ())), enabled)
        if speculation is not None:
            from .speculative import PreparedRound
            for batch_size in reversed(batch_sizes):
                for width in reversed(config.speculative_widths):
                    name = f"speculative{batch_size}x{width}"
                    operation = PreparedRound(speculation, batch_size, width, name)
                    operation.prepare()
                    add(name, operation, "speculative", batch_size, width,
                        (*inputs.buffers(), "sampling_grammar_mask", "draft_page_tables"), True)
    with phase(progress, "Prepare and tune kernels") as stage, transient:
        tuning = session.tune(show_progress=progress is not None, console=stage.console)
    stream, graph_pool = torch.cuda.Stream(device=device), package_storage.graph_pool.id
    # Workers serialize and consume metadata only. Cache state is cleared before
    # warmup, so its physical memory can hold compiler contexts in the meantime.
    released_state = []
    if config.compile_decoder:
        torch.cuda.synchronize(device)
        for region in package_storage.dynamic_regions:
            if region.group in ("STATE_ROWS", "KV_PAGES"):
                released_state.append((region, region.nbytes))
                region.release_backing()

    def materialize():
        for region, size in released_state:
            region.grow(max(region.nbytes, size))
        package_storage.grow("KV_PAGES", pages)
        pool.memory_capacity[1] = pages
        pool.clear_for_preparation()

    try:
        graphs, compiled = capture_forwards(forwards, signatures, session=session, device=device,
            stream=stream, pool=graph_pool, collective=None, coordinator=coordinator,
            compile_decoder=config.compile_decoder, progress=progress, cache_root=config.kernel_cache,
            retain_graphs=config.retain_graphs, materialize=materialize, warmup_storage=package_storage,
            compile_kinds=("prefill", "decode", "resume", "speculative"))
    finally:
        materialize()
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    release_unused()
    assets = {role: {"path": str(snapshot / file), "sha256": hashlib.sha256((snapshot / file).read_bytes()).hexdigest()}
              for role, file in (("tokenizer", "tokenizer.json"), ("chat_template", "chat_template.jinja"))}
    eos = weights.config.text["eos_token_id"]
    tokenizer = json.loads((snapshot / "tokenizer_config.json").read_text())
    eos_text = tokenizer["eos_token"]
    if isinstance(eos_text, dict):
        eos_text = eos_text["content"]
    from tokenizers import Tokenizer
    tokenizer_eos = Tokenizer.from_file(str(snapshot / "tokenizer.json")).token_to_id(eos_text)
    eos_ids = list(dict.fromkeys([*(eos if isinstance(eos, list) else [eos]), tokenizer_eos]))
    server_config = dict(identity=identity, model_id=snapshot.name, max_sequence_tokens=config.context,
        prefill_chunk_size=config.prefill_chunk_size, pages=pages, state_rows=state_rows, snapshots=snapshots,
        page_tokens=PAGE_TOKENS, frontend=prepare_frontend(snapshot, eos_token_ids=eos_ids),
        sampling_defaults=RECIPE.sampling_defaults.as_dict(), assets=assets,
        model_contract=prepare_contract(signatures, buffers, pool, vocabulary=vocabulary))
    if speculation is not None:
        server_config.update(speculative_widths=config.speculative_widths,
            speculative_token_capacity=execution_rows, speculative_proposal_source="mtp",
            speculative_fixed_width=config.speculative_fixed_width, history_drafts=False)
    free, total = torch.cuda.mem_get_info(device)
    memory = dict(rank=0, world_size=1, device=config.device, weights_bytes=weights.allocated_bytes,
        mtp_weights_bytes=0 if mtp_weights is None else mtp_weights.allocated_bytes,
        speculative_widths=config.speculative_widths if config.mtp else (),
        speculative_state_columns=config.draft_tokens + 1,
        mapped_host_bytes=0, offloaded_bytes=0, continuation_bytes=pool.backing.nbytes, state_rows=state_rows,
        preparation_state_rows=int(pool.memory_capacity[0].item()),
        preparation_state_bytes=int(pool.memory_capacity[0].item()) * layout.row_bytes,
        pages=pages, page_bytes=bytes_per_page, page_tokens=PAGE_TOKENS,
        kv_bytes=pages * bytes_per_page, completion_output_bytes=completion.nbytes, operator_scratch_bytes=arena.nbytes, graphs=len(graphs),
        prefill_capacities=prefill_sizes, free_device_bytes=free, total_device_bytes=total,
        headroom_bytes=config.headroom_bytes, tuning=tuning, compiled_decoder=config.compile_decoder)
    return PreparedModel(device=config.device, graphs=graphs, buffers=buffers, signatures=signatures,
        max_active=config.max_active, session=session, server_config=server_config, memory_report=memory,
        export_zero_ranges=(*storage_ranges([prefill_storage, decode_storage, arena.tensor, metadata]), *completion.zero_ranges()),
        resources=(weights, mtp_weights, speculation, pool, requests, inputs, admission, commands, arena, forwards, prefill_storage,
                   decode_storage, metadata, compiled, stream, graph_pool, session, coordinator))
