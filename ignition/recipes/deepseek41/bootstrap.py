"""Prepare DeepSeek text weights, file-backed Engram tables, and native graph catalogs."""

import contextlib
import hashlib
import json
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import torch

from ignition.components.capacity import speculative_token_capacity, batch_capacities, prefill_capacities, request_capacity
from ignition.components.parallel import TensorParallelShard
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from ignition.components.state_transfer import StateCommands
from ignition.preparation.capture import capture_forwards
from .contract import prepare as prepare_contract
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.collectives import (CollectivePreparation, create_all_gather_runtime,
    create_all_reduce_runtime, create_dma_all_reduce_runtime)
from ignition.preparation.disposable import storage_ranges
from ignition.preparation.execution import GraphSignature
from ignition.preparation.host_memory import release_unused
from ignition.preparation.frontend import prepare_frontend
from ignition.preparation.model import PreparedModel
from ignition.preparation.rank_group import minimum_across_ranks
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.progress import phase

from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from .recipe import RECIPE
from .attention import INDEXER_QUERY_ROWS
from .decoder import PreparedDecoder, Scratch
from ignition.components.dspark import PreparedContext, PreparedMarkov
from .dspark import PreparedDraft
from .engram import Tables
from ignition.components.paged_generation import PreparedGeneration
from ignition.components.paged_maintenance import PreparedMaintenance
from .pool import ContinuationPool, PAGE_TOKENS, continuation_layout, page_bytes
from ignition.components.paged_resume import PreparedResume
from .weights import ModelWeights


_CONTEXT = partial(PreparedContext, compact_prefill=True)


@dataclass(frozen=True)
class PreparationConfig:
    devices: tuple[int, ...] = (0, 1, 2, 3)
    context: int = 1048576
    prefill_chunk_size: int = 4096
    max_active: int = 4
    minimum_pages: int = 8
    # An explicit cache capacity makes memory comparisons independent of automatic sizing.
    kv_pages: int | None = None
    snapshot_budget_bytes: int = 4 << 30
    headroom_bytes: int = 4 << 30
    compile_decoder: bool = True
    retain_graphs: bool = False
    kernel_cache: Path | None = None
    dspark: bool = True
    speculative_widths: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    speculative_fixed_width: int | None = None
    speculative_token_capacity: int | None = None
    # Run single-row decode's shared expert on an auxiliary stream; without overlap it runs serially.
    stream_overlap: bool = True
    # Separate proposal and verification graphs for native confidence-prefix scheduling.
    split_speculation: bool = False
    # Additional combined graphs for exact qualification; omit when measuring service memory.
    speculation_reference_graphs: bool = False

    def __post_init__(self):
        if self.speculation_reference_graphs and not self.split_speculation:
            raise ValueError("speculation reference graphs require split speculation")
        if self.split_speculation and (not self.dspark or self.draft_tokens < 1):
            raise ValueError("split speculation requires a learned draft model and positive draft depth")
        if type(self.stream_overlap) is not bool:
            raise TypeError("stream_overlap must be a boolean")
        if len(self.devices) != 4 or len(set(self.devices)) != 4 or any(d < 0 for d in self.devices):
            raise ValueError("DeepSeek V4.1 Flash preparation requires four distinct GPUs")
        if not PAGE_TOKENS <= self.context <= 1048576 or self.context % PAGE_TOKENS:
            raise ValueError("DeepSeek context must be 128-aligned in [128,1048576]")
        if not 64 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 64:
            raise ValueError("prefill chunk size must be 64-aligned in [64,context]")
        request_capacity(self.max_active)
        self.speculative_shapes
        if self.speculative_fixed_width is not None and self.speculative_fixed_width not in self.speculative_widths:
            raise ValueError("fixed verification width must belong to the prepared widths")
        if self.minimum_pages <= 0 or self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0:
            raise ValueError("page and snapshot budgets must be positive and headroom nonnegative")
        if self.kv_pages is not None and (type(self.kv_pages) is not int or self.kv_pages < self.minimum_pages):
            raise ValueError("kv_pages must cover the minimum page capacity")

    @property
    def request_capacity(self):
        return request_capacity(self.max_active)

    @property
    def prefill_capacities(self):
        return prefill_capacities(self.prefill_chunk_size)

    @property
    def batch_capacities(self):
        return tuple(reversed(batch_capacities(self.max_active)))

    @property
    def speculative_shapes(self):
        from ignition.components.capacity import speculative_shapes

        return speculative_shapes(self.max_active, self.speculative_widths, self.speculative_token_capacity) if self.dspark else ()

    @property
    def decode_tokens(self):
        return max((b * w for b, w in self.speculative_shapes), default=self.request_capacity)

    @property
    def draft_tokens(self):
        return max(self.speculative_widths) - 1 if self.dspark else 0


@torch.no_grad()
def prepare_rank(snapshot, config, *, rank, group, progress=None, package_storage=None):
    if package_storage is None:
        raise ValueError("DeepSeek preparation requires package storage")
    snapshot = Path(snapshot).resolve()
    shard = TensorParallelShard(rank, len(config.devices))
    device = torch.device("cuda", config.devices[rank])
    torch.cuda.set_device(device)
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
    metadata_path = manifest.metadata_path
    transient = contextlib.ExitStack()
    transient.enter_context(package_storage.transient())
    weights = ModelWeights.load(manifest, device, shard=shard, progress=progress, dspark=config.dspark,
                                confidence=config.split_speculation)
    release_unused()
    if progress is not None:
        progress(f"Resident model tensors: {weights.allocated_bytes / (1 << 30):.3f} GiB per rank")
    sequence = torch.zeros(1, dtype=torch.int64, device=device)
    tables = Tables.prepare(manifest, weights.config, tokens=max(config.prefill_chunk_size, config.decode_tokens),
        device=device, sequence=sequence, host_allocator=package_storage.host_allocate, progress=progress)
    with phase(progress, "Allocate model state and collectives"):
        layout = continuation_layout(weights.config, dspark=config.dspark)
        snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
        if snapshots < 1:
            raise ValueError("snapshot budget must hold a complete DeepSeek continuation")
        slots, blocks = config.request_capacity, config.context // PAGE_TOKENS
        execution_rows = slots * (max(config.speculative_widths) if config.dspark else 1)
        state_rows = snapshots + execution_rows
        pages = (config.prefill_chunk_size + PAGE_TOKENS - 1) // PAGE_TOKENS + 2 * slots
        if config.kv_pages is not None:
            pages = config.kv_pages
        bytes_per_page = page_bytes(weights.config)
        total_memory = minimum_across_ranks(torch.cuda.mem_get_info(device)[1], group)
        pool = ContinuationPool.allocate(weights.config, state_rows + 1, pages, device,
            storage=package_storage, maximum_pages=total_memory // bytes_per_page,
            maximum_rows=total_memory // layout.row_bytes,
            backed_slots=min(state_rows + 1, 1 + execution_rows + slots), dspark=config.dspark)
        requests = DeviceRequestState.allocate(device, vocab_size=129280, capacity=slots)
        inputs = BatchInputs.allocate(config.prefill_chunk_size, blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=129280, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        records = torch.zeros((2, slots, 24 if config.dspark else 8), dtype=torch.int64, device=device)
        coordinator = CollectivePreparation(group)
        ipc = package_storage.peer_storage
        tokens = max(config.prefill_chunk_size, config.decode_tokens, slots * config.draft_tokens)
        collective = create_all_reduce_runtime(group, device, max_tokens=tokens, hidden_size=5120, ipc=ipc)
        prefill_collective = create_dma_all_reduce_runtime(group, device,
            max_tokens=config.prefill_chunk_size, hidden_size=5120, ipc=ipc)
        vocabulary_collective = create_all_gather_runtime(group, device,
            max_rows=max(config.decode_tokens, slots * config.draft_tokens), columns=129280, ipc=ipc)
        markov_collective = (create_all_gather_runtime(group, device,
            max_rows=slots, columns=2 * 129280, ipc=ipc) if config.dspark else None)
        identity = hashlib.sha256(json.dumps({"config": json.loads((metadata_path / "config.json").read_text()),
            "headers": [s.header_sha256 for s in manifest.shards], "world_size": shard.world_size,
            "contract": "deepseek41.graph-inputs.v1"}, sort_keys=True).encode()).hexdigest()
        session = RecordedPreparationSession(device=device, checkpoint=identity,
            workload_schema="deepseek41-text", numerical_policy="default", coordinator=coordinator,
            headroom_bytes=config.headroom_bytes, benchmark_storage=package_storage)
        session.__enter__()
        session.register_benchmark_resource("continuations", pool, restore=pool.clear_for_preparation)
        arena = ScratchArena(device, overlap=config.stream_overlap)
        session.register_scratch(arena)
        prefill_storage = Scratch.allocate(config.prefill_chunk_size, 1, shard.world_size, device)
        decoder_storage = Scratch.allocate(config.decode_tokens, config.decode_tokens, shard.world_size, device)
        suffix_storage = Scratch.allocate(128, 1, shard.world_size, device)
        window_staging = (torch.empty((len(weights.layers), slots, (128 + max(config.speculative_widths)) * 528),
                                     dtype=torch.uint8, device=device) if config.dspark else None)
        decoder_factory = partial(PreparedDecoder, window_staging=window_staging, suffix_scratch=suffix_storage)
        buffers = dict(inputs.buffers(), **tables.buffers(), **commands.buffers(), **{
            name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing, memory_capacity=pool.memory_capacity,
            sampling_grammar_mask=requests.grammar_mask, frame_index=frame,
            submission_sequence=sequence, completion_records=records)
        forwards, signatures, metadata, scratch = {}, {}, {}, {}
        speculation = None
        if config.dspark:
            from ignition.components.dspark import DSparkForwards, SpeculativePreparation

            speculation = SpeculativePreparation(weights, tables, pool, requests, inputs, decoder_storage,
                draft_tokens=config.draft_tokens, records=records, frame=frame, sequence=sequence,
                session=session, workspace=arena,
                runtime=collective, vocabulary_runtime=vocabulary_collective,
                markov_runtime=markov_collective, coordinator=coordinator,
                forwards=DSparkForwards(decoder_factory,Scratch,_CONTEXT,PreparedDraft,PreparedMarkov))

    del manifest
    modes = (True, False) if config.dspark else (False,)
    count = (len(modes) * (len(config.prefill_capacities) + 4 * len(config.batch_capacities))
             + len(config.speculative_shapes) + len(config.batch_capacities))
    if config.split_speculation:
        count += len(config.batch_capacities) * (config.draft_tokens + 1)
        if config.speculation_reference_graphs:
            count += len(config.speculative_shapes)
    catalog_bytes = {}
    catalog_previous = torch.cuda.memory_allocated(device)
    with phase(progress, "Construct graph catalog", total=count) as stage:
        def add(name, forward, kind, rows, tokens, fields, *, enabled=None):
            nonlocal catalog_previous
            forwards[name] = forward
            signatures[name] = GraphSignature(kind, rows, tokens, tuple(fields), speculation_enabled=enabled)
            allocated = torch.cuda.memory_allocated(device)
            catalog_bytes[kind] = catalog_bytes.get(kind, 0) + allocated - catalog_previous
            catalog_previous = allocated
            stage.update(advance=1)

        forward_inputs = (*inputs.buffers(), "sampling_grammar_mask")
        for enabled in modes:
            suffix = "-no-mtp" if config.dspark and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            for mode, capacities in (("prefill", reversed(config.prefill_capacities)),
                                     ("decode", () if enabled else config.batch_capacities)):
                for capacity in capacities:
                    rows = 1 if mode == "prefill" else capacity
                    name = f"{mode}{capacity}" + (suffix if mode == "prefill" else "")
                    metadata[name] = Metadata.allocate(capacity, rows, blocks, device)
                    storage = prefill_storage if mode == "prefill" else decoder_storage
                    scratch[name] = storage.rows(capacity, rows)
                    decoder = decoder_factory(mode_weights, tables, scratch[name], metadata[name], pool,
                        mode=mode, runtime=prefill_collective if mode == "prefill" and capacity >= 128 else collective,
                        vocabulary_runtime=vocabulary_collective,
                        coordinator=coordinator, session=session, name=name, workspace=arena)
                    batch = PreparedBatch(inputs, requests, metadata[name], mode=mode, shard=shard, pool=pool)
                    forward = PreparedGeneration(decoder, batch, records, frame, sequence, runtime=collective,
                        coordinator=coordinator, session=session, name=name + ".generation", workspace=arena, context_factory=_CONTEXT)
                    add(name, forward, mode, rows, capacity if mode == "prefill" else 1, forward_inputs, enabled=enabled)
        split = None
        if speculation is not None:
            from ignition.components.parallel_draft import PreparedParallelDraftRound, prepare_sampling_scratch

            from ignition.preparation.scratch import bind_scratch

            sampling_scratch = prepare_sampling_scratch(slots, max(config.speculative_widths), requests.vocab_size, device)
            # Sequential graphs share scratch; retained proposals have request-owned storage.
            round_storage = bind_scratch(
                PreparedParallelDraftRound.storage_layout(slots, max(config.speculative_widths), requests.vocab_size),
                device=device)
            if config.split_speculation:
                from copy import copy
                from ignition.components.capacity import proposal_shapes
                from ignition.components.split_draft import (
                    PreparedCombinedDraftRound, PreparedDraftProposal, PreparedDraftVerification, PreparedProposalStorage)

                confidence = package_storage.host_allocate("proposal_confidence", (slots, config.draft_tokens), torch.float32)
                buffers["proposal_confidence"] = confidence.device_view
                split = PreparedProposalStorage(speculation, config.draft_tokens)
                buffers.update(split.buffers())
                proposals = {}
                for batch, depth in proposal_shapes(config.max_active, config.draft_tokens):
                    owner = copy(speculation)
                    owner.block_size = depth
                    name = f"proposal{batch}x{depth + 1}"
                    source = PreparedParallelDraftRound(owner, batch, depth + 1, name,
                        sampling_scratch=sampling_scratch, round_storage=round_storage, proposal_only=True)
                    source.prepare()
                    proposal = PreparedDraftProposal(source, confidence.device_view, split)
                    proposals[batch, depth] = proposal
                    add(name, proposal, "proposal", batch, depth + 1, forward_inputs, enabled=True)
                for batch, width in config.speculative_shapes:
                    name = f"verification{batch}x{width}"
                    target = PreparedParallelDraftRound(speculation, batch, width, name,
                        generate_proposals=False, sampling_scratch=sampling_scratch, round_storage=round_storage)
                    target.prepare()
                    verification = PreparedDraftVerification(split, target)
                    add(name, verification, "verification", batch, width, forward_inputs, enabled=True)
                    if width == 1:
                        # No proposal is needed to decode while retaining draft context.
                        decode = PreparedParallelDraftRound(speculation, batch, 1, f"speculative{batch}x1",
                            sampling_scratch=sampling_scratch, round_storage=round_storage)
                        decode.prepare()
                        add(f"speculative{batch}x1", decode, "speculative", batch, 1, forward_inputs, enabled=True)
                    if config.speculation_reference_graphs:
                        add(f"combined{batch}x{width}", PreparedCombinedDraftRound(
                            proposals[batch, config.draft_tokens], verification),
                            "speculative", batch, width, forward_inputs, enabled=True)
            else:
                for batch, width in config.speculative_shapes:
                    name = f"speculative{batch}x{width}"
                    forward = PreparedParallelDraftRound(speculation, batch, width, name,
                        sampling_scratch=sampling_scratch, round_storage=round_storage)
                    forward.prepare()
                    add(name, forward, "speculative", batch, width, forward_inputs, enabled=True)
        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for enabled in modes:
            suffix = "-no-mtp" if config.dspark and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            for capacity in config.batch_capacities:
                name = f"resume{capacity}" + suffix
                forward = PreparedResume(mode_weights, requests, pool, inputs, records, frame, sequence,
                    rows=capacity, runtime=collective, vocabulary_runtime=vocabulary_collective,
                    coordinator=coordinator, session=session, name=name)
                add(name, forward, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"), enabled=enabled)
                for mode in ("init", "restore", "snapshot"):
                    name = f"{mode}{capacity}" + suffix
                    forward = PreparedMaintenance(mode, requests, pool, commands, inputs, records, frame, sequence,
                        runtime=collective, coordinator=coordinator, session=session, name=name,
                        capacity=capacity, admission_buffers=admission, draft_state=enabled)
                    fields = row_inputs if mode == "snapshot" else ("request_state_rows", *admission.__dataclass_fields__)
                    add(name, forward, mode, capacity, 1, (*fields, "page_tables", *commands.buffers()), enabled=enabled)
    if progress is not None:
        progress(f"before tuning: {torch.cuda.memory_allocated(device) >> 20} MiB allocated, "
                 f"{torch.cuda.mem_get_info(device)[0] >> 20} MiB free")
        progress(f"Graph operand allocation by operation: {catalog_bytes}")
    with phase(progress, "Prepare and tune kernels") as stage, transient:
        tuning = session.tune(show_progress=progress is not None and rank == 0, console=stage.console)
    stream, graph_pool = torch.cuda.Stream(device=device), package_storage.graph_pool.id
    graphs, compiled = capture_forwards(forwards, signatures, session=session, device=device, stream=stream,
        pool=graph_pool, collective=collective, coordinator=coordinator, compile_decoder=config.compile_decoder,
        progress=progress, cache_root=config.kernel_cache, retain_graphs=config.retain_graphs,
        materialize=pool.clear_for_preparation, warmup_storage=package_storage,
        compile_kinds=("prefill", "decode", "resume", "speculative", "proposal", "verification"))
    session.freeze()
    pool.clear_for_preparation()
    if split is not None:
        for buffer in split.buffers().values():
            buffer.zero_()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    release_unused()
    if group is not None:
        torch.distributed.barrier(group=group)
    free, total = torch.cuda.mem_get_info(device)
    frontend = Path(__file__).parent / "assets"
    assets = {role: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for role, path in (("tokenizer", metadata_path / "tokenizer.json"),
            ("chat_template", frontend / "chat_template.jinja"))}
    buffers["request_pending_token"] = requests.pending_token
    server_config = {"identity": identity, "model_id": snapshot.name,
        "max_sequence_tokens": config.context, "prefill_chunk_size": config.prefill_chunk_size,
        "pages": pages, "state_rows": state_rows, "snapshots": snapshots, "page_tokens": PAGE_TOKENS,
        "frontend": prepare_frontend(metadata_path, eos_token_ids=[1]),
        "sampling_defaults": RECIPE.sampling_defaults.as_dict(), "assets": assets,
        "model_contract": prepare_contract(signatures, buffers, pool,
            page_tokens=PAGE_TOKENS, draft_lag=0 if config.dspark else None, rows=None,
            embedding_tables=range(len(tables.sources)))}
    memory = {"rank": rank, "world_size": shard.world_size, "device": config.devices[rank],
        "catalog_operand_bytes": catalog_bytes,
        "weights_bytes": weights.allocated_bytes, "mapped_host_bytes": 0,
        "offloaded_bytes": sum(source.nbytes for pair in tables.sources.values() for source in pair),
        "continuation_bytes": pool.backing.nbytes, "state_rows": state_rows,
        "preparation_state_rows": int(pool.memory_capacity[0].item()),
        "preparation_state_bytes": int(pool.memory_capacity[0].item()) * layout.row_bytes,
        "pages": pages, "page_bytes": bytes_per_page, "page_tokens": PAGE_TOKENS,
        "kv_bytes": pages * bytes_per_page, "operator_scratch_bytes": arena.nbytes,
        "graphs": len(graphs), "prefill_capacities": config.prefill_capacities,
        "indexer_query_rows": INDEXER_QUERY_ROWS,
        "free_device_bytes": free, "total_device_bytes": total, "headroom_bytes": config.headroom_bytes,
        "allocated_bytes": torch.cuda.memory_allocated(device), "reserved_bytes": torch.cuda.memory_reserved(device),
        "tuning": tuning, "compiled_decoder": config.compile_decoder}
    memory["split_speculation"] = config.split_speculation
    if config.dspark:
        server_config.update(speculative_widths=config.speculative_widths,
            speculative_token_capacity=speculative_token_capacity(config.max_active, max(config.speculative_widths), config.speculative_token_capacity),
            speculative_fixed_width=config.speculative_fixed_width)
        memory.update(speculative_shapes=config.speculative_shapes)
    disposable = [prefill_storage, decoder_storage, arena.tensor, metadata, tables.buffers()]
    if speculation is not None:
        disposable.append(speculation.draft_scratch)
    if split is not None:
        disposable.append(split.buffers())
    for forward in forwards.values():
        if hasattr(forward, "completion") and hasattr(forward.completion, "scratch_tensors"):
            disposable.append(forward.completion.scratch_tensors)
    model = PreparedModel(device=config.devices[rank], graphs=graphs, buffers=buffers, signatures=signatures,
        max_active=config.max_active, session=session, server_config=server_config, memory_report=memory,
        export_zero_ranges=storage_ranges(disposable), export_events=tables.events(),
        embedding_catalog=tables.resident_catalog(),
        package_data=tables if rank == 0 else None,
        resources=(weights, tables, pool, requests, inputs, admission, commands, arena, forwards,
            prefill_storage, decoder_storage, scratch, metadata, compiled, stream, graph_pool, session,
            collective, prefill_collective, vocabulary_collective, markov_collective, coordinator, speculation),
        close_resources=tuple(c for c in (markov_collective, prefill_collective, vocabulary_collective, collective) if c is not None))
    if split is not None:
        model.proposal_confidence = confidence.host_view
        model.proposal_state = split
    return model
