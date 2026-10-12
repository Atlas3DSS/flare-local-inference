"""Prepare a GLM tensor-parallel rank before native serving takes ownership."""

import contextlib
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from ignition.components.capacity import batch_capacities, prefill_capacities, request_capacity
from ignition.components.parallel import TensorParallelShard
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from ignition.components.state_transfer import StateCommands
from ignition.preparation.capture import capture_forwards
from ignition.preparation.deferred import deferred_mode
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.collectives import (
    CollectivePreparation,
    create_all_gather_runtime,
    create_all_reduce_runtime,
    create_head_exchange,
)
from ignition.preparation.execution import GraphSignature
from ignition.preparation.fixtures.recurrent import register_benchmark_state
from ignition.preparation.model import PreparedModel
from ignition.preparation.rank_group import minimum_across_ranks
from ignition.preparation.scratch import ScratchArena
from ignition.progress import phase
from ignition.preparation.tuning import RecordedPreparationSession

from .batch import BatchInputs, PreparedBatch
from .generation import PreparedGeneration
from .maintenance import PreparedMaintenance
from .pool import ContinuationPool, continuation_layout
from .resume import PreparedResume
from .stack import DecoderScratch, PreparedDecoder
from .state import DecoderMetadata
from .weights import ModelWeights


_COMPILED_KINDS = ("prefill", "decode", "resume", "speculative")
# Allocator granularity and fragmentation slack beyond the headroom kept after the final KV allocation.
_KV_ALLOCATION_MARGIN = 64 << 20


@dataclass(frozen=True)
class PreparationConfig:
    devices: tuple[int, ...] = (0, 1)
    context: int = 262144
    prefill_chunk_size: int = 4096
    max_active: int = 4
    pages: int | None = None
    # Without fixed pages, KV pages fill the memory measured free after graph capture, at least this many.
    minimum_pages: int | None = None
    snapshot_budget_bytes: int = 4 << 30
    headroom_bytes: int = 4 << 30
    compile_decoder: bool = True
    retain_graphs: bool = False
    kernel_cache: Path | None = None
    mtp: bool = False
    # Routed NVFP4-CSF experts: A16 activations with A4 prefill launches, or A4 throughout.
    expert_activation: str = "a16"
    # Shard latent records and C4 pools across the tensor-parallel ranks (decode context parallelism).
    context_parallel: bool = False
    speculative_widths: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    speculative_fixed_width: int | None = None
    speculative_graph_budget: int = 8

    def __post_init__(self):
        if type(self.mtp) is not bool:
            raise TypeError("mtp must be a boolean")
        self.speculative_shapes
        if type(self.compile_decoder) is not bool:
            raise TypeError("compile_decoder must be a boolean")
        if (not isinstance(self.devices, tuple) or len(self.devices) not in (2, 4)
                or len(set(self.devices)) != len(self.devices)
                or any(type(d) is not int or d < 0 for d in self.devices)):
            raise ValueError("GLM requires two or four distinct process-local CUDA devices")
        for name in ("context", "prefill_chunk_size", "max_active", "snapshot_budget_bytes", "headroom_bytes"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer")
        if not self.page_tokens <= self.context <= 1048576 or self.context % self.page_tokens:
            raise ValueError(f"GLM context must be {self.page_tokens}-aligned in [{self.page_tokens},1048576]")
        if not 64 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 64:
            raise ValueError("prefill chunk size must be 64-aligned in [64,context]")
        request_capacity(self.max_active)
        if self.pages is not None and (type(self.pages) is not int or not 0 < self.pages <= 2147483647):
            raise ValueError("page capacity must be a positive signed 32-bit integer")
        if self.minimum_pages is not None and (type(self.minimum_pages) is not int or self.minimum_pages <= 0
                                               or self.pages is not None and self.minimum_pages > self.pages):
            raise ValueError("minimum page capacity must be positive and at most the fixed page capacity")
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0:
            raise ValueError("snapshot budget must be positive and headroom nonnegative")

    @property
    def speculative_shapes(self):
        from ignition.components.capacity import speculative_shapes

        return speculative_shapes(self.max_active, self.speculative_widths,
                                  self.speculative_fixed_width, self.speculative_graph_budget)

    @property
    def token_capacity(self):
        return max(self.prefill_chunk_size, self.request_capacity,
                   max((b * w for b, w in self.speculative_shapes), default=0) if self.mtp else 0)

    @property
    def prefill_capacities(self):
        return prefill_capacities(self.prefill_chunk_size)

    @property
    def batch_capacities(self):
        return tuple(reversed(batch_capacities(self.max_active)))

    @property
    def request_capacity(self):
        return request_capacity(self.max_active)

    @property
    def page_capacity(self):
        """Fixed KV pages, or the context of every active request when memory does not size them."""
        if self.pages is not None:
            return self.pages
        return self.minimum_pages or (self.context // self.page_tokens + int(self.mtp)) * self.max_active

    @property
    def speculative_state_rows(self):
        """Recurrent states kept per request for every verified position after the first."""
        return max(self.speculative_widths) - 1

    @property
    def page_tokens(self):
        """Global positions per KV page: 256 local records on each rank that shares the context."""
        return 256 * (len(self.devices) if self.context_parallel else 1)

    @property
    def preparation_pages(self):
        """Pages held while tuning and measuring graphs; the final pools replace them before capture."""
        return min(self.page_capacity, 2 * self.request_capacity)


@torch.no_grad()
def prepare_rank(snapshot: Path, config: PreparationConfig, *, rank, group, progress=None,
                 package_storage=None) -> PreparedModel:
    """Load, tune and capture all text variants on one supervised TP rank.

    The caller must terminate the complete rank group if any preparation fails.
    Successful cleanup requires every rank to close its IPC runtime together,
    after native replay drains. No eager execution path is exported. Package
    storage backs device memory, collective mappings and graphs with addresses
    that native restore reproduces.
    """
    def report(message):
        if progress is not None:
            progress(message)

    snapshot = Path(snapshot).resolve()
    shard = TensorParallelShard(rank, len(config.devices))
    device = torch.device("cuda", config.devices[rank])
    torch.cuda.set_device(device)
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
    if config.expert_activation == "a16":
        # b12x retains activation scales for A4 prefill launches only while preparing with this switch.
        os.environ["B12X_W4A16_A4_PREFILL"] = "1"
    # Loading, construction and tuning free most of what they allocate; a transient package pool returns
    # that memory to the device once tuning completes, before graph measurement sizes the KV pages.
    transient = contextlib.ExitStack()
    if package_storage is not None:
        transient.enter_context(package_storage.transient())
    weights = ModelWeights.load(manifest, device, shard=shard, progress=progress, mtp=config.mtp,
                                expert_activation=config.expert_activation)
    with phase(progress, "Allocate model state and collectives") as stage:
        layer_types = weights.tree.config.text_config.layer_types
        layout = continuation_layout(layer_types, shard.world_size, mtp=config.mtp)
        snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
        if snapshots < 1:
            raise ValueError("GLM snapshot budget must hold at least one continuation row")
        pages, blocks = config.preparation_pages, config.context // config.page_tokens
        slots = config.request_capacity
        extra_base = slots + 1 + snapshots
        pool = ContinuationPool.allocate(layer_types, shard.world_size,
            extra_base + (slots * config.speculative_state_rows if config.mtp else 0), pages, device, mtp=config.mtp)
        context_ranks = shard.world_size if config.context_parallel else 1
        pool.replace_pages(pages, ranks=context_ranks)
        page_bytes = sum(cache.nbytes for _, cache in (*pool.partial_pages(), *pool.draft_pages())) // pages
        # Plans bound the page pools that replace the preparation pages after graph measurement.
        max_cache_pages = minimum_across_ranks(torch.cuda.mem_get_info(device)[1] // page_bytes, group)
        requests = DeviceRequestState.allocate(device, vocab_size=154880, capacity=slots, consumed_tokens=pool.consumed_tokens[1:1 + slots, 0])
        inputs = BatchInputs.allocate(config.prefill_chunk_size, blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=154880, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        records = torch.zeros((2, slots, 24 if config.mtp else 8), dtype=torch.int64, device=device)
        coordinator = CollectivePreparation(group)
        ipc = None if package_storage is None else package_storage.peer_storage
        collective = create_all_reduce_runtime(group, device, max_tokens=config.token_capacity, hidden_size=4096,
                                               ipc=ipc)
        sample_rows = max(b * w for b, w in config.speculative_shapes) if config.mtp else config.request_capacity
        vocabulary_collective = create_all_gather_runtime(group, device,
            max_rows=sample_rows, columns=154880, ipc=ipc)
        identity = hashlib.sha256(json.dumps({"config": json.loads((snapshot / "config.json").read_text()),
            "headers": [s.header_sha256 for s in manifest.shards], "world_size": shard.world_size,
            "contract": "glm5next.graph-inputs.v1"}, sort_keys=True).encode()).hexdigest()
        session = RecordedPreparationSession(device=device, checkpoint=identity, workload_schema="glm-text-stack-tp-v1",
            numerical_policy="default", coordinator=coordinator)
        session.__enter__()
        register_benchmark_state(session, pool.decoder.layers[0].recurrent,
            storage=pool.backing, restore=pool.clear_for_preparation)
        arena = ScratchArena(device)
        session.register_scratch(arena)
        forwards, signatures, decoders, scratch, metadata = {}, {}, {}, {}, {}
        buffers = dict(inputs.buffers(), **commands.buffers(), **{
            name: getattr(admission, name) for name in admission.__dataclass_fields__},
            sampling_grammar_mask=requests.grammar_mask, frame_index=frame,
            submission_sequence=sequence, completion_records=records)
        forward_inputs = (*inputs.buffers(), "sampling_grammar_mask")
        context = None
        if config.context_parallel:
            from .dcp import SLICE_ROWS, ContextParallel, ContextShards

            # Only decode exchanges heads; prefill gathers context blocks instead.
            exchange_rows = min(SLICE_ROWS, max(config.request_capacity,
                max((b * w for b, w in config.speculative_shapes), default=1) if config.mtp else 1))
            context = ContextParallel(ContextShards.on(rank, shard.world_size, device),
                heads=create_head_exchange(group, device, max_rows=exchange_rows, heads=64, head_dim=512, ipc=ipc),
                candidates=create_all_gather_runtime(group, device, max_rows=min(config.token_capacity, SLICE_ROWS),
                                                     columns=2048 * shard.world_size, ipc=ipc),
                coordinator=coordinator)
        shards = None if context is None else context.shards
        # Prefill chunks overlay their layer intermediates; decode rows keep small separate scratch.
        prefill_storage = DecoderScratch.allocate(config.prefill_chunk_size, blocks, "prefill", device,
            shard=shard, sample_rows=1, context_ranks=context_ranks, overlay_layers=True)
        decode_rows = max(config.request_capacity,
                          max((b * w for b, w in config.speculative_shapes), default=1) if config.mtp else 1)
        decoder_storage = DecoderScratch.allocate(decode_rows, blocks, "decode", device,
            shard=shard, sample_rows=sample_rows, context_ranks=context_ranks)

        speculation = None
        if config.mtp:
            from .speculative import SpeculativePreparation

            speculation = SpeculativePreparation(weights, pool, requests, inputs, decoder_storage, prefill_storage,
                extra_base=extra_base, session=session, workspace=arena, records=records, frame=frame,
                sequence=sequence, runtime=collective, vocabulary_runtime=vocabulary_collective, coordinator=coordinator,
                max_cache_pages=max_cache_pages, prefill_tokens=config.prefill_chunk_size,
                decode_tokens=max(b * w for b, w in config.speculative_shapes), context=context,
                state_stride=config.speculative_state_rows)
            buffers["draft_page_tables"] = speculation.draft_tables
            forward_inputs += ("draft_page_tables",)

    graph_count = (len(config.prefill_capacities) + 4 * len(config.batch_capacities)
                   + (len(config.speculative_shapes) if config.mtp else len(config.batch_capacities)))
    with phase(progress, "Construct graph catalog", total=graph_count) as stage:
        def add(name, function, kind, rows, tokens, fields):
            forwards[name] = function
            signatures[name] = GraphSignature(kind, rows, tokens, tuple(fields))
            stage.update(advance=1)

        for mode, capacities in (("prefill", reversed(config.prefill_capacities)), ("decode", () if config.mtp else config.batch_capacities)):
            for capacity in capacities:
                rows = 1 if mode == "prefill" else capacity
                name = f"{mode}{capacity}"
                metadata[name] = DecoderMetadata.allocate(capacity, rows, blocks, device)
                storage = prefill_storage if mode == "prefill" else decoder_storage
                scratch[name] = storage.rows(capacity, mode, sample_rows=rows)
                feedback = torch.empty((capacity, 4096), dtype=torch.bfloat16, device=device) if config.mtp else None
                decoders[name] = PreparedDecoder(weights, scratch[name], pool.decoder, metadata[name], mode=mode,
                    runtime=collective, vocabulary_runtime=vocabulary_collective, coordinator=coordinator,
                    session=session, name=name, workspace=arena, feedback=feedback, max_cache_pages=max_cache_pages,
                    context=context)
                batch = PreparedBatch(inputs, requests, metadata[name], mode=mode, shard=shard, pages=pages,
                                      shards=shards)
                forward = PreparedGeneration(decoders[name], batch, records, frame, sequence,
                    pool=pool, runtime=collective, coordinator=coordinator, session=session, name=f"{name}.generation", replicated_sampling=config.mtp)
                if speculation is not None:
                    from .speculative import PreparedPrefill

                    forward = PreparedPrefill(speculation, forward, feedback, name)
                add(name, forward, mode, rows, capacity if mode == "prefill" else 1, forward_inputs)
        if speculation is not None:
            from .speculative import PreparedRound

            for batch, width in config.speculative_shapes:
                name = f"speculative{batch}x{width}"
                operation = PreparedRound(speculation, batch, width, name)
                operation.prepare()
                add(name, operation, "speculative", batch, width, forward_inputs)

        row_inputs = ("row_slot_ids", "row_generations", "live_rows")
        for capacity in config.batch_capacities:
            name = f"resume{capacity}"
            resume = PreparedResume(weights, requests, pool, inputs, records, frame, sequence,
                rows=capacity, runtime=collective, vocabulary_runtime=vocabulary_collective,
                coordinator=coordinator, session=session, name=name, replicated_sampling=config.mtp)
            add(name, resume, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"))
            for mode in ("init", "restore", "snapshot"):
                name = f"{mode}{capacity}"
                operation = PreparedMaintenance(mode, requests, pool, commands, inputs, records, frame, sequence,
                    runtime=collective, coordinator=coordinator, session=session, name=name,
                    capacity=capacity, admission_buffers=admission,
                    draft_tables=None if speculation is None else speculation.draft_tables,
                    page_tokens=config.page_tokens)
                fields = row_inputs if mode == "snapshot" else tuple(admission.__dataclass_fields__)
                add(name, operation, mode, capacity, 1, (*fields, "page_tables", *(("draft_page_tables",) if config.mtp else ()), *commands.buffers()))
    with phase(progress, "Prepare and tune kernels") as stage, transient:
        tuning = session.tune(show_progress=progress is not None and rank == 0, console=stage.console)
    report(f"selected kernels: {tuning['selection_counts']}")
    report(f"after tuning: free={torch.cuda.mem_get_info(device)[0] >> 20} MiB, "
           f"allocated={torch.cuda.memory_allocated(device) >> 20} MiB, reserved={torch.cuda.memory_reserved(device) >> 20} MiB")
    stream = torch.cuda.Stream(device=device)
    graph_pool = torch.cuda.graph_pool_handle() if package_storage is None else package_storage.graph_pool.id

    def rebind_pages(final_pages, *, compiled_only=False):
        for name, forward in forwards.items():
            if compiled_only and signatures[name].kind not in _COMPILED_KINDS:
                continue
            if hasattr(forward, "rebind_pages"):
                forward.rebind_pages(final_pages)

    # Graphs captured over the preparation pages measure everything except the final page pools.
    measured, _ = capture_forwards(forwards, signatures, session=session, device=device, stream=stream,
        pool=graph_pool, collective=collective, coordinator=coordinator, compile_decoder=False,
        progress=progress, label="Measure graph memory", cache_root=config.kernel_cache)
    free = torch.cuda.mem_get_info(device)[0]
    report(f"after graph measurement: free={free >> 20} MiB, allocated={torch.cuda.memory_allocated(device) >> 20} MiB, "
           f"reserved={torch.cuda.memory_reserved(device) >> 20} MiB, headroom={config.headroom_bytes >> 20} MiB")
    available = minimum_across_ranks(free - config.headroom_bytes - _KV_ALLOCATION_MARGIN, group)
    pages = available // page_bytes if config.pages is None else config.pages
    if pages < (config.minimum_pages or 1) or pages * page_bytes > available:
        raise MemoryError(f"prepared graphs leave {max(available, 0) / (1 << 30):.2f} GiB for KV pages; "
                          f"{max(pages, config.minimum_pages or 1)} pages need "
                          f"{max(pages, config.minimum_pages or 1) * page_bytes / (1 << 30):.2f} GiB")
    if config.compile_decoder:
        from b12x._lib.compile_plan import compile_only_launches

        # Compiler workers trace the final pool shapes while the pools hold no device memory.
        with compile_only_launches():
            pool.replace_pages(pages, metadata_mode=deferred_mode(), ranks=context_ranks)
            rebind_pages(pages, compiled_only=True)

    shared_pages = []

    def materialize():
        with phase(progress, "Allocate KV cache") as stage:
            from ignition.components.shared_pages import SharedPages

            shared_pages.extend(pool.replace_pages(pages, ranks=context_ranks,
                                                   share=lambda nbytes: SharedPages(group, device, nbytes, ipc=ipc)))
            rebind_pages(pages)
            for graph in measured.values():
                graph.reset()
            measured.clear()
            torch.cuda.synchronize(device)
            stage.update(f"{pages * page_bytes / (1 << 30):.2f} GiB per GPU · {pages * config.page_tokens:,} tokens")

    graphs, compiled = capture_forwards(forwards, signatures, session=session, device=device,
        stream=stream, pool=graph_pool, collective=collective, coordinator=coordinator,
        compile_decoder=config.compile_decoder, materialize=materialize, progress=progress,
        cache_root=config.kernel_cache, retain_graphs=config.retain_graphs)
    kv_bytes = pages * page_bytes
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    memory = {"rank": rank, "world_size": shard.world_size, "device": config.devices[rank],
        "weights_bytes": weights.allocated_bytes, "continuation_bytes": pool.backing.nbytes,
        "snapshot_capacity": snapshots, "pages": pages, "page_bytes": kv_bytes // pages, "kv_bytes": kv_bytes,
        "operator_scratch_bytes": arena.nbytes, "graphs": len(graphs), "prefill_capacities": config.prefill_capacities,
        "free_device_bytes": free, "total_device_bytes": total, "headroom_bytes": config.headroom_bytes,
        "allocated_bytes": torch.cuda.memory_allocated(device), "reserved_bytes": torch.cuda.memory_reserved(device),
        "tuning": tuning, "compiled_decoder": config.compile_decoder}
    assets = {role: {"path": str(snapshot / filename), "sha256": hashlib.sha256((snapshot / filename).read_bytes()).hexdigest()}
        for role, filename in (("tokenizer", "tokenizer.json"), ("tokenizer_config", "tokenizer_config.json"),
                               ("chat_template", "chat_template.jinja"), ("generation_config", "generation_config.json"))}
    generation = json.loads((snapshot / "generation_config.json").read_text())
    server_config = {"identity": identity, "model_id": snapshot.name, "max_sequence_tokens": config.context,
        "prefill_chunk_size": config.prefill_chunk_size, "pages": pages, "snapshots": snapshots,
        "eos_token_ids": generation["eos_token_id"], "assets": assets,
        "model_contract": {"schema": "glm5next.graph-inputs.v1", "vocabulary": 154880,
            "vision": {"features_per_request": 0, "max_images": 0, "patches_per_feature": 0, "patch_capacities": []},
            **({"page_tokens": config.page_tokens} if config.context_parallel else {})}}
    if config.mtp:
        server_config.update(speculative_widths=config.speculative_widths,
            speculative_fixed_width=config.speculative_fixed_width, speculative_graph_budget=config.speculative_graph_budget)
        memory.update(speculative_shapes=config.speculative_shapes)
    resources = (weights, pool, requests, inputs, admission, commands, arena, forwards, decoders,
        decoder_storage, prefill_storage, scratch, metadata, compiled, stream, graph_pool, session, collective,
        vocabulary_collective, coordinator, speculation, context)
    exchanges = () if context is None else (context.heads, context.candidates)
    # Operator scratch holds nothing between launches; a package stores it as zeros.
    disposable = ((arena.tensor.data_ptr(), arena.tensor.nbytes),) if arena.nbytes else ()
    return PreparedModel(device=config.devices[rank], graphs=graphs, buffers=buffers, signatures=signatures,
        max_active=config.max_active, resources=resources, session=session, export_zero_ranges=disposable,
        close_resources=(*exchanges, *shared_pages, vocabulary_collective, collective),
        server_config=server_config, memory_report=memory)
