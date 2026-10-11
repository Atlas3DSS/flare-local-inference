"""Prepare a GLM tensor-parallel rank before native serving takes ownership."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from flare.components.capacity import batch_capacities, request_capacity
from flare.components.parallel import TensorParallelShard
from flare.components.sampling import AdmissionBuffers, DeviceRequestState
from flare.components.state_transfer import StateCommands
from flare.preparation.capture import capture_forwards
from flare.preparation.checkpoint import inspect_indexed_checkpoint
from flare.preparation.collectives import (
    CollectivePreparation,
    create_all_gather_runtime,
    create_all_reduce_runtime,
)
from flare.preparation.execution import GraphSignature
from flare.preparation.fixtures.recurrent import register_benchmark_state
from flare.preparation.model import PreparedModel
from flare.preparation.scratch import ScratchArena
from flare.progress import phase
from flare.preparation.tuning import RecordedPreparationSession

from .batch import BatchInputs, PreparedBatch
from .generation import PreparedGeneration
from .maintenance import PreparedMaintenance
from .pool import ContinuationPool, continuation_layout
from .resume import PreparedResume
from .stack import DecoderScratch, PreparedDecoder
from .state import DecoderMetadata
from .weights import ModelWeights


@dataclass(frozen=True)
class PreparationConfig:
    devices: tuple[int, ...] = (0, 1)
    context: int = 262144
    prefill_chunk_size: int = 4096
    max_active: int = 4
    pages: int | None = None
    snapshot_budget_bytes: int = 4 << 30
    headroom_bytes: int = 4 << 30
    compile_decoder: bool = True

    def __post_init__(self):
        if type(self.compile_decoder) is not bool:
            raise TypeError("compile_decoder must be a boolean")
        if (not isinstance(self.devices, tuple) or len(self.devices) not in (2, 4)
                or len(set(self.devices)) != len(self.devices)
                or any(type(d) is not int or d < 0 for d in self.devices)):
            raise ValueError("GLM requires two or four distinct process-local CUDA devices")
        for name in ("context", "prefill_chunk_size", "max_active", "snapshot_budget_bytes", "headroom_bytes"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer")
        if not 256 <= self.context <= 1048576 or self.context % 256:
            raise ValueError("GLM context must be 256-aligned in [256,1048576]")
        if not 64 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 64:
            raise ValueError("prefill chunk size must be 64-aligned in [64,context]")
        request_capacity(self.max_active)
        if self.pages is not None and (type(self.pages) is not int or not 0 < self.pages <= 2147483647):
            raise ValueError("page capacity must be a positive signed 32-bit integer")
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0:
            raise ValueError("snapshot budget must be positive and headroom nonnegative")

    @property
    def prefill_capacities(self):
        sizes, capacity = [], 128
        while capacity < self.prefill_chunk_size:
            sizes.append(capacity)
            capacity *= 2
        return (*sizes, self.prefill_chunk_size)

    @property
    def batch_capacities(self):
        return tuple(reversed(batch_capacities(self.max_active)))

    @property
    def request_capacity(self):
        return request_capacity(self.max_active)

    @property
    def page_capacity(self):
        return self.context // 256 * self.max_active if self.pages is None else self.pages


@torch.no_grad()
def prepare_rank(snapshot: Path, config: PreparationConfig, *, rank, group, progress=None) -> PreparedModel:
    """Load, tune and capture all text variants on one supervised TP rank.

    The caller must terminate the complete rank group if any preparation fails.
    Successful cleanup requires every rank to close its IPC runtime together,
    after native replay drains. No eager execution path is exported.
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
    weights = ModelWeights.load(manifest, device, shard=shard, progress=progress)
    with phase(progress, "Allocate model state and collectives") as stage:
        layer_types = weights.tree.config.text_config.layer_types
        layout = continuation_layout(layer_types, shard.world_size)
        snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
        if snapshots < 1:
            raise ValueError("GLM snapshot budget must hold at least one continuation row")
        pages, blocks = config.page_capacity, config.context // 256
        slots = config.request_capacity
        pool = ContinuationPool.allocate(layer_types, shard.world_size, slots + 1 + snapshots, pages, device)
        kv_bytes = sum(cache.nbytes for _, cache in pool.partial_pages())
        stage.update(f"KV cache: {kv_bytes / (1 << 30):.2f} GiB per GPU · {pages * 256:,} tokens")
        requests = DeviceRequestState.allocate(device, vocab_size=154880, capacity=slots, consumed_tokens=pool.consumed_tokens[1:1 + slots, 0])
        inputs = BatchInputs.allocate(config.prefill_chunk_size, blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=154880, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        records = torch.zeros((2, slots, 8), dtype=torch.int64, device=device)
        coordinator = CollectivePreparation(group)
        collective = create_all_reduce_runtime(group, device, max_tokens=config.prefill_chunk_size, hidden_size=4096)
        vocabulary_collective = create_all_gather_runtime(group, device,
            max_rows=max(config.batch_capacities), columns=154880)
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
        decoder_storage = DecoderScratch.allocate(config.prefill_chunk_size, blocks, "decode", device,
            shard=shard, sample_rows=max(config.batch_capacities))

    graph_count = len(config.prefill_capacities) + 5 * len(config.batch_capacities)
    with phase(progress, "Construct graph catalog", total=graph_count) as stage:
        def add(name, function, kind, rows, tokens, fields):
            forwards[name] = function
            signatures[name] = GraphSignature(kind, rows, tokens, tuple(fields))
            stage.update(advance=1)

        for mode, capacities in (("prefill", reversed(config.prefill_capacities)), ("decode", config.batch_capacities)):
            for capacity in capacities:
                rows = 1 if mode == "prefill" else capacity
                name = f"{mode}{capacity}"
                metadata[name] = DecoderMetadata.allocate(capacity, rows, blocks, device)
                scratch[name] = decoder_storage.rows(capacity, mode, sample_rows=rows)
                decoders[name] = PreparedDecoder(weights, scratch[name], pool.decoder, metadata[name], mode=mode,
                    runtime=collective, vocabulary_runtime=vocabulary_collective, coordinator=coordinator,
                    session=session, name=name, workspace=arena)
                batch = PreparedBatch(inputs, requests, metadata[name], mode=mode, shard=shard, pages=pages)
                forward = PreparedGeneration(decoders[name], batch, records, frame, sequence,
                    pool=pool, runtime=collective, coordinator=coordinator, session=session, name=f"{name}.generation")
                add(name, forward, mode, rows, capacity if mode == "prefill" else 1, forward_inputs)
        row_inputs = ("row_slot_ids", "row_generations", "live_rows")
        for capacity in config.batch_capacities:
            name = f"resume{capacity}"
            resume = PreparedResume(weights, requests, pool, inputs, records, frame, sequence,
                rows=capacity, runtime=collective, vocabulary_runtime=vocabulary_collective,
                coordinator=coordinator, session=session, name=name)
            add(name, resume, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"))
            for mode in ("init", "restore", "snapshot"):
                name = f"{mode}{capacity}"
                operation = PreparedMaintenance(mode, requests, pool, commands, inputs, records, frame, sequence,
                    runtime=collective, coordinator=coordinator, session=session, name=name,
                    capacity=capacity, admission_buffers=admission)
                fields = row_inputs if mode == "snapshot" else tuple(admission.__dataclass_fields__)
                add(name, operation, mode, capacity, 1, (*fields, "page_tables", *commands.buffers()))
    with phase(progress, "Prepare and tune kernels") as stage:
        tuning = session.tune(show_progress=progress is not None and rank == 0, console=stage.console)
    report(f"selected kernels: {tuning['selection_counts']}")
    stream = torch.cuda.Stream(device=device)
    graph_pool = torch.cuda.graph_pool_handle()
    graphs, compiled = capture_forwards(forwards, signatures, session=session, device=device,
        stream=stream, pool=graph_pool, collective=collective,
        compile_decoder=config.compile_decoder, progress=progress)
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    if free < config.headroom_bytes:
        raise MemoryError(f"GLM rank {rank} has {free / (1 << 30):.3f} GiB free, "
                          f"below requested headroom {config.headroom_bytes / (1 << 30):.3f} GiB; "
                          f"KV cache: {kv_bytes / (1 << 30):.2f} GiB per GPU, {pages * 256:,} tokens; "
                          f"snapshots={snapshots}")
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
            "vision": {"features_per_request": 0, "max_images": 0, "patches_per_feature": 0, "patch_capacities": []}}}
    resources = (weights, pool, requests, inputs, admission, commands, arena, forwards, decoders,
        decoder_storage, scratch, metadata, compiled, stream, graph_pool, session, collective,
        vocabulary_collective, coordinator)
    return PreparedModel(device=config.devices[rank], graphs=graphs, buffers=buffers, signatures=signatures,
        max_active=config.max_active, resources=resources, session=session,
        close_resources=(vocabulary_collective, collective),
        server_config=server_config, memory_report=memory)
