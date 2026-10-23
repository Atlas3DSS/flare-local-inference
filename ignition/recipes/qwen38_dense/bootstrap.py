"""Prepare a single-GPU dense Qwen text graph catalog with device-resident state."""

import contextlib
from dataclasses import dataclass, replace
from functools import partial
from types import SimpleNamespace
import hashlib
import json
from pathlib import Path

import torch

from ignition.components.completion import CompletionOutputs
from ignition.components.prompt_logprobs import (
    PreparedScoredPrefill, PromptScoring, prefill_modes,
)

from ignition.components.capacity import batch_capacities, prefill_capacities, request_capacity, speculative_shapes, speculative_token_capacity
from ignition.components.gdn_recovery import GDNLayerRecords
from ignition.components.speculation import SpeculativeRngState
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
from ignition.preparation.frontend import annotated_template_asset, prepare_frontend
from ignition.preparation.fixtures.recurrent import register_benchmark_state
from ignition.preparation.host_memory import release_unused
from ignition.preparation.model import PreparedModel
from ignition.preparation.scratch import GraphBufferPool, ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession, with_autotune_effort
from ignition.preparation.tuning_replicas import TuningReplica
from ignition.progress import phase

from .config import PAGE_TOKENS
from .contract import prepare as prepare_contract
from .decoder import PreparedDecoder, Scratch
from .pool import ContinuationPool, continuation_layout, page_bytes
from .weights import ModelWeights
from .mtp import PreparedFeedback, PreparedRound


@dataclass(frozen=True)
class PreparationConfig:
    prompt_logprobs: bool = False
    compile_workers: int | None = None
    autotune_effort: str = "medium"
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
    mtp: bool = True
    speculative_widths: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    speculative_fixed_width: int | None = None
    speculative_token_capacity: int | None = 128

    def __post_init__(self):
        request_capacity(self.max_active)
        if not PAGE_TOKENS <= self.context <= 262144 or self.context % PAGE_TOKENS:
            raise ValueError("Qwen dense context must be 128-aligned in [128,262144]")
        if not 128 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 128:
            raise ValueError("Qwen dense prefill chunk must be 128-aligned and fit the context")
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0:
            raise ValueError("snapshot budget must be positive and headroom nonnegative")
        if type(self.mtp) is not bool:
            raise TypeError("MTP selection must be a boolean")
        speculative_shapes(self.max_active, self.speculative_widths, self.speculative_token_capacity)
        if self.speculative_fixed_width is not None and self.speculative_fixed_width not in self.speculative_widths:
            raise ValueError("fixed verification width must belong to the prepared catalog")

    @property
    def speculative_shapes(self):
        return (speculative_shapes(self.max_active, self.speculative_widths, self.speculative_token_capacity)
                if self.mtp else ())


@torch.no_grad()
@with_autotune_effort
def prepare(snapshot, config, *, progress=None, package_storage):
    from .recipe import RECIPE

    snapshot = Path(snapshot).resolve()
    device = torch.device("cuda", config.device)
    torch.cuda.set_device(device)
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
    transient = contextlib.ExitStack()
    transient.enter_context(package_storage.transient())
    weights = ModelWeights.load(manifest, device, progress=progress, mtp=config.mtp)
    release_unused()
    layout = continuation_layout(weights.config, mtp=config.mtp)
    snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
    if snapshots < 1:
        raise ValueError("snapshot budget must hold a complete Qwen dense continuation")
    slots, blocks = request_capacity(config.max_active), config.context // PAGE_TOKENS
    shapes = config.speculative_shapes
    decode_tokens = max((b * w for b, w in shapes), default=slots)
    state_rows = slots + snapshots
    pages = config.prefill_chunk_size // PAGE_TOKENS + 2 * slots
    total_memory = torch.cuda.mem_get_info(device)[1]
    with phase(progress, "Allocate model state"):
        pool = ContinuationPool.for_model(weights.config, state_rows + 1, pages, device,
            mtp=config.mtp, storage=package_storage, maximum_pages=total_memory // page_bytes(weights.config, mtp=config.mtp),
            maximum_rows=total_memory // layout.row_bytes, backed_slots=min(state_rows + 1, 1 + 2 * slots))
        vocabulary = weights.config.vocabulary
        requests = DeviceRequestState.allocate(device, vocab_size=vocabulary, capacity=slots)
        inputs = BatchInputs.allocate(max(config.prefill_chunk_size, slots), blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=vocabulary, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        completion = CompletionOutputs.allocate((2, slots, 24 if config.mtp else 8), device=device)
        coordinator = None
        identity = hashlib.sha256(json.dumps({
            "config": weights.config.text, "headers": [s.header_sha256 for s in manifest.shards],
            "contract": "qwen38-dense.graph-inputs.v1"}, sort_keys=True).encode()).hexdigest()
        session = RecordedPreparationSession(
            compile_workers=config.compile_workers, autotune_effort=config.autotune_effort, tuning_replica=config.tuning_replica,
            device=device, checkpoint=identity,
            workload_schema="qwen38-dense", numerical_policy="default", coordinator=coordinator,
            headroom_bytes=config.headroom_bytes, benchmark_storage=package_storage)
        session.__enter__()
        register_benchmark_state(session, pool.reserved_backing, storage=pool.reserved_backing,
                                 restore=pool.clear_for_preparation)
        arena = ScratchArena(device)
        session.register_scratch(arena)
        graph_buffers, private_buffers = GraphBufferPool(device), GraphBufferPool(device)
        prefill_storage = Scratch.allocate(config.prefill_chunk_size, 1, weights.config, device, buffers=graph_buffers.cursor())
        decode_storage = Scratch.allocate(decode_tokens, decode_tokens, weights.config, device, buffers=graph_buffers.cursor())
        buffers = dict(inputs.buffers(), **commands.buffers(),
            **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing, memory_capacity=pool.memory_capacity,
            request_pending_token=requests.pending_token,
            sampling_grammar_mask=requests.grammar_mask, frame_index=frame,
            submission_sequence=sequence, **completion.buffers())
        speculation = None
        if config.mtp:
            draft_tables = buffers["draft_page_tables"] = torch.full_like(inputs.page_tables, -1)
            records = GDNLayerRecords.allocate((i for i, kind in enumerate(weights.config.layers)
                if kind == "linear_attention"), slots, device, key_heads=16, value_heads=48)
            speculation = SimpleNamespace(weights=weights, pool=pool, inputs=inputs, requests=requests,
                scratch=decode_storage, draft_scratch=None, records=records, draft_tables=draft_tables,
                session=session, workspace=arena, collectives=dict(runtime=None, vocabulary_runtime=None, coordinator=None),
                overlap=False, completion=completion, frame=frame, sequence=sequence,
                rng=SpeculativeRngState.allocate(device, capacity=slots),
                offset=torch.zeros(slots, dtype=torch.int64, device=device),
                generation=torch.zeros(slots, dtype=torch.int64, device=device), sampling_scratch=None)
    scoring = PromptScoring.allocate(config, device, buffers, frame=frame)
    forwards, signatures, metadata = {}, {}, {}
    prefill_sizes, batch_sizes = prefill_capacities(config.prefill_chunk_size), batch_capacities(config.max_active)
    modes = (True, False) if config.mtp else (False,)
    with phase(progress, "Construct graph catalog"):
        def add(name, forward, kind, rows, tokens, fields, enabled):
            if kind == "scored_prefill":
                forward = PreparedScoredPrefill(forward, requests, inputs)
                fields = (*fields, "prompt_target_tokens")
            forwards[name] = forward
            signatures[name] = GraphSignature(kind, rows, tokens, tuple(fields), speculation_enabled=enabled)

        forward_inputs = (*inputs.buffers(), "sampling_grammar_mask")
        for enabled in modes:
            suffix = "-no-mtp" if config.mtp and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            fields = (*forward_inputs, *(("draft_page_tables",) if enabled else ()))
            for kind, mode, sizes in prefill_modes(prefill_sizes, () if enabled else reversed(batch_sizes), scoring):
                for capacity in sizes:
                    rows = 1 if mode == "prefill" else capacity
                    name = f"{kind}{capacity}" + (suffix if mode == "prefill" else "")
                    private = private_buffers.cursor()
                    m = metadata[name] = Metadata.allocate(capacity, rows, blocks, device, buffers=private)
                    scratch = (prefill_storage if mode == "prefill" else decode_storage).rows(capacity, rows)
                    decoder = PreparedDecoder(mode_weights, scratch, m, pool, mode=mode,
                        scoring=scoring if kind == "scored_prefill" else None, buffers=private,
                        vocabulary_runtime=None, coordinator=coordinator, session=session, name=name, workspace=arena)
                    batch = PreparedBatch(inputs, requests, m, mode=mode, shard=weights.shard, pool=pool, buffers=private)
                    context = partial(PreparedFeedback, inputs=inputs, tables=draft_tables, buffers=private) if enabled else None
                    forward = PreparedGeneration(decoder, batch, completion, frame, sequence, runtime=None,
                        coordinator=coordinator, session=session, name=name + ".generation", workspace=arena,
                        context_factory=context, buffers=private)
                    add(name, forward, kind, rows, capacity if mode == "prefill" else 1, fields, enabled)
        if speculation is not None:
            for batch, width in shapes:
                name = f"speculative{batch}x{width}"
                forward = PreparedRound(speculation, batch, width, name, buffers=private_buffers.cursor())
                forward.prepare()
                add(name, forward, "speculative", batch, width, (*forward_inputs, "draft_page_tables"), True)
        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for enabled in modes:
            suffix = "-no-mtp" if config.mtp and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            for capacity in reversed(batch_sizes):
                name = f"resume{capacity}" + suffix
                forward = PreparedResume(mode_weights, requests, pool, inputs, completion, frame, sequence,
                    rows=capacity, runtime=None, vocabulary_runtime=None, coordinator=coordinator,
                    session=session, name=name, activation_mode="auto", buffers=private_buffers.cursor())
                add(name, forward, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"), enabled)
                for mode in ("init", "restore", "snapshot"):
                    name = f"{mode}{capacity}" + suffix
                    forward = PreparedMaintenance(mode, requests, pool, commands, inputs, completion, frame, sequence,
                        runtime=None, coordinator=coordinator, session=session, name=name, capacity=capacity,
                        admission_buffers=admission, draft_state=enabled, buffers=private_buffers.cursor(),
                        trailing_pages=pool.draft_pages() if enabled else (), trailing_tables=draft_tables if enabled else None)
                    fields = row_inputs if mode == "snapshot" else ("request_state_rows", *admission.__dataclass_fields__)
                    add(name, forward, mode, capacity, 1,
                        (*fields, "page_tables", *(("draft_page_tables",) if enabled else ()), *commands.buffers()), enabled)
    with phase(progress, "Prepare and tune kernels") as stage, transient:
        tuning = session.tune(show_progress=progress is not None, console=stage.console)
    stream, graph_pool = torch.cuda.Stream(device=device), package_storage.graph_pool.id
    graphs, compiled = capture_forwards(forwards, signatures, session=session, device=device,
        stream=stream, pool=graph_pool, collective=None, coordinator=coordinator,
        compile_decoder=config.compile_decoder, progress=progress, cache_root=config.kernel_cache,
        retain_graphs=config.retain_graphs, materialize=pool.clear_for_preparation, warmup_storage=package_storage,
        compile_kinds=("prefill", "scored_prefill", "decode", "resume", "speculative"))
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    release_unused()
    assets = {role: {"path": str(snapshot / file)}
              for role, file in (("tokenizer", "tokenizer.json"),)}
    assets["chat_template"] = annotated_template_asset(
        (snapshot / "chat_template.jinja").read_bytes(), Path(__file__).parents[1] / "qwen38" / "assets" / "chat_template.jinja")
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
    if config.mtp:
        server_config.update(speculative_widths=config.speculative_widths,
            speculative_fixed_width=config.speculative_fixed_width,
            speculative_token_capacity=speculative_token_capacity(
                config.max_active, max(config.speculative_widths), config.speculative_token_capacity))
    free, total = torch.cuda.mem_get_info(device)
    memory = dict(rank=0, world_size=1, device=config.device, weights_bytes=weights.allocated_bytes,
        mapped_host_bytes=0, offloaded_bytes=0, continuation_bytes=pool.backing.nbytes, state_rows=state_rows,
        preparation_state_rows=int(pool.memory_capacity[0].item()),
        preparation_state_bytes=int(pool.memory_capacity[0].item()) * layout.row_bytes,
        pages=pages, page_bytes=page_bytes(weights.config, mtp=config.mtp), page_tokens=PAGE_TOKENS,
        kv_bytes=pages * page_bytes(weights.config, mtp=config.mtp), completion_output_bytes=completion.nbytes, operator_scratch_bytes=arena.nbytes, graphs=len(graphs),
        prefill_capacities=prefill_sizes, free_device_bytes=free, total_device_bytes=total,
        headroom_bytes=config.headroom_bytes, tuning=tuning, compiled_decoder=config.compile_decoder,
        speculative_shapes=shapes, graph_buffer_bytes=graph_buffers.nbytes, private_buffer_bytes=private_buffers.nbytes)
    if scoring is not None:
        memory.update(scoring.memory_report())
    return PreparedModel(rebuild_weights={"weights": weights}, device=config.device, graphs=graphs, buffers=buffers, signatures=signatures,
        max_active=config.max_active, session=session, server_config=server_config, memory_report=memory,
        close_resources=(() if scoring is None else (scoring,)),
        export_zero_ranges=(*(() if scoring is None else scoring.zero_ranges()),
                           *storage_ranges([prefill_storage, decode_storage, arena.tensor, metadata,
                                            graph_buffers.chunks, private_buffers.chunks]), *completion.zero_ranges()),
        resources=(scoring, weights, pool, requests, inputs, admission, commands, arena, forwards, prefill_storage,
                   decode_storage, metadata, compiled, stream, graph_pool, session, coordinator,
                   graph_buffers, private_buffers, speculation))
