"""Prepare Kimi K3 text and image graphs with tensor-parallel decoder storage."""

import contextlib
from dataclasses import dataclass, replace
import hashlib
import json
from tempfile import TemporaryDirectory
from pathlib import Path

import torch

from ignition.components.capacity import batch_capacities, prefill_capacities, request_capacity, speculative_shapes, speculative_token_capacity
from ignition.components.parallel import TensorParallelShard
from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.components.paged_generation import PreparedGeneration
from ignition.components.paged_maintenance import PreparedMaintenance
from ignition.components.paged_resume import PreparedResume
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from ignition.components.state_transfer import StateCommands
from ignition.preparation.collectives import CollectivePreparation, create_all_gather_runtime, create_all_reduce_runtime
from ignition.preparation.rank_group import minimum_across_ranks
from ignition.preparation.capture import capture_forwards
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.disposable import storage_ranges
from ignition.preparation.execution import GraphSignature
from ignition.preparation.frontend import prepare_frontend
from ignition.preparation.fixtures.recurrent import register_benchmark_state
from ignition.preparation.host_memory import release_unused
from ignition.preparation.model import PreparedModel
from ignition.preparation.scratch import ScratchArena, scratch_views
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.progress import phase

from .config import PAGE_TOKENS
from .contract import prepare as prepare_contract
from .decoder import DecoderStorage, PreparedDecoder, Scratch
from .pool import ContinuationPool, continuation_layout, page_bytes
from .weights import ModelWeights
from .vision import PATCH_CAPACITIES, FEATURES, HIDDEN, PreparedVision, VisionInputs, VisionScratch
from .vision_state import PreparedVisionGeneration
from ignition.components.vision_state import PrefillVisionInputs, VisionState


@dataclass(frozen=True)
class PreparationConfig:
    devices: tuple[int, ...] = tuple(range(16))
    context: int = 1048576
    prefill_chunk_size: int = 4096
    max_active: int = 4
    snapshot_budget_bytes: int = 4 << 30
    headroom_bytes: int = 4 << 30
    compile_decoder: bool = True
    retain_graphs: bool = False
    kernel_cache: Path | None = None
    draft_checkpoint: Path | None = None
    speculative_widths: tuple[int, ...] = tuple(range(1, 9))
    speculative_fixed_width: int | None = None
    speculative_token_capacity: int | None = None

    @property
    def speculative_shapes(self):
        return (speculative_shapes(self.max_active, self.speculative_widths, self.speculative_token_capacity)
                if self.draft_checkpoint is not None else ())

    @property
    def decode_tokens(self):
        return max((b * w for b, w in self.speculative_shapes), default=request_capacity(self.max_active))

    def __post_init__(self):
        if len(self.devices) != 16 or len(set(self.devices)) != 16 or any(d < 0 for d in self.devices):
            raise ValueError("Kimi K3 preparation requires sixteen distinct GPUs")
        request_capacity(self.max_active)
        if not self.speculative_widths or max(self.speculative_widths) > 8:
            raise ValueError("Kimi DFlash2 supports at most seven draft tokens")
        if self.speculative_fixed_width is not None and self.speculative_fixed_width not in self.speculative_widths:
            raise ValueError("fixed verification width must belong to the captured widths")
        _ = self.speculative_shapes
        if not PAGE_TOKENS <= self.context <= 1048576 or self.context % PAGE_TOKENS:
            raise ValueError("Kimi K3 context must be 128-aligned in [128,1048576]")
        if not 128 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 128:
            raise ValueError("Kimi K3 prefill chunk must be 128-aligned and fit the context")
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0:
            raise ValueError("snapshot budget must be positive and headroom nonnegative")


@torch.no_grad()
def prepare_rank(snapshot, config, *, rank, group, progress=None, package_storage):
    from .recipe import RECIPE

    snapshot = Path(snapshot).resolve()
    shard = TensorParallelShard(rank, len(config.devices))
    device = torch.device("cuda", config.devices[rank])
    torch.cuda.set_device(device)
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
        headers = [shard.header_sha256 for shard in manifest.shards]
    with package_storage.transient():
        weights = ModelWeights.load(manifest, device, shard=shard, progress=progress, draft_checkpoint=config.draft_checkpoint)
        del manifest
        release_unused()
    transient = contextlib.ExitStack()
    transient.enter_context(package_storage.transient())
    layout = continuation_layout(weights.config, shard.world_size)
    snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
    if snapshots < 1:
        raise ValueError("snapshot budget must hold a complete Kimi K3 continuation")
    slots, blocks = request_capacity(config.max_active), config.context // PAGE_TOKENS
    drafting = weights.draft is not None
    draft_rows = slots * weights.config.draft.block_size if drafting else 0
    max_tokens = max(config.prefill_chunk_size, config.decode_tokens, draft_rows)
    state_rows = slots + snapshots
    pages = config.prefill_chunk_size // PAGE_TOKENS + 2 * slots
    total_memory = minimum_across_ranks(torch.cuda.mem_get_info(device)[1], group)
    with phase(progress, "Allocate model state"):
        pool = ContinuationPool.for_model(weights.config, shard.world_size, state_rows + 1, pages, device,
            storage=package_storage, maximum_pages=total_memory // page_bytes(weights.config),
            maximum_rows=total_memory // layout.row_bytes, backed_slots=min(state_rows + 1, 1 + 2 * slots))
        vocabulary = weights.config.vocabulary
        requests = DeviceRequestState.allocate(device, vocab_size=vocabulary, capacity=slots)
        inputs = BatchInputs.allocate(max_tokens, blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=vocabulary, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        records = torch.zeros((2, slots, 24 if drafting else 8), dtype=torch.int64, device=device)
        coordinator = CollectivePreparation(group)
        collective = create_all_reduce_runtime(group, device, max_tokens=max_tokens,
            hidden_size=weights.config.hidden, ipc=package_storage.peer_storage)
        vocabulary_collective = create_all_gather_runtime(group, device, max_rows=max(config.decode_tokens, draft_rows),
            columns=weights.config.vocabulary, ipc=package_storage.peer_storage)
        embedding_collective = create_all_gather_runtime(group, device, max_rows=max_tokens,
            columns=weights.config.hidden, ipc=package_storage.peer_storage)
        router_collective = create_all_gather_runtime(group, device, max_rows=max_tokens,
            columns=2 * shard.world_size * ((weights.config.text["num_experts"] // shard.world_size + 31) // 32 * 32),
            ipc=package_storage.peer_storage)
        identity = hashlib.sha256(json.dumps({
            "config": weights.config.text, "headers": headers,
            "vision": weights.config.vision, "draft": None if not drafting else weights.config.draft.text, "world_size": shard.world_size, "contract": "kimi3.graph-inputs.v1"}, sort_keys=True).encode()).hexdigest()
        session = RecordedPreparationSession(device=device, checkpoint=identity,
            workload_schema="kimi3", numerical_policy="default", coordinator=coordinator,
            headroom_bytes=config.headroom_bytes, benchmark_storage=package_storage)
        session.__enter__()
        register_benchmark_state(session, pool.reserved_backing, storage=pool.reserved_backing,
                                 restore=pool.clear_for_preparation)
        arena = ScratchArena(device, overlap=False)
        session.register_scratch(arena)
        with package_storage.scratch_pool() as scratch_pool:
            # Vision and text graphs execute serially and share disposable workspace.
            text_shape = (max(config.prefill_chunk_size, config.decode_tokens), weights.config.hidden)
            text_shapes = (text_shape,) * 5
            if drafting:
                text_shapes += ((text_shape[0], len(weights.config.draft.target_layers) * text_shape[1]),)
            shared_scratch = torch.zeros(max(PATCH_CAPACITIES[-1] * 3 * weights.config.vision["qkv_hidden_size"],
                sum(rows * columns for rows, columns in text_shapes)), dtype=torch.bfloat16, device=device)
            activation, total, kda_output, mla_output, moe_output, *auxiliary = scratch_views(shared_scratch, text_shapes)
            prefill_storage = Scratch.allocate(config.prefill_chunk_size, 1, weights.config, shard.world_size, device,
                                               activations=activation[:config.prefill_chunk_size])
            decode_storage = Scratch.allocate(config.decode_tokens, config.decode_tokens, weights.config, shard.world_size, device)
            vision_inputs = VisionInputs.allocate(device)
            vision_storage = VisionScratch.allocate(PATCH_CAPACITIES[-1], device, work=shared_scratch)
        vision_prefill = PrefillVisionInputs.allocate(config.prefill_chunk_size, device)
        vision_state = VisionState(inputs, vision_prefill, requests, image_token=163605,
                                   features=FEATURES, hidden=HIDDEN // shard.world_size)
        buffers = dict(inputs.buffers(), **vision_inputs.buffers(), **vision_prefill.buffers(), **commands.buffers(),
            **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing, memory_capacity=pool.memory_capacity,
            request_pending_token=requests.pending_token,
            sampling_grammar_mask=requests.grammar_mask, frame_index=frame,
            submission_sequence=sequence, completion_records=records)
    forwards, signatures, metadata = {}, {}, {}
    decoder_buffers = DecoderStorage.allocate(weights.config, shard, device, scratch={
        "total": total, "kda": {"output": kda_output},
        "mla": {"output": mla_output}, "moe": {"output": moe_output},
        **({"auxiliary": auxiliary[0]} if drafting else {})})
    prefill_sizes, batch_sizes = prefill_capacities(config.prefill_chunk_size), batch_capacities(config.max_active)
    collectives = dict(runtime=collective, embedding_runtime=embedding_collective, router_runtime=router_collective,
        vocabulary_runtime=vocabulary_collective, coordinator=coordinator, vision_state=vision_state)
    speculation = None
    if drafting:
        from .dflash import SpeculativePreparation
        # Context injection follows target execution and retains only auxiliary features.
        context_buffers = dict(hidden=activation,
            local=total.flatten()[:text_shape[0] * text_shape[1] // shard.world_size].view(text_shape[0], -1),
            kv=kda_output.flatten()[:text_shape[0] * 576].view(text_shape[0], 576),
            latent=mla_output.flatten()[:text_shape[0] * 576].view(text_shape[0], 576))
        speculation = SpeculativePreparation(weights, pool, requests, inputs, decode_storage,
            records=records, frame=frame, sequence=sequence, session=session, workspace=arena,
            collectives=collectives, decoder_buffers=decoder_buffers, context_buffers=context_buffers)
    modes = (True, False) if drafting else (False,)
    count = len(modes) * (len(prefill_sizes) + 4 * len(batch_sizes)) + len(batch_sizes) + len(config.speculative_shapes) + len(PATCH_CAPACITIES)
    with phase(progress, "Construct graph catalog", total=count) as stage:
        def add(name, forward, kind, rows, tokens, fields, *, enabled=None):
            forwards[name] = forward
            signatures[name] = GraphSignature(kind, rows, tokens, tuple(fields), speculation_enabled=enabled)
            stage.update(advance=1)

        for enabled in modes:
            mode_weights = weights if enabled else replace(weights, draft=None)
            suffix = "-no-mtp" if drafting and not enabled else ""
            for mode, sizes in (("prefill", reversed(prefill_sizes)), ("decode", () if enabled else reversed(batch_sizes))):
                for capacity in sizes:
                    rows = 1 if mode == "prefill" else capacity
                    name = f"{mode}{capacity}" + (suffix if mode == "prefill" else "")
                    m = metadata[name] = Metadata.allocate(capacity, rows, blocks, device)
                    scratch = (prefill_storage if mode == "prefill" else decode_storage).rows(capacity, rows)
                    decoder = PreparedDecoder(mode_weights, scratch, m, pool, mode=mode,
                        session=session, name=name, workspace=arena, buffers=decoder_buffers, **collectives)
                    batch = PreparedBatch(inputs, requests, m, mode=mode, shard=weights.shard, pool=pool,
                        validate_prefill=vision_state.validate_prefill if mode == "prefill" else None)
                    forward = PreparedGeneration(decoder, batch, records, frame, sequence, runtime=collective,
                        coordinator=coordinator, session=session, name=name + ".generation", workspace=arena,
                        context_factory=None if speculation is None else speculation.forwards.context)
                    add(name, forward, mode, rows, capacity if mode == "prefill" else 1,
                        (*inputs.buffers(), "sampling_grammar_mask", *(vision_prefill.buffers() if mode == "prefill" else ())),
                        enabled=enabled)
        if speculation is not None:
            from ignition.components.parallel_draft import PreparedParallelDraftRound, prepare_sampling_scratch
            from ignition.preparation.scratch import bind_scratch

            sampling = prepare_sampling_scratch(slots, max(config.speculative_widths), vocabulary, device)
            retained = bind_scratch(PreparedParallelDraftRound.storage_layout(slots, max(config.speculative_widths), vocabulary), device=device)
            for batch, width in sorted(config.speculative_shapes, reverse=True):
                name = f"speculative{batch}x{width}"
                forward = PreparedParallelDraftRound(speculation, batch, width, name,
                    sampling_scratch=sampling, round_storage=retained)
                forward.prepare()
                add(name, forward, "speculative", batch, width, (*inputs.buffers(), "sampling_grammar_mask"), enabled=True)

        for capacity in PATCH_CAPACITIES:
            name = f"vision{capacity}"
            encoder = PreparedVision(weights.vision, vision_storage.rows(capacity), vision_inputs,
                session=session, workspace=arena, name=name)
            forward = PreparedVisionGeneration(encoder, vision_state, records, frame, sequence,
                                                feature_range=shard.partition(HIDDEN))
            add(name, forward, "vision", 1, capacity, vision_inputs.buffers())
        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for enabled in modes:
            mode_weights = weights if enabled else replace(weights, draft=None)
            suffix = "-no-mtp" if drafting and not enabled else ""
            for capacity in reversed(batch_sizes):
                name = f"resume{capacity}" + suffix
                forward = PreparedResume(mode_weights, requests, pool, inputs, records, frame, sequence,
                    rows=capacity, runtime=collective, vocabulary_runtime=vocabulary_collective, coordinator=coordinator,
                    session=session, name=name, activation_mode="auto")
                add(name, forward, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"), enabled=enabled)
                for mode in ("init", "restore", "snapshot"):
                    name = f"{mode}{capacity}" + suffix
                    forward = PreparedMaintenance(mode, requests, pool, commands, inputs, records, frame, sequence,
                        runtime=collective, coordinator=coordinator, session=session, name=name, capacity=capacity,
                        admission_buffers=admission, draft_state=enabled)
                    fields = row_inputs if mode == "snapshot" else ("request_state_rows", *admission.__dataclass_fields__)
                    add(name, forward, mode, capacity, 1, (*fields, "page_tables", *commands.buffers()), enabled=enabled)
    with phase(progress, "Prepare and tune kernels") as stage, transient:
        tuning = session.tune(show_progress=progress is not None, console=stage.console)
    stream, graph_pool = torch.cuda.Stream(device=device), package_storage.graph_pool.id
    if config.compile_decoder:
        scratch_pool.release_backing()

    def materialize():
        scratch_pool.materialize()
        vision_inputs.vision_slot.fill_(-1)
        pool.clear_for_preparation()

    # Large vision allocations must precede smaller text allocations in the shared graph pool.
    vision_names = tuple(f"vision{capacity}" for capacity in reversed(PATCH_CAPACITIES))
    capture_order = (*vision_names, *(name for name in forwards if name not in vision_names))
    capture_catalog = {name: forwards[name] for name in capture_order}
    try:
        graphs, compiled = capture_forwards(capture_catalog, signatures, session=session, device=device,
            stream=stream, pool=graph_pool, collective=collective, coordinator=coordinator,
            compile_decoder=config.compile_decoder, progress=progress, cache_root=config.kernel_cache,
            retain_graphs=config.retain_graphs, materialize=materialize,
            warmup_storage=package_storage,
            compile_kinds=("prefill", "decode", "resume", "vision", "speculative"))
    finally:
        scratch_pool.materialize()
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    release_unused()
    from .tokenizer import convert
    frontend_assets = TemporaryDirectory(prefix="flare-kimi3-frontend-")
    assets_path = Path(frontend_assets.name)
    convert(snapshot).save(str(assets_path / "tokenizer.json"))
    template = Path(__file__).with_name("assets") / "chat_template.jinja"
    (assets_path / "chat_template.jinja").write_bytes(template.read_bytes())
    assets = {role: {"path": str(assets_path / file), "sha256": hashlib.sha256((assets_path / file).read_bytes()).hexdigest()}
              for role, file in (("tokenizer", "tokenizer.json"), ("chat_template", "chat_template.jinja"))}
    eos = weights.config.text["eos_token_id"]
    eos_ids = list(dict.fromkeys([*(eos if isinstance(eos, list) else [eos]), 163586]))
    frontend = prepare_frontend(snapshot, eos_token_ids=eos_ids, image_profile="moonvit")
    frontend["template_encoding"] = 1
    server_config = dict(identity=identity, model_id=snapshot.name, max_sequence_tokens=config.context,
        prefill_chunk_size=config.prefill_chunk_size, pages=pages, state_rows=state_rows, snapshots=snapshots,
        page_tokens=PAGE_TOKENS, frontend=frontend,
        sampling_defaults=RECIPE.sampling_defaults.as_dict(), assets=assets,
        model_contract=prepare_contract(signatures, buffers, pool, vocabulary=vocabulary, draft_lag=0 if drafting else None))
    free, total = torch.cuda.mem_get_info(device)
    memory = dict(rank=rank, world_size=shard.world_size, device=config.devices[rank], weights_bytes=weights.allocated_bytes,
        mapped_host_bytes=0, offloaded_bytes=0, continuation_bytes=pool.backing.nbytes, state_rows=state_rows,
        preparation_state_rows=int(pool.memory_capacity[0].item()),
        preparation_state_bytes=int(pool.memory_capacity[0].item()) * layout.row_bytes,
        pages=pages, page_bytes=page_bytes(weights.config), page_tokens=PAGE_TOKENS,
        kv_bytes=pages * page_bytes(weights.config), operator_scratch_bytes=arena.nbytes, graphs=len(graphs),
        prefill_capacities=prefill_sizes, free_device_bytes=free, total_device_bytes=total,
        headroom_bytes=config.headroom_bytes, tuning=tuning, compiled_decoder=config.compile_decoder)
    if drafting:
        server_config.update(speculative_widths=config.speculative_widths,
            speculative_token_capacity=speculative_token_capacity(config.max_active, max(config.speculative_widths), config.speculative_token_capacity),
            speculative_fixed_width=config.speculative_fixed_width)
        memory.update(speculative_shapes=config.speculative_shapes)
    draft_disposable = [] if speculation is None else [speculation.draft_scratch, speculation.context_buffers,
        speculation.kda_records.corrections, speculation.kda_records.keys_and_gates]
    return PreparedModel(device=config.devices[rank], graphs=graphs, buffers=buffers, signatures=signatures,
        max_active=config.max_active, session=session, server_config=server_config, memory_report=memory,
        export_zero_ranges=storage_ranges([prefill_storage, decode_storage, vision_storage, arena.tensor, metadata,
                                           decoder_buffers.scratch_tensors(), vision_state.arena, *draft_disposable]),
        resources=(weights, pool, requests, inputs, admission, commands, arena, forwards, prefill_storage,
                   decode_storage, metadata, compiled, stream, graph_pool, session, coordinator, collective,
                   vocabulary_collective, embedding_collective, router_collective, frontend_assets,
                   vision_inputs, vision_prefill, vision_state, vision_storage, scratch_pool, shared_scratch, speculation))
