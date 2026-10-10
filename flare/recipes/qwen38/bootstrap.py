"""Prepare the complete Qwen graph catalog before starting native serving."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from flare.components.sampling import AdmissionBuffers
from flare.preparation import inspect_qwen38_checkpoint
from flare.preparation.capture import capture_forwards
from flare.preparation.execution import GraphSignature
from flare.preparation.fixtures.recurrent import register_benchmark_state
from flare.preparation.model import PreparedModel
from flare.preparation.qwen38 import (
    CONFIG_SHA256,
    FRONTEND_SHA256,
    GENERATION_SHA256,
    HEADERS_SHA256,
    MODEL_ID,
    REVISION,
)
from flare.preparation.scratch import ScratchArena

from .attention import QSAState, RotaryTables
from .decoder import DecoderScratch, ModelWeights, PreparedDecoder
from .decoder_state import DecoderInputs, PreparedDecoderState, PreparedResume
from .maintenance import PreparedMaintenance
from .ple import MappedPLETable
from .sampling import DeviceRequestState
from .state import (
    SNAPSHOT_BASE,
    HybridStateLayout,
    HybridStatePool,
    StateTransferMetadata,
)
from .tuning import preparation_session
from .vision import PreparedVision, VisionScratch

_KV_ALLOCATION_MARGIN = 64 << 20


def _allocate_page_pools(cache, capacity, *, page_bytes, device, headroom_bytes, automatic, report):
    """Fit unbound KV pools to measured device consumption before graph capture."""
    while capacity > 0:
        pools = {index: value.resize_pages_for_preparation(capacity) for index, value in cache.items()}
        torch.cuda.synchronize(device)
        free = torch.cuda.mem_get_info(device)[0]
        report(f"KV allocation: pages={capacity}, free={free}, "
               f"allocated={torch.cuda.memory_allocated(device)}, reserved={torch.cuda.memory_reserved(device)}")
        if free >= headroom_bytes:
            return pools, capacity
        del pools
        torch.cuda.empty_cache()
        if not automatic:
            raise MemoryError(f"requested KV pages violate configured GPU headroom: pages={capacity}, "
                              f"free={free}, required={headroom_bytes}")
        reduction = (headroom_bytes - free + _KV_ALLOCATION_MARGIN + page_bytes - 1) // page_bytes
        capacity -= reduction
        report(f"adjusted automatic KV capacity to {capacity} pages after measured allocator overhead")
    raise MemoryError("prepared graph catalog leaves insufficient memory for KV pages and headroom")


@dataclass(frozen=True)
class PreparationConfig:
    device: int = 0
    context: int = 262144
    prefill_chunk_size: int = 4096
    snapshot_budget_bytes: int = 4 << 30
    headroom_bytes: int = 4 << 30
    pages: int | None = None
    mtp: bool = False
    speculative_proposal_source: str = "mtp"
    speculative_widths: tuple[int, ...] = tuple(range(1, 9))
    speculative_fixed_width: int | None = None
    history_drafts: bool = False

    def __post_init__(self):
        for name in ("device", "context", "prefill_chunk_size", "snapshot_budget_bytes", "headroom_bytes"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer")
        if self.device < 0 or not 64 <= self.context <= 262144 or self.context % 64:
            raise ValueError("device must be nonnegative and context must be 64-aligned in [64,262144]")
        if not 64 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 64:
            raise ValueError("prefill chunk size must be 64-aligned in [64,context]")
        if self.pages is not None and (type(self.pages) is not int or self.pages <= 0):
            raise ValueError("explicit page capacity must be a positive integer")
        if type(self.mtp) is not bool or type(self.history_drafts) is not bool:
            raise TypeError("MTP and history draft switches must be booleans")
        minimum_snapshot = HybridStateLayout.build(speculative_tokens=7 if self.mtp else 0).row_bytes
        if self.snapshot_budget_bytes < minimum_snapshot or self.headroom_bytes < 0:
            raise ValueError("snapshot budget requires at least one state row; headroom must be nonnegative")
        if self.speculative_proposal_source not in ("mtp", "copy"):
            raise ValueError("speculative proposal source must be mtp or copy")
        widths = self.speculative_widths
        if (not isinstance(widths, tuple) or not widths or widths[0] != 1
                or any(type(width) is not int or not 1 <= width <= 8 for width in widths)
                or tuple(sorted(set(widths))) != widths):
            raise ValueError("speculative widths must be an increasing subset of 1 through 8 including one")
        if self.speculative_fixed_width is not None and (
                type(self.speculative_fixed_width) is not int or self.speculative_fixed_width not in widths):
            raise ValueError("fixed verification width must belong to the prepared catalog")

    @property
    def speculative_shapes(self) -> tuple[tuple[int, int], ...]:
        return tuple((batch, width) for batch in (4, 2, 1)
                     for width in reversed(self.speculative_widths) if batch * width <= 8)

    @property
    def prefill_capacities(self) -> tuple[int, ...]:
        capacities = [1, 16]
        capacity = 64
        while capacity < self.prefill_chunk_size:
            capacities.append(capacity)
            capacity *= 2
        return (*capacities, self.prefill_chunk_size)



def _identity(manifest):
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(Path(__file__).parent.glob("*.py"))}
    components = Path(__file__).parents[2] / "components"
    sources.update({f"components/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(components.glob("*.py"))})
    value = {"checkpoint": REVISION, "headers": HEADERS_SHA256,
             "config": CONFIG_SHA256, "recipe": sources, "kv_dtype": "bfloat16", "page_size": 64}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@torch.no_grad()
def prepare_model(snapshot: Path, config: PreparationConfig | None = None, *, progress=None,
                  compile_decoder: bool = True) -> PreparedModel:
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

    from .vision_state import (
        PreparedEmbeddingInjection,
        PreparedVisionHandoff,
        VisionFeatureState,
        VisionInputs,
    )

    def report(message):
        if progress is not None:
            progress(message)

    config = PreparationConfig() if config is None else config
    device = torch.device("cuda", config.device)
    torch.cuda.set_device(device)
    manifest = inspect_qwen38_checkpoint(snapshot)
    weights = ModelWeights.load(manifest, device, progress=progress)
    mtp_weights = None
    has_mtp = config.mtp and config.speculative_proposal_source == "mtp"
    if has_mtp:
        from .mtp_weights import MTPWeights

        mtp_weights = MTPWeights.load(manifest, device,
            embedding_weight=weights.tree.model.language_model.embed_tokens.weight,
            lm_head_weight=weights.tree.lm_head.weight, progress=progress)
    table = MappedPLETable.load(manifest, device)
    session = preparation_session(device=device, checkpoint=REVISION,
        namespace={"config": CONFIG_SHA256, "headers": HEADERS_SHA256})
    session.__enter__()
    graphs = {}
    try:
        speculative_tokens = 7 if config.mtp else 0
        layout = HybridStateLayout.build(speculative_tokens=speculative_tokens)
        snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
        extra_base = SNAPSHOT_BASE + snapshots
        state = HybridStatePool.allocate(extra_base + (28 if config.mtp else 0), device,
                                         speculative_tokens=speculative_tokens)
        register_benchmark_state(
            session, next(iter(state.gdn.values())).recurrent,
            storage=state.backing, restore=state.reset_for_preparation,
        )
        requests = DeviceRequestState.allocate(device,
            consumed_tokens=state.consumed_tokens[1:5, 0], ple_history=state.ple_history[1:5])
        inputs = DecoderInputs.allocate(device, context=config.context, tokens=config.prefill_chunk_size)
        scratch = DecoderScratch.allocate(config.prefill_chunk_size, device, sample_rows=8 if config.mtp else 4)
        workspaces = ScratchArena(device)
        rotary = RotaryTables.allocate(weights.tree.config.text_config, config.context, device)
        cache = {index: QSAState.allocate(pages=1, slots=5, device=device, draft_tokens=speculative_tokens)
                 for index in (*range(3, 48, 4), *((48,) if has_mtp else ()))}
        page_bytes = sum(sum(getattr(value, name).nbytes for name in
            ("main_k_cache", "main_v_cache", "compressed_k_cache")) for value in cache.values())
        planned_pages = torch.cuda.mem_get_info(device)[1] // page_bytes
        admission = AdmissionBuffers.allocate(device, vocab_size=248320)
        transfer = StateTransferMetadata.allocate(4, device)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        completion = torch.zeros((2, 4, 24 if config.mtp else 8), dtype=torch.int64, device=device)
        features = VisionFeatureState.allocate(device)
        vision_inputs = VisionInputs.allocate(device)
        vision_scratch = VisionScratch.allocate(16384, device)
        buffers = {**inputs.buffers(), **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
            **{f"state_{name}": getattr(transfer, name) for name in transfer.__dataclass_fields__},
            **{name: getattr(vision_inputs, name) for name in vision_inputs.__dataclass_fields__},
            "frame_index": frame, "submission_sequence": sequence, "completion_records": completion,
            "sampling_grammar_mask": requests.grammar_mask}
        functions, signatures, decoders, owners = {}, {}, [], []
        speculation = None
        if config.mtp:
            from .speculative_model import SpeculativePreparation

            speculation = SpeculativePreparation(weights=weights, mtp_weights=mtp_weights,
                pool=state, requests=requests, inputs=inputs, scratch=scratch, workspaces=workspaces,
                rotary=rotary, cache=cache, table=table, session=session, completion=completion,
                frame=frame, sequence=sequence, extra_base=extra_base, context=config.context,
                prefill_capacity=config.prefill_chunk_size, max_cache_pages=planned_pages,
                proposal_source=config.speculative_proposal_source)
            buffers.update(speculation.buffers())
        def add(name, function, kind, batch, tokens, fields):
            functions[name] = function
            signatures[name] = GraphSignature(kind, batch, tokens, tuple(fields))

        vision_fields = tuple(vision_inputs.__dataclass_fields__)
        for capacity in (16384, 4096, 1024, 256):
            s = vision_scratch.rows(capacity)
            handoff = PreparedVisionHandoff(vision_inputs, features, requests, s.output, completion, frame, sequence)
            handoff.prepare()
            with torch.device("meta"):
                module = Qwen4ExpVisionModel(weights.tree.config.vision_config)
            module.eval().requires_grad_(False)
            encoder = PreparedVision(module, s, handoff.metadata, weights.vision, session=session,
                name=f"vision{capacity}", projection_workspace=workspaces,
                attention_workspace=workspaces)
            def vision(handoff=handoff, encoder=encoder):
                handoff.begin()
                encoder(handoff.pixels)
                handoff.finish()
            add(f"vision{capacity}", vision, "vision", 1, capacity, vision_fields)
            owners.extend((handoff, encoder))
            report(f"declared vision capacity {capacity}")

        forward_fields = (*inputs.buffers(), "sampling_grammar_mask",
                          *(("draft_page_tables",) if has_mtp else ()))
        modes = (("prefill", tuple(reversed(config.prefill_capacities))),)
        if not config.mtp:
            modes += (("decode", (4, 2, 1)),)
        report("declaring graph shapes before kernel autotuning")
        for mode, capacities in modes:
            for capacity in capacities:
                rows = 1 if mode == "prefill" else capacity
                metadata = PreparedDecoderState(inputs, requests, state, tokens=capacity, rows=rows, mode=mode)
                s = scratch.rows(capacity, rows)
                injection = None
                if mode == "prefill":
                    injection = PreparedEmbeddingInjection(features, requests, inputs.vision_token_indices[:capacity],
                        inputs.row_slot_ids, inputs.row_generations, inputs.query_lengths, inputs.live_rows,
                        s.embedding, rows=rows)
                    injection.prepare()
                decoder = PreparedDecoder(weights, s, metadata, cache, rotary, table, workspaces,
                    completion, frame, sequence, session=session, name=f"{mode}{capacity}",
                    embedding_injection=injection, max_cache_pages=planned_pages)
                forward = decoder if speculation is None else speculation.prefill(decoder, f"{mode}{capacity}")
                add(f"{mode}{capacity}", forward, mode, rows,
                    capacity if mode == "prefill" else 1, forward_fields)
                decoders.append(decoder)
                report(f"declared {mode} capacity {capacity}")

        if speculation is not None:
            fields = (*forward_fields, *(name for name in speculation.buffers() if name != "draft_page_tables"))
            for batch, width in config.speculative_shapes:
                name = f"speculative{batch}x{width}"
                operation = speculation.round(batch, width, name)
                operation.prepare()
                add(name, operation, "speculative", batch, width, fields)
                report(f"declared speculative batch={batch}, verification_width={width}")
            decoders.extend(speculation.decoders)
            owners.append(speculation)

        state_fields = tuple(f"state_{name}" for name in transfer.__dataclass_fields__ if name != "errors")
        admission_fields = tuple(admission.__dataclass_fields__)
        for capacity in (4, 2, 1):
            resume = PreparedResume(inputs, requests, state, scratch.sample_hidden[:capacity],
                scratch.logits[:capacity], weights.tree.lm_head.weight, completion, frame, sequence,
                session=session, name=f"resume{capacity}")
            forward = resume if speculation is None else speculation.resume(resume)
            add(f"resume{capacity}", forward, "resume", capacity, 1, forward_fields)
            owners.append(resume)
            for mode in ("init", "restore", "snapshot"):
                maintenance = PreparedMaintenance(mode, requests, state, transfer, completion, frame, sequence,
                    admission_buffers=admission, forward_inputs=inputs,
                    qsa_states=tuple(value for index, value in cache.items() if index < 48), capacity=capacity,
                    speculation=speculation)
                maintenance.prepare()
                fields = state_fields + (forward_fields if mode == "snapshot" else admission_fields)
                if mode == "restore":
                    fields += ("page_tables", *(("draft_page_tables",) if has_mtp else ()))
                add(f"{mode}{capacity}", maintenance, mode, capacity, 1, fields)
                owners.append(maintenance)
        session.register_scratch(workspaces)
        report("preparing weight kernels, then cache-state kernels")
        tuning = session.tune(show_progress=progress is not None)
        report(f"selected prepared kernels: {tuning['selection_counts']}")
        report(f"preparation allocated bytes: before tuning={tuning['allocated_before_tuning']}, "
               f"after trial release={tuning['allocated_after_trial_release']}")
        stream = torch.cuda.Stream(device=device)
        graph_pool = torch.cuda.graph_pool_handle()
        compiled_functions = {}
        def capture_catalog():
            captured, compiled = capture_forwards(functions, signatures, session=session, device=device,
                stream=stream, pool=graph_pool, compile_decoder=compile_decoder, progress=report)
            graphs.update(captured)
            compiled_functions.clear()
            compiled_functions.update(compiled)

        capture_catalog()
        report("measured complete graph catalog memory")
        free, total = torch.cuda.mem_get_info(device)
        available = free - config.headroom_bytes - _KV_ALLOCATION_MARGIN
        page_capacity = available // page_bytes if config.pages is None else config.pages
        if page_capacity <= 0 or page_capacity * page_bytes > available:
            raise MemoryError("prepared graph catalog leaves insufficient memory for requested KV pages and headroom")
        report(f"KV sizing: free={free}, allocated={torch.cuda.memory_allocated(device)}, "
               f"reserved={torch.cuda.memory_reserved(device)}, pages={page_capacity}, "
               f"page_bytes={page_bytes}, headroom={config.headroom_bytes}")
        final_cache, page_capacity = _allocate_page_pools(cache, page_capacity, page_bytes=page_bytes,
            device=device, headroom_bytes=config.headroom_bytes, automatic=config.pages is None, report=report)
        for decoder in decoders:
            for index in range(3, 48, 4):
                decoder.layers[index][1].rebind_pages_for_preparation(final_cache[index])
            decoder.qsa_states = final_cache
        if speculation is not None:
            speculation.rebind_pages_for_preparation(final_cache)
        for owner in owners:
            if isinstance(owner, PreparedMaintenance):
                owner.rebind_pages_for_preparation(final_cache)
        for graph in graphs.values():
            graph.reset()
        graphs.clear()
        capture_catalog()
        session.freeze()
        free, total = torch.cuda.mem_get_info(device)
        if free < config.headroom_bytes:
            raise MemoryError(f"captured model violates configured GPU headroom: free={free}, "
                f"required={config.headroom_bytes}, pages={page_capacity}, "
                f"allocated={torch.cuda.memory_allocated(device)}, reserved={torch.cuda.memory_reserved(device)}")
        memory = {"weights_bytes": weights.allocated_bytes, "mapped_ple_bytes": 28800138240,
            "hybrid_state_bytes": state.backing.nbytes, "snapshot_capacity": snapshots,
            "pages": page_capacity, "page_bytes": page_bytes, "kv_bytes": page_capacity * page_bytes,
            "cached_token_capacity": page_capacity * 64, "headroom_bytes": config.headroom_bytes,
            "free_device_bytes": free, "total_device_bytes": total,
            "allocated_bytes": torch.cuda.memory_allocated(device), "reserved_bytes": torch.cuda.memory_reserved(device),
            "operator_scratch_bytes": workspaces.nbytes, "graphs": len(graphs), "tuning": tuning,
            "prefill_chunk_size": config.prefill_chunk_size, "prefill_capacities": config.prefill_capacities,
            "compiled_decoder": compile_decoder}
        if speculation is not None:
            memory.update(mtp_weights_bytes=0 if mtp_weights is None else mtp_weights.allocated_bytes,
                speculative_widths=config.speculative_widths,
                speculative_shapes=config.speculative_shapes,
                speculative_proposal_source=config.speculative_proposal_source,
                speculative_state_columns=8, draft_raw_ring_capacity=layout.draft_ring_capacity if has_mtp else 0,
                speculative_fixed_width=config.speculative_fixed_width,
                history_drafts=config.history_drafts)
        hashes = {"tokenizer": ("tokenizer.json", FRONTEND_SHA256["tokenizer.json"]),
            "tokenizer_config": ("tokenizer_config.json", FRONTEND_SHA256["tokenizer_config.json"]),
            "chat_template": ("chat_template.jinja", FRONTEND_SHA256["chat_template.jinja"]),
            "preprocessor_config": ("preprocessor_config.json", FRONTEND_SHA256["preprocessor_config.json"]),
            "model_config": ("config.json", CONFIG_SHA256), "generation_config": ("generation_config.json", GENERATION_SHA256)}
        server_config = {"model_id": MODEL_ID, "identity": _identity(manifest),
            "max_sequence_tokens": config.context, "prefill_chunk_size": config.prefill_chunk_size,
            "pages": page_capacity, "snapshots": snapshots,
            "eos_token_ids": [248046, 248044],
            "model_contract": {"schema": "qwen4exp.graph-inputs.v2", "vocabulary": 248320,
                "vision": {"features_per_request": 16384, "max_images": 4,
                    "patches_per_feature": 4, "patch_capacities": [256, 1024, 4096, 16384]}},
            "assets": {name: {"path": str((manifest.snapshot / filename).resolve()), "sha256": digest}
                       for name, (filename, digest) in hashes.items()}}
        if speculation is not None:
            server_config.update(speculative_widths=config.speculative_widths,
                speculative_proposal_source=config.speculative_proposal_source,
                speculative_fixed_width=config.speculative_fixed_width, history_drafts=config.history_drafts)
        resources = (weights, table, session, state, requests, inputs, scratch, workspaces, rotary,
            cache, final_cache, admission, transfer, features, vision_inputs, vision_scratch,
            functions, compiled_functions, decoders, owners, stream, graph_pool, speculation, mtp_weights)
        report(f"prepared {len(graphs)} graphs and {page_capacity * 64} cached-token capacity")
        return PreparedModel(device=config.device, graphs=graphs, buffers=buffers, signatures=signatures,
            resources=resources, server_config=server_config, memory_report=memory, session=session, close_resources=(table,))
    except BaseException:
        torch.cuda.synchronize(device)
        for graph in graphs.values():
            graph.reset()
        session.close()
        table.close()
        raise
