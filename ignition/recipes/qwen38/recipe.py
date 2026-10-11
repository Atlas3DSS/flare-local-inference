"""Qualified checkpoint, serving range, and memory costs of the Qwen 3.8 recipe."""

from types import MappingProxyType

from ignition.planning import HEADROOM_BYTES
from ignition.preparation.qwen38 import ARCHITECTURE, MODEL_ID, REVISION
from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, Speculation

# Twelve target sparse-attention layers each store one page; learned drafts add the draft layer's.
_TARGET_PAGE_LAYERS = 12
# Rough estimates of what a rank holds beyond stored weights, state rows, and
# pages: prepared weight layouts, replicated tensors, operator scratch, graphs, and
# the CUDA context. They serve every speculation setting.
_GRAPH_BYTES = {1: 7168 << 20, 2: 8000 << 20, 4: 8576 << 20}


def _costs(state_row_bytes, page_layer_bytes, graph_bytes, *, speculating):
    layers = _TARGET_PAGE_LAYERS + int(speculating)
    return RankCosts(state_row_bytes, layers * page_layer_bytes, graph_bytes,
                     draft_page_bytes=page_layer_bytes if speculating else 0)


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False, package_storage=None):
    """Prepare the plan's catalog and speculation settings on process-local GPUs 0 through N-1.

    KV pages fill the free memory measured after graph capture, not the plan's estimate.
    """
    from .bootstrap import PreparationConfig, prepare_model, prepare_rank

    speculation = plan.speculation
    drafting = {} if speculation is None else {
        "speculative_widths": speculation.widths, "speculative_fixed_width": speculation.fixed_width,
        "speculative_graph_budget": speculation.graph_budget,
        "speculative_proposal_source": speculation.proposal_source, "history_drafts": speculation.history_drafts}
    config = PreparationConfig(
        devices=tuple(range(plan.world_size)), compile_decoder=compile_forwards, retain_graphs=retain_graphs,
        max_active=plan.concurrency,
        context=plan.context, prefill_chunk_size=plan.prefill_chunk, minimum_pages=plan.minimum_pages,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=HEADROOM_BYTES + plan.reserve_bytes, kernel_cache=kernel_cache,
        mtp=speculation is not None, **drafting)
    if plan.world_size == 1:
        storage = {} if package_storage is None else {"package_storage": package_storage}
        return prepare_model(snapshot, config, progress=progress, compile_decoder=compile_forwards,
                             **storage)
    from ignition.preparation.rank_group import prepare_group

    return prepare_group(snapshot, config, prepare_rank, progress=progress)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


RECIPE = Recipe(
    name="qwen38", prepare=prepare, usage=usage,
    checkpoints=(Checkpoint(
        alias="qwen3.8-flash", repository=MODEL_ID, revision=REVISION, architecture=ARCHITECTURE,
        stored_bytes=102_138_950_916,
        # Decoder, vision, and draft tensors; the PLE tables stay in mapped host memory.
        device_bytes=73_338_812_672,
        mapped_host_bytes=28_800_138_244,
        # The MTP module's tensors.
        draft_bytes=1_596_720_640),),
    context=262144, page_tokens=64, concurrency=4, max_concurrency=64, prefill_chunk=4096,
    # Rounds adapt among zero through three drafts. Verification retains recurrent
    # state for seven draft positions per request whatever the captured counts.
    speculation=Speculation(draft_tokens=3), speculative_rows=7, history_copy=True,
    ranks=MappingProxyType({
        1: _costs(122_932_224, 135_168, _GRAPH_BYTES[1], speculating=True),
        2: _costs(61_705_216, 69_632, _GRAPH_BYTES[2], speculating=True),
        4: _costs(31_550_464, 69_632, _GRAPH_BYTES[4], speculating=True),
    }),
    plain_ranks=MappingProxyType({
        1: _costs(117_291_776, 135_168, _GRAPH_BYTES[1], speculating=False),
        2: _costs(58_776_320, 69_632, _GRAPH_BYTES[2], speculating=False),
        4: _costs(29_911_808, 69_632, _GRAPH_BYTES[4], speculating=False),
    }),
    mapped_host_name="PLE table", max_images=4, image_bytes=32 << 20,
)
