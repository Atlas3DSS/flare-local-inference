"""Qualified checkpoint, serving range, and memory costs of the GLM 5.3 recipe."""

from types import MappingProxyType

from ignition.planning import HEADROOM_BYTES
from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, Speculation

# Each sparse-attention layer stores four latent blocks and one index block per
# 256-token page on every rank. The target has eleven such layers; learned drafts
# add the draft layer's.
_PAGE_LAYER_BYTES = 4 * 64 * 528 + 64 * 132
_TARGET_PAGE_LAYERS = 11
# Rough estimate of what a rank holds beyond stored weights, state rows, and
# pages: operator scratch, graphs, collectives, and the CUDA context. It serves
# every rank count and speculation setting.
_GRAPH_BYTES = 4864 << 20


def _costs(state_row_bytes, graph_bytes, *, speculating):
    layers = _TARGET_PAGE_LAYERS + int(speculating)
    return RankCosts(state_row_bytes, layers * _PAGE_LAYER_BYTES, graph_bytes,
                     draft_page_bytes=_PAGE_LAYER_BYTES if speculating else 0)


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress):
    """Prepare the plan's catalog and speculation settings on process-local GPUs 0 through N-1.

    KV pages fill the free memory measured after graph capture, not the plan's estimate.
    """
    from ignition.preparation.rank_group import prepare_group

    from .bootstrap import PreparationConfig, prepare_rank

    speculation = plan.speculation
    drafting = {} if speculation is None else {
        "speculative_widths": speculation.widths, "speculative_fixed_width": speculation.fixed_width,
        "speculative_graph_budget": speculation.graph_budget}
    config = PreparationConfig(
        devices=tuple(range(plan.world_size)), compile_decoder=compile_forwards, max_active=plan.concurrency,
        context=plan.context, prefill_chunk_size=plan.prefill_chunk, minimum_pages=plan.minimum_pages,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=HEADROOM_BYTES + plan.reserve_bytes, kernel_cache=kernel_cache,
        mtp=speculation is not None, **drafting)
    return prepare_group(snapshot, config, prepare_rank, progress=progress)


def usage(prepared):
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in prepared.memory_report["ranks"])


RECIPE = Recipe(
    name="glm53", prepare=prepare, usage=usage,
    checkpoints=(Checkpoint(
        alias="glm-5.3-flash", repository="local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD",
        revision="dec48abd33efa73c3bb7c95b74eee10cad34f9be", architecture="Glm5NextForConditionalGeneration",
        stored_bytes=177_680_678_236,
        # Language model, draft layer, and output head; the vision tower is not loaded.
        device_bytes=177_290_048_556,
        # Decoder layer 45, the MTP module.
        draft_bytes=4_301_658_112),),
    context=1048576, page_tokens=256, concurrency=4, max_concurrency=64, prefill_chunk=4096,
    # Rounds adapt among zero through five drafts. Verification retains recurrent
    # state for seven draft positions per request whatever the captured counts.
    speculation=Speculation(draft_tokens=5), speculative_rows=7,
    ranks=MappingProxyType({
        2: _costs(81_464_576, _GRAPH_BYTES, speculating=True),
        4: _costs(41_635_072, _GRAPH_BYTES, speculating=True),
    }),
    plain_ranks=MappingProxyType({
        2: _costs(75_420_672, _GRAPH_BYTES, speculating=False),
        4: _costs(38_515_712, _GRAPH_BYTES, speculating=False),
    }),
)
