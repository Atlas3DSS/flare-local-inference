"""Qualified checkpoint, serving range, and memory costs of the Qwen 3.8 recipe."""

import os
from types import MappingProxyType

from flare.planning import HEADROOM_BYTES
from flare.preparation.qwen38 import ARCHITECTURE, MODEL_ID, REVISION
from flare.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe

# Twelve target sparse-attention layers and the draft layer each store one page.
_PAGE_LAYERS = 13
# Cold-host estimates of what a rank holds beyond stored weights, state rows, and
# pages: prepared weight layouts, replicated tensors, operator scratch, graphs, and
# the CUDA context. One and two ranks were measured at concurrency 4 on an RTX PRO
# 6000 Blackwell; four ranks extrapolate the replicated weight share from them.
_GRAPH_BYTES = {1: 7168 << 20, 2: 8000 << 20, 4: 8576 << 20}


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress):
    """Prepare the plan's catalog with learned multi-token speculation on process-local GPUs 0 through N-1.

    KV pages fill the free memory measured after graph capture, not the plan's estimate.
    """
    from .bootstrap import PreparationConfig, prepare_model, prepare_rank

    config = PreparationConfig(
        devices=tuple(range(plan.world_size)), compile_decoder=compile_forwards, max_active=plan.concurrency,
        context=plan.context, prefill_chunk_size=plan.prefill_chunk, minimum_pages=plan.minimum_pages,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=HEADROOM_BYTES + plan.reserve_bytes, kernel_cache=kernel_cache, mtp=True,
        # Local experiment knob: maximum MTP draft tokens (verification widths 1..drafts+1).
        speculative_widths=tuple(range(1, int(os.environ.get("FLARE_MTP_MAX_DRAFTS", "3")) + 2)))
    if plan.world_size == 1:
        return prepare_model(snapshot, config, progress=progress, compile_decoder=compile_forwards)
    from flare.preparation.rank_group import prepare_group

    return prepare_group(snapshot, config, prepare_rank, progress=progress)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["weights_bytes"] + rank.get("mtp_weights_bytes", 0), rank["hybrid_state_bytes"],
                           rank["page_bytes"], rank["pages"], rank["free_device_bytes"])
                 for rank in report.get("ranks", (report,)))


RECIPE = Recipe(
    name="qwen38", prepare=prepare, usage=usage,
    checkpoints=(Checkpoint(
        alias="qwen3.8-flash", repository=MODEL_ID, revision=REVISION, architecture=ARCHITECTURE,
        stored_bytes=102_138_950_916,
        # Decoder, vision, and draft tensors; the PLE tables stay in mapped host memory.
        device_bytes=73_338_812_672,
        mapped_host_bytes=28_800_138_244),),
    context=262144, page_tokens=64, concurrency=4, max_concurrency=64,
    # Verification retains recurrent state for seven draft positions per request.
    speculative_rows=7, prefill_chunk=4096,
    ranks=MappingProxyType({
        1: RankCosts(state_row_bytes=122_932_224, page_bytes=_PAGE_LAYERS * 135_168, graph_bytes=_GRAPH_BYTES[1]),
        2: RankCosts(state_row_bytes=61_705_216, page_bytes=_PAGE_LAYERS * 69_632, graph_bytes=_GRAPH_BYTES[2]),
        4: RankCosts(state_row_bytes=31_550_464, page_bytes=_PAGE_LAYERS * 69_632, graph_bytes=_GRAPH_BYTES[4]),
    }),
    mapped_host_name="PLE table", max_images=4, image_bytes=32 << 20,
)
