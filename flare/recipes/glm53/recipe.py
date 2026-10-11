"""Qualified checkpoint, serving range, and memory costs of the GLM 5.3 recipe."""

import os
from types import MappingProxyType

from flare.planning import HEADROOM_BYTES
from flare.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe

# Each of the eleven sparse-attention layers and the draft layer stores four latent
# blocks and one index block per 256-token page on every rank.
_PAGE_BYTES = 12 * (4 * 64 * 528 + 64 * 132)
# Cold-host estimate of what a rank holds beyond stored weights, state rows, and
# pages: operator scratch, graphs, collectives, and the CUDA context. Measured with
# two ranks at concurrency 4 on an RTX PRO 6000 Blackwell.
_GRAPH_BYTES = 4864 << 20
# Decode context parallelism shards each page's records over the tensor-parallel ranks (FLARE_GLM_DCP=0 disables).
_CONTEXT_PARALLEL = os.environ.get("FLARE_GLM_DCP", "1") != "0"


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress):
    """Prepare the plan's catalog with learned multi-token speculation on process-local GPUs 0 through N-1.

    KV pages fill the free memory measured after graph capture, not the plan's estimate.
    """
    from flare.preparation.rank_group import prepare_group

    from .bootstrap import PreparationConfig, prepare_rank

    config = PreparationConfig(
        devices=tuple(range(plan.world_size)), compile_decoder=compile_forwards, max_active=plan.concurrency,
        context=plan.context, prefill_chunk_size=plan.prefill_chunk, minimum_pages=plan.minimum_pages,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=HEADROOM_BYTES + plan.reserve_bytes, kernel_cache=kernel_cache, mtp=True,
        context_parallel=_CONTEXT_PARALLEL,
        # Local experiment knob: maximum MTP draft tokens (verification widths 1..drafts+1).
        speculative_widths=tuple(range(1, int(os.environ.get("FLARE_MTP_MAX_DRAFTS", "5")) + 2)))
    return prepare_group(snapshot, config, prepare_rank, progress=progress)


def usage(prepared):
    return tuple(RankUsage(rank["weights_bytes"], rank["continuation_bytes"], rank["page_bytes"], rank["pages"],
                           rank["free_device_bytes"]) for rank in prepared.memory_report["ranks"])


RECIPE = Recipe(
    name="glm53", prepare=prepare, usage=usage,
    checkpoints=(Checkpoint(
        alias="glm-5.3-flash", repository="local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD",
        revision="dec48abd33efa73c3bb7c95b74eee10cad34f9be", architecture="Glm5NextForConditionalGeneration",
        stored_bytes=177_680_678_236,
        # Language model and output head; the vision tower is not loaded.
        device_bytes=177_290_048_556),),
    context=1048576, page_tokens=256, concurrency=4, max_concurrency=64,
    # Verification retains recurrent state for seven draft positions per request.
    speculative_rows=7, prefill_chunk=4096,
    ranks=MappingProxyType({
        2: RankCosts(state_row_bytes=81_464_576, page_bytes=_PAGE_BYTES, graph_bytes=_GRAPH_BYTES,
                     page_tokens=512 if _CONTEXT_PARALLEL else None),
        4: RankCosts(state_row_bytes=41_635_072, page_bytes=_PAGE_BYTES, graph_bytes=_GRAPH_BYTES,
                     page_tokens=1024 if _CONTEXT_PARALLEL else None),
    }),
)
