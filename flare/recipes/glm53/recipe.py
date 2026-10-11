"""Qualified checkpoint, serving range, and memory costs of the GLM 5.3 recipe."""

from types import MappingProxyType

from flare.planning import HEADROOM_BYTES
from flare.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe

# Each of the eleven sparse-attention layers stores four latent blocks and one
# index block per 256-token page on every rank.
_PAGE_BYTES = 11 * (4 * 64 * 528 + 64 * 132)
# Cold-host estimate of what a rank holds beyond stored weights, state rows, and
# pages: operator scratch, graphs, collectives, and the CUDA context. Measured with
# two ranks at concurrency 4 on an RTX PRO 6000 Blackwell.
_GRAPH_BYTES = 4864 << 20


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress):
    """Prepare the plan's catalog on process-local GPUs 0 through N-1."""
    from flare.preparation.rank_group import prepare_group

    from .bootstrap import PreparationConfig, prepare_rank

    config = PreparationConfig(
        devices=tuple(range(plan.world_size)), compile_decoder=compile_forwards, max_active=plan.concurrency,
        context=plan.context, prefill_chunk_size=plan.prefill_chunk, pages=plan.pages,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=HEADROOM_BYTES + plan.reserve_bytes, kernel_cache=kernel_cache)
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
    context=1048576, page_tokens=256, concurrency=4, max_concurrency=64, speculative_rows=0, prefill_chunk=4096,
    ranks=MappingProxyType({
        2: RankCosts(state_row_bytes=75_420_672, page_bytes=_PAGE_BYTES, graph_bytes=_GRAPH_BYTES),
        4: RankCosts(state_row_bytes=38_515_712, page_bytes=_PAGE_BYTES, graph_bytes=_GRAPH_BYTES),
    }),
)
