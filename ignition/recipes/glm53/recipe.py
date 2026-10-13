"""Qualified checkpoint, serving range, and memory costs of the GLM 5.3 recipe."""

from types import MappingProxyType

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


def _costs(state_row_bytes, graph_bytes, *, ranks, speculating):
    layers = _TARGET_PAGE_LAYERS + int(speculating)
    # Decode context parallelism shards each page's latent records over the ranks.
    return RankCosts(state_row_bytes, layers * _PAGE_LAYER_BYTES, graph_bytes,
                     draft_page_bytes=_PAGE_LAYER_BYTES if speculating else 0, page_tokens=256 * ranks)


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            package_storage=None):
    """Prepare the plan's catalog and speculation settings on process-local GPUs 0 through N-1.

    KV pages fill the free memory measured after graph capture, not the plan's estimate.
    """
    from ignition.preparation.rank_group import prepare_group

    from .bootstrap import PreparationConfig, prepare_rank

    speculation = plan.speculation
    drafting = {} if speculation is None else {
        "speculative_widths": speculation.widths, "speculative_fixed_width": speculation.fixed_width}
    config = PreparationConfig(
        devices=tuple(range(plan.world_size)), compile_decoder=compile_forwards, retain_graphs=retain_graphs,
        max_active=plan.concurrency, context_parallel=plan.context_parallel,
        context=plan.context, prefill_chunk_size=plan.prefill_chunk, minimum_pages=plan.minimum_pages,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, kernel_cache=kernel_cache,
        mtp=speculation is not None, **drafting)
    return prepare_group(snapshot, config, prepare_rank, progress=progress, package_storage=package_storage)


def usage(prepared):
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in prepared.memory_report["ranks"])


RECIPE = Recipe(
    name="glm53", prepare=prepare, usage=usage,
    checkpoints=(Checkpoint(
        alias="glm-5.3-flash", repository="local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD",
        revision="dec48abd33efa73c3bb7c95b74eee10cad34f9be", architecture="Glm5NextForConditionalGeneration",
        stored_bytes=178_175_535_810,
        # Language model, draft layer, and output head; the vision tower is not loaded.
        device_bytes=177_784_906_130,
        # Decoder layer 45, the MTP module.
        draft_bytes=4_301_658_112),),
    context=1048576, page_tokens=256, concurrency=4, max_concurrency=64, prefill_chunk=4096,
    speculation=Speculation(draft_tokens=5), max_draft_tokens=7,
    ranks=MappingProxyType({
        2: _costs(81_464_576, _GRAPH_BYTES, ranks=2, speculating=True),
        4: _costs(41_635_072, _GRAPH_BYTES, ranks=4, speculating=True),
    }),
    plain_ranks=MappingProxyType({
        2: _costs(75_420_672, _GRAPH_BYTES, ranks=2, speculating=False),
        4: _costs(38_515_712, _GRAPH_BYTES, ranks=4, speculating=False),
    }),
)
