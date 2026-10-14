"""DeepSeek V4 Flash Vision checkpoint with integrated DSpark proposals."""

from types import MappingProxyType

from ignition.planning import HEADROOM_BYTES
from ignition.recipes.declaration import (
    Checkpoint,
    RankCosts,
    RankUsage,
    Recipe,
    SamplingDefaults,
    Speculation,
)

from .config import MODEL_ID, REVISION


def prepare(
    snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False, package_storage=None
):
    from ignition.preparation.rank_group import prepare_group

    from .bootstrap import PreparationConfig, prepare_rank

    speculation = plan.speculation
    drafting = (
        {}
        if speculation is None
        else {"speculative_widths": speculation.widths, "speculative_fixed_width": speculation.fixed_width}
    )
    config = PreparationConfig(
        devices=tuple(range(plan.world_size)),
        compile_decoder=compile_forwards,
        retain_graphs=retain_graphs,
        max_active=plan.concurrency,
        context=plan.context,
        prefill_chunk_size=plan.prefill_chunk,
        minimum_pages=plan.minimum_pages,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=HEADROOM_BYTES + plan.reserve_bytes,
        kernel_cache=kernel_cache, compile_devices=plan.compile_devices,
        dspark=speculation is not None,
        **drafting,
    )
    return prepare_group(snapshot, config, prepare_rank, progress=progress, package_storage=package_storage)


def usage(prepared):
    return tuple(
        RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in prepared.memory_report["ranks"]
    )


RECIPE = Recipe(
    name="deepseek4",
    # Agent evaluation sampling: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp#introduction.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=-1),
    prepare=prepare,
    usage=usage,
    checkpoints=(
        Checkpoint(
            alias="deepseek-v4-flash",
            repository=MODEL_ID,
            revision=REVISION,
            architecture="DeepseekV4ForCausalLM",
            stored_bytes=167_811_372_792,
            device_bytes=167_811_372_792,
            draft_bytes=10_862_841_372,
        ),
    ),
    context=1048576,
    page_tokens=256,
    concurrency=4,
    max_concurrency=64,
    prefill_chunk=4096,
    speculation=Speculation(draft_tokens=5),
    max_draft_tokens=5,
    max_images=4,
    image_bytes=32 << 20,
    # Continuation rows and latent pages are replicated, so their costs do not depend on the rank count.
    ranks=MappingProxyType({
        2: RankCosts(16_668_416, 998_208, 11 << 30),
        4: RankCosts(16_668_416, 998_208, 24 << 30),
    }),
    plain_ranks=MappingProxyType({
        2: RankCosts(16_443_392, 998_208, 12 << 30),
        4: RankCosts(16_443_392, 998_208, 24 << 30),
    }),
)
