"""DeepSeek V4.1 Flash text checkpoint and tensor-parallel memory geometry."""

from types import MappingProxyType

from ignition.planning import HEADROOM_BYTES
from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, Speculation


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            package_storage=None):
    from ignition.preparation.rank_group import prepare_group
    from .bootstrap import PreparationConfig, prepare_rank

    speculation = plan.speculation
    drafting = {} if speculation is None else {
        "speculative_widths": speculation.widths, "speculative_fixed_width": speculation.fixed_width}
    config = PreparationConfig(devices=tuple(range(plan.world_size)), compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, max_active=plan.concurrency, context=plan.context,
        prefill_chunk_size=plan.prefill_chunk, minimum_pages=plan.minimum_pages,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=HEADROOM_BYTES + plan.reserve_bytes, kernel_cache=kernel_cache,
        dspark=speculation is not None, **drafting)
    return prepare_group(snapshot, config, prepare_rank, progress=progress, package_storage=package_storage)


def usage(prepared):
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in prepared.memory_report["ranks"])


RECIPE = Recipe(
    name="deepseek41", prepare=prepare, usage=usage,
    checkpoints=(Checkpoint(alias="deepseek-v4.1-flash", repository="deepseek-ai/DeepSeek-V4.1-Flash",
        revision="fb2764a5cf321eaa5070ca8f9e892818f477c16d", architecture="DeepseekV41ForCausalLM",
        stored_bytes=510_286_023_000, device_bytes=306_557_453_640, draft_bytes=7_932_874_632),),
    context=1048576, page_tokens=128, concurrency=4, max_concurrency=64, prefill_chunk=4096,
    speculation=Speculation(draft_tokens=5), max_draft_tokens=7,
    ranks=MappingProxyType({4: RankCosts(3_043_840, 113_920, 8 << 30)}),
    plain_ranks=MappingProxyType({4: RankCosts(2_841_088, 113_920, 8 << 30)}),
)
