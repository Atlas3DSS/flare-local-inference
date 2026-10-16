"""DeepSeek V4.1 Flash text checkpoint and tensor-parallel memory geometry."""

from types import MappingProxyType

from ignition.planning import HEADROOM_BYTES
from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, Speculation


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
        dspark=speculation is not None, stream_overlap=plan.stream_overlap, **drafting)
    return prepare_group(snapshot, config, prepare_rank, progress=progress, package_storage=package_storage)


def usage(prepared):
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in prepared.memory_report["ranks"])


RECIPE = Recipe(
    name="deepseek41", prepare=prepare, usage=usage,
    # Recommended sampling: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash#minimal-inference.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=-1),
    checkpoints=(Checkpoint(alias="deepseek-v4.1-flash", repository="local-inference-lab/DeepSeek-V4.1-Flash-lossless-CSF",
        revision="9c60bcbf1670e06f154b5d61080e50c9b3b7b4a3", architecture="DeepseekV41ForCausalLM",
        stored_bytes=495_629_051_096, device_bytes=291_883_853_264, draft_bytes=7_932_874_632),),
    context=1048576, page_tokens=128, concurrency=4, max_concurrency=64, prefill_chunk=4096,
    speculation=Speculation(draft_tokens=5), max_draft_tokens=7,
    # Single-row decode runs the shared expert on an auxiliary stream.
    stream_overlap=True,
    ranks=MappingProxyType({4: RankCosts(3_043_840, 113_920, 8 << 30)}),
    plain_ranks=MappingProxyType({4: RankCosts(2_841_088, 113_920, 8 << 30)}),
)
