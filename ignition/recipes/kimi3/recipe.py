"""Kimi K3 MXFP4-CSF preparation capacities and serving defaults."""

from types import MappingProxyType

from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, Speculation


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            package_storage=None, ranks=None):
    from ignition.preparation.rank_group import prepare_group
    from .bootstrap import PreparationConfig, prepare_rank

    if ranks is not None:
        raise ValueError("Kimi K3 preparation requires GPUs on one host")
    drafting = {}
    if plan.speculation is not None:
        from huggingface_hub import snapshot_download
        from .draft_config import CHECKPOINT, REPOSITORY, REVISION

        draft = snapshot_download(REPOSITORY, revision=REVISION, local_dir=CHECKPOINT,
                                  allow_patterns=("config.json", "model.safetensors"))
        drafting = dict(draft_checkpoint=draft, speculative_widths=plan.speculation.widths,
            speculative_fixed_width=plan.speculation.fixed_width, speculative_token_capacity=plan.speculative_token_capacity)
    config = PreparationConfig(compile_workers=plan.compile_workers, tuning_replica=plan.tuning_replica,
        devices=tuple(range(plan.world_size)), context=plan.context,
        max_active=plan.concurrency, prefill_chunk_size=plan.prefill_chunk,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, kernel_cache=kernel_cache, **drafting)
    if plan.world_size == 1:
        return prepare_rank(snapshot, config, rank=0, group=None, progress=progress, package_storage=package_storage)
    return prepare_group(snapshot, config, prepare_rank, progress=progress, package_storage=package_storage,
                         checkpoint=plan.checkpoint)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


RECIPE = Recipe(
    name="kimi3", prepare=prepare, usage=usage,
    # Agentic sampling: https://huggingface.co/moonshotai/Kimi-K3#3-evaluation-results.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=1., top_k=-1),
    checkpoints=(Checkpoint(alias="kimi-k3-mxfp4-csf", repository="/data/models/Kimi-K3-MXFP4-CSF",
        revision="local", architecture="KimiK3ForConditionalGeneration",
        stored_bytes=1_491_026_420_236, device_bytes=1_505_431_429_644, draft_bytes=14_405_009_408),),
    context=1048576, page_tokens=128, concurrency=4, max_concurrency=64, prefill_chunk=4096,
    speculation=Speculation(draft_tokens=7), max_draft_tokens=7, max_images=4, image_bytes=32 << 20,
    verification_rows=0,
    ranks=MappingProxyType({16: RankCosts(state_row_bytes=34_602_240, page_bytes=4_276_224,
                                        draft_page_bytes=737_280, graph_bytes=5 << 30)}),
    plain_ranks=MappingProxyType({16: RankCosts(state_row_bytes=31_639_296, page_bytes=3_538_944, graph_bytes=5 << 30)}),
)
