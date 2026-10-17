"""Kimi K3 MXFP4-CSF preparation capacities and serving defaults."""

from pathlib import Path
from types import MappingProxyType

from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, Speculation


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            package_storage=None, ranks=None):
    from ignition.preparation.rank_group import prepare_group
    from .bootstrap import PreparationConfig, prepare_rank
    from .config import DRAFT_DIRECTORY

    if ranks is not None:
        raise ValueError("Kimi K3 preparation requires GPUs on one host")
    drafting = {}
    if plan.speculation is not None:
        drafting = dict(speculative_widths=plan.speculation.widths, speculative_fixed_width=plan.speculation.fixed_width,
                        speculative_token_capacity=plan.speculative_token_capacity)
        if (Path(snapshot) / DRAFT_DIRECTORY).is_dir():
            # A checkpoint carrying Red Hat's DSpark speculator drafts with it.
            drafting["dspark"] = True
        else:
            from huggingface_hub import snapshot_download
            from .draft_config import REPOSITORY, REVISION

            drafting["draft_checkpoint"] = snapshot_download(REPOSITORY, revision=REVISION, cache_dir=plan.hub_cache,
                                                             allow_patterns=("config.json", "model.safetensors"))
    config = PreparationConfig(compile_workers=plan.compile_workers, tuning_replica=plan.tuning_replica,
        devices=tuple(range(plan.world_size)), context=plan.context,
        max_active=plan.concurrency, image_concurrency=plan.image_concurrency, prefill_chunk_size=plan.prefill_chunk,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, kernel_cache=kernel_cache, online_mxfp8=plan.online_mxfp8,
        stream_overlap=plan.stream_overlap, context_parallel=plan.context_parallel,
        vision=not plan.language_model_only, **drafting)
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
        stored_bytes=1_491_026_420_236, device_bytes=1_505_431_429_644, draft_bytes=14_405_009_408),
        # The native checkpoint's block scales are compressed while loading, as resident as the CSF copy's.
        Checkpoint(alias="kimi-k3", repository="moonshotai/Kimi-K3",
        revision="2496450e92e425c886db095102a52a6682ca3970", architecture="KimiK3ForConditionalGeneration",
        stored_bytes=1_560_860_324_864, device_bytes=1_505_431_429_644, draft_bytes=14_405_009_408),
        # The native checkpoint with Red Hat's DSpark speculator (RedHatAI/Kimi-K3-speculator.dspark@38a88101)
        # in its dspark/ directory; the draft shares the target's output head and keeps its windows in
        # continuation rows, so decode context parallelism spreads each page over the sixteen ranks.
        Checkpoint(alias="kimi-k3-dspark", repository="/data/kimi-k3-dspark", revision="local",
        architecture="KimiK3ForConditionalGeneration", stored_bytes=1_570_350_125_826,
        device_bytes=1_498_926_420_236, draft_bytes=7_900_000_000,
        ranks=MappingProxyType({16: RankCosts(state_row_bytes=34_716_928, page_bytes=1_769_472, graph_bytes=6 << 30,
                                              page_tokens=128 * 16)}))),
    context=1048576, page_tokens=128, concurrency=4, max_concurrency=64, prefill_chunk=4096,
    speculation=Speculation(draft_tokens=7), max_draft_tokens=7, max_images=4, image_bytes=32 << 20,
    verification_rows=0,
    # KDA's fused input projection (query, key, value, gates and beta) and the DSpark draft's state
    # combination, attention outputs and MLPs may load as MXFP8.
    online_mxfp8=True,
    # Decode and verification run each MoE layer's shared experts beside its routing and routed experts.
    stream_overlap=True,
    # Text-only plans, as vLLM serves Kimi K3 with --language-model-only, leave the MoonViT encoder unloaded.
    language_model_only=True,
    # E4M3 latent pages, as vLLM serves Kimi K3 (--kv-cache-dtype fp8). DFlash2 attends its own E4M3 pages over
    # whole context, so each rank keeps its context whole; without a draft, decode context parallelism spreads
    # each page's positions over the sixteen ranks.
    ranks=MappingProxyType({16: RankCosts(state_row_bytes=32_464_128, page_bytes=2_138_112,
                                        draft_page_bytes=368_640, graph_bytes=5 << 30)}),
    plain_ranks=MappingProxyType({16: RankCosts(state_row_bytes=29_869_824, page_bytes=1_769_472, graph_bytes=6 << 30,
                                              page_tokens=128 * 16)}),
)
