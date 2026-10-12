"""Disposable GLM forward storage excluded from persistent tensor payloads."""

from ignition.preparation.disposable import storage_ranges
from .generation import PreparedGeneration
from .resume import PreparedResume
from .speculative import PreparedPrefill, PreparedRound


def disposable_ranges(scratch, arena, forwards):
    values = [scratch, arena.tensor]
    for forward in forwards.values():
        if isinstance(forward, PreparedPrefill):
            values.extend((forward.feedback, forward.previous, forward.before))
            values.extend(_draft_scratch(forward.absorb))
            forward = forward.generation
        if isinstance(forward, (PreparedGeneration, PreparedResume)):
            sampling = forward.completion
            if hasattr(sampling, "scratch_tensors"):
                values.append(sampling.scratch_tensors)
        if isinstance(forward, PreparedRound):
            values.extend(getattr(forward, name) for name in (
                "draft_logits", "before", "pending", "previous", "active", "candidates", "drafts",
                "probabilities", "valid", "seed", "offset", "base_offset", "sampled", "tail_backup"))
            values.extend(_draft_scratch(forward.absorb))
            if forward.draft is not None:
                values.extend(_draft_scratch(forward.draft))
            target, proposals, verify, _ = forward.operations
            for distribution in (target, *proposals):
                values.extend(getattr(distribution, name) for name in (
                    "scores", "probabilities", "counts", "point_mass_ids", "maximum", "valid",
                    "packed_top_k", "packed_top_p"))
            values.extend(getattr(verify, name) for name in (
                "safe_draft_probs", "safe_target_probs", "safe_ids", "seed_bits", "offset",
                "independent_accepted", "emitted_drafts", "output"))
    return storage_ranges(values)


def _draft_scratch(draft):
    values = [getattr(draft, name) for name in (
        "embedded", "joined", "residual", "normalized", "attention_output", "mlp_output", "hidden")]
    values.append(draft.attention.scratch)
    if not draft.cache_only:
        values.extend((draft.moe.scratch, draft.local_logits, draft.logits))
    return values
