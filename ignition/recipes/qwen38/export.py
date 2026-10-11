"""Disposable Qwen storage cleared before publishing a runtime package."""

from dataclasses import fields, is_dataclass

import torch


def storage_ranges(values):
    """Deduplicate backing storage for explicitly declared scratch tensor trees."""
    storage = {}

    def visit(value):
        if value is None:
            return
        if isinstance(value, torch.Tensor):
            if value.is_meta:
                raise ValueError("export scratch must have materialized storage")
            backing = value.untyped_storage()
            if backing.nbytes():
                storage[backing.data_ptr()] = backing.nbytes()
        elif is_dataclass(value):
            for field in fields(value):
                visit(getattr(value, field.name))
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)
        else:
            raise TypeError(f"unsupported scratch declaration: {type(value).__name__}")

    visit(values)
    return tuple(sorted(storage.items()))


def disposable_ranges(*, cache, scratch, workspaces, vision_scratch, decoders, owners, speculation):
    values = [scratch, workspaces.tensor, vision_scratch]
    for state in cache.values():
        values.extend((state.main_k_cache, state.main_v_cache, state.compressed_k_cache, state.raw_k_ring))
    for owner in (*decoders, *owners):
        sampler = getattr(owner, "sampler", None)
        if sampler is not None:
            values.append(sampler.scratch_tensors)
    if speculation is not None:
        from .speculative_model import PreparedPrefillAbsorption, PreparedSpeculativeRound

        values.append(speculation.mtp_scratch)
        for operation in speculation.owners:
            if isinstance(operation, PreparedPrefillAbsorption):
                values.extend((operation.target_residual, operation.previous, operation.previous_rope, operation.before))
            elif isinstance(operation, PreparedSpeculativeRound):
                values.extend(getattr(operation, name) for name in (
                    "target_residual", "before", "pending", "previous", "previous_rope", "active", "drafts",
                    "draft_probabilities", "draft_valid", "copy_mask", "raw_backup", "seed", "offset",
                    "base_offset", "sampled"))
                target, proposals, verify, _, _ = operation.operations
                for distribution in (target, *proposals):
                    values.extend(getattr(distribution, name) for name in (
                        "scores", "probabilities", "counts", "point_mass_ids", "maximum", "valid",
                        "packed_top_k", "packed_top_p"))
                for proposal in proposals:
                    values.extend((proposal.proposal.candidates, proposal.proposal.selected))
                values.extend(getattr(verify, name) for name in (
                    "safe_draft_probs", "safe_target_probs", "safe_ids", "seed_bits", "offset",
                    "independent_accepted", "emitted_drafts", "output"))
    return storage_ranges(values)
