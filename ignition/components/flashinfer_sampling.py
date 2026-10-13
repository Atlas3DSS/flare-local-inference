"""Trace FlashInfer sampling operators with explicit persistent workspace inputs."""

import torch
from flashinfer import sampling
from flashinfer.utils import device_support_pdl


@torch.compiler.allow_in_graph
def filter_probabilities(logits, top_k, top_p, row_states, softmax_workspace):
    # FX traces the operators; Dynamo does not inspect cached native module loading.
    module = sampling.get_sampling_module()
    masked = module.top_k_mask_logits(logits, top_k, 0, row_states)
    probabilities = module.softmax(softmax_workspace, masked, None, 1.,
                                   device_support_pdl(logits.device))
    return sampling.top_p_renorm_probs(probabilities, top_p, is_deterministic=True)


@torch.compiler.allow_in_graph
def sample_probabilities(probabilities, seed, offset, return_valid=False):
    return sampling.sampling_from_probs(probabilities, seed=seed, offset=offset,
                                       deterministic=True, check_nan=False, return_valid=return_valid)


@torch.compiler.allow_in_graph
def verify_chain(draft_probs, draft_ids, target_probs, accepted, emitted, seed, offset):
    # Each row draws from its own seed and offset only, so its tokens do not depend on its batch slot.
    return sampling.chain_speculative_sampling(draft_probs, draft_ids, target_probs,
        accepted, emitted, deterministic=True, seed=seed, offset=offset, row_independent_rng=True)
