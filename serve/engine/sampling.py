"""Token sampling with temperature, top-p, top-k, and repetition penalty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SamplingParams:
    """Parameters controlling token sampling."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1           # -1 = disabled.
    min_p: float = 0.0        # 0 = disabled. Filters tokens below min_p * max_prob.
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0    # Additive penalty for tokens that appeared.
    frequency_penalty: float = 0.0   # Additive penalty proportional to frequency.
    max_new_tokens: int = 256
    stop_token_ids: list[int] | None = None
    stop_sequences: list[list[int]] | None = None

    @staticmethod
    def greedy(max_new_tokens: int = 256, **kwargs) -> SamplingParams:
        return SamplingParams(temperature=0.0, max_new_tokens=max_new_tokens, **kwargs)


def sample(
    logits: torch.Tensor,
    params: SamplingParams,
    generated_ids: list[list[int]] | None = None,
) -> torch.Tensor:
    """Sample next tokens from logits. Returns [batch] int64 tensor.

    logits: [batch, vocab_size].
    generated_ids: per-request lists of previously generated tokens (for rep penalty).
    """
    batch = logits.shape[0]

    # Repetition, presence, and frequency penalties.
    if generated_ids is not None:
        for i in range(batch):
            if not generated_ids[i]:
                continue
            prev = torch.tensor(generated_ids[i], device=logits.device, dtype=torch.long)

            # Repetition penalty (multiplicative).
            if params.repetition_penalty != 1.0:
                unique = prev.unique()
                score = logits[i, unique]
                logits[i, unique] = torch.where(
                    score > 0,
                    score / params.repetition_penalty,
                    score * params.repetition_penalty,
                )

            # Presence penalty (additive, applied once per unique token).
            if params.presence_penalty != 0.0:
                unique = prev.unique()
                logits[i, unique] -= params.presence_penalty

            # Frequency penalty (additive, proportional to count).
            if params.frequency_penalty != 0.0:
                counts = torch.bincount(prev, minlength=logits.shape[-1]).to(logits.dtype)
                logits[i] -= params.frequency_penalty * counts

    # Greedy.
    if params.temperature == 0.0:
        return logits.argmax(dim=-1)

    # Temperature.
    logits = logits / params.temperature

    # Top-K.
    if params.top_k > 0:
        top_k = min(params.top_k, logits.shape[-1])
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    # Top-P (nucleus).
    if params.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > params.top_p
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

    # Min-P: filter tokens below min_p * max_prob.
    if params.min_p > 0.0:
        probs = torch.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1, keepdim=True).values
        logits = logits.masked_fill(probs < params.min_p * max_prob, float("-inf"))

    # Sample.
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def sample_batch(
    logits: torch.Tensor,
    params_list: list[SamplingParams],
    generated_ids: list[list[int]] | None = None,
) -> torch.Tensor:
    """Sample with per-request params. Returns [batch] int64 tensor.

    Fast path: if all params are identical, delegates to sample().
    Slow path: samples each request individually.
    """
    if len(params_list) == 1:
        return sample(logits, params_list[0], generated_ids)

    # Check if all params are identical (common case: same API call).
    p0 = params_list[0]
    if all(p == p0 for p in params_list[1:]):
        return sample(logits, p0, generated_ids)

    # Per-request sampling.
    results = []
    for i, params in enumerate(params_list):
        gids = [generated_ids[i]] if generated_ids else None
        tok = sample(logits[i:i+1], params, gids)
        results.append(tok)
    return torch.cat(results)
