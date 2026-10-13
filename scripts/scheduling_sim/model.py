"""Synthetic execution costs, observable request state, and legal actions."""

from dataclasses import dataclass
from math import ceil, isfinite


def expected_tokens(probabilities: tuple[float, ...], drafts: int) -> float:
    if not 0 <= drafts <= len(probabilities) or any(not 0 <= p <= 1 for p in probabilities):
        raise ValueError("draft count and acceptance probabilities must define a valid prefix")
    survival = 1.0
    total = 1.0
    for probability in probabilities[:drafts]:
        survival *= probability
        total += survival
    return total


def token_outcomes(probabilities: tuple[float, ...], drafts: int) -> tuple[tuple[int, float], ...]:
    """Emitted counts and probabilities before EOS, including the mandatory token."""
    expected_tokens(probabilities, drafts)
    survival = 1.0
    outcomes = []
    for depth, probability in enumerate(probabilities[:drafts]):
        outcomes.append((depth + 1, survival * (1 - probability)))
        survival *= probability
    outcomes.append((drafts + 1, survival))
    return tuple(outcomes)


def delivery_reference(first_token_ms: float, decode_ms: float, token: int) -> float:
    """Frozen reference delivery time for a one-based output position."""
    if token < 1 or first_token_ms <= 0 or decode_ms <= 0:
        raise ValueError("delivery references require positive costs and a positive token position")
    return first_token_ms + (token - 1) * decode_ms


@dataclass(frozen=True)
class Request:
    name: str
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int
    acceptance: tuple[float, ...] = (0.75,) * 5
    cached_tokens: int = 0

    def __post_init__(self):
        if (
            not self.name
            or not isfinite(self.arrival_ms)
            or self.arrival_ms < 0
            or self.prompt_tokens < 0
            or self.cached_tokens < 0
            or self.output_tokens < 1
            or any(not 0 <= p <= 1 for p in self.acceptance)
        ):
            raise ValueError("invalid request dimensions or acceptance probabilities")


@dataclass(frozen=True)
class View:
    """Scheduler inputs exclude output length and future acceptance outcomes."""

    index: int
    arrival_ms: float
    remaining_prefill: int
    context: int
    last_service_ms: float
    lag_ms: float
    probabilities: tuple[float, ...]
    prefill_credit_ms: float
    decode_credit_ms: float
    generated_tokens: int
    first_token_reference_ms: float


@dataclass(frozen=True)
class Work:
    index: int
    prefill: int = 0
    drafts: int = 0


@dataclass(frozen=True)
class Action:
    work: tuple[Work, ...]

    @property
    def indices(self) -> set[int]:
        return {w.index for w in self.work}

    @property
    def mixed(self) -> bool:
        return any(w.prefill for w in self.work) and any(not w.prefill for w in self.work)


@dataclass(frozen=True)
class Config:
    quantum_ms: float = 20.0
    urgent_after_ms: float = 40.0
    lag_scale_ms: float = 20.0
    max_batch: int = 8
    resident_limit: int = 64
    max_drafts: int = 5
    chunks: tuple[int, ...] = (64, 128, 256, 512, 1024)
    mixing: bool = True
    acceptance_mode: str = "learned"
    reference: str = "ordinary"
    explore_every: int = 16
    service_gap_target_ms: float | None = None

    def __post_init__(self):
        if any(not isfinite(x) or x <= 0 for x in (
            self.quantum_ms, self.urgent_after_ms, self.lag_scale_ms
        )):
            raise ValueError("time budgets must be finite and positive")
        if self.max_batch < 1 or self.resident_limit < 1 or self.max_drafts < 0 or self.explore_every < 0:
            raise ValueError("invalid execution capacities")
        if not self.chunks or tuple(sorted(set(self.chunks))) != self.chunks or self.chunks[0] < 1:
            raise ValueError("prefill chunks must be positive, sorted, and unique")
        if self.acceptance_mode not in ("learned", "oracle"):
            raise ValueError("acceptance mode must be learned or oracle")
        if self.reference not in ("ordinary", "speculative"):
            raise ValueError("reference must be ordinary or speculative")
        if self.service_gap_target_ms is not None and (
            not isfinite(self.service_gap_target_ms) or self.service_gap_target_ms <= 0
        ):
            raise ValueError("service gap target must be finite and positive")


@dataclass(frozen=True)
class CostModel:
    """Milliseconds in a synthetic roofline model; coefficients are not GPU measurements."""

    launch_ms: float = 0.15
    weight_ms: float = 3.0
    target_token_ms: float = 0.022
    attention_ms_per_ktoken: float = 0.004
    draft_step_ms: float = 0.35
    draft_token_ms: float = 0.025
    mixed_penalty_ms: float = 0.0
    shape: str = "rectangular"

    def __post_init__(self):
        values = (
            self.launch_ms, self.weight_ms, self.target_token_ms,
            self.attention_ms_per_ktoken, self.draft_step_ms,
            self.draft_token_ms, self.mixed_penalty_ms,
        )
        if any(not isfinite(x) or x < 0 for x in values) or self.weight_ms <= 0:
            raise ValueError("cost coefficients must be finite and nonnegative; weight_ms must be positive")
        if self.shape not in ("rectangular", "ragged"):
            raise ValueError("shape must be rectangular or ragged")

    def duration(self, action: Action, views: dict[int, View]) -> float:
        if not action.work or len(action.indices) != len(action.work):
            raise ValueError("an action must contain distinct requests")
        prefills = [w for w in action.work if w.prefill]
        decodes = [w for w in action.work if not w.prefill]
        if len(prefills) > 1:
            raise ValueError("the action catalog permits one prefill per action")
        for work in action.work:
            view = views[work.index]
            if work.prefill < 0 or work.drafts < 0:
                raise ValueError("negative work")
            if work.prefill:
                if work.drafts or work.prefill > view.remaining_prefill:
                    raise ValueError("invalid prefill work")
            elif view.remaining_prefill:
                raise ValueError("decode requires completed prefill")
        prefill_rows = sum(64 * ceil(w.prefill / 64) for w in prefills)
        deepest = max((w.drafts for w in decodes), default=0)
        if self.shape == "rectangular" and decodes:
            batch = 1 << (len(decodes) - 1).bit_length()
            decode_rows = batch * (deepest + 1)
            draft_rows = batch * deepest
        else:
            decode_rows = sum(w.drafts + 1 for w in decodes)
            draft_rows = sum(w.drafts for w in decodes)
        attention = sum(
            (w.prefill or (w.drafts + 1))
            * (views[w.index].context + (w.prefill or w.drafts + 1) / 2)
            / 1024
            for w in action.work
        )
        return (
            self.launch_ms
            + max(self.weight_ms, self.target_token_ms * (prefill_rows + decode_rows))
            + self.attention_ms_per_ktoken * attention
            + self.draft_step_ms * deepest
            + self.draft_token_ms * draft_rows
            + (self.mixed_penalty_ms if action.mixed else 0)
        )


def credit(work: Work, view: View) -> float:
    if work.prefill:
        return work.prefill * view.prefill_credit_ms
    return expected_tokens(view.probabilities, work.drafts) * view.decode_credit_ms
