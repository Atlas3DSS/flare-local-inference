"""Sufficient next-service feasibility checks using serial singleton rescues."""

from .model import Action, View


def singleton_costs(options: list[tuple[Action, float]]) -> dict[int, float]:
    costs = {}
    for action, duration in options:
        if len(action.work) == 1:
            index = action.work[0].index
            costs[index] = min(costs.get(index, float("inf")), duration)
    return costs


def rescue_latenesses(
    now: float, action: Action, duration: float, views: list[View],
    costs: dict[int, float], target_ms: float,
) -> dict[int, float]:
    """Pending-visit latenesses after the action and deadline-ordered singleton rescues."""
    end = now + duration
    included = action.indices
    latenesses = {v.index: end - v.last_service_ms - target_ms for v in views if v.index in included}
    for view in sorted(views, key=lambda v: (v.last_service_ms, v.index)):
        if view.index not in included:
            end += costs[view.index]
            latenesses[view.index] = end - view.last_service_ms - target_ms
    return latenesses
