"""Admission limits and power-of-two capacities for prepared request batches."""

MAX_CONCURRENCY = 64
BATCH_CAPACITIES = (1, 2, 4, 8, 16, 32, 64)


def request_capacity(concurrency: int) -> int:
    """Pad request storage to the largest graph needed for the admission limit."""
    if type(concurrency) is not int or not 1 <= concurrency <= MAX_CONCURRENCY:
        raise ValueError(f"concurrency must be an integer in [1, {MAX_CONCURRENCY}]")
    return 1 << (concurrency - 1).bit_length()


def prefill_capacities(chunk_size: int) -> tuple[int, ...]:
    """Token capacities of prepared prefill graphs; Helios pads a chunk to the smallest that holds it.

    Capacities double from 128 tokens to a step of at least 512 tokens and about an eighth of the
    chunk, then grow by that step, so a partial chunk pads by less than one step; the chunk size
    itself is last. Helios's `prefill_capacities` must agree.
    """
    step = max(512, chunk_size // 8 // 512 * 512)
    capacities, capacity = [], 128
    while capacity < min(chunk_size, step):
        capacities.append(capacity)
        capacity *= 2
    capacity = step
    while capacity < chunk_size:
        capacities.append(capacity)
        capacity += step
    return (*capacities, chunk_size)


def batch_capacities(concurrency: int) -> tuple[int, ...]:
    capacity = request_capacity(concurrency)
    return tuple(batch for batch in BATCH_CAPACITIES if batch <= capacity)


def speculative_shapes(concurrency, widths, fixed_width, budget):
    """Spend graph capacity on small batches and wide verification first."""
    batches = batch_capacities(concurrency)
    if (not widths or widths[0] != 1 or tuple(sorted(set(widths))) != tuple(widths)
            or any(type(w) is not int or not 1 <= w <= 8 for w in widths)):
        raise ValueError("speculative widths must increase from one through at most eight")
    if fixed_width is not None and (type(fixed_width) is not int or fixed_width not in widths):
        raise ValueError("fixed speculative width must belong to the configured widths")
    if type(budget) is not int or not len(batches) <= budget <= 56:
        raise ValueError("speculative graph budget must cover every batch capacity and be at most 56")
    choices = (fixed_width,) if fixed_width is not None else tuple(reversed(widths))
    remaining, exhausted, shapes = budget, False, []
    for index, batch in enumerate(batches):
        available = remaining - (len(batches) - 1 - index)
        selected = (1,) if exhausted else choices[:available]
        exhausted |= len(selected) < len(choices)
        shapes.extend((batch, width) for width in selected)
        remaining -= len(selected)
    return tuple(shapes)
