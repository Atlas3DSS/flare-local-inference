"""Admission limits and power-of-two capacities for prepared request batches."""

MAX_CONCURRENCY = 64
BATCH_CAPACITIES = (1, 2, 4, 8, 16, 32, 64)


def request_capacity(concurrency: int) -> int:
    """Pad request storage to the largest graph needed for the admission limit."""
    if type(concurrency) is not int or not 1 <= concurrency <= MAX_CONCURRENCY:
        raise ValueError(f"concurrency must be an integer in [1, {MAX_CONCURRENCY}]")
    return 1 << (concurrency - 1).bit_length()


def prefill_capacities(chunk_size: int) -> tuple[int, ...]:
    """Token capacities offered to Helios for prompt chunk scheduling.

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


def speculative_shapes(concurrency, widths):
    """Capture every verification width at every prepared batch capacity."""
    batches = batch_capacities(concurrency)
    if (not widths or widths[0] != 1 or tuple(sorted(set(widths))) != tuple(widths)
            or any(type(w) is not int or not 1 <= w <= 8 for w in widths)):
        raise ValueError("speculative widths must increase from one through at most eight")
    return tuple((batch, width) for batch in batches for width in reversed(widths))


def mixed_shapes(concurrency, chunk_size, widths=(1,)):
    """Cover prompt lengths and verification depths without their Cartesian product.

    Each batch retains every prompt bucket at width one and every other width
    at a common prompt bucket. The anchor minimizes the worst multiplicative
    distance to the shortest and longest chunks, balancing padding of short
    prompts against subdivision of long prompts. Pure graphs retain complete
    batch/width coverage. This is a bounded coverage policy, not a latency model.
    """
    speculative_shapes(concurrency, widths)
    prompts = prefill_capacities(chunk_size)
    anchor = min(prompts, key=lambda p: (max(p / prompts[0], prompts[-1] / p), p))
    return tuple((batch, width, prompt)
        for batch in reversed(batch_capacities(concurrency)) if batch > 1
        for width in reversed(widths)
        for prompt in reversed(prompts) if width == 1 or prompt == anchor)
