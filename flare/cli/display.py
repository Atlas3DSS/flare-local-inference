"""Reports for the commands that print and exit.

A terminal receives styled text. Redirected output carries the same text
without styles, wrapping, or trailing spaces.
"""

import sys
from pathlib import Path

from flare import planning

ACCENT = "bold #ff9c42"
GOOD, CAUTION, BAD = "green", "yellow", "red"
_GIB = 1 << 30
_PLAIN_WIDTH = 1000


def _console(stream, **options):
    from rich.console import Console

    return Console(file=stream, highlight=False, markup=False, emoji=False, **options)


def show(*blocks, stream=None) -> None:
    """Print report blocks separated by blank lines."""
    stream = sys.stdout if stream is None else stream
    parts = [part for index, block in enumerate(blocks) for part in (("", block) if index else (block,))]
    console = _console(stream)
    if console.is_terminal:
        console.print(*parts, sep="\n")
        return
    plain = _console(stream, width=_PLAIN_WIDTH)
    with plain.capture() as capture:
        plain.print(*parts, sep="\n")
    stream.write("".join(line.rstrip() + "\n" for line in capture.get().splitlines()))


def error(message: str) -> None:
    """Print a failure on stderr with its continuation lines aligned under the first."""
    from rich.text import Text

    _console(sys.stderr, soft_wrap=True).print(
        Text.assemble(("error:", "bold red"), " " + message.replace("\n", "\n       ")))


def gib(value: int) -> str:
    return f"{value / _GIB:.1f}"


def styled(text: str, style: str = "dim"):
    from rich.text import Text

    return Text(text, style=style)


def title(name: str, detail: str = ""):
    from rich.text import Text

    return Text.assemble((name, ACCENT), (f"  {detail}" if detail else "", "dim"))


def fields(rows):
    """Lay out (label, value) rows with the labels in one dim column."""
    from rich.table import Table

    grid = Table.grid(padding=(0, 3, 0, 0))
    grid.add_column(style="dim", no_wrap=True)
    # A narrow terminal folds a long identifier onto further lines; it never truncates one.
    grid.add_column(overflow="fold")
    for label, value in rows:
        grid.add_row(label, value)
    return grid


def table(headings, rows, *, numeric=(), footer=None):
    """Lay out rows under dim headings, right-aligning the columns indexed by `numeric`."""
    from rich import box
    from rich.table import Table

    grid = Table(box=box.SIMPLE, show_edge=False, pad_edge=False, padding=(0, 2, 0, 0), header_style="dim",
                 show_footer=footer is not None)
    for index, heading in enumerate(headings):
        grid.add_column(heading, footer="" if footer is None else footer[index],
                        justify="right" if index in numeric else "left", overflow="fold",
                        no_wrap=index == 0 or index in numeric)
    for row in rows:
        grid.add_row(*row)
    return grid


def section(heading, *blocks):
    """Place blocks under a heading, indented and separated by blank lines."""
    from rich.console import Group
    from rich.padding import Padding

    parts = [heading]
    for index, block in enumerate(blocks):
        if index:
            parts.append("")
        parts.append(Padding(block, (0, 0, 0, 2), expand=False))
    return Group(*parts)


def checkpoint_name(checkpoint) -> str:
    """Name a checkpoint by its directory, or by its repository and abbreviated revision."""
    if Path(checkpoint.repository).is_absolute():
        return checkpoint.repository
    return f"{checkpoint.repository}@{checkpoint.revision[:8]}"


def fit(plan) -> str:
    """Summarize a plan as its GPU count and the tokens its concurrent requests are guaranteed."""
    recipe, noun = plan.recipe, "requests" if plan.concurrency > 1 else "request"
    text = (f"{plan.world_size} GPU{'s' if plan.world_size > 1 else ''} · {plan.concurrency} {noun}"
            f" of {plan.guaranteed_tokens:,} tokens")
    whole = plan.tokens // recipe.context
    if plan.guaranteed_tokens < recipe.context and whole >= 1:
        text += f", or {whole} of {recipe.context:,}"
    return text


def plan(plan, *, restriction: str | None = None):
    """Describe a fitting plan: its settings, where each GPU's memory goes, and what requests are guaranteed."""
    recipe, checkpoint = plan.recipe, plan.checkpoint
    indices = planning.format_gpus(gpu.index for gpu in plan.gpus)
    names = " / ".join(sorted({gpu.name.removeprefix("NVIDIA ") for gpu in plan.gpus}))
    several = plan.world_size > 1
    noun = "requests" if plan.concurrency > 1 else "request"
    requests = f"{plan.concurrency} {noun}"

    settings = [
        ("checkpoint", checkpoint_name(checkpoint)),
        ("architecture", checkpoint.architecture),
        ("GPUs", f"{indices} · {names} · tensor-parallel {plan.world_size}"),
    ]
    if restriction is not None:
        settings.append(("", styled(f"indices follow CUDA_VISIBLE_DEVICES={restriction}")))
    settings += [
        ("concurrency", requests),
        ("context", f"{recipe.context:,} tokens"),
        ("prefill chunk", f"{plan.prefill_chunk:,} tokens"),
    ]
    if checkpoint.mapped_host_bytes:
        settings.append(("host memory", (f"{gib(checkpoint.mapped_host_bytes)} GiB · {recipe.mapped_host_name}"
                                         " mapped from the checkpoint")))

    cache = f"{plan.cache_states} states for prompt reuse"
    if plan.cache_states != plan.cache_target:
        cache += f"; {plan.cache_target} without memory pressure"
    pages = f"{plan.tokens:,} tokens"
    if plan.pages < plan.demand_pages:
        pages += (f"; full context for {requests} would take"
                  f" {gib(plan.demand_pages * plan.costs.page_bytes)} GiB")
    graphs = (styled("measured on this host", GOOD) if plan.measured
              else styled("estimate; never prepared on this host", CAUTION))
    memory = [
        ("weights", plan.weights_bytes, styled("checkpoint tensors")),
        ("graphs and scratch", plan.graph_bytes, graphs),
        ("live state", plan.live_bytes, styled(f"{plan.concurrency} active {noun}")),
        ("prefix cache", plan.cache_bytes, styled(cache)),
        ("KV pages", plan.page_bytes, styled(pages)),
        ("headroom", planning.HEADROOM_BYTES, styled("kept free through warm-up and graph capture")),
        ("reserve", plan.reserve_bytes, styled("[memory] reserve")),
    ]
    free = styled(f"least free of GPUs {indices}" if several else f"GPU {indices}")
    guarantee = (f"guarantees {plan.concurrency} × {plan.guaranteed_tokens:,} tokens at full concurrency;"
                 f" a single request up to {plan.context:,}")
    return section(
        title(checkpoint.alias),
        fields(settings),
        table(("memory per GPU" if several else "memory", "GiB", ""),
              [(name, gib(value), note) for name, value, note in memory], numeric={1},
              footer=("free", gib(plan.free_bytes), free)),
        guarantee)
