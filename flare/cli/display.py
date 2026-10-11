"""Reports for the commands that print and exit.

A terminal receives styled text. Redirected output carries the same text
without styles, wrapping, or trailing spaces.
"""

import sys

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


def table(headings, rows, *, numeric=()):
    """Lay out rows under dim headings, right-aligning the columns indexed by `numeric`."""
    from rich import box
    from rich.table import Table

    grid = Table(box=box.SIMPLE, show_edge=False, pad_edge=False, padding=(0, 2, 0, 0), header_style="dim")
    for index, heading in enumerate(headings):
        grid.add_column(heading, justify="right" if index in numeric else "left", overflow="fold",
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


def fit(plan) -> str:
    """Summarize an estimated plan: its GPU count and about how many tokens each concurrent request gets."""
    def thousands(tokens):
        return f"{tokens // 1000}K"

    recipe, noun = plan.recipe, "requests" if plan.concurrency > 1 else "request"
    text = (f"{plan.world_size} GPU{'s' if plan.world_size > 1 else ''} · {plan.concurrency} {noun}"
            f" of about {thousands(plan.guaranteed_tokens)} tokens")
    whole = plan.tokens // recipe.context
    if plan.guaranteed_tokens < recipe.context and whole >= 1:
        text += f", or {whole} of {thousands(recipe.context)}"
    return text


def speculation(settings, *, brief: bool = False) -> str:
    """Describe speculation settings: the drafts verified per round and where they come from.

    The brief form fits a startup line and names the source only when it is not the MTP module alone.
    """
    if settings is None:
        return "speculation off" if brief else "off"
    drafts = settings.draft_tokens
    depth = f"{'up to' if settings.adaptive else 'exactly'} {drafts} draft token{'' if drafts == 1 else 's'}"
    if brief:
        source = " from history" if not settings.learned else " with history drafts" if settings.history_drafts else ""
        return depth + source
    source = ("request history" if not settings.learned
              else "MTP module and request history" if settings.history_drafts else "MTP module")
    return f"{depth} per round · {source}"
