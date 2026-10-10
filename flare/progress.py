"""Terminal progress for model preparation; callbacks retain plain log output."""

from contextlib import contextmanager
import sys
from time import monotonic


class StartupProgress:
    def __init__(self, *, stream=None):
        from rich.console import Console

        self.console = Console(file=sys.stderr if stream is None else stream, highlight=False)
        self.active = None

    def __call__(self, message):
        if self.active is not None:
            self.active.update(message)


class _Phase:
    def __init__(self, progress, label, total):
        self.progress, self.label, self.total = progress, label, total
        self.completed = 0
        self.started = self.operation_started = monotonic()
        self.detail = ""
        self.live = None
        self.failed = False

    def update(self, detail=None, *, advance=0):
        if detail is not None:
            self.detail = str(detail)
            self.operation_started = monotonic()
        self.completed += advance
        if self.progress is not None and not isinstance(self.progress, StartupProgress) and detail is not None:
            self.progress(f"{self.label}: {self.detail}")

    def __rich__(self):
        from rich.console import Group
        from rich.progress_bar import ProgressBar
        from rich.table import Table
        from rich.text import Text

        now = monotonic()
        status = "failed" if self.failed else self.label
        color = "red" if self.failed else "cyan"
        row = Table.grid(expand=True, padding=(0, 1))
        row.add_column(no_wrap=True)
        row.add_column(ratio=1)
        row.add_column(no_wrap=True, justify="right")
        count = f"{self.completed}/{self.total}  " if self.total is not None else ""
        row.add_row(Text(status, style=f"bold {color}"),
            ProgressBar(total=self.total, completed=self.completed, pulse=self.total is None,
                        complete_style=color, finished_style="green", pulse_style=color),
            Text(f"{count}{now - self.started:.1f}s"))
        detail = Text(self.detail, style="dim", overflow="ellipsis", no_wrap=True)
        if self.detail:
            detail.append(f"  ·  {now - self.operation_started:.1f}s")
        return Group(row, detail)


@contextmanager
def phase(progress, label, *, total=None):
    """Display completed work and elapsed time without estimating unknown progress."""
    stage = _Phase(progress, label, total)
    display = progress if isinstance(progress, StartupProgress) else None
    previous = display.active if display is not None else None
    if display is not None:
        from rich.live import Live

        display.active = stage
        if display.console.is_terminal:
            stage.live = Live(stage, console=display.console, refresh_per_second=8,
                              redirect_stdout=False, redirect_stderr=False)
            stage.live.start()
        else:
            display.console.print(f"{label} started", markup=False)
    elif progress is not None:
        progress(f"{label} started")
    try:
        yield stage
    except BaseException:
        stage.failed = True
        raise
    else:
        if stage.total is None:
            stage.total = 1
        stage.completed = stage.total
    finally:
        if stage.live is not None:
            stage.live.stop()
        elif progress is not None:
            message = f"{label} {'failed' if stage.failed else 'complete'} in {monotonic() - stage.started:.2f}s"
            if display is None:
                progress(message)
            else:
                display.console.print(message, markup=False)
        if display is not None:
            display.active = previous
