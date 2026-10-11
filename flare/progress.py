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
        self.console = progress.console if isinstance(progress, StartupProgress) else None
        self.completed = 0
        self.finished = False
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
        from rich.text import Text

        now = monotonic()
        row = Text("[", overflow="ellipsis", no_wrap=True)
        if self.failed:
            row.append("  FAILED  ", style="bold red")
        elif self.finished:
            row.append("    OK    ", style="bold green")
        else:
            position = int((now - self.started) * 8) % 16
            position = min(position, 15 - position)
            for index in range(10):
                row.append("-", style="bold cyan" if position <= index < position + 3 else "dim")
        row.append(f"] {self.label}")
        count = f"{self.completed}/{self.total}  " if self.total is not None else ""
        row.append(f"  {count}{now - self.started:.1f}s", style="dim")
        if not self.detail:
            return row
        detail = Text(f"             {self.detail}", style="dim", overflow="ellipsis", no_wrap=True)
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
            display.console.print(stage)
    elif progress is not None:
        progress(f"{label} started")
    try:
        yield stage
    except BaseException:
        stage.failed = True
        raise
    else:
        stage.finished = True
        if stage.total is not None:
            stage.completed = stage.total
    finally:
        if stage.live is not None:
            stage.live.stop()
        elif progress is not None:
            message = f"{label} {'failed' if stage.failed else 'complete'} in {monotonic() - stage.started:.2f}s"
            if display is None:
                progress(message)
            else:
                display.console.print(stage)
        if display is not None:
            display.active = previous
