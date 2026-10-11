"""Terminal progress for model preparation; callbacks retain plain log output."""

from contextlib import contextmanager
import sys
from time import monotonic


_BOOT_SPLASH = """
       ·                                           ╱      ·    ⠂
                                            ·   ⣠⠞  ⢀⣠⣤⣤⣄⡀   ✧
                                              ⣰⡿⣠⣾⠟⠋⠉⠉⠛⢿⣦⡀   ⠄
     ███████╗██╗      █████╗ ██████╗ ███████╗  ⣰⣿⣿⠟⠁  ⣠⠴⠒⠲⣄⠹⣷⡀
     ██╔════╝██║     ██╔══██╗██╔══██╗██╔════╝ ⢠⣿⡿⠁ ⢀⡾⠋      ⠹⡆⢹⣧ ·
     █████╗  ██║     ███████║██████╔╝█████╗   ⣾⣿⠃ ⣠⠟   ✦      ⢻⠈⡿ ⠂
     ██╔══╝  ██║     ██╔══██║██╔══██╗██╔══╝  ⣿⡏ ⣰⠏           ⡼ ⠸   ╱
     ██║     ███████╗██║  ██║██║  ██║███████╗ ⣿⣧⡾⠃       ⢀⡴⠋  ·  ⠊
     ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝ ⣿⡿⠁   ⢀⣠⠴⠚⠉    ⠄
                                          ⢀⣾⣿⡇ ⣠⡾⠋    ⠂    ✧
                           ⣀⣀⣀⡠⠤⠤⠴⠶⠶⣶⣾⣿⣿⣷⡿⠋⠶⠤⢄⣀⣀
            ⣀⣀⡠⠤⠔⠒⠋⠉⠉⡲⢔⢎⢕⡕⢜⡪⣠⡴⠛⠋⣠⣾⡿⠿⠿⣷⣄⢎⢕⠈⠙⠲⣄⡲⠉⠉⠒⠢⠤⢄⣀⣀
⣀⣀⡠⠤⠔⠒⠉⠉⢎⢕⡕⠠⣄⣀⡠⠔⠋⢕⡕⢜⡪⡱⡲⢔⠊⢕⡕⣠⠞⠋⡲⢔⠈⠙⢦⣄⠈⠙⠷⣤⡀⢕⡕⢜⡪⠈⠢⡀⢎⢕⡕⢜⡪⡱⡲⢔⠉⠉⠒⠤⢄⣀
⢔⢎⢕⡕⢜⡪⠤⠤⠒⠋⢕⡕⢜⡪⡱⡲⢔⢎⣀⡠⠤⠤⠒⠋⢔⢎⢕⡕⢜⡪⡱⡲⢔⢎⢕⡕⠈⠑⠢⣄⢔⢎⢕⠉⠓⠦⣄⣀⢔⢎⢕⠈⠂⡪⡱⡲⢔⢎⢕⡕⢜⡪
""".strip("\n")


class StartupProgress:
    def __init__(self, *, stream=None):
        from rich.console import Console

        self.console = Console(file=sys.stderr if stream is None else stream, highlight=False)
        self.active = None

    def splash(self):
        from rich.text import Text

        if not self.console.is_terminal:
            return
        colors = ("#a95348", "#e54b32", "#ff7033", "#ff9c42", "#ffb957", "#ffd27b",
                  "#ffd27b", "#ffb957", "#ff9c42", "#ffbf69", "#ffe8b0", "#ff9c42",
                  "#e54b32", "#96352e")
        art = Text(overflow="crop", no_wrap=True)
        for index, (line, color) in enumerate(zip(_BOOT_SPLASH.splitlines(), colors, strict=True)):
            row = Text(line, style=color)
            if index in (11, 12):
                left, right = (21, 46) if index == 11 else (8, 56)
                row.stylize("#ffd27b", 0, left)
                row.stylize("#ffd27b", right, len(line))
            for column, character in enumerate(line):
                if character in "⡪⢕⡲⢜⢎⡱⡕⢔":
                    row.stylize(("#c47636", "#a64d2d", "#71332a")[index - 11], column, column + 1)
                if not (3 <= index <= 8 and 5 <= column < 44) and character in "⣿⣾⣷✦":
                    row.stylize("bold #fff8e7", column, column + 1)
            if 3 <= index <= 8:
                row.stylize("bold #fff0c7", 5, 44)
            art.append_text(row)
            art.append("\n")
        self.console.print(art)

    def __call__(self, message):
        if self.active is not None:
            self.active.update(message)


class _Phase:
    def __init__(self, progress, label, total):
        self.progress, self.label, self.total = progress, label, total
        self.console = progress.console if isinstance(progress, StartupProgress) else None
        self.completed = 0
        self.finished = False
        self.started = monotonic()
        self.detail = ""
        self.live = None
        self.failed = False

    def update(self, detail=None, *, advance=0):
        if detail is not None:
            self.detail = str(detail)
        self.completed += advance
        if self.progress is not None and not isinstance(self.progress, StartupProgress) and detail is not None:
            self.progress(f"{self.label}: {self.detail}")

    def __rich__(self):
        from rich.progress_bar import ProgressBar
        from rich.table import Table
        from rich.text import Text

        now = monotonic()
        if self.failed:
            status = Text("  FAILED  ", style="bold red")
        elif self.finished:
            status = Text("    OK    ", style="bold green")
        else:
            status = ProgressBar(total=self.total, completed=self.completed, width=10,
                                 pulse=self.total is None, complete_style="cyan",
                                 finished_style="green", pulse_style="cyan")
        label = Text(f"] {self.label}.", overflow="ellipsis", no_wrap=True)
        if self.detail:
            label.append(f"  {self.detail}", style="dim")
        count = f"{self.completed}/{self.total}  " if self.total is not None else ""
        timing = Text(f"  {count}{now - self.started:.1f}s", style="dim")
        row = Table.grid(padding=0)
        row.add_column(width=1, no_wrap=True)
        row.add_column(width=10, no_wrap=True)
        row.add_column(no_wrap=True)
        row.add_column(width=len(timing.plain), no_wrap=True)
        row.add_row(Text("["), status, label, timing)
        return row


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
