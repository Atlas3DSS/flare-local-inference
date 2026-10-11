"""Startup phase rows in consoles of different widths."""

import io
import re

from rich.console import Console

from flare.progress import StartupProgress, phase


def rows(width, detail):
    stream = io.StringIO()
    progress = StartupProgress(stream=stream)
    progress.console = Console(file=stream, width=width, highlight=False)
    with phase(progress, "Select GPUs") as stage:
        stage.update(detail)
    return stream.getvalue().splitlines()


def test_a_wide_console_prints_the_whole_row():
    line = rows(120, "GPU 0 · concurrency 4 · up to 3 draft tokens")[-1]
    assert re.fullmatch(r"\[    OK    \] Select GPUs\.  GPU 0 · concurrency 4 · up to 3 draft tokens  \d+\.\ds", line)


def test_a_narrow_console_shortens_the_detail_and_keeps_status_and_timing():
    line = rows(60, "GPUs 0-1 of CUDA_VISIBLE_DEVICES=3,1 · concurrency 4 · exactly 2 draft tokens")[-1]
    assert len(line) == 60 and line.startswith("[    OK    ] Select GPUs.  GPUs 0-1 of CUDA_VISIBLE_D")
    assert re.search(r"…  \d+\.\ds$", line)
