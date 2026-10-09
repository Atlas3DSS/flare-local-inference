"""Rich-backed logging and progress helpers for `serve`."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)


_CONSOLE = Console(stderr=True)
_PROGRESS: Progress | None = None
_ENABLED = False


def configure_logging(level: str = "INFO", *, rank: int = 0) -> None:
    """Configure the shared `serve` logger."""

    global _ENABLED
    _ENABLED = rank == 0

    root = logging.getLogger("serve")
    root.handlers.clear()
    root.propagate = False
    root.setLevel(getattr(logging, level.upper()))

    if not _ENABLED:
        root.addHandler(logging.NullHandler())
        return

    handler = RichHandler(
        console=_CONSOLE,
        rich_tracebacks=True,
        show_path=False,
        show_time=False,
        markup=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    if name == "serve" or name.startswith("serve."):
        return logging.getLogger(name)
    return logging.getLogger(f"serve.{name}")


def _get_progress() -> Progress | None:
    global _PROGRESS
    if not _ENABLED:
        return None
    if _PROGRESS is None:
        _PROGRESS = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None, finished_style="green"),
            TaskProgressColumn(),
            DownloadColumn(binary_units=True),
            TransferSpeedColumn(),
            TimeElapsedColumn(),
            console=_CONSOLE,
            transient=False,
            expand=True,
        )
        _PROGRESS.start()
    return _PROGRESS


def _stop_progress_if_idle() -> None:
    global _PROGRESS
    if _PROGRESS is None:
        return
    if any(not task.finished for task in _PROGRESS.tasks):
        return
    _PROGRESS.stop()
    _PROGRESS = None


@dataclass
class LoadSession:
    """Model-load progress session shown on rank 0."""

    model_name: str
    backend: str
    total_bytes: int
    tp_world_size: int
    progress: Progress | None
    weights_task: TaskID | None = None
    fanout_task: TaskID | None = None
    layers_task: TaskID | None = None
    _pending_storage_bytes: int = 0
    _pending_fanout_bytes: int = 0
    _last_refresh_s: float = 0.0

    def __post_init__(self) -> None:
        if self.progress is None:
            return
        total = max(1, self.total_bytes)
        self.weights_task = self.progress.add_task(
            f"[bold cyan]Load {self.model_name}[/] [dim](TP={self.tp_world_size}, {self.backend})[/]",
            total=total,
        )
        fanout_desc = "[bold magenta]Fanout[/] [dim](tensor-parallel transport)[/]"
        self.fanout_task = self.progress.add_task(
            fanout_desc,
            total=None,
            visible=self.backend == "distributed" and self.tp_world_size > 1,
        )
        self._last_refresh_s = time.monotonic()

    def advance_storage(self, num_bytes: int) -> None:
        if self.weights_task is None:
            return
        self._pending_storage_bytes += num_bytes
        self._flush()

    def advance_fanout(self, num_bytes: int) -> None:
        if self.fanout_task is None or num_bytes <= 0:
            return
        self._pending_fanout_bytes += num_bytes
        self._flush()

    def start_layers(self, description: str, *, total: int) -> None:
        if self.progress is None:
            return
        if self.layers_task is not None:
            self.progress.remove_task(self.layers_task)
        self.layers_task = self.progress.add_task(
            f"[bold blue]{description}[/]",
            total=max(1, total),
        )

    def advance_layers(self, *, advance: int = 1, description: str | None = None) -> None:
        if self.layers_task is None or self.progress is None:
            return
        update = {}
        if description is not None:
            update["description"] = f"[bold blue]{description}[/]"
        self.progress.update(self.layers_task, advance=advance, **update)

    def finish(self) -> None:
        self._flush(force=True)
        if self.progress is None:
            return
        for task_id in (self.layers_task, self.fanout_task, self.weights_task):
            if task_id is None:
                continue
            self.progress.update(task_id, visible=False)
            self.progress.remove_task(task_id)
        _stop_progress_if_idle()

    def _flush(self, *, force: bool = False) -> None:
        if self.progress is None:
            return
        now = time.monotonic()
        if not force and now - self._last_refresh_s < 0.05:
            return
        if self.weights_task is not None and self._pending_storage_bytes:
            self.progress.update(self.weights_task, advance=self._pending_storage_bytes)
            self._pending_storage_bytes = 0
        if self.fanout_task is not None and self._pending_fanout_bytes:
            self.progress.update(self.fanout_task, advance=self._pending_fanout_bytes)
            self._pending_fanout_bytes = 0
        self._last_refresh_s = now


def start_load_session(
    *,
    model_name: str,
    backend: str,
    total_bytes: int,
    tp_world_size: int,
) -> LoadSession:
    return LoadSession(
        model_name=model_name,
        backend=backend,
        total_bytes=total_bytes,
        tp_world_size=tp_world_size,
        progress=_get_progress(),
    )


@dataclass
class StartupSession:
    """Warmup / compile / graph-capture progress shown on rank 0."""

    progress: Progress | None
    task_id: TaskID | None = None
    _last_refresh_s: float = 0.0

    def start(self, description: str, *, total: int) -> None:
        self.progress = _get_progress()
        if self.progress is None:
            return
        if self.task_id is not None:
            self.progress.update(self.task_id, visible=False)
            self.progress.remove_task(self.task_id)
        self.task_id = self.progress.add_task(
            f"[bold yellow]{description}[/]",
            total=max(1, total),
        )
        self._last_refresh_s = time.monotonic()

    def advance(self, *, advance: int = 1, description: str | None = None) -> None:
        if self.progress is None or self.task_id is None:
            return
        update = {"advance": advance}
        if description is not None:
            update["description"] = f"[bold yellow]{description}[/]"
        self.progress.update(self.task_id, **update)

    def finish(self) -> None:
        if self.progress is None or self.task_id is None:
            return
        self.progress.update(self.task_id, visible=False)
        self.progress.remove_task(self.task_id)
        self.task_id = None
        _stop_progress_if_idle()
        self.progress = None


def start_startup_session() -> StartupSession:
    return StartupSession(progress=_get_progress())
