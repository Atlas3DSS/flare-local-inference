"""Prepared-model ownership across native serving and coordinated graph drain."""

import json
from threading import get_ident
from types import MappingProxyType

import torch

from .execution import PreparedExecution


class PreparedModel(PreparedExecution):
    """Own frozen preparation resources until the native server drains."""

    def __init__(self, *, server_config, memory_report, session, close_resources=(), **kwargs):
        super().__init__(**kwargs)
        self.server_config = server_config
        self.memory_report = memory_report
        self.session = session
        self._close_resources = tuple(close_resources)
        self._rank_drained = False
        self.runtime = None
        self.closed = False
        self._startup_undrained = False
        self._preparation_thread = get_ident()

    def start_server(self, *, bind: str = "127.0.0.1:8000", template_defaults=None, peers=None, options=None):
        """Hand the prepared graphs to the native server; `options` adds server configuration keys."""
        import flare_helios

        with self._handoff_lock:
            if self._consumed or self.closed:
                raise RuntimeError("prepared model has already been consumed or closed")
            config = {**self.server_config, **(options or {}), "bind": bind,
                      "template_defaults": {} if template_defaults is None else template_defaults}
            manifest = self.manifest()
            torch.cuda.synchronize(self.device)
            try:
                arguments = {} if peers is None else {"peer_fds": [peer.fileno() for peer in peers]}
                server = flare_helios.Server(self, manifest, json.dumps(config), **arguments)
            except BaseException as error:
                if getattr(error, "resources_retained", False):
                    self._startup_undrained = True
                    self._consumed = True
                raise
            self.runtime = server
            self._consumed = True
            return server

    def serve_rank(self, socket, *, rank, world):
        super().serve_rank(socket, rank=rank, world=world)
        self._rank_drained = True

    def close(self):
        with self._handoff_lock:
            self._close_locked()

    def _close_locked(self):
        if self.closed:
            return
        if get_ident() != self._preparation_thread:
            raise RuntimeError("prepared model cleanup must run on its preparation thread")
        if self._startup_undrained:
            raise RuntimeError("native startup could not confirm stream drain; prepared resources remain retained")
        shutdown_error = None
        if self.runtime is not None:
            try:
                self.runtime.close()
            except BaseException as error:
                if not self.runtime.drained:
                    raise
                shutdown_error = error
            if not self.runtime.drained:
                raise RuntimeError("native streams did not drain; prepared resources remain retained")
        elif self._consumed and not self._rank_drained:
            raise RuntimeError("externally started executor must retain this model until its own shutdown")
        torch.cuda.synchronize(self.device)
        for graph in self.graphs.values():
            graph.reset()
        self.session.close()
        for resource in self._close_resources:
            resource.close()
        self._resources = ()
        self._buffers = MappingProxyType({})
        self._graphs = MappingProxyType({})
        self.session = None
        self._close_resources = ()
        self.runtime = None
        self.closed = True
        if shutdown_error is not None:
            raise shutdown_error

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        try:
            self.close()
        except BaseException as error:
            if value is None:
                raise
            value.add_note(f"prepared model cleanup failed: {error!r}")

