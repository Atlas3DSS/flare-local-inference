"""Spawn preparation ranks and transfer their lifetime supervision to Rust."""

import os
import socket
import traceback
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import torch.distributed as dist
import torch.multiprocessing as mp


def _peer(parent, snapshot, config, rank, rendezvous, connection, control, prepare):
    import flare_helios

    flare_helios.bind_parent_lifetime(parent)
    try:
        dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=len(config.devices),
                                timeout=timedelta(minutes=15))
        model = prepare(snapshot, config, rank=rank, group=dist.group.WORLD)
        model.manifest()
        control.send(model.memory_report)
        command = control.recv()
        control.close()
        if command == "serve":
            model.serve_rank(connection, rank=rank, world=len(config.devices))
        elif command != "close":
            raise RuntimeError("invalid prepared rank handoff")
        connection.close()
        model.close()
        dist.destroy_process_group()
    except BaseException:  # noqa: BLE001 - Never unwind CUDA resources around a stranded collective.
        traceback.print_exc()
        os._exit(1)


class PreparedRankGroup:
    """Root model and owned peer processes through native coordinated shutdown."""

    def __init__(self, model, processes, connections, controls, supervisor, rendezvous, reports):
        self.model, self.processes, self.connections, self.controls = model, processes, connections, controls
        self.supervisor, self.rendezvous = supervisor, rendezvous
        self.memory_report = {"world_size": len(processes) + 1, "ranks": [model.memory_report, *reports]}
        self._started = self._closed = False

    def start_server(self, *, bind="127.0.0.1:8000", template_defaults=None):
        if self._started or self._closed:
            raise RuntimeError("prepared rank group has already been consumed or closed")
        try:
            for control in self.controls:
                control.send("serve")
                control.close()
            self._started = True
            server = self.model.start_server(bind=bind, template_defaults=template_defaults, peers=self.connections)
            for connection in self.connections:
                connection.close()
            return server
        except BaseException:  # noqa: BLE001 - Startup failure can strand peer collectives.
            traceback.print_exc()
            self.supervisor.abort("native rank-group startup failed")

    def close(self):
        if self._closed:
            return
        try:
            self.supervisor.begin_shutdown(60000)
            if not self._started:
                for control in self.controls:
                    control.send("close")
                    control.close()
            self.model.close()
            dist.destroy_process_group()
            self.supervisor.finish_shutdown()
            for process in self.processes:
                process.join()
            for connection in self.connections:
                connection.close()
            self.rendezvous.cleanup()
            self._closed = True
        except BaseException:  # noqa: BLE001 - Failed drain requires process-level reclamation.
            traceback.print_exc()
            self.supervisor.abort("rank-group cleanup could not confirm drain")

    def __enter__(self):
        return self

    def manifest(self):
        return self.model.manifest()

    def __exit__(self, kind, value, traceback):
        self.close()


def prepare_group(snapshot, config, prepare, *, progress=None):
    """Prepare every rank before returning a root-owned native handoff.

    This function owns the process's Gloo group and must run in the main thread.
    Peer failure or unconfirmed CUDA drain terminates this process and its peers.
    """
    import flare_helios

    if dist.is_initialized():
        raise RuntimeError("rank preparation requires exclusive ownership of the control group")
    rendezvous = TemporaryDirectory(prefix="flare-ranks-")
    address = Path(rendezvous.name, "gloo").as_uri()
    context = mp.get_context("spawn")
    processes, connections, controls = [], [], []
    try:
        for rank in range(1, len(config.devices)):
            root, peer = socket.socketpair()
            connections.append(root)
            control, child_control = context.Pipe()
            controls.append(control)
            process = context.Process(target=_peer, args=(os.getpid(), snapshot, config, rank,
                address, peer, child_control, prepare), name=f"flare-rank-{rank}")
            try:
                process.start()
            finally:
                peer.close()
                child_control.close()
            processes.append(process)
        supervisor = flare_helios.RankSupervisor([process.pid for process in processes])
    except BaseException:
        for process in processes:
            process.kill()
            process.join()
        for resource in (*connections, *controls):
            resource.close()
        rendezvous.cleanup()
        raise
    try:
        dist.init_process_group("gloo", init_method=address, rank=0, world_size=len(config.devices),
                                timeout=timedelta(minutes=15))
        model = prepare(snapshot, config, rank=0, group=dist.group.WORLD, progress=progress)
        model.manifest()
        reports = [control.recv() for control in controls]
        return PreparedRankGroup(model, processes, connections, controls, supervisor, rendezvous, reports)
    except BaseException:  # noqa: BLE001 - Preparation interruption must terminate every rank.
        traceback.print_exc()
        supervisor.abort("tensor-parallel preparation failed")
