"""Join tensor-parallel ranks on separate hosts through rank zero's rendezvous.

Each host starts its own rank process. Rank zero listens on an address every
peer can reach, such as a DGX Spark's RoCE fabric address. A peer knocks on
that rendezvous with its rank, the world size, its source identity, and its
host topology. Rank zero admits every rank once and later hands each peer the
preparation it runs. Ranks then form a Gloo group through a TCP store on rank
zero, which collectives use to exchange their RDMA endpoints.

Every peer holds two TCP connections to rank zero. The control connection
carries the handshake, the preparation assignment, memory reports and the serve
command, and stays open for the group's lifetime: losing it ends the process on
the other side, because a rank cannot complete collectives without its peers.
The executor connection carries Helios's per-step coordination.
"""

import fcntl
import hashlib
import json
import os
import pickle
import secrets
import select
import socket
import struct
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

PROTOCOL = 1
# Peers wait here while rank zero compiles the complete serving catalog.
_PREPARATION_TIMEOUT = timedelta(hours=2)
_HANDSHAKE_SECONDS = 30
_MAX_MESSAGE = 1 << 30
_SHUTDOWN_SECONDS = 120
_JSON, _PICKLE = b"J", b"P"


def _send(sock, kind, payload):
    sock.sendall(kind + struct.pack("<Q", len(payload)) + payload)


def _receive_exact(sock, count):
    data = bytearray()
    while len(data) < count:
        chunk = sock.recv(min(count - len(data), 1 << 20))
        if not chunk:
            raise ConnectionError("rank rendezvous connection closed")
        data += chunk
    return bytes(data)


def _receive(sock, kind):
    header = _receive_exact(sock, 9)
    size = struct.unpack("<Q", header[1:])[0]
    if header[:1] != kind or size > _MAX_MESSAGE:
        raise ConnectionError("unexpected rank rendezvous message")
    return _receive_exact(sock, size)


def send_json(sock, value):
    _send(sock, _JSON, json.dumps(value, sort_keys=True).encode())


def receive_json(sock):
    return json.loads(_receive(sock, _JSON))


def send_object(sock, value):
    _send(sock, _PICKLE, pickle.dumps(value))


def receive_object(sock):
    """Unpickle a message from rank zero, which every peer was told to trust."""
    return pickle.loads(_receive(sock, _PICKLE))


def source_identity():
    """Digest of the Python and native sources every rank must run identically."""
    from importlib import util

    # Locate FlashInfer without importing it: CUDA device selection must precede PyTorch's import.
    flashinfer = util.find_spec("flashinfer")
    if flashinfer is None:
        raise RuntimeError("the preparation environment requires FlashInfer")
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for source in (root / "ignition", root / "helios" / "src", root / "heliopause" / "src",
                   Path(flashinfer.origin).resolve().parent):
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".rs", ".c", ".cu", ".cuh", ".h"):
                digest.update(str(path.relative_to(source)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def _rdma_devices():
    devices = []
    for device in sorted(Path("/sys/class/infiniband").glob("*")):
        try:
            if "ACTIVE" in (device / "ports" / "1" / "state").read_text():
                devices.append(device.name)
        except OSError:
            continue
    return devices


def local_topology(gpu):
    """What a rank reports about its host: identity, its GPU, and its active RDMA devices."""
    return {"host": socket.gethostname(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "gpu": {"name": gpu.name, "uuid": gpu.uuid, "total_bytes": gpu.total_bytes,
                    "free_bytes": gpu.free_bytes, "capability": list(gpu.capability)},
            "rdma": _rdma_devices()}


def topology_gpu(topology, index):
    from ignition.planning import Gpu

    gpu = topology["gpu"]
    return Gpu(index, gpu["name"], gpu["uuid"], gpu["total_bytes"], gpu["free_bytes"],
               tuple(gpu["capability"]))


def interface_of(address):
    """Name of the local interface that holds an IPv4 address, for Gloo's device selection."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for _, name in socket.if_nameindex():
            try:
                packed = fcntl.ioctl(probe.fileno(), 0x8915, struct.pack("256s", name[:15].encode()))  # SIOCGIFADDR
            except OSError:
                continue
            if socket.inet_ntoa(packed[20:24]) == address:
                return name
    finally:
        probe.close()
    raise RuntimeError(f"no local interface holds {address}")


def _use_interface(address):
    # Gloo binds its sockets to the interface that reaches rank zero.
    os.environ["GLOO_SOCKET_IFNAME"] = interface_of(address)


def parse_address(text):
    host, colon, port = text.rpartition(":")
    if not colon or not port.isdecimal():
        raise ValueError(f"rank rendezvous address must be IPv4:PORT; got {text!r}")
    socket.inet_aton(host)
    return host, int(port)


def _watch_until_closed(sockets, finished, reason):
    """Exit this process when another rank's connection closes before coordinated shutdown."""
    def watch():
        poller = select.poll()
        for sock in sockets:
            poller.register(sock.fileno(), select.POLLRDHUP | select.POLLHUP | select.POLLERR)
        while True:
            events = poller.poll(200)
            if finished.is_set():
                return
            if events:
                print(f"flare: {reason}", flush=True)
                os._exit(1)

    thread = threading.Thread(target=watch, name="flare-rank-lifetime", daemon=True)
    thread.start()
    return thread


@dataclass
class _Peer:
    rank: int
    topology: dict
    control: socket.socket
    executor: socket.socket | None = None


class HostRendezvous:
    """Rank zero's listener: admits one process per peer rank, each on its own host."""

    def __init__(self, address, world, *, identity, topology):
        if world not in (2, 4):
            raise ValueError("multi-host tensor parallelism requires two or four ranks")
        self.address, self.world = parse_address(address), world
        self.identity, self.topology = identity, topology
        self.session = secrets.token_hex(16)
        self.listener = socket.create_server(self.address, reuse_port=False)
        self.peers = {}
        self.store = None
        # Set once coordinated shutdown begins: peers then exit and close their connections.
        self.stopping = threading.Event()

    @property
    def hosts(self):
        """Boot identity of every rank's host, in rank order."""
        return tuple(topology["boot_id"] for topology in self.topologies)

    @property
    def topologies(self):
        return (self.topology, *(self.peers[rank].topology for rank in range(1, self.world)))

    def accept(self, *, timeout=None, report=None):
        """Wait until every peer rank has knocked and opened its executor connection."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while len(self.peers) < self.world - 1 or any(peer.executor is None for peer in self.peers.values()):
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                missing = [rank for rank in range(1, self.world)
                           if rank not in self.peers or self.peers[rank].executor is None]
                raise TimeoutError(f"tensor-parallel ranks {missing} did not join {self.address[0]}:{self.address[1]}")
            self.listener.settimeout(remaining)
            try:
                sock, origin = self.listener.accept()
            except TimeoutError:
                continue
            try:
                self._admit(sock, origin, report)
            except (OSError, ValueError, ConnectionError) as error:
                if report is not None:
                    report(f"rejected rank connection from {origin[0]}: {error}")
                sock.close()
        self.listener.close()
        boots = self.hosts
        if len(set(boots)) != len(boots):
            raise ValueError("every multi-host tensor-parallel rank must run on a different host")
        return self.topologies

    def _admit(self, sock, origin, report):
        sock.settimeout(_HANDSHAKE_SECONDS)
        hello = receive_json(sock)
        rank = hello.get("rank")

        def reject(reason):
            send_json(sock, {"accepted": False, "reason": reason})
            raise ValueError(reason)

        if hello.get("protocol") != PROTOCOL:
            reject(f"rank protocol {hello.get('protocol')} differs from {PROTOCOL}")
        if type(rank) is not int or not 1 <= rank < self.world:
            reject(f"rank {rank!r} is outside 1..{self.world - 1}")
        if hello.get("world") != self.world:
            reject(f"world size {hello.get('world')} differs from rank zero's {self.world}")
        if hello.get("channel") == "control":
            if hello.get("identity") != self.identity:
                reject("source identity differs from rank zero's; every host must run the same checkout")
            if rank in self.peers:
                reject(f"rank {rank} has already joined")
            self.peers[rank] = _Peer(rank, hello["topology"], sock)
            send_json(sock, {"accepted": True, "session": self.session})
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if report is not None:
                report(f"rank {rank} joined from {hello['topology']['host']} ({origin[0]})")
        elif hello.get("channel") == "executor":
            peer = self.peers.get(rank)
            if hello.get("session") != self.session or peer is None or peer.executor is not None:
                reject(f"rank {rank} executor connection does not match an admitted control connection")
            peer.executor = sock
            send_json(sock, {"accepted": True})
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        else:
            reject(f"unknown rank channel {hello.get('channel')!r}")

    def prepare(self, snapshot, config, prepare, *, checkpoint, progress=None, package_storage=None):
        """Prepare every rank before returning the root-owned group; any failure ends every rank.

        With `package_storage`, every rank prepares into its own package storage for export.
        """
        import torch.distributed as dist

        from ignition.progress import phase

        if dist.is_initialized():
            raise RuntimeError("rank preparation requires exclusive ownership of the control group")
        controls = [self.peers[rank].control for rank in range(1, self.world)]
        _watch_until_closed(controls, self.stopping, "a tensor-parallel peer disconnected before shutdown")
        try:
            with phase(progress, "Connect tensor-parallel ranks"):
                _use_interface(self.address[0])
                # Peers connect only after receiving the store's port, so rank zero does not wait for them here.
                self.store = dist.TCPStore(self.address[0], 0, self.world, True, timeout=_PREPARATION_TIMEOUT,
                                           wait_for_workers=False)
                assignment = {"store": (self.address[0], self.store.port), "checkpoint": checkpoint,
                              "config": config, "prepare": prepare, "packaged": package_storage is not None}
                for control in controls:
                    send_object(control, assignment)
                dist.init_process_group("gloo", store=self.store, rank=0, world_size=self.world,
                                        timeout=_PREPARATION_TIMEOUT)
            model = prepare(snapshot, config, rank=0, group=dist.group.WORLD, progress=progress,
                            **({} if package_storage is None else {"package_storage": package_storage}))
            model.manifest()
            with phase(progress, "Wait for tensor-parallel ranks", total=len(controls)) as stage:
                reports = []
                for control in controls:
                    message = receive_object(control)
                    if "report" not in message:
                        raise RuntimeError(f"peer preparation failed: {message}")
                    reports.append(message["report"])
                    stage.update(advance=1)
            return HostRankGroup(self, model, reports)
        except BaseException:  # noqa: BLE001 - Preparation interruption must end every rank.
            traceback.print_exc()
            print("flare: terminating tensor-parallel rank group: preparation failed", flush=True)
            os._exit(1)

    def finish(self):
        """Wait for every peer to report its exit, then release the rendezvous."""
        import torch.distributed as dist

        self.stopping.set()
        deadline = time.monotonic() + _SHUTDOWN_SECONDS
        failures = []
        for rank in range(1, self.world):
            peer = self.peers[rank]
            peer.control.settimeout(max(1.0, deadline - time.monotonic()))
            try:
                message = receive_object(peer.control)
                if message.get("exit") != 0:
                    failures.append(f"rank {rank}: {message}")
            except (OSError, ConnectionError, EOFError) as error:
                failures.append(f"rank {rank}: {error}")
        for peer in self.peers.values():
            for sock in (peer.control, peer.executor):
                if sock is not None:
                    sock.close()
        if dist.is_initialized():
            dist.destroy_process_group()
        self.store = None
        if failures:
            raise RuntimeError(f"tensor-parallel peers did not exit cleanly: {failures}")


class HostRankGroup:
    """Root model and its remote peers through native coordinated shutdown."""

    def __init__(self, rendezvous, model, reports):
        self.rendezvous, self.model = rendezvous, model
        self.memory_report = {"world_size": rendezvous.world, "ranks": [model.memory_report, *reports]}
        self._started = self._closed = False

    @property
    def controls(self):
        return [self.rendezvous.peers[rank].control for rank in range(1, self.rendezvous.world)]

    @property
    def connections(self):
        return [self.rendezvous.peers[rank].executor for rank in range(1, self.rendezvous.world)]

    def manifest(self):
        return self.model.manifest()

    def export(self, path, *, options, export_id, images, storage, progress=None):
        """Write every rank's own package file on its host: rank zero's here, each peer's at `path` on its host."""
        from .model_package import export_rank_package

        if self._started or self._closed:
            raise RuntimeError("prepared rank group has already been consumed or closed")
        world = self.rendezvous.world
        for rank, control in enumerate(self.controls, 1):
            send_object(control, ("export", str(path), options, (world, rank, export_id)))
        export_rank_package(self.model, images, storage, path, options=options, cluster=(world, 0, export_id),
                            progress=progress)
        for rank, control in enumerate(self.controls, 1):
            reply = receive_object(control)
            if "exported" not in reply:
                raise RuntimeError(f"rank {rank} export failed: {reply}")

    def start_server(self, *, bind="127.0.0.1:8000", template_defaults=None, options=None):
        if self._started or self._closed:
            raise RuntimeError("prepared rank group has already been consumed or closed")
        try:
            for control in self.controls:
                send_object(control, "serve")
            self._started = True
            return self.model.start_server(bind=bind, template_defaults=template_defaults,
                                           peers=self.connections, options=options)
        except BaseException:  # noqa: BLE001 - Startup failure can strand peer collectives.
            traceback.print_exc()
            print("flare: terminating tensor-parallel rank group: native startup failed", flush=True)
            os._exit(1)

    def close(self):
        if self._closed:
            return
        self.rendezvous.stopping.set()
        try:
            if not self._started:
                for control in self.controls:
                    send_object(control, "close")
            self.model.close()
            self.rendezvous.finish()
            self._closed = True
        except BaseException:  # noqa: BLE001 - Failed drain requires process-level reclamation.
            traceback.print_exc()
            print("flare: terminating tensor-parallel rank group: cleanup could not confirm drain", flush=True)
            os._exit(1)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close()


def knock(address, rank, world, *, identity, topology, timeout=600.0, report=None):
    """Join rank zero's rendezvous, retrying until it listens; returns the control and executor sockets."""
    host, port = parse_address(address)
    deadline = time.monotonic() + timeout

    def connect(hello):
        while True:
            try:
                sock = socket.create_connection((host, port), timeout=_HANDSHAKE_SECONDS)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(1.0)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        send_json(sock, hello)
        reply = receive_json(sock)
        if not reply.get("accepted"):
            sock.close()
            raise RuntimeError(f"rank zero rejected rank {rank}: {reply.get('reason')}")
        sock.settimeout(None)
        return sock, reply

    control, reply = connect({"protocol": PROTOCOL, "channel": "control", "rank": rank, "world": world,
                              "identity": identity, "topology": topology})
    executor, _ = connect({"protocol": PROTOCOL, "channel": "executor", "rank": rank, "world": world,
                           "session": reply["session"]})
    if report is not None:
        report(f"joined rank zero at {host}:{port} as rank {rank} of {world}")
    return control, executor


def serve_peer(control, executor, rank, world, *, resolve_snapshot, progress=None):
    """Run one peer rank: prepare its assignment, report memory, and serve until coordinated drain.

    Any failure exits the process; rank zero then ends the group.
    """
    import signal

    finished = threading.Event()
    _watch_until_closed([control], finished, "rank zero disconnected before shutdown")
    # Rank zero coordinates shutdown; a terminal interrupt must not end a peer mid-drain.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        assignment = receive_object(control)
        snapshot = resolve_snapshot(assignment["checkpoint"])
        config, prepare = assignment["config"], assignment["prepare"]
        store_host, store_port = assignment["store"]
        _use_interface(control.getsockname()[0])
        import torch.distributed as dist

        store = dist.TCPStore(store_host, store_port, world, False, timeout=_PREPARATION_TIMEOUT)
        dist.init_process_group("gloo", store=store, rank=rank, world_size=world, timeout=_PREPARATION_TIMEOUT)
        recorder = storage = None
        if assignment.get("packaged"):
            import torch

            from .module_images import ModuleImages
            from .package_storage import PackageStorage

            recorder = ModuleImages()
            with recorder:
                torch.cuda.set_device(0)
                storage = PackageStorage(rank=rank, world=world, device=0)
                with storage:
                    model = prepare(snapshot, config, rank=rank, group=dist.group.WORLD, progress=progress,
                                    package_storage=storage)
        else:
            model = prepare(snapshot, config, rank=rank, group=dist.group.WORLD, progress=progress)
        model.manifest()
        send_object(control, {"report": model.memory_report})
        command = receive_object(control)
        if isinstance(command, tuple) and command[0] == "export":
            from .model_package import export_rank_package

            _, path, options, cluster = command
            if progress is not None:
                progress(f"exporting rank {rank} to {path}")
            export_rank_package(model, recorder, storage, path, options=options, cluster=cluster)
            send_object(control, {"exported": path})
            command = receive_object(control)
        if command == "serve":
            if progress is not None:
                progress("serving")
            model.serve_rank(executor, rank=rank, world=world)
            if progress is not None:
                progress("drained")
        elif command != "close":
            raise RuntimeError("invalid prepared rank handoff")
        executor.close()
        model.close()
        if storage is not None:
            for region in storage.host_regions:
                region.close()
        dist.destroy_process_group()
        finished.set()
        send_object(control, {"exit": 0})
        control.close()
    except BaseException:  # noqa: BLE001 - Never unwind CUDA resources around a stranded collective.
        traceback.print_exc()
        try:
            send_object(control, {"exit": 1, "error": traceback.format_exc(limit=3)})
        except OSError:
            pass
        os._exit(1)
