"""Agent-owned encrypted TCP paths for preparation's exact checkpoint reads."""

import hashlib
import json
from pathlib import Path
import socket
import struct
import subprocess
import threading

from .checkpoint_source import SourceServer, _exact, _json


class RingSource:
    """Accept only the granted rank/stream slots on a private loopback source."""

    def __init__(self, source, key, identity, topologies, *, streams=8, orbit_file=None,
                 key_file=None, cluster=None):
        from ignition.cli.runtime import binary

        if not 1 <= streams <= 64:
            raise ValueError("checkpoint transfer streams must be between 1 and 64")
        self.source, self.key, self.identity = source, key, identity
        self.world, self.streams = len(topologies), streams
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.listener.settimeout(1)
        self.readers = {}
        self.error = None
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self.run, name="flare-checkpoint-ring", daemon=True)
        command = [str(binary()), "orbit", "checkpoint-route"]
        for flag, value in (("--orbit-file", orbit_file), ("--key-file", key_file), ("--cluster", cluster)):
            if value is not None:
                command.extend((flag, str(value)))
        request = dict(gpu_uuids=[topology["gpu"]["uuid"] for topology in topologies],
                       source_port=self.listener.getsockname()[1], streams=streams,
                       catalog_sha256=hashlib.sha256(_json(source.catalog)).hexdigest(), identity=identity)
        try:
            result = subprocess.run(command, input=json.dumps(request), text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(f"checkpoint ring authorization failed: {result.stderr.strip()}")
            self.grant = json.loads(result.stdout)
        except BaseException:
            self.listener.close()
            raise
        self.thread.start()

    def run(self):
        try:
            while len(self.readers) < (self.world - 1) * self.streams and not self.stopping.is_set():
                try:
                    connection, _ = self.listener.accept()
                except TimeoutError:
                    continue
                try:
                    connection.settimeout(30)
                    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    rank, stream = struct.unpack("<II", _exact(connection, 8))
                    if not 1 <= rank < self.world or not 0 <= stream < self.streams or (rank, stream) in self.readers:
                        raise ValueError("checkpoint channel is outside its granted stream slots")
                    reader = SourceServer(self.source, connection, self.key, rank, self.identity)
                    self.readers[rank, stream] = reader
                    reader.thread.start()
                except BaseException:
                    connection.close()
                    raise
        except BaseException as error:
            if not self.stopping.is_set():
                self.error = error
        finally:
            self.listener.close()

    def join(self):
        self.thread.join()
        if self.error is not None:
            raise RuntimeError("checkpoint ring admission failed") from self.error
        for reader in self.readers.values():
            reader.join()

    def close(self):
        self.stopping.set()
        self.thread.join()
        for reader in self.readers.values():
            try:
                reader.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        for reader in self.readers.values():
            reader.thread.join()
            reader.connection.close()


class RingConnection:
    """One local socket to a native helper that owns the encrypted fabric session."""

    def __init__(self, directory, grant, rank, stream):
        from ignition.cli.runtime import binary

        path = Path(directory) / "checkpoint-route.json"
        if not path.exists():
            path.write_text(json.dumps(grant))
            path.chmod(0o600)
        self.connection, child = socket.socketpair()
        self.connection.settimeout(30)
        self.log = (Path(directory) / f"checkpoint-stream-{stream}.log").open("w+")
        try:
            self.process = subprocess.Popen(
                [str(binary()), "checkpoint-connect", "--grant", str(path), "--rank", str(rank),
                 "--stream", str(stream), "--fd", str(child.fileno())],
                pass_fds=(child.fileno(),), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=self.log)
        except BaseException:
            self.connection.close()
            self.log.close()
            raise
        finally:
            child.close()

    def close(self, *, check=False):
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.connection.close()
        try:
            result = self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            result = -1
        self.log.seek(0)
        message = self.log.read()
        self.log.close()
        if check and result:
            raise RuntimeError(f"checkpoint fabric helper failed ({result}): {message}")
