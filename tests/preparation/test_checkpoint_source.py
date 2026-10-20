"""Checkpoint peers receive authenticated requested bytes and never weight files."""

from array import array
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import struct
from types import SimpleNamespace
from unittest import mock

import pytest

from ignition.preparation import checkpoint_source as transport
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.runtime_package import FileSource


def checkpoint(directory):
    directory.mkdir()
    selected, unused = bytes(range(32)), b"unrequested-engram" * 8192
    header = json.dumps({"selected": {"dtype": "U8", "shape": [4, 8], "data_offsets": [0, 32]},
                         "engram": {"dtype": "U8", "shape": [len(unused)],
                                    "data_offsets": [32, 32 + len(unused)]}}).encode()
    (directory / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + selected + unused)
    (directory / "config.json").write_text('{}')
    (directory / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 32 + len(unused)},
        "weight_map": {"selected": "weights.safetensors", "engram": "weights.safetensors"}}))
    return len(header) + 8


@contextmanager
def connected(tmp_path):
    original = tmp_path / "original"
    offset = checkpoint(original)
    description = SimpleNamespace(repository=str(original), revision="immutable")
    root = transport.RootSource(original, description)
    sender, receiver = socket.socketpair()
    sender.settimeout(5)
    receiver.settimeout(5)
    server = transport.SourceServer(root, sender, b"k" * 32, 1, "code")
    server.thread.start()
    client = None
    try:
        client = transport.RemoteSource(receiver, b"k" * 32, 1, "code", description, directory=tmp_path)
        server.wait_ready()
        yield client, server, offset
        if not client.finished and server.error is None:
            client.finish()
        if server.error is None:
            server.join()
    finally:
        receiver.close()
        sender.shutdown(socket.SHUT_RDWR)
        server.thread.join(5)
        assert not server.thread.is_alive()
        sender.close()
        if client is not None:
            client.close()
        root.close()


def test_only_selected_row_bytes_cross_the_wire(tmp_path):
    with connected(tmp_path) as (client, server, start):
        before = dict(client.stats)
        assert client.read("weights.safetensors", start + 1, 2, 4, 8) == bytes((1, 2, 9, 10, 17, 18, 25, 26))
        assert client.stats["payload_bytes"] - before["payload_bytes"] == 8
        assert server.stats == client.stats
        assert not list(client.snapshot.rglob("*.safetensors"))
        assert client.stats["payload_bytes"] == 8
        client.finish()
        server.join()
        # The authenticated end marker leaves the socket exactly at the prepared report boundary.
        receiver = client.channel.connection
        transport._packet(receiver, b"prepared report")
        assert transport._receive(server.connection) == b"prepared report"


def test_source_disables_read_ahead_before_serving_selective_ranges(tmp_path, monkeypatch):
    advise, read = os.posix_fadvise, os.pread
    random_access = set()
    reads = []

    def fadvise(descriptor, offset, length, advice):
        if advice == os.POSIX_FADV_RANDOM:
            random_access.add(descriptor)
        return advise(descriptor, offset, length, advice)

    def pread(descriptor, length, offset):
        assert descriptor in random_access
        reads.append((descriptor, length, offset))
        return read(descriptor, length, offset)

    monkeypatch.setattr(os, "posix_fadvise", fadvise)
    monkeypatch.setattr(os, "pread", pread)
    with connected(tmp_path) as (client, server, start):
        assert {held.descriptor for held in server.source.files.values()} == random_access
        assert client.read("weights.safetensors", start, 8, 2, 16) == bytes((*range(8), *range(16, 24)))
        assert reads
        assert client.stats == server.stats


def test_headers_and_file_sources_require_the_scoped_owner(tmp_path):
    with connected(tmp_path) as (client, _, start):
        with pytest.raises(RuntimeError, match="active source owner"):
            inspect_indexed_checkpoint(client.snapshot)
        with client.activate():
            manifest = inspect_indexed_checkpoint(client.snapshot)
            assert manifest.tensors["selected"].offset == start
            source = FileSource(client.snapshot / "weights.safetensors", start + 32, 64)
            assert source.nbytes == 64
            assert client.stats["payload_bytes"] == 0
        with pytest.raises(RuntimeError, match="active source owner"):
            transport.source_size(source.path)


@pytest.mark.parametrize("offset,width,rows,stride", [(-1, 1, 1, 0), (0, 1, 0, 0),
    (0, transport.BLOCK_BYTES + 1, 1, 0), (0, 2, 2, transport.BLOCK_BYTES), (1 << 63, 1, 1, 0)])
def test_invalid_row_requests_fail_before_transfer(tmp_path, offset, width, rows, stride):
    with connected(tmp_path) as (client, server, _):
        before = dict(server.stats)
        with pytest.raises(ValueError):
            client.read("weights.safetensors", offset, width, rows, stride)
        assert server.stats == before


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../b", "a\\b", "./weights.safetensors"])
def test_paths_cannot_escape_source(name):
    with pytest.raises(ValueError):
        transport.relative_name(name)


def test_source_mutation_aborts_the_same_reader(tmp_path):
    with connected(tmp_path) as (client, server, start):
        path = server.source.snapshot / "weights.safetensors"
        with path.open("r+b") as stream:
            stream.seek(start)
            stream.write(b"!")
        with pytest.raises(ConnectionError):
            client.read("weights.safetensors", start, 1)
        server.thread.join(5)
        assert isinstance(server.error, RuntimeError)
        assert "changed" in str(server.error)


def test_disconnect_aborts_reader_and_private_metadata_is_removed(tmp_path):
    with connected(tmp_path) as (client, server, _):
        directory = client.snapshot
        client.channel.connection.shutdown(socket.SHUT_RDWR)
        server.thread.join(5)
        assert isinstance(server.error, ConnectionError)
        client.close()
        assert not directory.exists()


def test_channel_rejects_tamper_and_replay():
    sender, receiver = socket.socketpair()
    try:
        root = transport.Channel(sender, b"k" * 32, server=True)
        peer = transport.Channel(receiver, b"k" * 32, server=False)
        root.send("data", b"exact bytes")
        assert peer.receive("data")[1] == b"exact bytes"
        root.sent = 0
        root.send("data", b"replayed")
        with pytest.raises(ValueError, match="sequence"):
            peer.receive("data")
        bad = transport.Channel(sender, b"x" * 32, server=True)
        bad.send("data", b"modified")
        with pytest.raises(ValueError, match="authentication"):
            peer.receive("data")
    finally:
        sender.close()
        receiver.close()


def test_end_marker_byte_accounting_is_authenticated(tmp_path):
    with connected(tmp_path) as (client, server, _):
        client.stats["payload_bytes"] += 1
        with pytest.raises(ConnectionError):
            client.finish()
        server.thread.join(5)
        assert "accounting" in str(server.error)


def test_adapter_preserves_packed_rows_and_cpu_pointer_records(tmp_path):
    from ignition.preparation.remote_weights import RemoteWeightSession

    with connected(tmp_path) as (client, _, start), client.activate():
        manifest = inspect_indexed_checkpoint(client.snapshot)
        adapter = RemoteWeightSession.__new__(RemoteWeightSession)
        adapter.source, adapter.checkpoint = client, manifest
        adapter.names = ("weights.safetensors",)
        path = tmp_path / "bounded-buffer"
        adapter.buffer_write = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        adapter.buffer_read = os.open(path, os.O_RDONLY)
        observed = []
        def execute(self, records):
            values = list(records)
            observed.append((values, None if values[4] == 2 else os.pread(values[0], values[2] * values[5], 0)))
        try:
            with mock.patch("b12x.loader._checkpoint.DirectWeightSession._execute", execute):
                adapter._execute(array("Q", (0, start + 1, 2, 1000, 0, 4, 8, 4,
                                               0, 9000, 4, 2000, 2, 1, 0, 0)))
            assert observed[0][1] == bytes((1, 2, 9, 10, 17, 18, 25, 26))
            assert observed[0][0][2:] == [2, 1000, 0, 4, 2, 4]
            assert observed[1][0] == [0, 9000, 4, 2000, 2, 1, 0, 0]
            assert client.stats["payload_bytes"] == 8
            observed.clear()
            with mock.patch("b12x.loader._checkpoint.DirectWeightSession._execute", execute):
                adapter._execute(array("Q", (0, start, 2, 3000, 0, 3, 8, 8,
                                               0, start + 2, 2, 3002, 0, 3, 8, 8)))
            assert [value for _, value in observed] == [bytes((0, 1, 8, 9, 16, 17)),
                                                       bytes((2, 3, 10, 11, 18, 19))]
            with pytest.raises(ValueError, match="overlapping"), \
                    mock.patch("b12x.loader._checkpoint.DirectWeightSession._execute", execute):
                adapter._execute(array("Q", (0, start, 2, 3000, 0, 3, 8, 8,
                                               0, start + 2, 2, 3001, 0, 3, 8, 8)))
        finally:
            os.close(adapter.buffer_write)
            os.close(adapter.buffer_read)


def test_wide_rows_preserve_bf16_expansion_and_destination_stride():
    from ignition.preparation import remote_weights

    with mock.patch.object(remote_weights, "BLOCK_BYTES", 8):
        actual = list(remote_weights.chunks(100, 12, 2, 20, 1000, 32, 1))
    assert actual == [(100, 8, 1, 20, 1000, 32), (120, 8, 1, 20, 1032, 32),
                      (108, 4, 1, 20, 1016, 32), (128, 4, 1, 20, 1048, 32)]
    assert list(remote_weights.chunks(0, 0, 1, 0, 0, 0, 0)) == []


def test_strided_destination_intersection_matches_written_bytes():
    import random
    from ignition.preparation.remote_weights import destinations_overlap

    randomizer = random.Random(192)
    for _ in range(500):
        def descriptor():
            width = randomizer.randint(1, 12)
            return (randomizer.randint(0, 64), width, randomizer.randint(1, 12),
                    width + randomizer.randint(0, 16))
        left, right = descriptor(), descriptor()
        def touched(value):
            start, width, rows, stride = value
            return {start + row * stride + column for row in range(rows) for column in range(width)}
        assert destinations_overlap(left, right) == bool(touched(left) & touched(right))


def test_shared_cache_release_skips_virtual_tensor_files(tmp_path):
    from ignition.preparation.host_memory import release_page_cache

    with connected(tmp_path) as (client, _, _), client.activate():
        release_page_cache([client.snapshot / "weights.safetensors"])


def test_shared_scalar_read_uses_only_its_declared_bytes(tmp_path):
    import torch

    with connected(tmp_path) as (client, _, start), client.activate():
        manifest = inspect_indexed_checkpoint(client.snapshot)
        selected = transport.read_small_tensor(manifest, "selected", limit=32)
        assert torch.equal(selected, torch.arange(32, dtype=torch.uint8).reshape(4, 8))
        assert client.stats["payload_bytes"] == 32
        with pytest.raises(ValueError, match="budget"):
            transport.read_small_tensor(manifest, "engram", limit=1024)


def test_private_token_is_required(tmp_path):
    path = tmp_path / "credential"
    path.write_bytes(b"k" * 32)
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        transport.token(path)
    path.chmod(0o600)
    assert transport.token(path) == b"k" * 32


def test_wrong_peer_credential_cannot_read_the_catalog(tmp_path):
    original = tmp_path / "original"
    checkpoint(original)
    description = SimpleNamespace(repository=str(original), revision="immutable")
    root = transport.RootSource(original, description)
    sender, receiver = socket.socketpair()
    sender.settimeout(5)
    receiver.settimeout(5)
    server = transport.SourceServer(root, sender, b"k" * 32, 1, "code")
    server.thread.start()
    try:
        with pytest.raises(ConnectionError):
            transport.RemoteSource(receiver, b"wrong" * 8, 1, "code", description, directory=tmp_path)
        server.thread.join(5)
        assert "authentication" in str(server.error)
        assert not list(tmp_path.glob("flare-checkpoint-*"))
    finally:
        sender.close()
        receiver.close()
        root.close()


def test_completed_root_reads_release_aligned_page_cache(tmp_path):
    with connected(tmp_path) as (client, server, start):
        with mock.patch.object(transport.os, "posix_fadvise") as advise:
            client.read("weights.safetensors", start + 1, 2, 4, 8)
        descriptor, offset, length, advice = advise.call_args.args
        page = os.sysconf("SC_PAGESIZE")
        assert descriptor == server.source.files["weights.safetensors"].descriptor
        assert offset % page == length % page == 0
        assert offset <= start + 1 and offset + length >= start + 27
        assert advice == os.POSIX_FADV_DONTNEED


@pytest.mark.parametrize("failed_send", [False, True])
def test_peer_export_keeps_the_remote_owner_for_cleanup(tmp_path, monkeypatch, failed_send):
    import contextlib
    import signal
    import torch.distributed as dist
    from ignition.preparation import host_ranks, runtime_package

    events = []
    class Owner:
        stats = dict(payload_bytes=0, metadata_bytes=0, requests=0)
        @contextlib.contextmanager
        def activate(self):
            yield tmp_path
        def finish(self):
            events.append("drained")
        def close(self):
            events.append("owner-closed")
    owner = Owner()
    def remote(*args, **kwargs):
        kwargs["on_created"](owner)
        return owner
    class Control:
        def getsockname(self):
            return "127.0.0.1", 0
        def sendfile(self, stream, *, count):
            events.append("package-sent")
            if failed_send:
                raise ConnectionError("synthetic interrupted export")
            return len(stream.read(count))
        def close(self):
            pass
    model = SimpleNamespace(memory_report={}, manifest=lambda: None, close=lambda: events.append("model-closed"))
    assignment = {"config": None, "prepare": lambda *args, **kwargs: model,
                  "store": ("127.0.0.1", 1234), "packaged": False}
    source = tmp_path / "weights.bin"
    source.write_bytes(b"synthetic weights")
    commands = iter((("checkpoint-source", None), assignment, ("export", {}), [str(source)], "close"))
    def export(*args, **kwargs):
        from ignition.preparation.package_file import write_sources
        from ignition.preparation.runtime_package_pb2 import ModelIndex, BundleIndex, Span
        size = source.stat().st_size
        index = ModelIndex(version=4, ranks=[BundleIndex(resources={'weights': Span(nbytes=size)})])
        write_sources(args[3], index, [(0, source, 0, size)])
        return SimpleNamespace(memory_inventory={})
    monkeypatch.setattr(transport, "RemoteSource", remote)
    monkeypatch.setattr(transport, "token", lambda path: b"test" * 8)
    monkeypatch.setattr(host_ranks, "_watch_until_closed", lambda *args, **kwargs: None)
    monkeypatch.setattr(host_ranks, "_use_interface", lambda *args: None)
    monkeypatch.setattr(host_ranks, "receive_object", lambda *args: next(commands))
    monkeypatch.setattr(host_ranks, "send_object", lambda *args: None)
    monkeypatch.setattr(runtime_package, "export_model", export)
    monkeypatch.setattr(dist, "TCPStore", lambda *args, **kwargs: None)
    monkeypatch.setattr(dist, "init_process_group", lambda *args, **kwargs: None)
    monkeypatch.setattr(dist, "destroy_process_group", lambda: None)
    def exit_process(code):
        raise SystemExit(code)
    monkeypatch.setattr(host_ranks.os, "_exit", exit_process)
    previous = signal.getsignal(signal.SIGINT)
    try:
        if failed_send:
            with pytest.raises(SystemExit):
                host_ranks.serve_peer(Control(), SimpleNamespace(close=lambda: None), 1, 2,
                                      hub_cache=tmp_path, export_staging=tmp_path, identity="test")
        else:
            host_ranks.serve_peer(Control(), SimpleNamespace(close=lambda: None), 1, 2,
                                  hub_cache=tmp_path, export_staging=tmp_path, identity="test")
    finally:
        signal.signal(signal.SIGINT, previous)
    assert events.count("owner-closed") == 1
    assert events.index("drained") < events.index("package-sent") < events.index("owner-closed")
