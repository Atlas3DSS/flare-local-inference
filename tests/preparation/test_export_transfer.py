"""Prepared payloads retain authentication, bounded resources, and atomic publication."""

from contextlib import ExitStack
import hashlib
from pathlib import Path
import resource
import socket
import threading

import pytest

from ignition.preparation import host_ranks
from ignition.preparation.checkpoint_source import BLOCK_BYTES, Channel
from ignition.preparation.package_file import read_sources, write_sources
from ignition.preparation.package_snapshot import BlobStore, ExportOptions
from ignition.preparation.runtime_package_pb2 import BundleIndex, ModelIndex, Span


def sockets(stack):
    left, right = socket.socketpair()
    for connection in (left, right):
        stack.enter_context(connection)
        connection.settimeout(5)
    return left, right


def channels(stack, count):
    roots, peers = [], []
    for _ in range(count):
        left, right = sockets(stack)
        roots.append(Channel(left, b'k' * 32, server=True))
        peers.append(Channel(right, b'k' * 32, server=False))
    return roots, peers


def manifest(directory, count, *, large=False):
    store = BlobStore(directory / 'peer-cache')
    payload, resources, contents = [], {}, {}
    offset = 0
    for i in range(count):
        data = bytes([i % 256]) * (BLOCK_BYTES + 17 if large and i < 2 else i + 1)
        name = store.root / store.put(len(data), lambda: [data])
        payload.append((offset, name, 4096, len(data)))
        resources[str(i)] = Span(offset=offset, nbytes=len(data))
        contents[str(i)] = data
        offset += (len(data) + 4095) // 4096 * 4096
    path = directory / 'peer.json'
    write_sources(path, ModelIndex(version=4, ranks=[BundleIndex(resources=resources)]), payload)
    return path, contents


@pytest.mark.parametrize('streams', [0, 1, 4, 8])
def test_parallel_export_and_verified_cache_reuse(tmp_path, streams):
    path, contents = manifest(tmp_path, 11, large=True)
    settings = ExportOptions(tmp_path / 'root-cache')
    for iteration in range(2):
        failures = []
        with ExitStack() as stack:
            control, peer = sockets(stack)
            roots, peers = channels(stack, streams)

            def transmit():
                try:
                    host_ranks.send_export_sources(peer, path, {}, channels=peers or None)
                    host_ranks.send_object(peer, 'export drained')
                except BaseException as error:
                    failures.append(error)

            thread = threading.Thread(target=transmit)
            thread.start()
            description = host_ranks.receive_object(control)['sources']
            destination = tmp_path / f'received-{iteration}.json'
            report = host_ranks.receive_export_sources(control, description, destination,
                ExportOptions(settings.cache, no_weights=iteration == 1), channels=roots or None)
            assert host_ranks.receive_object(control) == 'export drained'
            thread.join(5)
            assert not thread.is_alive() and not failures
            assert report['streams'] == (streams or 1)
            assert report['transport'] == ('agent-ring-tcp' if streams else 'rendezvous-tcp')
            assert report['files'] == (len(contents) if iteration == 0 else 0)
            expected = sum(info['size'] for info in description['files'].values()) if iteration == 0 else 0
            assert report['payload_bytes'] == expected
            if streams and iteration == 0:
                assert all(channel.received > 1 for channel in roots)
            index, locate = read_sources(destination)
            for name, span in index.ranks[0].resources.items():
                location, offset, size = locate(span)[0]
                with Path(location).open('rb') as source:
                    source.seek(offset)
                    assert source.read(size) == contents[name]


def test_export_thousands_of_files_with_a_small_descriptor_limit(tmp_path):
    path, _ = manifest(tmp_path, 1200)
    limits = resource.getrlimit(resource.RLIMIT_NOFILE)
    with ExitStack() as stack:
        control, peer = sockets(stack)
        roots, peers = channels(stack, 8)
        failures = []

        def transmit():
            try:
                host_ranks.send_export_sources(peer, path, {}, channels=peers)
            except BaseException as error:
                failures.append(error)
                peer.shutdown(socket.SHUT_RDWR)

        resource.setrlimit(resource.RLIMIT_NOFILE, (min(limits[0], 128), limits[1]))
        thread = threading.Thread(target=transmit)
        thread.start()
        try:
            description = host_ranks.receive_object(control)['sources']
            report = host_ranks.receive_export_sources(control, description, tmp_path / 'root.json',
                                                       ExportOptions(tmp_path / 'root-cache'), channels=roots)
            assert report['files'] == 1200
        finally:
            thread.join(10)
            resource.setrlimit(resource.RLIMIT_NOFILE, limits)
        assert not thread.is_alive() and not failures


@pytest.mark.parametrize('fault', ['offset', 'digest', 'truncated', 'accounting', 'authentication'])
def test_failed_stream_aborts_siblings_and_removes_partial_files(tmp_path, fault):
    data = b'payload'
    digest = hashlib.sha256(data).hexdigest()
    description = {'document': {'payload': [[0, 'file', 0, len(data)]]},
                   'files': {'file': {'size': len(data), 'sha256': digest, 'blob': True}}}
    settings = ExportOptions(tmp_path / 'root-cache')
    with ExitStack() as stack:
        control, peer = sockets(stack)
        roots, peers = channels(stack, 4)

        def transmit():
            assert host_ranks.receive_object(peer) == ['file']
            if fault == 'truncated':
                peers[0].connection.shutdown(socket.SHUT_RDWR)
                return
            if fault == 'authentication':
                peers[0].key = b'wrong' * 8
            peers[0].send('export-data', b'changed' if fault == 'digest' else data,
                          name='file', offset=1 if fault == 'offset' else 0)
            if fault == 'accounting':
                peers[0].send('export-finished', stats=dict(files=1, payload_bytes=0))
            # Other streams deliberately remain idle: failure must wake their receivers.

        thread = threading.Thread(target=transmit)
        thread.start()
        try:
            with pytest.raises((ValueError, ConnectionError)) as caught:
                host_ranks.receive_export_sources(control, description, tmp_path / 'root.json', settings, channels=roots)
            assert not isinstance(caught.value, TimeoutError)
            assert not (tmp_path / 'root.json').exists()
            assert not list(settings.cache.rglob('.rank-blob-*'))
            if fault != 'accounting':
                assert not list(settings.cache.rglob('*.safetensors'))
        finally:
            thread.join(5)
        assert not thread.is_alive()


@pytest.mark.parametrize('replacement', [False, True])
def test_export_rejects_sources_changed_after_metadata_scan(tmp_path, replacement):
    path, _ = manifest(tmp_path, 1)
    with ExitStack() as stack:
        control, peer = sockets(stack)
        roots, peers = channels(stack, 4)
        failures = []

        def transmit():
            try:
                host_ranks.send_export_sources(peer, path, {}, channels=peers)
            except BaseException as error:
                failures.append(error)

        thread = threading.Thread(target=transmit)
        thread.start()
        description = host_ranks.receive_object(control)['sources']
        name = next(iter(description['files']))
        target = Path(name)
        if replacement:
            changed = tmp_path / 'replacement'
            changed.write_bytes(target.read_bytes())
            changed.replace(target)
        else:
            target.chmod(0o600)
            with target.open('r+b') as source:
                source.write(b'changed')
        host_ranks.send_object(control, [name])
        with pytest.raises(ConnectionError):
            roots[0].receive()
        thread.join(5)
        assert not thread.is_alive()
        assert len(failures) == 1 and 'changed before transfer' in str(failures[0])


def test_checkpoint_completion_hands_authenticated_channel_to_export(tmp_path):
    from tests.preparation.test_checkpoint_source import connected

    path, _ = manifest(tmp_path, 3)
    with connected(tmp_path) as (client, server, offset), ExitStack() as stack:
        assert client.read('weights.safetensors', offset, 8) == bytes(range(8))
        client.finish(keep_transport=True)
        server.join()
        control, peer = sockets(stack)
        failures = []

        def transmit():
            try:
                host_ranks.send_export_sources(peer, path, {}, channels=[client.channel])
            except BaseException as error:
                failures.append(error)

        thread = threading.Thread(target=transmit)
        thread.start()
        description = host_ranks.receive_object(control)['sources']
        report = host_ranks.receive_export_sources(control, description, tmp_path / 'root.json',
            ExportOptions(tmp_path / 'root-cache'), channels=[server.channel])
        thread.join(5)
        assert report['files'] == 3 and report['transport'] == 'agent-ring-tcp'
        assert not thread.is_alive() and not failures
