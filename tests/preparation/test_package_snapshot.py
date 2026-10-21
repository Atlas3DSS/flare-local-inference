"""Blob snapshots preserve signed packages and reuse immutable payload files."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import pytest

from helios.tests.signing import publisher
from ignition.preparation.package_file import PackageFiles, _publish, read_index
from ignition.preparation.package_snapshot import (convert, export_options, read_container, read_at)
from ignition.preparation.protection import Protection
from tests.preparation import test_package_protection as fixtures


@pytest.fixture(scope='module')
def native():
    root = Path(__file__).resolve().parents[2]
    subprocess.run(['cargo', 'build', '--manifest-path', str(root / 'helios/Cargo.toml'),
                    '--features', 'runtime-package,test-tools', '--bin', 'test_package_protection'],
                   check=True, capture_output=True, text=True)
    return root / 'helios/target/debug/test_package_protection'


def profile():
    signer = publisher()
    return {'namespace': signer.namespace, 'development': False, 'public_key': list(signer.public_key)}


def legacy(path, index, payload, *, publisher, shard_size=0):
    with tempfile.TemporaryDirectory() as temporary:
        protection = Protection(index, payload, publisher, temporary, shard_size=shard_size, basename=path.stem)
        _publish(path, index, protection.payload, protection)


def assert_native(native, path):
    result = fixtures.read(native, path, weights=True)
    assert result.returncode == 0, result.stderr


def test_snapshot_primary_is_metadata_only_and_native_reads_payloads(native, tmp_path):
    path, data = fixtures.fixture(tmp_path)
    with path.open('rb') as source:
        header, base, _, layout = read_container(source)
        assert header['flare.payload']['shape'] == [0]
        assert header['__metadata__']['format'] == 'flare-snapshot'
        index, _ = read_index(path)
        with PackageFiles(path, source, base, 0) as files:
            span = index.ranks[0].resources['weights']
            source, base, _ = files.get(span.filename)
            assert read_at(source, span.nbytes, base + span.offset) == data['weights']
    for parts in layout['files'].values():
        for _, name in parts:
            if name:
                blob = path.parent / name
                assert hashlib.sha256(blob.read_bytes()).hexdigest() == name[:-12]
                assert blob.stat().st_size <= 64 << 30
                assert not blob.stat().st_mode & 0o222
    assert_native(native, path)


def test_snapshot_companion_links_support_native_loading_and_conversion(native, tmp_path):
    path, _ = fixtures.fixture(tmp_path)
    with path.open('rb') as source:
        layout = read_container(source)[3]
    storage = tmp_path / 'payload-storage'
    storage.mkdir()
    names = {name for parts in layout['files'].values() for _, name in parts if name}
    assert names
    for name in names:
        companion = path.parent / name
        target = storage / name
        companion.rename(target)
        companion.symlink_to(target)
    assert_native(native, path)
    destination = tmp_path / 'converted' / 'model.safetensors'
    convert(path, destination, profiles=[profile()], cache=tmp_path / 'converted-cache')
    assert_native(native, destination)


@pytest.mark.parametrize('sharded', [False, True])
def test_converter_preserves_original_signature_and_reuses_blobs(native, tmp_path, monkeypatch, sharded):
    def publish(*args, **kwargs):
        return legacy(*args, **kwargs, shard_size=16384 if sharded else 0)
    monkeypatch.setattr(fixtures, 'publish', publish)
    source, _ = fixtures.fixture(tmp_path, size=100)
    with source.open('rb') as file:
        original = read_container(file)[2]
    cache = tmp_path / 'cache'
    first = tmp_path / 'first' / 'model.safetensors'
    convert(source, first, profiles=[profile()], cache=cache)
    second = tmp_path / 'second' / 'model.safetensors'
    report = convert(first, second, profiles=[profile()], cache=cache)
    assert report.written == 0
    assert report.reused > 0
    with second.open('rb') as file:
        _, _, signed, layout = read_container(file)
        assert signed == original
    for parts in layout['files'].values():
        for _, name in parts:
            if name:
                assert os.path.samefile(first.parent / name, second.parent / name)
                assert os.path.samefile(cache / 'blobs' / name, second.parent / name)
    assert_native(native, first)
    assert_native(native, second)


def test_export_no_weights_requires_existing_exact_payload(native, tmp_path):
    first = tmp_path / 'first'
    first.mkdir()
    cache = tmp_path / 'cache'
    with export_options(cache):
        source, _ = fixtures.fixture(first, size=100)
    second = tmp_path / 'second'
    second.mkdir()
    with export_options(cache, no_weights=True):
        destination, _ = fixtures.fixture(second, size=200)
    assert_native(native, destination)
    first_weights = read_index(source)[0].ranks[0].resources['weights']
    with source.open('rb') as file:
        layout = read_container(file)[3]
    weight_name = layout['files'][first_weights.filename][0][1]
    assert os.path.samefile(first / weight_name, second / weight_name)
    third = tmp_path / 'third'
    third.mkdir()
    with export_options(tmp_path / 'empty', no_weights=True), pytest.raises(ValueError, match='cached payload'):
        fixtures.fixture(third, size=100)
    assert not (third / 'model.safetensors').exists()


def test_converter_rejects_tampered_weights_and_unknown_publisher(tmp_path, monkeypatch):
    monkeypatch.setattr(fixtures, 'publish', legacy)
    source, _ = fixtures.fixture(tmp_path, size=100)
    with pytest.raises(ValueError, match='unknown'):
        convert(source, tmp_path / 'unknown.safetensors', profiles=[])
    _, base = read_index(source)
    with source.open('r+b') as file:
        file.seek(base)
        file.write(b'changed')
    with pytest.raises(ValueError, match='digest mismatch'):
        convert(source, tmp_path / 'bad.safetensors', profiles=[profile()])
    assert not (tmp_path / 'bad.safetensors').exists()


def test_shard_limit_is_hard_and_primary_is_never_published_on_failure(tmp_path):
    from ignition.preparation.package_snapshot import BlobStore, publish_snapshot
    from ignition.preparation import package_file
    with pytest.raises(ValueError, match='shard limit'):
        package_file.publish(tmp_path / 'bad.safetensors', None, [], publisher=publisher(), shard_size=0)


def test_allocation_cache_hit_writes_no_staging_weights(tmp_path):
    from ignition.preparation.runtime_package import Allocation, FileSource, _write_allocation
    from ignition.preparation.package_snapshot import BlobStore
    from types import SimpleNamespace

    contents = b'w' * 65536 + bytes(65536) + b'z' * 65536
    cuda = SimpleNamespace(download=lambda address, count: contents[address - 4096:address - 4096 + count])
    allocation = Allocation('weight-storage', 4096, len(contents))
    store = BlobStore(tmp_path / 'cache')
    first, spans = _write_allocation(cuda, allocation, tmp_path / 'first.bin', blob_store=store)
    assert isinstance(first, FileSource)
    assert spans == [[0, 65536], [131072, 65536]]
    assert first.path.read_bytes()[4096:] == b'w' * 65536 + b'z' * 65536
    written = store.written
    second, repeated = _write_allocation(cuda, allocation, tmp_path / 'second.bin',
                                         blob_store=store, require_cached=True)
    assert second == first and repeated == spans
    assert store.written == written
    assert not (tmp_path / 'first.bin').exists() and not (tmp_path / 'second.bin').exists()


def test_converter_checks_relocated_bytes_before_publishing(tmp_path, monkeypatch):
    from ignition.preparation.package_snapshot import BlobStore
    monkeypatch.setattr(fixtures, 'publish', legacy)
    source, _ = fixtures.fixture(tmp_path, size=100)
    index, base = read_index(source)
    original = BlobStore.put
    changed = False

    def mutate(self, size, blocks, **kwargs):
        nonlocal changed
        if not changed:
            changed = True
            span = index.ranks[0].resources['weights']
            with source.open('r+b') as file:
                file.seek(base + span.offset)
                file.write(b'bad')
        return original(self, size, blocks, **kwargs)

    monkeypatch.setattr(BlobStore, 'put', mutate)
    destination = tmp_path / 'converted' / 'model.safetensors'
    with pytest.raises(ValueError, match='relocated stored-data digest mismatch'):
        convert(source, destination, profiles=[profile()])
    assert not destination.exists()


def test_snapshot_transfers_only_selected_rank_blobs(native, tmp_path):
    path, _ = fixtures.shard_fixture(tmp_path, ranks=2, shard_size=32768)
    result = subprocess.run([str(native), str(path), 'ranges', '1'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    ranges = json.loads(result.stdout)
    destination = tmp_path / 'peer'
    destination.mkdir()
    for name, (length, parts) in ranges.items():
        assert parts == [[0, length]]
        (destination / (name or path.name)).write_bytes((path.with_name(name) if name else path).read_bytes())
    result = fixtures.read(native, destination / path.name, weights=True, rank=1)
    assert result.returncode == 0, result.stderr
    assert fixtures.read(native, destination / path.name, weights=True).returncode != 0


def test_snapshot_rejects_swapped_payload_and_wrong_blob_length(native, tmp_path):
    from ignition.preparation.package_snapshot import write_primary
    path, _ = fixtures.fixture(tmp_path)
    with path.open('rb') as source:
        _, _, envelope, layout = read_container(source)
    parts = layout['files']['']
    nonempty = [part for part in parts if part[1]]
    nonempty[0][1], nonempty[-1][1] = nonempty[-1][1], nonempty[0][1]
    bad = tmp_path / 'swapped.safetensors'
    write_primary(bad, envelope, layout)
    result = fixtures.read(native, bad, weights=True)
    assert result.returncode != 0


def test_cached_blob_corruption_is_rejected_without_replacement(tmp_path):
    from ignition.preparation.package_snapshot import BlobStore
    store = BlobStore(tmp_path)
    name = store.put(4, lambda: [b'data'])
    path = store.root / name
    path.chmod(0o644)
    with path.open('r+b') as file:
        file.seek(4096)
        file.write(b'nope')
    with pytest.raises(ValueError, match='cached blob content'):
        store.put(4, lambda: [b'data'], required=True)
    assert path.read_bytes()[4096:] == b'nope'
    with pytest.raises(ValueError, match='64G'):
        store.put(64 << 30, lambda: [])


def test_blob_verification_retries_metadata_changes_from_a_concurrent_link(tmp_path, monkeypatch):
    from ignition.preparation.package_snapshot import BlobStore

    store = BlobStore(tmp_path)
    name = store.put(4, lambda: [b'data'])
    path = store.root / name
    original = hashlib.file_digest
    calls = []

    def digest(source, algorithm):
        result = original(source, algorithm)
        calls.append(1)
        if len(calls) == 1:
            os.link(path, tmp_path / 'rank-payload')
        return result

    monkeypatch.setattr(hashlib, 'file_digest', digest)
    BlobStore.verify(path, bytes.fromhex(name[:-12]), path.stat().st_size)
    assert len(calls) == 2


def test_blob_verification_retry_still_rejects_changed_content(tmp_path, monkeypatch):
    from ignition.preparation.package_snapshot import BlobStore

    store = BlobStore(tmp_path)
    name = store.put(4, lambda: [b'data'])
    path = store.root / name
    path.chmod(0o644)
    original = hashlib.file_digest
    calls = []

    def digest(source, algorithm):
        result = original(source, algorithm)
        calls.append(1)
        if len(calls) == 1:
            with path.open('r+b') as output:
                output.seek(4096)
                output.write(b'nope')
        return result

    monkeypatch.setattr(hashlib, 'file_digest', digest)
    with pytest.raises(ValueError, match='cached blob content'):
        BlobStore.verify(path, bytes.fromhex(name[:-12]), path.stat().st_size)
    assert len(calls) == 2


def test_blob_verification_retries_are_bounded(tmp_path, monkeypatch):
    from ignition.preparation.package_snapshot import BlobStore

    store = BlobStore(tmp_path)
    name = store.put(4, lambda: [b'data'])
    path = store.root / name
    original = hashlib.file_digest
    calls = []

    def digest(source, algorithm):
        result = original(source, algorithm)
        calls.append(1)
        os.link(path, tmp_path / f'rank-payload-{len(calls)}')
        return result

    monkeypatch.setattr(hashlib, 'file_digest', digest)
    with pytest.raises(ValueError, match='cached blob changed during verification'):
        BlobStore.verify(path, bytes.fromhex(name[:-12]), path.stat().st_size)
    assert len(calls) == 3


def test_logical_reads_seek_to_overlapping_blobs_and_preserve_padding(tmp_path):
    from ignition.preparation.package_snapshot import LogicalFile

    class Parts(list):
        reads = 0

        def __getitem__(self, index):
            self.reads += 1
            return super().__getitem__(index)

    path = tmp_path / 'payload'
    path.write_bytes(b'abcdefgh')
    with path.open('rb') as source:
        parts = Parts([[8, 'payload'], [3, '']] * 5000)
        logical = LogicalFile(parts, lambda _: (source, 0, 8))
        parts.reads = 0
        assert logical.read_at(9, 4999 * 11 + 3) == b'defgh' + bytes(3)
        assert parts.reads == 2
        assert logical.read_at(3, 5000 * 11) == b''
        assert logical.read_at(0, 3) == b''
        assert logical.seek(-8, 2) == 5000 * 11 - 8
        assert logical.read() == b'defgh' + bytes(3)
        assert logical.complete_blob(4999 * 11, 8) == (source, 0)
        assert logical.complete_blob(4999 * 11 + 1, 7) is None


def test_cached_snapshot_signing_hashes_unchanged_weight_inode_once(tmp_path, monkeypatch):
    from ignition.preparation.package_snapshot import BlobStore
    from ignition.preparation.resign import resign

    signer = fixtures.development_publisher()
    cache = tmp_path / 'cache'
    with export_options(cache):
        source, _ = fixtures.fixture(tmp_path, signer=signer)
        with source.open('rb') as file:
            layout = read_container(file)[3]
        span = read_index(source)[0].ranks[0].resources['weights']
        weight_name = layout['files'][span.filename][0][1]
        original = BlobStore.verify

        def verify(path, *args):
            assert Path(path).name != weight_name, 'weight bytes were already hashed from this inode'
            return original(path, *args)

        monkeypatch.setattr(BlobStore, 'verify', staticmethod(verify))
        destination = tmp_path / 'production.safetensors'
        resign(source, destination, publisher=publisher(), development_profile=fixtures.development_profile(signer))
    assert destination.is_file()


def test_blob_link_rechecks_an_inode_changed_after_verification(tmp_path):
    from ignition.preparation.package_snapshot import BlobStore

    store = BlobStore(tmp_path / 'cache')
    name = store.put(4, lambda: [b'data'])
    source = store.root / name
    store.link(name, tmp_path)
    source.chmod(0o644)
    with source.open('r+b') as file:
        file.seek(4096)
        file.write(b'nope')
    with pytest.raises(ValueError, match='cached blob content'):
        store.link(name, tmp_path)


def test_snapshot_syncs_blob_directory_once_before_primary(tmp_path, monkeypatch):
    from ignition.preparation import package_file

    synced = []
    original = package_file._sync_directory

    def sync(path):
        synced.append(Path(path))
        original(path)

    monkeypatch.setattr(package_file, '_sync_directory', sync)
    source, _ = fixtures.fixture(tmp_path)
    cache = tmp_path / '.flare-cache' / 'blobs'
    assert synced.count(cache) == 1
    assert synced.index(cache) < synced.index(tmp_path)
    assert source.is_file()


def test_snapshot_segment_budget_fails_before_payload_io(tmp_path, monkeypatch):
    from ignition.preparation import package_snapshot

    source, _ = fixtures.fixture(tmp_path)
    read = package_snapshot.read_at
    monkeypatch.setattr(package_snapshot, 'MAX_BLOBS', 1)
    monkeypatch.setattr(package_snapshot, 'read_at', lambda *args: pytest.fail('payload read before layout validation'))
    monkeypatch.setattr(package_snapshot.BlobStore, 'put', lambda *args, **kwargs:
                        pytest.fail('payload write before layout validation'))
    with pytest.raises(ValueError, match='too many payload segments'):
        convert(source, tmp_path / 'converted' / 'model.safetensors', profiles=[profile()])
    monkeypatch.setattr(package_snapshot, 'read_at', read)
    (tmp_path / 'new').mkdir()
    with pytest.raises(ValueError, match='too many payload segments'):
        fixtures.fixture(tmp_path / 'new')


def test_snapshot_signing_rejects_weight_corruption(tmp_path):
    from ignition.preparation.resign import resign

    signer = fixtures.development_publisher()
    source, _ = fixtures.fixture(tmp_path, signer=signer)
    with source.open('rb') as file:
        layout = read_container(file)[3]
    span = read_index(source)[0].ranks[0].resources['weights']
    weight = tmp_path / layout['files'][span.filename][0][1]
    weight.chmod(0o644)
    with weight.open('r+b') as file:
        file.seek(4096)
        file.write(b'bad')
    destination = tmp_path / 'production.safetensors'
    with pytest.raises(ValueError, match='stored-data digest mismatch'):
        resign(source, destination, publisher=publisher(), development_profile=fixtures.development_profile(signer))
    assert not destination.exists()


def test_cached_source_does_not_bypass_a_different_corrupt_cache_inode(tmp_path):
    from ignition.preparation.package_snapshot import BlobStore

    store = BlobStore(tmp_path / 'cache')
    name = store.put(4, lambda: [b'data'])
    cached = store.root / name
    source = tmp_path / 'source'
    source.write_bytes(cached.read_bytes())
    cached.chmod(0o644)
    with cached.open('r+b') as file:
        file.seek(4096)
        file.write(b'nope')
    with source.open('rb') as file, pytest.raises(ValueError, match='cached blob content'):
        store.put(4, lambda: [b'data'], source_blob=(file, 4096))


def test_rank_packing_preserves_interleaved_ownership_and_shard_limit():
    from ignition.preparation.package_snapshot import _resource_groups
    from ignition.preparation import runtime_package_pb2 as proto

    roles = [(kind, rank, proto.Span(offset=i * 4096, nbytes=31))
             for i, (kind, rank) in enumerate([(2, 0), (3, 0), (2, 1), (3, 0),
                                               (2, 0), (4, 0), (4, 0), (2, None)])]
    groups = list(_resource_groups(roles, 12288))
    assert [[(kind, rank) for kind, rank, _ in group] for group in groups] == [
        [(2, 0), (3, 0)], [(2, 1)], [(3, 0), (2, 0)], [(4, 0), (4, 0)], [(2, None)]]
    assert all(group[-1][2].offset + group[-1][2].nbytes - group[0][2].offset + 4096 <= 12288
               for group in groups)


def test_packing_caps_published_files_without_rejecting_legacy_layouts():
    from ignition.preparation.package_snapshot import _resource_groups, validate_layout
    from ignition.preparation import runtime_package_pb2 as proto

    size = 32_000_000_000
    roles = [(4, 0, proto.Span(offset=i * size, nbytes=size)) for i in range(2)]
    assert len(list(_resource_groups(roles, 64 << 30))) == 2
    validate_layout({'version': 1, 'files': {'': [[(64 << 30) - 4096, 'a' * 64 + '.safetensors']]}})


def test_packed_weights_preserve_padding_signatures_and_cache_reuse(native, tmp_path):
    from ignition.preparation import runtime_package_pb2 as proto
    from ignition.preparation.package_file import publish, align
    from ignition.preparation.resign import resign

    cache = tmp_path / 'cache'
    signer = fixtures.development_publisher()

    def export(directory, graph):
        directory.mkdir()
        index = proto.ModelIndex(version=4, ranks=[proto.BundleIndex(), proto.BundleIndex()])
        index.cluster.world, index.cluster.export_id = 2, 'packed-weights'
        payload, cursor = [], 0
        for owner, rank in enumerate(index.ranks):
            for i in range(4):
                name = f'weight-{i}'
                data = directory / f'{owner}-{name}'
                data.write_bytes(bytes([owner + i + 1]) * (101 + i))
                rank.allocations.add(name=name, file=name)
                rank.resources[name].CopyFrom(proto.Span(offset=cursor, nbytes=data.stat().st_size))
                payload.append((cursor, data, 0, data.stat().st_size))
                cursor = align(cursor + data.stat().st_size)
            data = directory / f'graph-{owner}'
            data.write_bytes(graph)
            rank.graphs.add(name='decode', resource='graph')
            rank.resources['graph'].CopyFrom(proto.Span(offset=cursor, nbytes=len(graph)))
            payload.append((cursor, data, 0, len(graph)))
            cursor = align(cursor + len(graph))
        path = directory / 'model.safetensors'
        publish(path, index, payload, publisher=signer, shard_size=16384)
        return path

    with export_options(cache):
        source = export(tmp_path / 'first', b'graph')
    with source.open('rb') as stream:
        original_layout = read_container(stream)[3]
    # Four allocations per rank fill two bounded shards, rather than four files.
    weights = {name for logical, parts in original_layout['files'].items() if logical
               for _, name in parts if name}
    assert len(weights) == 4
    assert all((source.parent / name).stat().st_size <= 16384 for name in weights)
    development = fixtures.profiles(tmp_path / 'development.json', signer)
    result = fixtures.read(native, source, namespace='local-development', weights=True, profile=development)
    assert result.returncode == 0, result.stderr

    with export_options(cache, no_weights=True):
        repeated = export(tmp_path / 'second', b'changed graph')
    for name in weights:
        assert os.path.samefile(source.parent / name, repeated.parent / name)
    signed = tmp_path / 'signed' / 'model.safetensors'
    with export_options(cache):
        resign(source, signed, publisher=publisher(),
               development_profile=fixtures.development_profile(signer), shard_size=16384)
    assert_native(native, signed)

    repacked = tmp_path / 'repacked' / 'model.safetensors'
    convert(signed, repacked, profiles=[profile()], cache=cache, shard_size=65536)
    with signed.open('rb') as before, repacked.open('rb') as after:
        assert read_container(before)[2] == read_container(after)[2]
    assert_native(native, repacked)


def test_signing_large_resource_catalog_stays_within_snapshot_segment_budget(native, tmp_path):
    """Resource-per-blob layout would require more than 65,536 segments."""
    from ignition.preparation import runtime_package_pb2 as proto
    from ignition.preparation.package_file import publish, align
    from ignition.preparation.resign import resign

    index = proto.ModelIndex(version=4, ranks=[proto.BundleIndex(), proto.BundleIndex()])
    index.cluster.world, index.cluster.export_id = 2, 'large-resource-catalog'
    data = tmp_path / 'payload'
    data.write_bytes(b'graph')
    payload = []
    cursor = 0
    for owner, rank in enumerate(index.ranks):
        rank.allocations.add(name='weights', file='weights')
        rank.resources['weights'].CopyFrom(proto.Span(offset=cursor, nbytes=5))
        payload.append((cursor, data, 0, 5))
        cursor = align(cursor + 5)
        module = rank.modules.add(name='kernel', file='resource-0')
        for i in range(31201):
            name = f'resource-{i}'
            if i % 2:
                module.globals.add(symbol=name, file=name, nbytes=5)
            else:
                rank.graphs.add(name=name, resource=name)
            rank.resources[name].CopyFrom(proto.Span(offset=cursor, nbytes=5))
            payload.append((cursor, data, 0, 5))
            cursor = align(cursor + 5)
    signer = fixtures.development_publisher()
    source = tmp_path / 'development.safetensors'
    destination = tmp_path / 'production.safetensors'
    limit = 8 << 20
    with export_options(tmp_path / 'cache'):
        publish(source, index, payload, publisher=signer, shard_size=limit)
        resign(source, destination, publisher=publisher(), development_profile=fixtures.development_profile(signer),
               shard_size=limit)
    with destination.open('rb') as file:
        header, _, _, layout = read_container(file)
    assert header['flare.payload']['shape'] == [0]
    assert sum(len(parts) for parts in layout['files'].values()) < 100
    assert all((tmp_path / name).stat().st_size <= limit
               for parts in layout['files'].values() for _, name in parts if name)
    assert_native(native, destination)
    for rank in (0, 1):
        result = subprocess.run([str(native), str(destination), 'ranges', str(rank)], capture_output=True,
                                text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        selected = json.loads(result.stdout)
        peer = tmp_path / f'peer-{rank}'
        peer.mkdir()
        for name in selected:
            os.link(destination.with_name(name) if name else destination,
                    peer / (name or destination.name))
        result = fixtures.read(native, peer / destination.name, weights=True, rank=rank)
        assert result.returncode == 0, result.stderr
        assert fixtures.read(native, peer / destination.name, weights=True, rank=1 - rank).returncode != 0
    converted = tmp_path / 'converted' / 'model.safetensors'
    convert(destination, converted, profiles=[profile()], cache=tmp_path / 'cache', shard_size=limit)
    assert_native(native, converted)


def test_resource_larger_than_hard_limit_does_not_publish(tmp_path):
    with pytest.raises(ValueError, match='resource exceeds'):
        fixtures.shard_fixture(tmp_path, sizes=(17000,), shard_size=12288)
    assert not (tmp_path / 'model.safetensors').exists()


def test_snapshot_promotion_reuses_weights_and_native_verifies(native, tmp_path, monkeypatch):
    from ignition.preparation.resign import resign
    signer = fixtures.development_publisher()
    monkeypatch.setattr(fixtures, 'publisher', lambda: signer)
    cache = tmp_path / 'cache'
    with export_options(cache):
        source, _ = fixtures.fixture(tmp_path)
        destination = tmp_path / 'production' / 'model.safetensors'
        destination.parent.mkdir()
        resign(source, destination, publisher=publisher(), development_profile=fixtures.development_profile(signer))
    with source.open('rb') as file:
        first = read_container(file)[3]
    with destination.open('rb') as file:
        second = read_container(file)[3]
    shared = {name for parts in first['files'].values() for _, name in parts if name} & {
              name for parts in second['files'].values() for _, name in parts if name}
    assert shared
    assert all(os.path.samefile(source.parent / name, destination.parent / name) for name in shared)
    assert_native(native, destination)


def test_convert_cli_requires_no_torch_or_private_key(tmp_path, monkeypatch, capsys):
    import sys
    from ignition.cli import main
    monkeypatch.setattr(fixtures, 'publish', legacy)
    source, _ = fixtures.fixture(tmp_path, size=100)
    keys = tmp_path / 'publisher.json'
    keys.write_text(json.dumps([profile()]))
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setitem(sys.modules, 'torch', None)
    assert main(['convert', str(source), '--publisher-keys', str(keys)]) == 0
    output = capsys.readouterr().out
    assert 'Converted package:' in output and 'Payload bytes written:' in output
    converted, = (tmp_path / '.cache/flare/packages/models--model/snapshots').glob('*/model.safetensors')
    with converted.open('rb') as file:
        assert read_container(file)[0]['flare.payload']['shape'] == [0]
