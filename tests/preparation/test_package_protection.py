"""Publisher authentication across the Python exporter and native resource reader."""

import hashlib
import json
from pathlib import Path
import struct
import subprocess

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from helios.tests.signing import publisher
from ignition.preparation import runtime_package_pb2 as proto
from ignition.preparation.package_file import align, publish, read_index
from ignition.preparation.protection import CHUNK, Publisher, signature_message

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def native():
    subprocess.run(['cargo', 'build', '--manifest-path', str(ROOT / 'helios/Cargo.toml'),
                    '--features', 'runtime-package,test-tools', '--bin', 'test_package_protection'],
                   check=True, capture_output=True, text=True)
    return ROOT / 'helios/target/debug/test_package_protection'


def read(native, path, namespace='flare-tests', weights=False, profile=None):
    args = [str(native), str(path), namespace, 'verify-weights' if weights else 'skip-weights']
    if profile is not None:
        args.append(str(profile))
    return subprocess.run(args, capture_output=True, text=True)


def fixture(tmp_path, *, size=CHUNK + 39, signer=None):
    index = proto.ModelIndex(version=3, ranks=[proto.BundleIndex()])
    rank = index.ranks[0]
    rank.allocations.add(name='weights', file='weights')
    module = rank.modules.add(name='kernel', file='kernel')
    module.globals.add(symbol='constant', file='constant', nbytes=13)
    rank.graphs.add(name='decode', resource='graph')
    data = {'weights': bytes(range(251)) * 31, 'kernel': (b'kernel-code-' * (size // 12 + 1))[:size],
            'constant': b'constant-data', 'graph': b'graph-records', 'tokenizer': b'tokenizer-data'}
    payload, cursor = [], 0
    for name, contents in data.items():
        source = tmp_path / name
        source.write_bytes(contents)
        span = proto.Span(offset=cursor, nbytes=len(contents))
        if name == 'tokenizer':
            index.assets.add(name=name, payload=span, sha256=hashlib.sha256(contents).hexdigest())
        else:
            rank.resources[name].CopyFrom(span)
        payload.append((cursor, source, 0, len(contents)))
        cursor = align(cursor + len(contents))
    path = tmp_path / 'model.safetensors'
    publish(path, index, payload, publisher=signer or publisher())
    return path, data


def envelope(path):
    data = path.read_bytes()
    start = 8 + struct.unpack_from('<Q', data)[0]
    header = json.loads(data[8:start])
    end = start + header['flare.payload']['shape'][0]
    return proto.ProtectedIndex.FromString(data[end:]), data[start:end]


def replace_envelope(path, value, payload):
    metadata = value.SerializeToString(deterministic=True)
    size = len(payload)
    header = json.dumps({'__metadata__': {'format': 'flare'},
        'flare.payload': {'dtype': 'U8', 'shape': [size], 'data_offsets': [0, size]},
        'flare.metadata': {'dtype': 'U8', 'shape': [len(metadata)],
                           'data_offsets': [size, size + len(metadata)]}}, separators=(',', ':')).encode()
    header = header.ljust(align(8 + len(header)) - 8, b' ')
    path.write_bytes(struct.pack('<Q', len(header)) + header + payload + metadata)


def resign(value, signer):
    value.signer = signer.fingerprint
    metadata = proto.PublisherMetadata.FromString(value.groups[0].body)
    metadata.namespace, metadata.official_source = signer.namespace, signer.official_source
    metadata.development = signer.development
    del metadata.group_hashes[:]
    metadata.group_hashes.extend(hashlib.sha256(g.body).digest() for g in value.groups[1:])
    value.groups[0].body = metadata.SerializeToString(deterministic=True)
    for group in value.groups:
        group.signature = signer.key.sign(signature_message(value.package_id, value.signer, group.kind, group.body))


def profiles(path, signer, name='Fixture publisher'):
    path.write_text(json.dumps([{'name': name, 'namespace': signer.namespace,
        'public_key': list(signer.public_key), 'development': signer.development}]))
    return path


def test_multichunk_export_is_read_and_authenticated_by_rust(native, tmp_path):
    path, data = fixture(tmp_path)
    result = read(native, path, weights=True)
    assert result.returncode == 0, result.stderr
    index, start = read_index(path)
    expected = {span.offset: hashlib.sha256(data[name]).hexdigest()
                for name, span in index.ranks[0].resources.items() if name != 'weights'}
    expected[index.assets[0].payload.offset] = hashlib.sha256(data['tokenizer']).hexdigest()
    assert dict(json.loads(result.stdout)) == expected
    contents = path.read_bytes()
    kernel = index.ranks[0].resources['kernel']
    assert data['kernel'] not in contents
    assert contents[start:start + len(data['weights'])] == data['weights']
    assert kernel.nbytes == len(data['kernel']) + 32


def test_multi_host_export_signs_cluster_metadata_and_each_ranks_resources(native, tmp_path, monkeypatch):
    from ignition.preparation import model_package
    from ignition.preparation.package_file import extend_cluster

    def export(rank, path):
        index = proto.ModelIndex(version=3, ranks=[proto.BundleIndex()])
        bundle = index.ranks[0]
        bundle.roce.world, bundle.roce.rank = 2, rank
        bundle.allocations.add(name='weights', file='weights')
        bundle.modules.add(name='kernel', file='kernel')
        payload = []
        for offset, name in [(0, 'weights'), (4096, 'kernel')]:
            source = tmp_path / f'{name}-{rank}'
            source.write_bytes(f'{name}-{rank}'.encode())
            bundle.resources[name].CopyFrom(proto.Span(offset=offset, nbytes=source.stat().st_size))
            payload.append((offset, source, 0, source.stat().st_size))
        publish(path, index, payload)

    peer = tmp_path / 'peer.safetensors'
    export(1, peer)
    monkeypatch.setattr(model_package, 'export_model',
                        lambda prepared, images, storage, path, **kwargs: export(0, path))
    path = tmp_path / 'cluster.safetensors'
    with peer.open('rb') as stream:
        model_package.export_cluster_package(None, None, None, path, peer_files=[stream], options={},
                                             export_id='cluster-test', world=2, publisher=publisher())
    result = read(native, path, weights=True)
    assert result.returncode == 0, result.stderr
    index, _ = read_index(path)
    assert (index.cluster.world, index.cluster.export_id) == (2, 'cluster-test')
    assert [bundle.roce.rank for bundle in index.ranks] == [0, 1]
    expected = {bundle.resources['kernel'].offset: hashlib.sha256(f'kernel-{rank}'.encode()).hexdigest()
                for rank, bundle in enumerate(index.ranks)}
    assert dict(json.loads(result.stdout)) == expected
    assert max(b.resources['weights'].offset + b.resources['weights'].nbytes for b in index.ranks) <= min(
        b.resources['kernel'].offset for b in index.ranks)
    assert not list(tmp_path.glob('*.partial'))
    with pytest.raises(ValueError, match='unsigned rank staging'):
        extend_cluster(path, [], (2, 'different-cluster'))
    value, payload = envelope(path)
    metadata = proto.PublisherMetadata.FromString(value.groups[0].body)
    metadata.index.cluster.export_id = 'tampered-cluster'
    value.groups[0].body = metadata.SerializeToString(deterministic=True)
    replace_envelope(path, value, payload)
    assert read(native, path).returncode != 0


@pytest.mark.parametrize('resource', ['kernel', 'graph', 'constant', 'tokenizer'])
def test_tampering_with_execution_or_support_fails_without_weight_scan(native, tmp_path, resource):
    path, _ = fixture(tmp_path, size=41)
    index, start = read_index(path)
    span = index.assets[0].payload if resource == 'tokenizer' else index.ranks[0].resources[resource]
    with path.open('r+b') as output:
        output.seek(start + span.offset)
        value = output.read(1)
        output.seek(-1, 1)
        output.write(bytes([value[0] ^ 1]))
    result = read(native, path)
    assert result.returncode != 0 and 'content verification failed' in result.stderr


@pytest.mark.parametrize('padding', [False, True])
def test_optional_data_scan_covers_weights_and_alignment_padding(native, tmp_path, padding):
    path, data = fixture(tmp_path, size=41)
    _, start = read_index(path)
    with path.open('r+b') as output:
        output.seek(start + (len(data['weights']) if padding else 0))
        output.write(b'X')
    assert read(native, path).returncode == 0
    result = read(native, path, weights=True)
    assert result.returncode != 0 and 'stored-data content verification failed' in result.stderr


@pytest.mark.parametrize('group', range(4))
def test_every_signature_group_is_verified_before_access(native, tmp_path, group):
    path, _ = fixture(tmp_path, size=41)
    value, payload = envelope(path)
    value.groups[group].signature = bytes(64)
    replace_envelope(path, value, payload)
    result = read(native, path)
    assert result.returncode != 0 and f'signature failed for group {group + 1}' in result.stderr


def test_acceptance_requires_the_exact_namespace(native, tmp_path):
    path, _ = fixture(tmp_path, size=41)
    result = read(native, path, namespace='yes')
    assert result.returncode != 0 and 'acceptance does not match' in result.stderr


@pytest.mark.parametrize('change', ['key', 'namespace'])
def test_resigning_ciphertext_under_different_identity_cannot_decrypt(native, tmp_path, change):
    path, _ = fixture(tmp_path, size=41)
    original = publisher()
    signer = Publisher(Ed25519PrivateKey.generate() if change == 'key' else original.key,
                       'another-publisher', 'https://huggingface.co/another-publisher/fixture')
    value, payload = envelope(path)
    resign(value, signer)
    replace_envelope(path, value, payload)
    profile = profiles(tmp_path / 'publisher.json', signer)
    result = read(native, path, namespace=signer.namespace, profile=profile)
    assert result.returncode != 0 and 'executable decryption failed' in result.stderr


def test_independently_prepared_publisher_is_supported(native, tmp_path):
    signer = Publisher(Ed25519PrivateKey.generate(), 'another-publisher',
                       'https://huggingface.co/another-publisher/fixture')
    path, _ = fixture(tmp_path, size=41, signer=signer)
    assert 'unknown package publisher' in read(native, path).stderr
    profile = profiles(tmp_path / 'publisher.json', signer)
    result = read(native, path, namespace=signer.namespace, weights=True, profile=profile)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('change', ['overlap', 'unclassified', 'reclassify', 'groups'])
def test_authenticated_but_inconsistent_index_is_rejected(native, tmp_path, change):
    path, _ = fixture(tmp_path, size=41)
    value, payload = envelope(path)
    metadata = proto.PublisherMetadata.FromString(value.groups[0].body)
    rank = metadata.index.ranks[0]
    if change == 'overlap':
        rank.resources['graph'].CopyFrom(rank.resources['kernel'])
    elif change == 'unclassified':
        rank.resources['unused'].CopyFrom(rank.resources['kernel'])
    elif change == 'reclassify':
        rank.allocations[0].file = 'kernel'
    value.groups[0].body = metadata.SerializeToString(deterministic=True)
    resign(value, publisher())
    if change == 'groups':
        value.groups[1].kind = 3
    replace_envelope(path, value, payload)
    assert read(native, path).returncode != 0


def test_native_loader_rejects_unsigned_staging(native, tmp_path):
    path = tmp_path / 'stage'
    publish(path, proto.ModelIndex(version=3, ranks=[proto.BundleIndex()]), [])
    result = read(native, path)
    assert result.returncode != 0 and 'not a Flare safetensors package' in result.stderr


@pytest.fixture(scope='module')
def flare():
    subprocess.run(['cargo', 'build', '--release', '--manifest-path', str(ROOT / 'flare/Cargo.toml')],
                   check=True, capture_output=True, text=True)
    return ROOT / 'flare/target/release/flare'


def launch(flare, path, profile, store, *args):
    # Stop immediately after trust handling; this CPU fixture has no executable CUDA program.
    return subprocess.run([str(flare), 'run', str(path), '--publisher-keys', str(profile),
        '--trust-store', str(store), '--api-key-file', str(store.parent / 'absent-api-key'), *args],
        capture_output=True, text=True, stdin=subprocess.DEVNULL)


def approval(signer):
    return ['--dangerously-approve-key', signer.fingerprint.hex(),
            '--dangerously-accept-publisher', signer.namespace]


def test_external_key_warns_once_and_remembers_exact_pair(flare, tmp_path):
    signer = publisher()
    path, _ = fixture(tmp_path, size=41)
    profile = profiles(tmp_path / 'publisher.json', signer)
    store = tmp_path / 'trust.toml'
    first = launch(flare, path, profile, store)
    assert 'DANGER: UNRECOGNIZED SIGNING KEY' in first.stderr
    assert 'unattended first use requires' in first.stderr and not store.exists()
    accepted = launch(flare, path, profile, store, *approval(signer))
    assert 'DANGER: UNRECOGNIZED SIGNING KEY' in accepted.stderr
    assert 'cannot read API key file' in accepted.stderr and store.exists()
    remembered = launch(flare, path, profile, store)
    assert 'DANGER:' not in remembered.stderr
    assert 'identity UNVERIFIED' in remembered.stderr
    assert 'cannot read API key file' in remembered.stderr
    assert store.stat().st_mode & 0o777 == 0o600
    revoked = subprocess.run([str(flare), 'trust', 'revoke', signer.fingerprint.hex(),
                             '--trust-store', str(store)], capture_output=True, text=True)
    assert revoked.returncode == 0, revoked.stderr
    assert 'unattended first use requires' in launch(flare, path, profile, store).stderr


@pytest.mark.parametrize('mismatch', ['key', 'publisher', 'missing'])
def test_explicit_approval_never_ignores_mismatch_or_missing_value(flare, tmp_path, mismatch):
    path, _ = fixture(tmp_path, size=41)
    signer = publisher()
    profile = profiles(tmp_path / 'publisher.json', signer)
    store = tmp_path / 'trust.toml'
    args = approval(signer)
    if mismatch == 'key':
        args[1] = '0' * 64
    elif mismatch == 'publisher':
        args[3] = 'someone-else'
    else:
        args = args[:2]
    result = launch(flare, path, profile, store, *args)
    assert result.returncode != 0 and not store.exists()
    assert 'cannot read API key file' not in result.stderr


def test_rotation_does_not_inherit_namespace_trust(flare, tmp_path):
    path, _ = fixture(tmp_path, size=41)
    first = publisher()
    profile = profiles(tmp_path / 'publisher.json', first)
    store = tmp_path / 'trust.toml'
    launch(flare, path, profile, store, *approval(first))
    directory = tmp_path / 'rotation'
    directory.mkdir()
    second = Publisher(Ed25519PrivateKey.generate(), first.namespace, first.official_source)
    changed, _ = fixture(directory, size=41, signer=second)
    profiles(profile, second)
    result = launch(flare, changed, profile, store)
    assert 'DANGER: UNRECOGNIZED SIGNING KEY' in result.stderr
    assert 'unattended first use requires' in result.stderr


def test_release_stamping_recognizes_only_the_embedded_identity(flare, tmp_path):
    from ignition.release import stamp
    signer = publisher()
    path, _ = fixture(tmp_path, size=41)
    profile = profiles(tmp_path / 'publisher.json', signer, 'Local Inference Lab fixture')
    executable = tmp_path / 'flare-publisher'
    stamp(flare, executable, json.loads(profile.read_text()))
    store = tmp_path / 'trust.toml'
    known = launch(executable, path, profile, store, *approval(signer))
    assert 'Recognized publisher: Local Inference Lab fixture' in known.stderr
    assert 'DANGER:' not in known.stderr
    assert 'cannot read API key file' in known.stderr
    directory = tmp_path / 'impersonation'
    directory.mkdir()
    impostor = Publisher(Ed25519PrivateKey.generate(), signer.namespace, signer.official_source)
    changed, _ = fixture(directory, size=41, signer=impostor)
    profiles(profile, impostor, 'Local Inference Lab fixture')
    result = launch(executable, changed, profile, store)
    assert 'DANGER: UNRECOGNIZED SIGNING KEY' in result.stderr
    assert 'identity UNVERIFIED' in result.stderr and 'Recognized publisher:' not in result.stderr
    with pytest.raises(ValueError, match='not an empty release slot'):
        stamp(executable, tmp_path / 'restamped', json.loads(profile.read_text()))


def test_invalid_signature_cannot_record_trust_even_with_approval(flare, tmp_path):
    path, _ = fixture(tmp_path, size=41)
    signer = publisher()
    profile = profiles(tmp_path / 'publisher.json', signer)
    store = tmp_path / 'trust.toml'
    value, payload = envelope(path)
    value.groups[0].signature = bytes(64)
    replace_envelope(path, value, payload)
    result = launch(flare, path, profile, store, *approval(signer))
    assert 'signature failed' in result.stderr and not store.exists()


def test_corrupt_trust_store_fails_closed(flare, tmp_path):
    path, _ = fixture(tmp_path, size=41)
    signer = publisher()
    profile = profiles(tmp_path / 'publisher.json', signer)
    store = tmp_path / 'trust.toml'
    store.write_text('trusted = "everything"\n')
    result = launch(flare, path, profile, store, *approval(signer))
    assert result.returncode != 0 and 'cannot read API key file' not in result.stderr
    assert store.read_text() == 'trusted = "everything"\n'


def test_tensor_parallel_assembly_authenticates_all_ranks(native, tmp_path):
    from ignition.preparation.package_file import combine
    ranks = []
    for rank in range(2):
        index = proto.ModelIndex(version=3, ranks=[proto.BundleIndex()])
        bundle = index.ranks[0]
        bundle.allocations.add(name='weight', file='weight')
        bundle.graphs.add(name='decode', resource='graph')
        payload = []
        for name, offset, data in [('weight', 0, bytes([rank]) * 301),
                                   ('graph', 4096, bytes([rank + 2]) * 41)]:
            source = tmp_path / f'{name}-{rank}'
            source.write_bytes(data)
            bundle.resources[name].CopyFrom(proto.Span(offset=offset, nbytes=len(data)))
            payload.append((offset, source, 0, len(data)))
        path = tmp_path / f'rank-{rank}'
        publish(path, index, payload)
        ranks.append(path)
    output = tmp_path / 'tp2.safetensors'
    combine(output, ranks, [], publisher=publisher())
    result = read(native, output, weights=True)
    assert result.returncode == 0, result.stderr
    assert {digest for _, digest in json.loads(result.stdout)} == {
        hashlib.sha256(bytes([rank + 2]) * 41).hexdigest() for rank in range(2)}


def test_native_record_reader_selects_independently_and_rejects_malformed_records(native, tmp_path):
    node = proto.Node(empty=proto.Empty()).SerializeToString()
    valid = struct.pack('<I', len(node)) + node
    data = {'selected': valid, 'oversize': struct.pack('<I', (1 << 20) + 1),
            'truncated': struct.pack('<I', 100) + b'x', 'trailing': valid + b'x'}
    index = proto.ModelIndex(version=3, ranks=[proto.BundleIndex(architecture=[12, 0], context=proto.Context())])
    bundle = index.ranks[0]
    payload = []
    for ordinal, (name, contents) in enumerate(data.items()):
        path = tmp_path / name
        path.write_bytes(contents)
        bundle.resources[name].CopyFrom(proto.Span(offset=ordinal * 4096, nbytes=len(contents)))
        bundle.graphs.add(name=name, resource=name, node_count=1)
        payload.append((ordinal * 4096, path, 0, len(contents)))
    output = tmp_path / 'records.safetensors'
    publish(output, index, payload, publisher=publisher())
    for name in data:
        result = subprocess.run([str(native), str(output), 'records', name], capture_output=True, text=True)
        assert (result.returncode == 0) == (name == 'selected'), result.stderr


@pytest.mark.parametrize('field,value', [
    (('flare.payload', 'dtype'), 'F32'),
    (('flare.payload', 'shape'), [2, 2]),
    (('flare.payload', 'data_offsets'), [1, 5]),
    (('flare.metadata', 'data_offsets'), [3, 2**64 - 1]),
    (('flare.metadata', 'shape'), [(128 << 20) + 1]),
])
def test_native_header_validation_rejects_bad_shapes_and_extents(native, tmp_path, field, value):
    path, _ = fixture(tmp_path, size=41)
    contents = path.read_bytes()
    start = 8 + struct.unpack_from('<Q', contents)[0]
    header = json.loads(contents[8:start])
    header[field[0]][field[1]] = value
    encoded = json.dumps(header, separators=(',', ':')).encode().ljust(start - 8, b' ')
    path.write_bytes(contents[:8] + encoded + contents[start:])
    assert read(native, path).returncode != 0
