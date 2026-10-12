"""Atomic publication and rank ownership in native model packages."""

from types import SimpleNamespace

import pytest

from ignition.preparation import model_package
from ignition.preparation.runtime_package_pb2 import Asset, ModelIndex, BundleIndex, Span
from ignition.preparation.package_file import combine, publish, read_index


@pytest.mark.parametrize('world', [1, 2])
def test_publication_contains_one_bundle_per_rank(monkeypatch, tmp_path, world):
    import torch
    import torch.distributed as dist

    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(dist, 'barrier', lambda: None)
    exported = []
    def export(prepared, images, storage, path, **kwargs):
        publish(path, ModelIndex(version=1, ranks=[BundleIndex()]), [])
        exported.append(path.name)
    monkeypatch.setattr(model_package, 'export_model', export)
    class Control:
        def send(self, command):
            assert command[0] == 'export'
            export(None, None, None, command[1])
        def recv(self):
            return {'peer_memory': [(0x210000000000, 65536)]}
    prepared = SimpleNamespace(model=object(), controls=[Control()] if world == 2 else [])
    storage = SimpleNamespace(world=world, peer_storage=SimpleNamespace(
        owned={0x200000000000: 65536}, reset=lambda: None))
    output = tmp_path / 'model'
    model_package.export_package(prepared, None, storage, output, options={})
    descriptor, start = read_index(output)
    assert descriptor.version == 1
    assert len(descriptor.ranks) == world
    assert sorted(exported) == [f'rank-{rank}.flaretensors' for rank in range(world)]
    assert output.is_file()
    assert len(descriptor.peer_memory) == (2 if world == 2 else 0)
    for owner, peer in enumerate(descriptor.peer_memory):
        assert peer.owner == owner and list(peer.readers) == list(range(world))
    assert list(tmp_path.iterdir()) == [output]


def test_failed_rank_export_leaves_no_published_model(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise RuntimeError('missing module image')
    monkeypatch.setattr(model_package, 'export_model', fail)
    with pytest.raises(RuntimeError, match='missing module image'):
        model_package.export_package(object(), None, SimpleNamespace(world=1), tmp_path / 'model', options={})
    assert not list(tmp_path.iterdir())


def test_combined_file_preserves_rank_resources_and_frontend_ranges(tmp_path):
    ranks = []
    for rank, data in enumerate([b'rank-zero-tokenizer', b'rank-one']):
        source = tmp_path / f'payload-{rank}'
        source.write_bytes(data)
        index = ModelIndex(version=1, ranks=[BundleIndex(
            resources={'weights': Span(offset=3, nbytes=len(data))})])
        if rank == 0:
            index.assets.append(Asset(name='tokenizer', payload=Span(offset=13, nbytes=9)))
            index.server.update({'model_id': 'test'})
        path = tmp_path / f'rank-{rank}.flaretensors'
        publish(path, index, [(3, source, 0, len(data))])
        ranks.append(path)
    model = tmp_path / 'model.flaretensors'
    combine(model, ranks, [])
    index, start = read_index(model)
    with model.open('rb') as source:
        for rank, expected in zip(index.ranks, [b'rank-zero-tokenizer', b'rank-one']):
            span = rank.resources['weights']
            source.seek(start + span.offset)
            assert source.read(span.nbytes) == expected
        source.seek(start + index.assets[0].payload.offset)
        assert source.read(index.assets[0].payload.nbytes) == b'tokenizer'
    assert index.server['model_id'] == 'test'
    with model.open('r+b') as source:
        source.truncate(model.stat().st_size - 1)
    with pytest.raises(ValueError, match='resource exceeds model package'):
        read_index(model)
