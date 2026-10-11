"""Atomic publication and rank ownership in native model packages."""

from types import SimpleNamespace

import pytest

from ignition.preparation import model_package
from ignition.preparation.runtime_package_pb2 import ModelPackage


@pytest.mark.parametrize('world', [1, 2])
def test_publication_contains_one_bundle_per_rank(monkeypatch, tmp_path, world):
    import torch
    import torch.distributed as dist

    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(dist, 'barrier', lambda: None)
    exported = []
    def export(prepared, images, storage, path, **kwargs):
        path.mkdir()
        (path / 'package.pb').write_bytes(b'rank resources')
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
    descriptor = ModelPackage.FromString((output / 'model.pb').read_bytes())
    assert descriptor.version == 1
    assert list(descriptor.ranks) == [f'rank-{rank}' for rank in range(world)]
    assert sorted(exported) == list(descriptor.ranks)
    assert all((output / rank / 'package.pb').is_file() for rank in descriptor.ranks)
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
