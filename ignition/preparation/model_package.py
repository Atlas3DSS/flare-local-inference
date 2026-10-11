"""Publish a model package containing one native resource bundle per rank."""

from pathlib import Path
import shutil
import tempfile

from .runtime_package import export_model
from .runtime_package_pb2 import ModelPackage


def export_package(prepared, images, storage, path, *, options, progress=None):
    import torch
    import torch.distributed as dist

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{path.name}-', dir=path.parent))
    try:
        roots = [f'rank-{rank}' for rank in range(storage.world)]
        if storage.world > 1:
            for rank, control in enumerate(prepared.controls, 1):
                control.send(('export', stage / roots[rank], options))
            dist.barrier()
            storage.peer_storage.reset()
            torch.cuda.synchronize()
            dist.barrier()
        model = prepared if storage.world == 1 else prepared.model
        export_model(model, images, storage, stage / roots[0], options=options, progress=progress)
        memory = [[]] if storage.world == 1 else [list(storage.peer_storage.owned.items())]
        if storage.world > 1:
            memory.extend(control.recv()['peer_memory'] for control in prepared.controls)
        descriptor = ModelPackage(version=1, ranks=roots)
        for rank, allocations in enumerate(memory):
            for address, size in allocations:
                descriptor.peer_memory.add(owner=rank, address=address, nbytes=size, readers=range(storage.world))
        (stage / 'model.pb').write_bytes(descriptor.SerializeToString())
        if path.exists():
            raise FileExistsError(path)
        stage.rename(path)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
