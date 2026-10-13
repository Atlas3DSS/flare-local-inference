"""Publish a model package containing one native resource bundle per rank."""

from pathlib import Path
import shutil
import tempfile

from .runtime_package import export_model
from .package_file import combine


def export_package(prepared, images, storage, path, *, options, progress=None):
    """Export every rank, close preparation resources, and assemble the model file."""
    import torch
    import torch.distributed as dist

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{path.name}-', dir=path.parent))
    try:
        roots = [f'rank-{rank}.safetensors' for rank in range(storage.world)]
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
        peers = []
        for rank, allocations in enumerate(memory):
            for address, size in allocations:
                peers.append(dict(owner=rank, address=address, nbytes=size, readers=list(range(storage.world))))
        prepared.close()
        for region in storage.host_regions:
            region.close()
        combine(path, [stage / root for root in roots], peers)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def export_cluster_package(prepared, images, storage, path, *, peer_files, options, export_id, world,
                           progress=None):
    """Export rank zero, pull every peer's exported rank, and write one package holding every rank.

    `peer_files` yields (host, path) for ranks 1 onward, in order, once each peer exported its
    rank. Every file is copied here with rsync over ssh and removed from its host. The package
    names its cluster and export id; `flare run --hosts` serves each rank on its own host.
    """
    import subprocess

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{path.name}-', dir=path.parent))
    try:
        files = [stage / "rank-0.safetensors"]
        export_model(prepared, images, storage, files[0], options=options, progress=progress)
        for rank, (host, source) in enumerate(peer_files, 1):
            files.append(stage / f"rank-{rank}.safetensors")
            subprocess.run(["rsync", "-a", "-e", "ssh -o BatchMode=yes", f"{host}:{source}", str(files[-1])],
                           check=True)
            subprocess.run(["ssh", "-o", "BatchMode=yes", host, "rm", "-rf", str(Path(source).parent)], check=True)
        if len(files) != world:
            raise RuntimeError(f"{len(files)} of {world} ranks exported")
        combine(path, files, [], cluster=(world, export_id))
    finally:
        if stage.exists():
            shutil.rmtree(stage)
