"""Publish a model package containing one native resource bundle per rank."""

from pathlib import Path
import shutil
import tempfile

from .runtime_package import export_model
from .package_file import combine


def export_package(prepared, images, storage, path, *, options, publisher, progress=None):
    """Export every rank, close preparation resources, and assemble the model file."""
    if publisher is None:
        raise ValueError("final model publication requires a signing publisher")
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
        combine(path, [stage / root for root in roots], peers, publisher=publisher)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def export_cluster_package(prepared, images, storage, path, *, peer_files, options, export_id, world,
                           publisher, progress=None):
    """Assemble private rank exports and sign one package holding every rank.

    `peer_files` yields a readable stream of each peer's exported rank file, ranks 1 onward, in
    order; each is appended to a staging file as it arrives. Publication copies stored data and
    encrypts executables into the final package before removing staging. The
    package names its cluster and export id; `flare run --hosts` serves each rank on its own host.
    """
    from .package_file import extend_cluster, publish, read_index
    from .protection import resource_roles

    if publisher is None:
        raise ValueError("final model publication requires a signing publisher")

    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.unlink(missing_ok=True)

    try:
        export_model(prepared, images, storage, partial, options=options, progress=progress)
        extend_cluster(partial, peer_files, (world, export_id))
        index, start = read_index(partial)
        payload = sorted((span.offset, partial, start + span.offset, span.nbytes)
                         for _, _, span in resource_roles(index))
        publish(path, index, payload, publisher=publisher)
    finally:
        partial.unlink(missing_ok=True)
