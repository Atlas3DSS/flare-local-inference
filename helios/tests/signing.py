"""Public test credentials shared by Python exporters and native fixture readers."""

from pathlib import Path
from ignition.preparation.runtime_package import export_model
from ignition.preparation.memory_inventory import print_inventory
from ignition.preparation.package_file import DEFAULT_SHARD_SIZE

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ignition.preparation.protection import Publisher


def publisher():
    return Publisher(Ed25519PrivateKey.from_private_bytes(bytes([42]) * 32),
                     'flare-tests', 'https://huggingface.co/flare-tests/fixture')


def publish_legacy(path, index, payload, *, publisher=None, shard_size=64 << 30):
    """Create version 4/5 containers for compatibility and corruption fixtures."""
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from ignition.preparation.package_file import _publish
    from ignition.preparation.protection import Protection

    if publisher is None:
        return _publish(path, index, payload)
    with TemporaryDirectory() as stage:
        copied = type(index)()
        copied.CopyFrom(index)
        protection = Protection(copied, payload, publisher, stage,
                                shard_size=shard_size, basename=Path(path).stem)
        return _publish(path, copied, protection.payload, protection)


def export_cluster_package(prepared, images, storage, path, *, peer_files, options, export_id, world,
                           publisher, peer_inventories, progress=None, shard_size=DEFAULT_SHARD_SIZE):
    """Create signed cluster containers for version 4/5 compatibility fixtures."""
    from ignition.preparation.package_file import extend_cluster, publish, read_index
    from ignition.preparation.protection import resource_roles

    if publisher is None:
        raise ValueError("final model publication requires a signing publisher")

    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.unlink(missing_ok=True)

    try:
        package = export_model(prepared, images, storage, partial, options=options, progress=progress)
        extend_cluster(partial, peer_files, (world, export_id))
        index, start = read_index(partial)
        payload = sorted((span.offset, partial, start + span.offset, span.nbytes)
                         for _, _, span in resource_roles(index))
        publish(path, index, payload, publisher=publisher, shard_size=shard_size)
        print_inventory([package.memory_inventory, *peer_inventories], console=getattr(progress, "console", None))
    finally:
        partial.unlink(missing_ok=True)
