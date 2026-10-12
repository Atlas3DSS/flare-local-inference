"""Indexed model files with bounded protobuf metadata and aligned binary payloads."""

import os
from itertools import chain
from pathlib import Path
import struct
import tempfile

from . import runtime_package_pb2 as proto


MAGIC = b"FLAREPKG"
ALIGNMENT = 4096
MAX_INDEX_BYTES = 16 << 20


def align(value):
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def read_index(path):
    """Read the index without reading any tensor, module, asset, or graph body."""
    with Path(path).open("rb") as source:
        prefix = source.read(16)
        if len(prefix) != 16 or prefix[:8] != MAGIC:
            raise ValueError("not a Flare model package")
        size = struct.unpack("<Q", prefix[8:])[0]
        if not 0 < size <= MAX_INDEX_BYTES:
            raise ValueError("invalid model index size")
        index = source.read(size)
        if len(index) != size:
            raise ValueError("truncated model index")
    result = proto.ModelIndex.FromString(index)
    if result.version != 1 or len(result.ranks) not in (1, 2, 4):
        raise ValueError("unsupported model index")
    start = align(16 + size)
    length = Path(path).stat().st_size - start
    spans = chain(
        (span for rank in result.ranks for span in rank.resources.values()),
        (asset.payload for asset in result.assets),
    )
    if length < 0 or any(span.offset + span.nbytes > length for span in spans):
        raise ValueError("resource exceeds model package")
    return result, start


def publish(path, index, payload):
    """Publish an index and an ordered sequence of file ranges without overwriting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    encoded = index.SerializeToString()
    if not 0 < len(encoded) <= MAX_INDEX_BYTES:
        raise ValueError("model index exceeds its metadata limit")
    data_start = align(16 + len(encoded))
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(MAGIC + struct.pack("<Q", len(encoded)) + encoded)
            output.truncate(data_start)
            for destination, source, offset, size in payload:
                output.seek(data_start + destination)
                with Path(source).open("rb") as contents:
                    contents.seek(offset)
                    remaining = size
                    while remaining:
                        block = contents.read(min(remaining, 4 << 20))
                        if not block:
                            raise ValueError(f"truncated package resource: {source}")
                        output.write(block)
                        remaining -= len(block)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def combine(path, ranks, peer_memory):
    """Combine independently exported rank files into one native model package."""
    result = proto.ModelIndex(version=1, peer_memory=peer_memory)
    payload, cursor = [], 0
    for rank, source in enumerate(ranks):
        index, start = read_index(source)
        if len(index.ranks) != 1:
            raise ValueError("rank export must contain one resource bundle")
        bundle = result.ranks.add()
        bundle.CopyFrom(index.ranks[0])
        cursor = align(cursor)
        for span in bundle.resources.values():
            span.offset += cursor
        if rank == 0:
            result.server.CopyFrom(index.server)
            for asset in index.assets:
                copied = result.assets.add()
                copied.CopyFrom(asset)
                copied.payload.offset += cursor
        size = Path(source).stat().st_size - start
        payload.append((cursor, source, start, size))
        cursor += size
    publish(path, result, payload)
