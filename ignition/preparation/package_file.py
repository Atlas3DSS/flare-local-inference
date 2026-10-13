"""Safetensors containers with an opaque payload and an authoritative protobuf index."""

import json
import os
from itertools import chain
from pathlib import Path
import struct
import tempfile

from . import runtime_package_pb2 as proto


ALIGNMENT = 4096
MAX_HEADER_BYTES = 64 << 10
MAX_INDEX_BYTES = 16 << 20


def align(value):
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate safetensors field: {key}")
        result[key] = value
    return result


def _byte_tensor(tensor):
    if not isinstance(tensor, dict) or set(tensor) != {"dtype", "shape", "data_offsets"}:
        raise ValueError("invalid package byte tensor")
    offsets, shape = tensor["data_offsets"], tensor["shape"]
    if (tensor["dtype"] != "U8" or not isinstance(offsets, list) or len(offsets) != 2
            or any(type(n) is not int or n < 0 for n in offsets)
            or offsets[1] < offsets[0] or not isinstance(shape, list) or len(shape) != 1
            or type(shape[0]) is not int or shape[0] != offsets[1] - offsets[0]):
        raise ValueError("invalid package byte tensor")
    return offsets


def _validate_spans(index, length):
    if index.version != 1 or len(index.ranks) not in (1, 2, 4):
        raise ValueError("unsupported model index")
    spans = chain(
        (span for rank in index.ranks for span in rank.resources.values()),
        (asset.payload for asset in index.assets),
        (span for table in index.engram.tables for span in (table.weights, table.scales)),
    )
    if any(span.offset + span.nbytes > length for span in spans):
        raise ValueError("resource exceeds model package payload")


def read_index(path):
    """Read the index without reading any tensor, module, asset, or graph body."""
    with Path(path).open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors header length")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 0 < header_size <= MAX_HEADER_BYTES:
            raise ValueError("invalid safetensors header size")
        start = 8 + header_size
        header = source.read(header_size)
        if len(header) != header_size or not header.startswith(b"{") or start % ALIGNMENT:
            raise ValueError("invalid or unaligned safetensors header")
        header = json.loads(header, object_pairs_hook=_unique_fields)
        if (set(header) != {"__metadata__", "flare.payload", "flare.metadata"}
                or header["__metadata__"] != {"format": "flare"}):
            raise ValueError("not a Flare safetensors package")
        payload_start, length = _byte_tensor(header["flare.payload"])
        index_start, end = _byte_tensor(header["flare.metadata"])
        if payload_start != 0 or index_start != length:
            raise ValueError("package tensors must cover the data buffer without gaps or overlap")
        size = end - index_start
        if not 0 < size <= MAX_INDEX_BYTES:
            raise ValueError("invalid model index size")
        if start + end != os.fstat(source.fileno()).st_size:
            raise ValueError("safetensors data length differs from model package")
        source.seek(start + index_start)
        index = source.read(size)
        if len(index) != size:
            raise ValueError("truncated model index")
    result = proto.ModelIndex.FromString(index)
    _validate_spans(result, length)
    return result, start


def publish(path, index, payload):
    """Publish an index and an ordered sequence of file ranges without overwriting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    encoded = index.SerializeToString(deterministic=True)
    if not 0 < len(encoded) <= MAX_INDEX_BYTES:
        raise ValueError("model index exceeds its metadata limit")
    payload = list(payload)
    length = 0
    for destination, _, offset, size in payload:
        if destination < length or offset < 0 or size < 0:
            raise ValueError("package payload ranges must be ordered and disjoint")
        length = destination + size
    _validate_spans(index, length)
    header = json.dumps({
        "__metadata__": {"format": "flare"},
        "flare.payload": {"dtype": "U8", "shape": [length], "data_offsets": [0, length]},
        "flare.metadata": {"dtype": "U8", "shape": [len(encoded)],
                           "data_offsets": [length, length + len(encoded)]},
    }, separators=(",", ":")).encode()
    data_start = align(8 + len(header))
    header = header.ljust(data_start - 8, b" ")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(struct.pack("<Q", len(header)) + header)
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
            output.seek(data_start + length)
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def combine(path, ranks, peer_memory, *, cluster=None):
    """Place every rank's tensor storage before execution resources and the index.

    `cluster` (world, export id) marks a package whose ranks run on separate hosts; it holds every
    rank, and each host loads the one it serves.
    """
    result = proto.ModelIndex(version=1, peer_memory=peer_memory)
    if cluster is not None:
        if len(ranks) != cluster[0]:
            raise ValueError("a multi-host package holds every rank of its cluster")
        result.cluster.world, result.cluster.export_id = cluster
    storage, execution = [], []
    for rank, source in enumerate(ranks):
        index, start = read_index(source)
        if len(index.ranks) != 1:
            raise ValueError("rank export must contain one resource bundle")
        bundle = result.ranks.add()
        bundle.CopyFrom(index.ranks[0])
        stored = {allocation.file for allocation in bundle.allocations if allocation.HasField("file")}
        for name, span in sorted(bundle.resources.items()):
            (storage if name in stored else execution).append((span, source, start + span.offset))
        if rank == 0:
            result.server.CopyFrom(index.server)
            if index.HasField("engram"):
                result.engram.CopyFrom(index.engram)
                for table in result.engram.tables:
                    for span in (table.weights, table.scales):
                        storage.append((span, source, start + span.offset))
            for asset in index.assets:
                copied = result.assets.add()
                copied.CopyFrom(asset)
                execution.append((copied.payload, source, start + copied.payload.offset))
        elif index.HasField("engram"):
            raise ValueError("offloaded Engram tables must be stored once on rank zero")
    payload, cursor = [], 0
    for span, source, offset in chain(storage, execution):
        cursor = align(cursor)
        span.offset = cursor
        payload.append((cursor, source, offset, span.nbytes))
        cursor += span.nbytes
    publish(path, result, payload)
