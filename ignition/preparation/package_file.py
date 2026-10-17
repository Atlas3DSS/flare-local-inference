"""Safetensors containers with an opaque payload and an authoritative protobuf index."""

import base64
import json
from contextlib import nullcontext
import os
from itertools import chain
from pathlib import Path
import struct
import tempfile

from . import runtime_package_pb2 as proto


ALIGNMENT = 4096
MAX_HEADER_BYTES = 64 << 10
MAX_INDEX_BYTES = 128 << 20
# Payload copies run on several threads; hashlib and positional reads and writes release the GIL.
COPY_BLOCK = 16 << 20
COPY_THREADS = 16
_ZEROS = bytes(4 << 20)


def _open_source(source):
    return nullcontext(source) if hasattr(source, 'read') else Path(source).open('rb')


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
    if index.version != 4 or not index.ranks:
        raise ValueError("unsupported model index")
    spans = chain(
        (span for rank in index.ranks for span in rank.resources.values()),
        (asset.payload for asset in index.assets),
        (span for table in index.engram.tables for span in (table.weights, table.scales)),
    )
    if any(span.offset + span.nbytes > length for span in spans):
        raise ValueError("resource exceeds model package payload")


def read_index(path):
    """Inspect staging or signed metadata without authenticating it or reading payloads."""
    with _open_source(path) as source:
        source.seek(0)
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
                or header["__metadata__"] not in ({"format": "flare"}, {"format": "flare-staging"})):
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
    if header["__metadata__"]["format"] == "flare":
        envelope = proto.ProtectedIndex.FromString(index)
        if (envelope.version != 4 or len(envelope.groups) != 3 or envelope.groups[0].kind != 1
                or not envelope.ranks):
            raise ValueError("invalid protected package envelope")
        result = proto.PublisherMetadata.FromString(envelope.groups[0].body).index
        for group in envelope.ranks:
            result.ranks.add().CopyFrom(proto.RankProtection.FromString(group.body).bundle)
    else:
        result = proto.ModelIndex.FromString(index)
    _validate_spans(result, length)
    return result, start


def publish(path, index, payload, *, publisher=None):
    """Publish private rank staging or a publisher-protected final package."""
    if publisher is None:
        return _publish(path, index, payload)
    from .protection import Protection
    with tempfile.TemporaryDirectory(prefix="flare-protection-") as stage:
        protected_index = type(index)()
        protected_index.CopyFrom(index)
        protection = Protection(protected_index, list(payload), publisher, stage)
        return _publish(path, protected_index, protection.payload, protection)


def _publish(path, index, payload, protection=None):
    """Publish an index and an ordered sequence of file ranges without overwriting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    encoded = (index.SerializeToString(deterministic=True) if protection is None
               else protection.envelope())
    if not 0 < len(encoded) <= MAX_INDEX_BYTES:
        raise ValueError(f"model index is {len(encoded)} bytes; metadata limit is {MAX_INDEX_BYTES} bytes")
    payload = list(payload)
    length = 0
    for destination, _, offset, size in payload:
        if destination < length or offset < 0 or size < 0:
            raise ValueError("package payload ranges must be ordered and disjoint")
        length = destination + size
    _validate_spans(index, length)
    header = json.dumps({
        "__metadata__": {"format": "flare" if protection is not None else "flare-staging"},
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
            output.flush()
            _copy_payload(output.fileno(), data_start, payload, protection)
            if protection is not None:
                signed = protection.envelope(final=True)
                if len(signed) != len(encoded):
                    raise ValueError("signed envelope size changed during publication")
                encoded = signed
            output.seek(data_start + length)
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _write_all(fd, data, position):
    view = memoryview(data)
    while view:
        written = os.pwrite(fd, view, position)
        view, position = view[written:], position + written


def _absorb_zeros(digest, count):
    while count > 0:
        block = _ZEROS[:min(count, len(_ZEROS))]
        digest.update(block)
        count -= len(block)


def _copy_payload(fd, data_start, payload, protection):
    """Copy every payload range into `fd` at `data_start` onward, in parallel.

    Each part's stored data is contiguous and hashed in order, padding included, so parts hash on
    separate threads; ranges outside stored data copy as independent jobs.
    """
    from bisect import bisect_right
    from concurrent.futures import ThreadPoolExecutor

    parts, starts = {}, []
    if protection is not None:
        starts = sorted((start, owner) for owner, (start, nbytes) in protection.ranges.items() if nbytes)
    positions = [start for start, _ in starts]
    jobs = []
    for entry in payload:
        destination = entry[0]
        index = bisect_right(positions, destination) - 1
        if protection is not None and destination < protection.data_end and index >= 0:
            # What every rank shares is the part owned by None.
            owner = starts[index][1]
            if owner not in parts:
                parts[owner] = []
                jobs.append((protection.hashes[owner], protection.ranges[owner], parts[owner]))
            parts[owner].append(entry)
        else:
            jobs.append((None, None, [entry]))

    def copy(digest, stored, entries):
        if digest is not None:
            cursor, nbytes = stored
            end = cursor + nbytes
        for destination, source, offset, size in entries:
            if digest is not None:
                # Stored-data hashes cover the zero padding between ranges.
                _absorb_zeros(digest, destination - cursor)
            # Open sources (a signed package being re-signed) are read by position, so threads share them.
            with _open_source(source) as contents:
                copied = 0
                while copied < size:
                    length = min(COPY_BLOCK, size - copied)
                    block = os.pread(contents.fileno(), length, offset + copied)
                    if len(block) != length:
                        raise ValueError(f"truncated package resource: {source}")
                    _write_all(fd, block, data_start + destination + copied)
                    if digest is not None:
                        digest.update(block)
                    copied += length
            if digest is not None:
                cursor = destination + size
        if digest is not None:
            _absorb_zeros(digest, end - cursor)

    if not jobs:
        return
    with ThreadPoolExecutor(min(COPY_THREADS, len(jobs))) as pool:
        for future in [pool.submit(copy, *job) for job in jobs]:
            future.result()


def write_sources(path, index, payload):
    """Record a rank export as its index and the files holding each payload range, without copying.

    `combine` reads the ranges from those files, which must remain until it returns.
    """
    payload = [(destination, str(source), offset, size) for destination, source, offset, size in payload]
    length = 0
    for destination, _, offset, size in payload:
        if destination < length or offset < 0 or size < 0:
            raise ValueError("package payload ranges must be ordered and disjoint")
        length = destination + size
    _validate_spans(index, length)
    document = {"index": base64.b64encode(index.SerializeToString(deterministic=True)).decode(), "payload": payload}
    with Path(path).open("x") as output:
        json.dump(document, output, separators=(",", ":"))


def read_sources(path):
    """Read a rank export recorded by `write_sources`.

    Returns its index and a function mapping a span to the files and offsets holding its bytes:
    one piece per recorded range it covers, such as a table assembled from several checkpoint
    shards. The ranges a span covers must be adjacent.
    """
    from bisect import bisect_right

    document = json.loads(Path(path).read_text(), object_pairs_hook=_unique_fields)
    if set(document) != {"index", "payload"}:
        raise ValueError("invalid rank export sources")
    index = proto.ModelIndex.FromString(base64.b64decode(document["index"], validate=True))
    payload, length = [], 0
    for destination, source, offset, size in document["payload"]:
        if destination < length or offset < 0 or size < 0:
            raise ValueError("package payload ranges must be ordered and disjoint")
        if size:
            payload.append((destination, Path(source), offset, size))
        length = destination + size
    _validate_spans(index, length)
    starts = [entry[0] for entry in payload]

    def locate(span):
        position = bisect_right(starts, span.offset) - 1
        if not span.nbytes:
            # An empty span keeps a zero-size piece of the range it starts in, so it stays addressable.
            if position < 0 or span.offset > sum(payload[position][::3]):
                raise ValueError("rank export resource lies outside its recorded sources")
            destination, source, offset, _ = payload[position]
            return ((source, offset + span.offset - destination, 0),)
        pieces, cursor, end = [], span.offset, span.offset + span.nbytes
        while cursor < end:
            if not 0 <= position < len(payload) or not payload[position][0] <= cursor < sum(payload[position][::3]):
                raise ValueError("rank export resource lies outside its recorded sources")
            destination, source, offset, size = payload[position]
            nbytes = min(end, destination + size) - cursor
            pieces.append((source, offset + cursor - destination, nbytes))
            cursor, position = cursor + nbytes, position + 1
        return tuple(pieces)

    return index, locate


def combine(path, ranks, peer_memory, *, publisher=None):
    """Place every rank's tensor storage before execution resources and the index.

    `ranks` names each rank's export recorded by `write_sources`; ranges copy directly from the
    files it names.
    """
    result = proto.ModelIndex(version=4, peer_memory=peer_memory)
    storage, execution = [], []
    for rank, recorded in enumerate(ranks):
        index, source = read_sources(recorded)
        if len(index.ranks) != 1:
            raise ValueError("rank export must contain one resource bundle")
        bundle = result.ranks.add()
        bundle.CopyFrom(index.ranks[0])
        stored = {allocation.file for allocation in bundle.allocations if allocation.HasField("file")}
        for name, span in sorted(bundle.resources.items()):
            (storage if name in stored else execution).append((span, source(span)))
        if rank == 0:
            result.contract.CopyFrom(index.contract)
            if index.HasField("sampling_defaults"):
                result.sampling_defaults.CopyFrom(index.sampling_defaults)
            if index.HasField("frontend"):
                result.frontend.CopyFrom(index.frontend)
            if index.HasField("engram"):
                result.engram.CopyFrom(index.engram)
                for table in result.engram.tables:
                    for span in (table.weights, table.scales):
                        storage.append((span, source(span)))
            for asset in index.assets:
                copied = result.assets.add()
                copied.CopyFrom(asset)
                execution.append((copied.payload, source(copied.payload)))
        elif index.HasField("engram"):
            raise ValueError("offloaded Engram tables must be stored once on rank zero")
    payload, cursor = [], 0
    for span, pieces in chain(storage, execution):
        cursor = align(cursor)
        span.offset = cursor
        for source, offset, nbytes in pieces:
            payload.append((cursor, source, offset, nbytes))
            cursor += nbytes
    publish(path, result, payload, publisher=publisher)


def _copy(source, output, nbytes):
    while nbytes:
        block = source.read(min(nbytes, 16 << 20))
        if not block:
            raise ValueError("truncated rank export")
        output.write(block)
        nbytes -= len(block)


def extend_cluster(path, peers, cluster):
    """Append ranks 1 onward to rank zero's private staging file at `path`.

    `peers` yields binary streams, each a complete exported rank file of one bundle, in rank order;
    each is read once, front to back. Every rank's payload stays contiguous, rank zero's where its
    export wrote it, so a host reads one region and this disk never holds a second copy of any
    rank. `cluster` is (world, export id): the package holds every rank, and each host loads the
    one it serves.
    """
    index, start = read_index(path)
    with Path(path).open("rb") as source:
        source.seek(8)
        header = json.loads(source.read(start - 8), object_pairs_hook=_unique_fields)
    if header["__metadata__"] != {"format": "flare-staging"}:
        raise ValueError("cluster assembly requires unsigned rank staging")
    if len(index.ranks) != 1:
        raise ValueError("rank export must contain one resource bundle")
    cursor = _byte_tensor(header["flare.payload"])[1]
    result = proto.ModelIndex()
    result.CopyFrom(index)
    with Path(path).open("r+b") as output:
        for stream in peers:
            size = struct.unpack("<Q", stream.read(8))[0]
            if not 0 < size <= MAX_HEADER_BYTES or (8 + size) % ALIGNMENT:
                raise ValueError("invalid rank export header")
            peer = json.loads(stream.read(size), object_pairs_hook=_unique_fields)
            if peer["__metadata__"] != {"format": "flare-staging"}:
                raise ValueError("cluster assembly requires unsigned rank staging")
            length = _byte_tensor(peer["flare.payload"])[1]
            index_start, end = _byte_tensor(peer["flare.metadata"])
            base = align(cursor)
            output.seek(start + base)
            _copy(stream, output, length)
            peer_index = proto.ModelIndex.FromString(stream.read(end - index_start))
            _validate_spans(peer_index, length)
            if len(peer_index.ranks) != 1 or peer_index.HasField("engram") or peer_index.assets:
                raise ValueError("a peer rank export must hold one bundle and no shared tables or assets")
            bundle = result.ranks.add()
            bundle.CopyFrom(peer_index.ranks[0])
            for span in bundle.resources.values():
                span.offset += base
            cursor = base + length
        if len(result.ranks) != cluster[0]:
            raise ValueError("a multi-host package holds every rank of its cluster")
        result.cluster.world, result.cluster.export_id = cluster
        _validate_spans(result, cursor)
        encoded = result.SerializeToString(deterministic=True)
        if not 0 < len(encoded) <= MAX_INDEX_BYTES:
            raise ValueError("model index exceeds its metadata limit")
        header = json.dumps({
            "__metadata__": {"format": "flare-staging"},
            "flare.payload": {"dtype": "U8", "shape": [cursor], "data_offsets": [0, cursor]},
            "flare.metadata": {"dtype": "U8", "shape": [len(encoded)],
                               "data_offsets": [cursor, cursor + len(encoded)]},
        }, separators=(",", ":")).encode()
        if 8 + len(header) > start:
            raise ValueError("package header outgrew rank zero's reserved space")
        output.seek(start + cursor)
        output.write(encoded)
        output.truncate()
        output.seek(0)
        output.write(struct.pack("<Q", start - 8) + header.ljust(start - 8, b" "))
        output.flush()
        os.fsync(output.fileno())
