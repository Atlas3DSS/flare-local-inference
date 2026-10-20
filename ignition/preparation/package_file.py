"""Safetensors containers with an opaque payload and an authoritative protobuf index."""

import base64
import json
from contextlib import nullcontext
import os
from itertools import chain
from pathlib import Path
import struct
import re
import stat
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


DEFAULT_SHARD_SIZE = 64 << 30


def validate_filename(filename):
    """Accept a portable sibling basename, never a path or device name."""
    if (not isinstance(filename, str) or len(filename) > 255
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*\.safetensors', filename)
            or '..' in filename
            or filename.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL',
                *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}):
        raise ValueError('invalid package companion filename')
    return filename


def container_header(length, index_size=0, format='flare-data'):
    header = {
        "__metadata__": {"format": format},
        "flare.payload": {"dtype": "U8", "shape": [length], "data_offsets": [0, length]},
    }
    if format != 'flare-data':
        header["flare.metadata"] = {"dtype": "U8", "shape": [index_size],
                                    "data_offsets": [length, length + index_size]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded = encoded.ljust(align(8 + len(encoded)) - 8, b" ")
    if len(encoded) > MAX_HEADER_BYTES:
        raise ValueError('package header exceeds its size limit')
    return struct.pack('<Q', len(encoded)) + encoded


class PackageFiles:
    """Retain regular companion files opened relative to one pinned directory."""

    def __init__(self, path, primary, base, length):
        from .package_snapshot import read_container, LogicalFile
        self.layout = read_container(primary)[3]
        self.directory = os.open(Path(path).absolute().parent, os.O_RDONLY | os.O_DIRECTORY)
        self.files = {'': (primary, base, length)}
        self.logical = {}
        if self.layout is not None:
            for name, parts in self.layout['files'].items():
                self.logical[name] = (LogicalFile(parts, self._physical), 0, sum(n for n, _ in parts))

    def get(self, filename):
        if self.layout is not None:
            try:
                return self.logical[filename]
            except KeyError:
                raise ValueError('logical package file is missing') from None
        return self._physical(filename)

    def _physical(self, filename):
        if filename in self.files:
            return self.files[filename]
        validate_filename(filename)
        if len(self.files) > (65536 if self.layout is not None else 99):
            raise ValueError('package references too many data files')
        fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.directory)
        source = os.fdopen(fd, 'rb')
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError('package companion must be a regular file')
            prefix = source.read(8)
            if len(prefix) != 8:
                raise ValueError('truncated companion header')
            size, = struct.unpack('<Q', prefix)
            if not 0 < size <= MAX_HEADER_BYTES or (8 + size) % ALIGNMENT:
                raise ValueError('invalid companion header size')
            header = json.loads(source.read(size), object_pairs_hook=_unique_fields)
            if (set(header) != {'__metadata__', 'flare.payload'}
                    or header['__metadata__'] != {'format': 'flare-data'}):
                raise ValueError('invalid package companion header')
            start, length = _byte_tensor(header['flare.payload'])
            if start or size + 8 + length != info.st_size:
                raise ValueError('invalid companion payload length')
            self.files[filename] = source, size + 8, length
            return self.files[filename]
        except BaseException:
            source.close()
            raise

    def close(self):
        for name, (source, _, _) in self.files.items():
            if name:
                source.close()
        os.close(self.directory)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def stored_parts(data):
    if len(data.sha256) != 32:
        raise ValueError('invalid stored-data digest')
    if data.parts:
        if data.offset or sum(part.nbytes for part in data.parts) != data.nbytes:
            raise ValueError('invalid signed stored-data parts')
        return list(data.parts)
    return [proto.Span(offset=data.offset, nbytes=data.nbytes)] if data.nbytes else []


def embedding_spans(index):
    """Yield each table plane with its owning rank, or None for shared planes."""
    for table in index.engram.tables:
        if table.shards:
            if table.HasField("weights") or table.HasField("scales"):
                raise ValueError("partitioned embedding tables cannot also hold shared planes")
            for shard in table.shards:
                for name in ("weights", "scales"):
                    if not shard.HasField(name):
                        raise ValueError("embedding partition lacks a payload plane")
                    yield shard.rank, getattr(shard, name)
        else:
            for name in ("weights", "scales"):
                yield None, getattr(table, name)


def _validate_spans(index, length):
    if index.version not in (4, 5) or not index.ranks:
        raise ValueError("unsupported model index")
    spans = chain(
        (span for rank in index.ranks for span in rank.resources.values()),
        (asset.payload for asset in index.assets),
        (span for _, span in embedding_spans(index)),
    )
    for span in spans:
        if span.filename:
            validate_filename(span.filename)
            if index.version != 5:
                raise ValueError('companion references require model index 5')
        elif span.offset + span.nbytes > length:
            raise ValueError("resource exceeds model package payload")
    if any(span.filename for rank in index.ranks for allocation in rank.allocations
           for span in allocation.data_spans):
        raise ValueError('allocation destination spans cannot name a file')


def read_index(path):
    """Inspect staging or signed metadata without authenticating it or reading payloads."""
    from .package_snapshot import read_container
    with _open_source(path) as source:
        header, start, encoded, layout = read_container(source)
        if layout is not None:
            envelope = proto.ProtectedIndex.FromString(encoded)
            if envelope.version != 4 or len(envelope.groups) != 3 or not envelope.ranks:
                raise ValueError('invalid protected package envelope')
            result = proto.PublisherMetadata.FromString(envelope.groups[0].body).index
            for group in envelope.ranks:
                result.ranks.add().CopyFrom(proto.RankProtection.FromString(group.body).bundle)
            _validate_spans(result, sum(n for n, _ in layout['files']['']))
            return result, start
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


def publish(path, index, payload, *, publisher=None, shard_size=DEFAULT_SHARD_SIZE):
    """Publish private rank staging or a publisher-protected final package."""
    if publisher is None:
        return _publish(path, index, payload)
    if not ALIGNMENT < shard_size <= DEFAULT_SHARD_SIZE:
        raise ValueError('payload shard limit must exceed 4096 bytes and be at most 64G')
    from .protection import Protection
    with tempfile.TemporaryDirectory(prefix="flare-protection-") as stage:
        protected_index = type(index)()
        protected_index.CopyFrom(index)
        protection = Protection(protected_index, list(payload), publisher, stage,
                                shard_size=shard_size, basename=Path(path).stem)
        from .package_snapshot import publish_snapshot
        return publish_snapshot(path, protected_index, protection, shard_size=shard_size)


def _publish(path, index, payload, protection=None):
    """Sync companions before publishing the primary, without replacing any existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payloads = {'': list(payload)} if protection is None else protection.payloads
    names = [name for name in payloads if name] + ['']
    destinations = {name: path.parent / name if name else path for name in names}
    for destination in destinations.values():
        if os.path.lexists(destination):
            raise FileExistsError(destination)
    encoded = (index.SerializeToString(deterministic=True) if protection is None
               else protection.envelope())
    if not 0 < len(encoded) <= MAX_INDEX_BYTES:
        raise ValueError(f"model index is {len(encoded)} bytes; metadata limit is {MAX_INDEX_BYTES} bytes")
    published = []
    try:
        for name in names:
            if not name and len(names) > 1:
                _sync_directory(path.parent)
            entries, length = payloads[name], 0
            for destination, _, offset, size in entries:
                if destination < length or offset < 0 or size < 0:
                    raise ValueError("package payload ranges must be ordered and disjoint")
                length = destination + size
            if not name:
                _validate_spans(index, length)
            header = container_header(length, 0 if name else len(encoded),
                                      'flare-data' if name else 'flare' if protection else 'flare-staging')
            destination = destinations[name]
            descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}-", dir=path.parent)
            try:
                with os.fdopen(descriptor, "wb") as output:
                    output.write(header)
                    output.flush()
                    _copy_payload(output.fileno(), len(header), entries, protection, name)
                    output.truncate(len(header) + length)
                    if not name:
                        if protection is not None:
                            signed = protection.envelope(final=True)
                            if len(signed) != len(encoded):
                                raise ValueError("signed envelope size changed during publication")
                            encoded = signed
                        output.seek(len(header) + length)
                        output.write(encoded)
                    output.flush()
                    os.fsync(output.fileno())
                os.link(temporary, destination)
                info = os.stat(temporary)
                published.append((destination, info.st_dev, info.st_ino))
            finally:
                Path(temporary).unlink(missing_ok=True)
        _sync_directory(path.parent)
    except BaseException:
        for destination, device, inode in reversed(published):
            try:
                info = destination.lstat()
                if (info.st_dev, info.st_ino) == (device, inode):
                    destination.unlink()
            except FileNotFoundError:
                pass
        raise


def _sync_directory(path):
    directory = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


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


def _copy_payload(fd, data_start, payload, protection, filename=''):
    """Copy in parallel across owners; hash each owner's parts in publication order."""
    from concurrent.futures import ThreadPoolExecutor

    parts = [] if protection is None else [(owner, part) for owner, ranges in protection.ranges.items()
                                           for part in ranges if part.filename == filename]
    stored_jobs = [(protection.hashes[owner], part, [], protection.source_verifiers.get(owner))
                   for owner, part in parts]
    jobs = list(stored_jobs)
    for entry in payload:
        destination, _, _, size = entry
        job = next((job for job in stored_jobs if job[1].offset <= destination
                    and destination + size <= job[1].offset + job[1].nbytes), None)
        if job is None:
            jobs.append((None, None, [entry], None))
        else:
            job[2].append(entry)

    def copy(digest, stored, entries, verifier):
        cursor = stored.offset if stored else 0
        for destination, source, offset, size in entries:
            if digest is not None:
                _absorb_zeros(digest, destination - cursor)
            with _open_source(source) as contents:
                copied = 0
                while copied < size:
                    length = min(COPY_BLOCK, size - copied)
                    block = os.pread(contents.fileno(), length, offset + copied)
                    if len(block) != length:
                        raise ValueError(f"truncated package resource: {source}")
                    if verifier is not None:
                        verifier.consume(contents, offset + copied, block)
                    _write_all(fd, block, data_start + destination + copied)
                    if digest is not None:
                        digest.update(block)
                    copied += length
            cursor = destination + size
        if digest is not None:
            _absorb_zeros(digest, stored.offset + stored.nbytes - cursor)

    if jobs:
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


def combine(path, ranks, peer_memory, *, publisher=None, shard_size=DEFAULT_SHARD_SIZE, cluster=None):
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
                for _, span in embedding_spans(result):
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
    if cluster is not None:
        if len(result.ranks) != cluster[0]:
            raise ValueError('a multi-host package holds every rank of its cluster')
        result.cluster.world, result.cluster.export_id = cluster
    publish(path, result, payload, publisher=publisher, shard_size=shard_size)


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
