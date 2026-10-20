"""Immutable payload blobs and metadata-only package snapshots.

The layout maps signed logical payload offsets to physical files. Publisher
signatures and executable encryption continue to authenticate the logical bytes,
so conversion does not require a private key or change the model's identity.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import tempfile

from .package_file import (ALIGNMENT, DEFAULT_SHARD_SIZE, MAX_HEADER_BYTES, MAX_INDEX_BYTES,
                           _byte_tensor, _unique_fields, align, container_header)

MAX_BLOBS = 65536
BLOCK = 8 << 20
_OPTIONS = ContextVar('package_snapshot_options', default=None)


@dataclass(frozen=True)
class ExportOptions:
    cache: Path
    no_weights: bool = False


@contextmanager
def export_options(cache, *, no_weights=False):
    token = _OPTIONS.set(ExportOptions(Path(cache), no_weights))
    try:
        yield
    finally:
        _OPTIONS.reset(token)


def read_at(source, size, offset):
    if isinstance(source, LogicalFile):
        return source.read_at(size, offset)
    return os.pread(source.fileno(), size, offset)


def read_container(source):
    """Return the physical header, signed envelope, and optional storage layout."""
    prefix = os.pread(source.fileno(), 8, 0)
    if len(prefix) != 8:
        raise ValueError('truncated package header')
    size, = struct.unpack('<Q', prefix)
    if not 0 < size <= MAX_HEADER_BYTES or (size + 8) % ALIGNMENT:
        raise ValueError('invalid package header size')
    header = json.loads(os.pread(source.fileno(), size, 8), object_pairs_hook=_unique_fields)
    if not isinstance(header, dict):
        raise ValueError('invalid package header')
    base = size + 8
    snapshot = header.get('__metadata__') == {'format': 'flare-snapshot'}
    expected = {'__metadata__', 'flare.payload', 'flare.metadata'}
    if snapshot:
        expected.add('flare.layout')
    if set(header) != expected:
        raise ValueError('invalid package container fields')
    begin, length = _byte_tensor(header['flare.payload'])
    index_start, end = _byte_tensor(header['flare.metadata'])
    if not 0 < end - index_start <= MAX_INDEX_BYTES:
        raise ValueError('invalid model index size')
    if begin or index_start != length:
        raise ValueError('invalid package tensor ranges')
    envelope = os.pread(source.fileno(), end - index_start, base + index_start)
    layout = None
    if snapshot:
        layout_start, layout_end = _byte_tensor(header['flare.layout'])
        if length or layout_start != end or not 0 < layout_end - end <= MAX_INDEX_BYTES:
            raise ValueError('invalid snapshot metadata layout')
        layout = json.loads(os.pread(source.fileno(), layout_end - end, base + end),
                            object_pairs_hook=_unique_fields)
        validate_layout(layout)
        end = layout_end
    elif header['__metadata__'] not in ({'format': 'flare'}, {'format': 'flare-staging'}):
        raise ValueError('not a Flare package')
    if base + end != os.fstat(source.fileno()).st_size:
        raise ValueError('safetensors data length differs from model package')
    return header, base, envelope, layout


def validate_layout(layout):
    from .package_file import validate_filename
    if not isinstance(layout, dict) or set(layout) != {'version', 'files'} or layout['version'] != 1:
        raise ValueError('unsupported snapshot layout')
    files = layout['files']
    if not isinstance(files, dict) or '' not in files or len(files) > 100:
        raise ValueError('invalid logical package files')
    count = 0
    lengths = {}
    for name, parts in files.items():
        if name:
            validate_filename(name)
        if not isinstance(parts, list):
            raise ValueError('invalid logical payload segments')
        for part in parts:
            count += 1
            if (not isinstance(part, list) or len(part) != 2 or type(part[0]) is not int
                    or not 0 < part[0] <= DEFAULT_SHARD_SIZE - ALIGNMENT
                    or not isinstance(part[1], str)
                    or (part[1] and not re.fullmatch(r'[0-9a-f]{64}\.safetensors', part[1]))):
                raise ValueError('invalid snapshot payload segment')
            if part[1] and lengths.setdefault(part[1], part[0]) != part[0]:
                raise ValueError('conflicting snapshot blob lengths')
    if count > MAX_BLOBS:
        raise ValueError('snapshot has too many payload segments')


class LogicalFile:
    """Read signed logical bytes from pinned blob handles and zero padding."""
    def __init__(self, parts, physical):
        self.parts, self.physical, self.position = parts, physical, 0

    def read_at(self, size, offset):
        if size < 0 or offset < 0:
            raise ValueError('invalid logical read')
        result, cursor = bytearray(), 0
        end = offset + size
        for length, name in self.parts:
            stop = cursor + length
            if cursor < end and offset < stop:
                first, last = max(offset, cursor), min(end, stop)
                if name:
                    source, base, actual = self.physical(name)
                    if actual != length:
                        raise ValueError('snapshot blob length mismatch')
                    data = os.pread(source.fileno(), last - first, base + first - cursor)
                    if len(data) != last - first:
                        raise ValueError('truncated snapshot blob')
                    result.extend(data)
                else:
                    result.extend(bytes(last - first))
            cursor = stop
            if cursor >= end:
                break
        return bytes(result)

    def seek(self, offset, whence=0):
        if whence == 1:
            offset += self.position
        elif whence == 2:
            offset += sum(n for n, _ in self.parts)
        elif whence != 0:
            raise ValueError('invalid seek')
        if offset < 0:
            raise ValueError('negative seek')
        self.position = offset
        return offset

    def read(self, size=-1):
        if size < 0:
            size = sum(n for n, _ in self.parts) - self.position
        data = self.read_at(max(0, size), self.position)
        self.position += len(data)
        return data


class BlobStore:
    def __init__(self, root):
        self.root = Path(root) / 'blobs'
        self.root.mkdir(parents=True, exist_ok=True)
        self.written = self.reused = 0

    def put(self, size, blocks, *, required=False):
        """Hash before writing so an existing blob incurs no payload writes."""
        if not 0 < size <= DEFAULT_SHARD_SIZE - ALIGNMENT:
            raise ValueError('package resource exceeds the 64G payload shard limit')
        header = container_header(size)
        digest = hashlib.sha256(header)
        count = 0
        for block in blocks():
            digest.update(block)
            count += len(block)
        if count != size:
            raise ValueError('blob input length mismatch')
        name = digest.hexdigest() + '.safetensors'
        target = self.root / name
        if target.exists():
            self.verify(target, digest.digest(), len(header) + size)
            self.reused += size
            return name
        if required:
            raise ValueError(f'--no-weights requires cached payload {name}; export with weights first')
        fd, temporary = tempfile.mkstemp(prefix='.blob-', dir=self.root)
        try:
            check = hashlib.sha256(header)
            count = 0
            with os.fdopen(fd, 'wb') as output:
                output.write(header)
                flushed = 0
                for block in blocks():
                    output.write(block)
                    check.update(block)
                    count += len(block)
                    if count - flushed >= 256 << 20:
                        # Spark's CPU page cache shares the DRAM holding the prepared model.
                        output.flush()
                        os.fdatasync(output.fileno())
                        os.posix_fadvise(output.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                        flushed = count
                if count != size or check.digest() != digest.digest():
                    raise ValueError('payload changed during export')
                output.flush()
                os.fsync(output.fileno())
                os.fchmod(output.fileno(), 0o444)
            try:
                os.link(temporary, target)
            except FileExistsError:
                self.verify(target, digest.digest(), len(header) + size)
            from .package_file import _sync_directory
            _sync_directory(self.root)
            self.written += size
        finally:
            Path(temporary).unlink(missing_ok=True)
        return name

    @staticmethod
    def verify(path, expected, size):
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size != size:
                raise ValueError('invalid cached blob')
            digest = hashlib.file_digest(source, 'sha256').digest()
        if digest != expected:
            raise ValueError('cached blob content does not match its hash')

    def link(self, name, directory):
        target = Path(directory) / name
        try:
            os.link(self.root / name, target)
        except FileExistsError:
            source = self.root / name
            self.verify(target, bytes.fromhex(name[:-12]), source.stat().st_size)
        except OSError as error:
            if error.errno != 18:
                raise
            raise ValueError('package snapshot and blob cache must be on the same filesystem') from error


def write_primary(path, envelope, layout):
    """Commit metadata after all immutable payload links are in place."""
    from .package_file import _sync_directory
    validate_layout(layout)
    encoded = json.dumps(layout, separators=(',', ':'), sort_keys=True).encode()
    if not 0 < len(encoded) <= MAX_INDEX_BYTES or not 0 < len(envelope) <= MAX_INDEX_BYTES:
        raise ValueError('snapshot metadata exceeds its limit')
    header = {'__metadata__': {'format': 'flare-snapshot'},
              'flare.payload': {'dtype': 'U8', 'shape': [0], 'data_offsets': [0, 0]},
              'flare.metadata': {'dtype': 'U8', 'shape': [len(envelope)], 'data_offsets': [0, len(envelope)]},
              'flare.layout': {'dtype': 'U8', 'shape': [len(encoded)],
                               'data_offsets': [len(envelope), len(envelope) + len(encoded)]}}
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw = raw.ljust(align(len(raw) + 8) - 8, b' ')
    path = Path(path)
    _sync_directory(path.parent)
    fd, temporary = tempfile.mkstemp(prefix='.snapshot-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(struct.pack('<Q', len(raw)) + raw + envelope + encoded)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
        _sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def publish_snapshot(path, index, protection, *, shard_size=DEFAULT_SHARD_SIZE):
    """Publish directly from staged resource ranges without a complete package copy."""
    from .package_file import _open_source, _absorb_zeros
    from .protection import resource_roles
    if not ALIGNMENT < shard_size <= DEFAULT_SHARD_SIZE:
        raise ValueError('payload shard limit must exceed 4096 bytes and be at most 64G')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    options = _OPTIONS.get() or ExportOptions(path.parent / '.flare-cache')
    store = BlobStore(options.cache)
    layout = {'version': 1, 'files': {}}
    for filename, entries in protection.payloads.items():
        parts, cursor = [], 0
        roles = [(kind, owner, span) for kind, owner, span in resource_roles(index)
                     if span.filename == filename]
        roles.sort(key=lambda item: item[2].offset)
        positions = {owner: {p.offset: p.offset for p in ranges if p.filename == filename}
                     for owner, ranges in protection.ranges.items()}
        for kind, owner, span in roles:
            if span.offset > cursor:
                parts.append([span.offset - cursor, ''])
            selected = [entry for entry in entries if span.offset <= entry[0] < span.offset + span.nbytes]
            if span.nbytes > shard_size - ALIGNMENT:
                raise ValueError('package resource exceeds the payload shard limit')
            observed = False
            part = next((p for p in protection.ranges.get(owner, []) if p.filename == filename
                         and p.offset <= span.offset < p.offset + p.nbytes), None) if kind == 4 else None
            if part is not None:
                _absorb_zeros(protection.hashes[owner], span.offset - positions[owner][part.offset])
            def blocks():
                nonlocal observed
                first = not observed
                observed = True
                for _, source, offset, size in selected:
                    with _open_source(source) as contents:
                        for pos in range(0, size, BLOCK):
                            count = min(BLOCK, size - pos)
                            block = read_at(contents, count, offset + pos)
                            if len(block) != count:
                                raise ValueError('truncated export source')
                            if first and part is not None:
                                protection.hashes[owner].update(block)
                                verifier = protection.source_verifiers.get(owner)
                                if verifier is not None:
                                    verifier.consume(contents, offset + pos, block)
                            yield block
            name = store.put(span.nbytes, blocks, required=options.no_weights and kind == 4)
            store.link(name, path.parent)
            parts.append([span.nbytes, name])
            cursor = span.offset + span.nbytes
            if part is not None:
                positions[owner][part.offset] = cursor
        layout['files'][filename] = parts
        for owner, ranges in protection.ranges.items():
            for part in (p for p in ranges if p.filename == filename):
                _absorb_zeros(protection.hashes[owner], part.offset + part.nbytes - positions[owner][part.offset])
    write_primary(path, protection.envelope(final=True), layout)
    return store


def convert(source, destination, *, profiles, cache=None, shard_size=DEFAULT_SHARD_SIZE):
    """Verify and relocate a signed package without changing its signed envelope."""
    from . import runtime_package_pb2 as proto
    from .package_file import PackageFiles, stored_parts
    from .protection import CHUNK, resource_roles
    from .resign import _authenticate
    if not ALIGNMENT < shard_size <= DEFAULT_SHARD_SIZE:
        raise ValueError('payload shard limit must exceed 4096 bytes and be at most 64G')
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    store = BlobStore(cache or destination.parent / '.flare-cache')
    with Path(source).open('rb') as primary:
        header, base, encoded, existing = read_container(primary)
        envelope = proto.ProtectedIndex.FromString(encoded)
        profile = next((p for p in profiles if hashlib.sha256(bytes(p['public_key'])).digest() == envelope.signer), None)
        if profile is None:
            raise ValueError('package publisher is unknown; supply its public profile with --publisher-keys')
        index, _, _, _, resources, stored, _ = _authenticate(primary, profile, development_only=False)
        length = header['flare.payload']['shape'][0]
        with PackageFiles(source, primary, base, length) as files:
            for data in stored.values():
                digest = hashlib.sha256()
                for part in stored_parts(data):
                    contents, start, size = files.get(part.filename)
                    if part.offset + part.nbytes > size:
                        raise ValueError('signed stored range exceeds payload')
                    for pos in range(0, part.nbytes, BLOCK):
                        block = read_at(contents, min(BLOCK, part.nbytes - pos), start + part.offset + pos)
                        digest.update(block)
                if digest.digest() != data.sha256:
                    raise ValueError('stored-data digest mismatch')
            layout = {'version': 1, 'files': {'': []}}
            cursors = {}
            for kind, owner, span in sorted(resource_roles(index), key=lambda r: (r[2].filename, r[2].offset)):
                contents, start, size = files.get(span.filename)
                if span.offset + span.nbytes > size or span.nbytes > shard_size - ALIGNMENT:
                    raise ValueError('package resource exceeds its payload or the shard limit')
                if kind != 4:
                    resource = resources[kind, owner, span.offset]
                    for i, expected in enumerate(resource.sha256):
                        pos = i * (CHUNK + 16) if kind == 2 else 0
                        count = min(CHUNK + 16, span.nbytes - pos) if kind == 2 else span.nbytes
                        digest = hashlib.sha256()
                        for offset in range(0, count, BLOCK):
                            digest.update(read_at(contents, min(BLOCK, count - offset), start + span.offset + pos + offset))
                        if digest.digest() != expected:
                            raise ValueError('publisher resource content verification failed')
                parts = layout['files'].setdefault(span.filename, [])
                cursor = cursors.get(span.filename, 0)
                if cursor < span.offset:
                    gap = span.offset - cursor
                    for offset in range(0, gap, BLOCK):
                        if any(read_at(contents, min(BLOCK, gap - offset), start + cursor + offset)):
                            raise ValueError('package padding is not zero')
                    parts.append([gap, ''])
                def blocks():
                    for offset in range(0, span.nbytes, BLOCK):
                        yield read_at(contents, min(BLOCK, span.nbytes - offset), start + span.offset + offset)
                name = store.put(span.nbytes, blocks)
                store.link(name, destination.parent)
                parts.append([span.nbytes, name])
                cursors[span.filename] = span.offset + span.nbytes
            for name, cursor in cursors.items():
                if cursor != files.get(name)[2]:
                    raise ValueError('unreferenced source payload bytes')
            # Verify the relocated bytes against the original signed hashes before publishing.
            # A source file can change between the initial verification and the blob read.
            from contextlib import ExitStack
            with ExitStack() as handles:
                physical_files = {}

                def physical(name):
                    if name not in physical_files:
                        fd = os.open(store.root / name, os.O_RDONLY | os.O_NOFOLLOW)
                        file = handles.enter_context(os.fdopen(fd, 'rb'))
                        physical_files[name] = (file, ALIGNMENT, os.fstat(fd).st_size - ALIGNMENT)
                    return physical_files[name]

                logical = {name: LogicalFile(parts, physical) for name, parts in layout['files'].items()}
                for data in stored.values():
                    digest = hashlib.sha256()
                    for part in stored_parts(data):
                        for offset in range(0, part.nbytes, BLOCK):
                            digest.update(logical[part.filename].read_at(
                                min(BLOCK, part.nbytes - offset), part.offset + offset))
                    if digest.digest() != data.sha256:
                        raise ValueError('relocated stored-data digest mismatch')
                for kind, owner, span in resource_roles(index):
                    if kind == 4:
                        continue
                    for i, expected in enumerate(resources[kind, owner, span.offset].sha256):
                        pos = i * (CHUNK + 16) if kind == 2 else 0
                        count = min(CHUNK + 16, span.nbytes - pos) if kind == 2 else span.nbytes
                        digest = hashlib.sha256()
                        for offset in range(0, count, BLOCK):
                            digest.update(logical[span.filename].read_at(
                                min(BLOCK, count - offset), span.offset + pos + offset))
                        if digest.digest() != expected:
                            raise ValueError('relocated publisher resource verification failed')
            write_primary(destination, encoded, layout)
    return store
