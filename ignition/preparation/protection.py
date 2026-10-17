"""Publisher authentication and consent-bound executable resource encryption."""

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import struct

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from . import runtime_package_pb2 as proto

CHUNK = 4 << 20
VERSION = 4


def fields(*values):
    return b''.join(struct.pack('<Q', len(value)) + value for value in values)


def signature_message(package_id, signer, kind, body):
    return fields(b'flare-package-signature-v4', struct.pack('<I', VERSION),
                  package_id, signer, struct.pack('<I', kind), body)


def resource_key(public_key, package_id, offset, namespace):
    # Security-critical executable access control: preserve the signing-key,
    # publisher, package, and resource binding used by Helios after acceptance.
    # Unauthorized circumvention may violate 17 U.S.C. §1201(a)(1), as defined in
    # §1201(a)(3). Qualifying willful violations for commercial advantage or private
    # financial gain may carry criminal penalties under §1204, subject to applicable
    # statutory limitations and exceptions.
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=package_id,
                info=fields(b'flare-executable-v3', struct.pack('<Q', offset), namespace.encode())).derive(public_key)


def chunk_aad(package_id, offset, size, chunk):
    return fields(b'flare-executable-chunk-v3', package_id,
                  struct.pack('<Q', offset), struct.pack('<Q', size), struct.pack('<Q', chunk))


def validate_namespace(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', value):
        raise ValueError('publisher namespace must be a Hugging Face owner name')


def validate_source(namespace, source):
    validate_namespace(namespace)
    if not re.fullmatch(r'https://huggingface\.co/' + re.escape(namespace) + r'/[A-Za-z0-9][A-Za-z0-9._-]{0,95}', source):
        raise ValueError('official_source must be https://huggingface.co/<publisher>/<repository>')


@dataclass(frozen=True)
class Publisher:
    key: Ed25519PrivateKey
    namespace: str
    official_source: str
    development: bool = False

    def __post_init__(self):
        if not isinstance(self.key, Ed25519PrivateKey):
            raise ValueError('publishing requires an Ed25519 private key')
        if self.development:
            if self.namespace != 'local-development' or self.official_source != 'local-development':
                raise ValueError('invalid local development publisher')
        else:
            validate_source(self.namespace, self.official_source)

    @classmethod
    def load(cls, key_path, namespace, official_source):
        from ignition.signing import load_key

        key = load_key(key_path)
        return cls(key, namespace, official_source)

    @property
    def public_key(self):
        return self.key.public_key().public_bytes_raw()

    @property
    def fingerprint(self):
        return hashlib.sha256(self.public_key).digest()


def resource_roles(index):
    """Classify every payload span from its actual consumer, rejecting ambiguity.

    Returns (kind, owner, span) for each span: owner is its rank, or None for what every rank shares.
    """
    resources = []
    for owner, rank in enumerate(index.ranks):
        roles = {}
        for kind, names in (
            (2, [m.file for m in rank.modules] + [g.resource for g in rank.graphs]),
            (3, [g.file for m in rank.modules for g in m.globals]),
            (4, [a.file for a in rank.allocations if a.HasField('file')]),
        ):
            for name in names:
                if name not in rank.resources or (name in roles and roles[name] != kind):
                    raise ValueError('missing or conflicting resource classification')
                roles[name] = kind
        if set(roles) != set(rank.resources):
            raise ValueError('unclassified package resource')
        resources.extend((roles[name], owner, span) for name, span in rank.resources.items())
    resources.extend((3, None, asset.payload) for asset in index.assets)
    resources.extend((4, None, span) for table in index.engram.tables for span in (table.weights, table.scales))
    ranges = sorted((s.offset, s.nbytes) for _, _, s in resources)
    end = 0
    for offset, size in ranges:
        if size <= 0 or offset < end:
            raise ValueError('empty or overlapping package resources')
        end = offset + size
    return resources


def _read_at(contents, position, length, error):
    block = os.pread(contents.fileno(), length, position)
    if len(block) != length:
        raise ValueError(error)
    return block


def _encrypt(cipher, package_id, offset, source, source_offset, size, encrypted):
    """Encrypt a resource chunk by chunk into `encrypted`; returns each ciphertext chunk's SHA-256.

    Sources are read by position, so jobs sharing an open source (a package being re-signed) do not race.
    """
    from .package_file import _open_source

    digests = []
    with _open_source(source) as contents, encrypted.open('xb') as output:
        for counter, start in enumerate(range(0, size, CHUNK)):
            block = _read_at(contents, source_offset + start, min(CHUNK, size - start), 'truncated executable resource')
            ciphertext = cipher.encrypt(counter.to_bytes(12, 'big'), block, chunk_aad(package_id, offset, size, counter))
            digests.append(hashlib.sha256(ciphertext).digest())
            output.write(ciphertext)
    return digests


def _digest(source, source_offset, size):
    from .package_file import _open_source

    digest = hashlib.sha256()
    with _open_source(source) as contents:
        for start in range(0, size, CHUNK):
            digest.update(_read_at(contents, source_offset + start, min(CHUNK, size - start),
                                   'truncated supporting resource'))
    return [digest.digest()]


class Protection:
    """Transform a final rank assembly and build its signed envelope.

    What every rank shares (the index without its bundles, assets and tables) is signed once, and
    each rank's bundle, executables, supporting resources and stored data separately, so a host
    holding one rank's bytes verifies that rank completely.
    """

    def __init__(self, index, payload, publisher, stage, *, expected_stored=None):
        from concurrent.futures import ThreadPoolExecutor
        from .package_file import COPY_THREADS, align
        self.publisher = publisher
        self.expected_stored = expected_stored
        self.package_id = os.urandom(32)
        self.index = index
        # Parts in stored-data order: what every rank shares, then each rank.
        parts = [None, *range(len(index.ranks))]
        self.code = {owner: proto.ResourceCatalog() for owner in parts[1:]}
        self.support = {owner: proto.ResourceCatalog() for owner in parts}
        roles = resource_roles(index)
        by_offset = {span.offset: (kind, owner, span) for kind, owner, span in roles}
        # Stored data may arrive as adjacent ranges of several files, such as a table read from
        # several checkpoint shards; every other resource is one range.
        starts = sorted(by_offset)
        ranges = {offset: [] for offset in starts}
        for entry in sorted(payload, key=lambda entry: entry[0]):
            ranges[starts[max(bisect_right(starts, entry[0]) - 1, 0)]].append(entry)
        for offset, entries in ranges.items():
            kind, _, span = by_offset[offset]
            ends = [destination + size for destination, _, _, size in entries]
            if [entry[0] for entry in entries] != [offset, *ends[:-1]] or kind != 4 and len(entries) != 1:
                raise ValueError('publication ranges differ from classified resources')
            if ends[-1] - offset != span.nbytes:
                raise ValueError('resource length differs from publication range')

        def order(offset):
            kind, owner, _ = by_offset[offset]
            return (kind != 4, parts.index(owner) if kind == 4 else 0, offset)

        self.payload = []
        cursor = 0
        data_end = 0
        firsts = {}
        # Executables encrypt and supporting resources hash on several threads once laid out.
        jobs = []
        # Stored data precedes execution and frontend resources, each part's contiguous.
        for offset in sorted(ranges, key=order):
            kind, owner, span = by_offset[offset]
            entries = ranges[offset]
            cursor = align(cursor)
            span.offset = cursor
            if kind == 4:
                firsts.setdefault(owner, cursor)
            else:
                ((_, source, source_offset, size),) = entries
            if kind == 2:
                encrypted = Path(stage) / f'executable-{cursor}'
                resource = self.code[owner].resources.add(offset=cursor, plain_bytes=size)
                cipher = AESGCM(resource_key(publisher.public_key, self.package_id, cursor, publisher.namespace))
                jobs.append((resource, encrypted, _encrypt, (cipher, self.package_id, cursor, source, source_offset,
                                                             size, encrypted)))
                # Each chunk's ciphertext carries a 16-byte tag.
                size += 16 * -(-size // CHUNK)
                resource.stored_bytes = size
                entries = [(offset, encrypted, 0, size)]
                span.nbytes = size
            elif kind == 3:
                resource = self.support[owner].resources.add(offset=cursor, plain_bytes=size, stored_bytes=size)
                jobs.append((resource, None, _digest, (source, source_offset, size)))
            for _, source, source_offset, size in entries:
                self.payload.append((cursor, source, source_offset, size))
                cursor += size
            if kind == 4:
                data_end = cursor
        if jobs:
            with ThreadPoolExecutor(COPY_THREADS) as pool:
                futures = [(resource, encrypted, pool.submit(work, *arguments))
                           for resource, encrypted, work, arguments in jobs]
                for resource, encrypted, future in futures:
                    resource.sha256.extend(future.result())
                    if encrypted is not None and encrypted.stat().st_size != resource.stored_bytes:
                        raise ValueError('encrypted executable size differs from its layout')
        # Include alignment padding up to the first non-data resource.
        self.data_end = min((s.offset for k, _, s in roles if k != 4), default=data_end)
        # Each part's stored data runs from its first byte to the next part's, or the end of stored data.
        self.ranges = dict.fromkeys(parts, (0, 0))
        starts = sorted((start, owner) for owner, start in firsts.items())
        for (start, owner), (end, _) in zip(starts, starts[1:] + [(self.data_end, None)]):
            self.ranges[owner] = (start, end - start)
        self.hashes = {owner: hashlib.sha256() for owner in parts}
        if expected_stored is not None and (set(expected_stored) != set(self.ranges)
                or any(self.ranges[owner] != (data.offset, data.nbytes) for owner, data in expected_stored.items())):
            raise ValueError('re-signing must preserve the signed stored-data ranges')

    def envelope(self, final=False):
        """The signed envelope; before every payload byte is absorbed, one of the same size."""
        if final and self.expected_stored is not None:
            for owner, expected in self.expected_stored.items():
                if self.hashes[owner].digest() != expected.sha256:
                    raise ValueError(f'development package stored-data digest mismatch for rank {owner}')
        def stored(owner):
            offset, nbytes = self.ranges[owner]
            digest = self.hashes[owner].digest() if final else bytes(32)
            return proto.StoredData(offset=offset, nbytes=nbytes, sha256=digest)

        result = proto.ProtectedIndex(version=VERSION, package_id=self.package_id, signer=self.publisher.fingerprint)

        def sign(group, kind, body):
            signature = self.publisher.key.sign(signature_message(self.package_id, result.signer, kind, body))
            group.add(kind=kind, body=body, signature=signature)

        shared = proto.ModelIndex()
        shared.CopyFrom(self.index)
        del shared.ranks[:]
        bodies = [self.support[None].SerializeToString(deterministic=True),
                  stored(None).SerializeToString(deterministic=True)]
        metadata = proto.PublisherMetadata(index=shared, namespace=self.publisher.namespace,
            official_source=self.publisher.official_source, development=self.publisher.development,
            group_hashes=[hashlib.sha256(body).digest() for body in bodies], ranks=len(self.index.ranks))
        bodies.insert(0, metadata.SerializeToString(deterministic=True))
        for kind, body in enumerate(bodies, 1):
            sign(result.groups, kind, body)
        for rank, bundle in enumerate(self.index.ranks):
            sign(result.ranks, 4, proto.RankProtection(rank=rank, bundle=bundle, executables=self.code[rank],
                support=self.support[rank], data=stored(rank)).SerializeToString(deterministic=True))
        return result.SerializeToString(deterministic=True)
