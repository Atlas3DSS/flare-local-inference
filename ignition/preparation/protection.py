"""Publisher authentication and consent-bound executable resource encryption."""

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import struct
import unicodedata

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
class License:
    """The license governing use of the model: named in the acceptance prompt, linked for reading.

    Flare prints both before acceptance, so the name is bounded printable text and the link, when
    present, is an HTTPS URL of printable ASCII. Typing the publisher namespace agrees to the terms
    at the link.
    """

    name: str
    link: str = ''

    def __post_init__(self):
        if (not isinstance(self.name, str) or not 0 < len(self.name) <= 128
                or any(unicodedata.category(c) == 'Cc' for c in self.name)):
            raise ValueError('license name must be 1 to 128 printable characters')
        if not isinstance(self.link, str) or (self.link and not re.fullmatch(r'https://[\x21-\x7e]{1,504}', self.link)):
            raise ValueError('license link must be an https URL of printable ASCII, at most 512 characters')


@dataclass(frozen=True)
class Publisher:
    key: Ed25519PrivateKey
    namespace: str
    official_source: str
    development: bool = False
    license: License | None = None

    def __post_init__(self):
        if not isinstance(self.key, Ed25519PrivateKey):
            raise ValueError('publishing requires an Ed25519 private key')
        if self.license is not None and not isinstance(self.license, License):
            raise ValueError('publisher license must be a License')
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
    from .package_file import embedding_spans, validate_filename
    resources.extend((4, owner, span) for owner, span in embedding_spans(index))
    for kind, _, span in resources:
        if span.filename:
            validate_filename(span.filename)
            if index.version != 5 or kind != 4:
                raise ValueError('only stored data in model index 5 may reference a companion')
    ranges = sorted((s.filename, s.offset, s.nbytes) for _, _, s in resources)
    end, previous = 0, None
    for filename, offset, size in ranges:
        if filename != previous:
            end, previous = 0, filename
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

    def __init__(self, index, payload, publisher, stage, *, shard_size=0, basename='model',
                 source_verifiers=None):
        from concurrent.futures import ThreadPoolExecutor
        from .package_file import COPY_THREADS, align, container_header, validate_filename
        if type(shard_size) is not int or shard_size < 0:
            raise ValueError('shard size must be a nonnegative byte count')
        self.publisher = publisher
        self.package_id = os.urandom(32)
        self.index = index
        self.source_verifiers = source_verifiers or {}
        owners = [None, *range(len(index.ranks))]
        roles = resource_roles(index)
        if any(span.filename for _, _, span in roles):
            raise ValueError('publication inputs must use a contiguous staging address space')
        by_offset = {span.offset: (kind, owner, span) for kind, owner, span in roles}
        starts = sorted(by_offset)
        ranges = {offset: [] for offset in starts}
        for entry in sorted(payload, key=lambda entry: entry[0]):
            if not starts or entry[0] < starts[0]:
                raise ValueError('publication range has no classified resource')
            ranges[starts[bisect_right(starts, entry[0]) - 1]].append(entry)
        for offset, entries in ranges.items():
            kind, _, span = by_offset[offset]
            ends = [destination + size for destination, _, _, size in entries]
            if (not entries or [entry[0] for entry in entries] != [offset, *ends[:-1]]
                    or kind != 4 and len(entries) != 1 or ends[-1] - offset != span.nbytes):
                raise ValueError('publication ranges differ from classified resources')
        ordered = sorted(ranges, key=lambda offset: (by_offset[offset][0] != 4,
                         owners.index(by_offset[offset][1]) if by_offset[offset][0] == 4 else 0, offset))
        sizes = {offset: by_offset[offset][2].nbytes for offset in ordered}
        self.hashes = {owner: hashlib.sha256() for owner in owners}

        def layout(sharded):
            self.code = {owner: proto.ResourceCatalog() for owner in owners[1:]}
            self.support = {owner: proto.ResourceCatalog() for owner in owners}
            self.ranges = {owner: [] for owner in owners}
            self.payloads = {'': []}
            positions = {'': 0}
            locations, catalog_entries = {}, {}
            data_file, data_files = '', []
            for offset in ordered:
                kind, owner, span = by_offset[offset]
                size = sizes[offset]
                stored = size + (16 * -(-size // CHUNK) if kind == 2 else 0)
                if sharded and kind == 4:
                    if not data_file or (positions[data_file] and align(positions[data_file]) + size + 4096 > shard_size):
                        data_file = str(len(data_files) + 1)
                        data_files.append(data_file)
                        if len(data_files) > 99:
                            raise ValueError('package requires more than 99 data files')
                        self.payloads[data_file], positions[data_file] = [], 0
                    filename = data_file
                else:
                    filename = ''
                cursor = align(positions[filename])
                span.offset, span.nbytes, span.filename = cursor, stored, filename
                positions[filename] = cursor + stored
                locations[offset] = (filename, cursor)
                if kind == 4:
                    parts = self.ranges[owner]
                    if parts and parts[-1].filename == filename:
                        parts[-1].nbytes = cursor + stored - parts[-1].offset
                    else:
                        parts.append(proto.Span(filename=filename, offset=cursor, nbytes=stored))
                else:
                    catalog = self.code[owner] if kind == 2 else self.support[owner]
                    catalog_entries[offset] = catalog.resources.add(
                        offset=cursor, plain_bytes=size, stored_bytes=stored,
                        sha256=[bytes(32)] * (-(-size // CHUNK) if kind == 2 else 1))
            # Assign alignment gaps to the preceding stored-data part, as in single-file packages.
            for filename in positions:
                parts = sorted((part for parts in self.ranges.values() for part in parts
                                if part.filename == filename), key=lambda part: part.offset)
                end = min((s.offset for k, _, s in roles if k != 4 and s.filename == filename),
                          default=positions[filename])
                for part, following in zip(parts, [p.offset for p in parts[1:]] + [end]):
                    part.nbytes = following - part.offset
            if sharded:
                names = {f: f'{basename}-data-{i:02}-of-{len(data_files):02}.safetensors'
                         for i, f in enumerate(data_files, 1)}
                for name in names.values():
                    validate_filename(name)
                for _, _, span in roles:
                    span.filename = names.get(span.filename, '')
                for parts in self.ranges.values():
                    for part in parts:
                        part.filename = names.get(part.filename, '')
                locations = {offset: (names.get(f, ''), pos) for offset, (f, pos) in locations.items()}
                self.payloads = {names.get(f, ''): [] for f in positions}
            index.version = 5 if sharded else 4
            return locations, positions[''], catalog_entries

        locations, length, catalog_entries = layout(False)
        encoded = self.envelope()
        total = len(container_header(length, len(encoded), 'flare')) + length + len(encoded)
        if shard_size and total > shard_size and any(kind == 4 for kind, _, _ in roles):
            locations, _, catalog_entries = layout(True)
        jobs = []
        for offset in ordered:
            kind, owner, span = by_offset[offset]
            filename, cursor = locations[offset]
            entries = ranges[offset]
            if kind != 4:
                ((_, source, source_offset, size),) = entries
                resource = catalog_entries[offset]
                if kind == 2:
                    encrypted = Path(stage) / f'executable-{cursor}'
                    cipher = AESGCM(resource_key(publisher.public_key, self.package_id, cursor, publisher.namespace))
                    jobs.append((resource, _encrypt, (cipher, self.package_id, cursor, source, source_offset,
                                                       size, encrypted)))
                    entries = [(offset, encrypted, 0, resource.stored_bytes)]
                else:
                    jobs.append((resource, _digest, (source, source_offset, size)))
            for _, source, source_offset, size in entries:
                self.payloads[filename].append((cursor, source, source_offset, size))
                cursor += size
        if jobs:
            with ThreadPoolExecutor(COPY_THREADS) as pool:
                futures = [(resource, pool.submit(work, *arguments)) for resource, work, arguments in jobs]
                for resource, future in futures:
                    del resource.sha256[:]
                    resource.sha256.extend(future.result())
        self.payload = self.payloads['']

    def envelope(self, final=False):
        """The signed envelope; before every payload byte is absorbed, one of the same size."""
        if final:
            for verifier in self.source_verifiers.values():
                verifier.finish()
        def stored(owner):
            parts = self.ranges[owner]
            digest = self.hashes[owner].digest() if final else bytes(32)
            if self.index.version == 4:
                part = parts[0] if parts else proto.Span()
                return proto.StoredData(offset=part.offset, nbytes=part.nbytes, sha256=digest)
            return proto.StoredData(nbytes=sum(p.nbytes for p in parts), sha256=digest, parts=parts)

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
        if self.publisher.license is not None:
            metadata.license.name = self.publisher.license.name
            metadata.license.link = self.publisher.license.link
        bodies.insert(0, metadata.SerializeToString(deterministic=True))
        for kind, body in enumerate(bodies, 1):
            sign(result.groups, kind, body)
        for rank, bundle in enumerate(self.index.ranks):
            sign(result.ranks, 4, proto.RankProtection(rank=rank, bundle=bundle, executables=self.code[rank],
                support=self.support[rank], data=stored(rank)).SerializeToString(deterministic=True))
        return result.SerializeToString(deterministic=True)
