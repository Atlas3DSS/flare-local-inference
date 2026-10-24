"""Promote authenticated development packages without preparing model computation."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ignition.progress import phase
from . import runtime_package_pb2 as proto
from .package_file import (DEFAULT_SHARD_SIZE, MAX_INDEX_BYTES, PackageFiles, _publish,
                           _validate_spans, align, read_index, stored_parts)
from .protection import (CHUNK, License, Protection, Qualification, chunk_aad, resource_key, resource_roles,
                         signature_message)


def _authenticate(source, profile, *, development_only=True):
    from .package_snapshot import read_container
    index, base = read_index(source)
    header, base, encoded, layout = read_container(source)
    if header['__metadata__']['format'] not in ('flare', 'flare-snapshot'):
        raise ValueError('re-signing requires a signed development package')
    payload_size = (sum(n for n, _ in layout['files']['']) if layout is not None
                    else header['flare.payload']['shape'][0])
    size = header['flare.metadata']['shape'][0]
    if not 0 < size <= MAX_INDEX_BYTES:
        raise ValueError('invalid protected package envelope size')
    envelope = proto.ProtectedIndex.FromString(encoded)
    if envelope.version != 4 or len(envelope.groups) != 3 or not envelope.ranks:
        raise ValueError('invalid protected package envelope')
    if (not isinstance(profile, dict)
            or (development_only and (profile.get('namespace') != 'local-development' or profile.get('development') is not True))
            or not isinstance(profile.get('public_key'), list)
            or any(type(v) is not int or not 0 <= v <= 255 for v in profile['public_key'])):
        raise ValueError('source profile must identify a local-development Ed25519 key')
    public = bytes(profile['public_key'])
    if (len(public) != 32 or len(envelope.package_id) != 32
            or envelope.signer != hashlib.sha256(public).digest()):
        raise ValueError('development publisher fingerprint does not match the package')
    verifier = Ed25519PublicKey.from_public_bytes(public)
    for kind, group in [*enumerate(envelope.groups, 1), *((4, group) for group in envelope.ranks)]:
        if group.kind != kind:
            raise ValueError('missing or unordered signature group')
        try:
            verifier.verify(group.signature, signature_message(envelope.package_id, envelope.signer, kind, group.body))
        except InvalidSignature as error:
            raise ValueError('development package signature verification failed') from error
    metadata = proto.PublisherMetadata.FromString(envelope.groups[0].body)
    if development_only and (not metadata.development or metadata.namespace != 'local-development'
            or metadata.official_source != 'local-development'):
        raise ValueError('only local-development packages can be promoted')
    if metadata.namespace != profile.get('namespace') or metadata.development != profile.get('development', False):
        raise ValueError('publisher profile does not match signed package identity')
    if (list(metadata.group_hashes) != [hashlib.sha256(g.body).digest() for g in envelope.groups[1:]]
            or metadata.ranks != len(envelope.ranks) or not metadata.HasField('index') or metadata.index.ranks
            or (metadata.index.HasField('cluster') and metadata.index.cluster.world != metadata.ranks)):
        raise ValueError('signed package group commitments or rank count differ')
    ranks = [proto.RankProtection.FromString(group.body) for group in envelope.ranks]
    if any(rank.rank != i or not rank.HasField('data') or not rank.HasField('bundle')
           for i, rank in enumerate(ranks)):
        raise ValueError('signed rank order or stored-data description is invalid')
    index = metadata.index
    for rank in ranks:
        index.ranks.add().CopyFrom(rank.bundle)
    _validate_spans(index, payload_size)
    stored = {None: proto.StoredData.FromString(envelope.groups[2].body),
              **{i: rank.data for i, rank in enumerate(ranks)}}
    catalogs = [(3, None, proto.ResourceCatalog.FromString(envelope.groups[1].body))]
    catalogs.extend((kind, i, catalog) for i, rank in enumerate(ranks)
                    for kind, catalog in ((2, rank.executables), (3, rank.support)))
    resources = {}
    for kind, owner, catalog in catalogs:
        for resource in catalog.resources:
            key = kind, owner, resource.offset
            if key in resources or resource.stored_bytes <= 0 or resource.offset + resource.stored_bytes > payload_size:
                raise ValueError('duplicate or invalid signed resource')
            chunks = (resource.plain_bytes + CHUNK - 1) // CHUNK
            if (resource.plain_bytes <= 0 or any(len(digest) != 32 for digest in resource.sha256)
                    or len(resource.sha256) != (chunks if kind == 2 else 1)
                    or resource.stored_bytes != resource.plain_bytes + (16 * chunks if kind == 2 else 0)):
                raise ValueError('invalid signed resource geometry')
            resources[key] = resource
    roles = resource_roles(index)
    if set(resources) != {(kind, owner, span.offset) for kind, owner, span in roles if kind != 4}:
        raise ValueError('resource catalogs differ from package consumers')
    for kind, owner, span in roles:
        if kind != 4 and resources[kind, owner, span.offset].stored_bytes != span.nbytes:
            raise ValueError('resource catalog length differs from package span')
    ranges = []
    for owner, data in stored.items():
        parts = stored_parts(data)
        for part in parts:
            if not part.nbytes or (not part.filename and part.offset + part.nbytes > payload_size):
                raise ValueError('invalid signed stored-data range')
            ranges.append((part.filename, part.offset, part.offset + part.nbytes))
        for _, _, span in (r for r in roles if r[0] == 4 and r[1] == owner):
            if not any(span.filename == part.filename and span.offset >= part.offset
                       and span.offset + span.nbytes <= part.offset + part.nbytes for part in parts):
                raise ValueError('stored resource exceeds its signed range')
    ordered = sorted(ranges)
    if any(left[0] == right[0] and left[2] > right[1] for left, right in zip(ordered, ordered[1:])):
        raise ValueError('signed stored-data ranges overlap')
    for kind, _, span in roles:
        if kind != 4 and any(not name and span.offset < end and offset < span.offset + span.nbytes
                             for name, offset, end in ranges):
            raise ValueError('protected resource overlaps stored data')
    return index, base, envelope.package_id, public, resources, stored, (License(metadata.license.name, metadata.license.link) if metadata.HasField('license') else None), (Qualification(metadata.qualification.reference_gates, metadata.qualification.reason) if metadata.HasField('qualification') else None)


class _SourceVerifier:
    """Hash source padding and the exact resource bytes copied into the destination."""

    def __init__(self, data, files):
        self.expected = data.sha256
        self.parts = []
        for part in stored_parts(data):
            source, base, length = files.get(part.filename)
            if not part.nbytes or part.offset + part.nbytes > length:
                raise ValueError('invalid signed stored-data range')
            self.parts.append((source, base + part.offset, base + part.offset + part.nbytes))
        self.position = 0
        self.cursor = self.parts[0][1] if self.parts else 0
        self.digest = hashlib.sha256()

    def _padding(self, end):
        from .package_snapshot import read_at
        source = self.parts[self.position][0]
        while self.cursor < end:
            count = min(CHUNK, end - self.cursor)
            block = read_at(source, count, self.cursor)
            if len(block) != count or any(block):
                raise ValueError('stored-data alignment padding is not zero')
            self.digest.update(block)
            self.cursor += count

    def _advance(self):
        self._padding(self.parts[self.position][2])
        self.position += 1
        if self.position < len(self.parts):
            self.cursor = self.parts[self.position][1]

    def consume(self, source, offset, block):
        while self.position < len(self.parts):
            file, start, end = self.parts[self.position]
            if file is source and start <= offset < end:
                break
            self._advance()
        else:
            raise ValueError('copied resource is outside signed stored-data parts')
        if offset < self.cursor or offset + len(block) > end:
            raise ValueError('copied resource overlaps or straddles signed stored-data parts')
        self._padding(offset)
        self.digest.update(block)
        self.cursor = offset + len(block)

    def finish(self):
        while self.position < len(self.parts):
            self._advance()
        if self.digest.digest() != self.expected:
            raise ValueError('stored-data digest mismatch')


def resign(source, destination, *, publisher, development_profile, progress=None,
           shard_size=DEFAULT_SHARD_SIZE):
    """Verify copied bytes before atomically publishing their production signatures."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if publisher.development:
        raise ValueError('re-signing requires a production publisher')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open('rb') as contents, TemporaryDirectory(prefix='.flare-resign-', dir=destination.parent) as temporary:
        with phase(progress, 'Authenticate development package'):
            index, base, package_id, public, resources, stored, license, qualification = _authenticate(
                contents, development_profile)
        # The model's license travels with the model; the production publisher may override it.
        if publisher.license is None and license is not None:
            publisher = replace(publisher, license=license)
        # Skipped export gates travel with the package; re-signing cannot remove the record.
        if qualification is not None:
            publisher = replace(publisher, qualification=qualification)
        contents.seek(8)
        length = json.loads(contents.read(base - 8))['flare.payload']['shape'][0]
        with PackageFiles(source, contents, base, length) as files:
            _repack(destination, index, base, contents, package_id, public, resources, stored,
                    files, publisher, temporary, progress, shard_size)


def _repack(destination, index, base, contents, package_id, public, resources, stored,
            files, publisher, temporary, progress, shard_size):
    verifiers = {owner: _SourceVerifier(data, files) for owner, data in stored.items()}
    payload, cursor = [], 0
    parts = {owner: stored_parts(data) for owner, data in stored.items()}
    def order(role):
        kind, owner, span = role
        if kind != 4:
            return (1, 0, 0, span.offset)
        part = next(i for i, part in enumerate(parts[owner]) if part.filename == span.filename
                    and part.offset <= span.offset < part.offset + part.nbytes)
        return (0, -1 if owner is None else owner, part, span.offset)
    with phase(progress, 'Verify and decrypt executable resources'):
        for kind, owner, span in sorted(resource_roles(index), key=order):
            offset, size = span.offset, span.nbytes
            source_file, source_base, length = files.get(span.filename)
            if offset + size > length:
                raise ValueError('resource exceeds package payload')
            cursor = align(cursor)
            span.offset, span.filename = cursor, ''
            if kind == 4:
                payload.append((cursor, source_file, source_base + offset, size))
                cursor += size
                continue
            resource = resources[kind, owner, offset]
            plain = Path(temporary) / f'plain-{offset}'
            source_file.seek(source_base + offset)
            cipher = AESGCM(resource_key(public, package_id, offset, 'local-development')) if kind == 2 else None
            digest = hashlib.sha256()
            remaining, counter = resource.plain_bytes, 0
            with plain.open('xb') as output:
                while remaining:
                    count = min(CHUNK, remaining)
                    block = source_file.read(count + (16 if cipher else 0))
                    if len(block) != count + (16 if cipher else 0):
                        raise ValueError('truncated development package resource')
                    if cipher:
                        if hashlib.sha256(block).digest() != resource.sha256[counter]:
                            raise ValueError('development executable digest mismatch')
                        try:
                            block = cipher.decrypt(counter.to_bytes(12, 'big'), block,
                                chunk_aad(package_id, offset, resource.plain_bytes, counter))
                        except InvalidTag as error:
                            raise ValueError('development executable authentication failed') from error
                    else:
                        digest.update(block)
                    output.write(block)
                    counter += 1
                    remaining -= count
            if not cipher and digest.digest() != resource.sha256[0]:
                raise ValueError('development supporting-resource digest mismatch')
            span.nbytes = resource.plain_bytes
            payload.append((cursor, plain, 0, span.nbytes))
            cursor += span.nbytes
    with phase(progress, 'Re-encrypt executable resources'):
        protection = Protection(index, payload, publisher, temporary, source_verifiers=verifiers,
                                shard_size=shard_size, basename=destination.stem)
    with phase(progress, 'Verify stored data and sign package') as stage:
        from .package_snapshot import publish_snapshot
        publish_snapshot(destination, index, protection, shard_size=shard_size, progress=stage)
