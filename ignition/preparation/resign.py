"""Promote authenticated development packages without preparing model computation."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ignition.progress import phase
from . import runtime_package_pb2 as proto
from .package_file import MAX_INDEX_BYTES, _publish, _validate_spans, read_index
from .protection import CHUNK, License, Protection, chunk_aad, resource_key, resource_roles, signature_message


def _authenticate(source, profile):
    index, base = read_index(source)
    source.seek(8)
    header = json.loads(source.read(base - 8))
    if header['__metadata__']['format'] != 'flare':
        raise ValueError('re-signing requires a signed development package')
    payload_size = header['flare.payload']['shape'][0]
    size = header['flare.metadata']['shape'][0]
    if not 0 < size <= MAX_INDEX_BYTES:
        raise ValueError('invalid protected package envelope size')
    source.seek(base + payload_size)
    envelope = proto.ProtectedIndex.FromString(source.read(size))
    if envelope.version != 4 or len(envelope.groups) != 3 or not envelope.ranks:
        raise ValueError('invalid protected package envelope')
    if (not isinstance(profile, dict) or profile.get('namespace') != 'local-development' or profile.get('development') is not True
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
    if (not metadata.development or metadata.namespace != 'local-development'
            or metadata.official_source != 'local-development'):
        raise ValueError('only local-development packages can be promoted')
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
        if len(data.sha256) != 32 or data.offset + data.nbytes > payload_size:
            raise ValueError('invalid signed stored-data range')
        if data.nbytes:
            ranges.append((data.offset, data.offset + data.nbytes))
        cursor = data.offset
        for _, _, span in sorted((r for r in roles if r[0] == 4 and r[1] == owner), key=lambda r: r[2].offset):
            if span.offset < cursor or span.offset + span.nbytes > data.offset + data.nbytes:
                raise ValueError('stored resource exceeds its signed range')
            _zero_padding(source, base + cursor, span.offset - cursor)
            cursor = span.offset + span.nbytes
        _zero_padding(source, base + cursor, data.offset + data.nbytes - cursor)
    ordered = sorted(ranges)
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError('signed stored-data ranges overlap')
    return index, base, envelope.package_id, public, resources, stored, (License(metadata.license.name, metadata.license.link) if metadata.HasField('license') else None)


def _zero_padding(source, offset, size):
    source.seek(offset)
    while size:
        block = source.read(min(CHUNK, size))
        if len(block) != min(CHUNK, size) or any(block):
            raise ValueError('stored-data alignment padding is not zero')
        size -= len(block)


def resign(source, destination, *, publisher, development_profile, progress=None):
    """Verify copied bytes before atomically publishing their production signatures."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if publisher.development:
        raise ValueError('re-signing requires a production publisher')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open('rb') as contents, TemporaryDirectory(prefix='.flare-resign-', dir=destination.parent) as temporary:
        with phase(progress, 'Authenticate development package'):
            index, base, package_id, public, resources, stored, license = _authenticate(contents, development_profile)
        # The model's license travels with the model; the production publisher may override it.
        if publisher.license is None and license is not None:
            publisher = replace(publisher, license=license)
        payload = []
        with phase(progress, 'Verify and decrypt executable resources'):
            for kind, owner, span in sorted(resource_roles(index), key=lambda r: r[2].offset):
                offset, size = span.offset, span.nbytes
                if kind == 4:
                    payload.append((offset, contents, base + offset, size))
                    continue
                resource = resources[kind, owner, offset]
                plain = Path(temporary) / f'plain-{offset}'
                contents.seek(base + offset)
                cipher = AESGCM(resource_key(public, package_id, offset, 'local-development')) if kind == 2 else None
                digest = hashlib.sha256()
                remaining, counter = resource.plain_bytes, 0
                with plain.open('xb') as output:
                    while remaining:
                        count = min(CHUNK, remaining)
                        block = contents.read(count + (16 if cipher else 0))
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
                payload.append((offset, plain, 0, span.nbytes))
        with phase(progress, 'Re-encrypt executable resources'):
            protection = Protection(index, payload, publisher, temporary, expected_stored=stored)
        with phase(progress, 'Verify stored data and sign package'):
            _publish(destination, index, protection.payload, protection)
