"""Per-user signing credentials and runtime verification profiles."""

import json
import os
from pathlib import Path
import tempfile

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ignition.preparation.protection import Publisher


def publisher(configuration):
    directory = configuration.path.parent
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / 'development-signing-key.pem'
    if not path.exists():
        key = Ed25519PrivateKey.generate()
        encoded = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                    serialization.NoEncryption())
        descriptor, temporary = tempfile.mkstemp(prefix='.development-key-', dir=directory)
        try:
            with os.fdopen(descriptor, 'wb') as output:
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            Path(temporary).unlink(missing_ok=True)
    if path.stat().st_mode & 0o077:
        raise ValueError(f'development signing key must have owner-only permissions: {path}')
    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    signer = Publisher(key, 'local-development', 'local-development', development=True)
    profile(configuration, signer)
    return signer


def profile(configuration, signer):
    """Publish the local verification key separately from execution trust."""
    path = configuration.path.parent / 'development-publisher.json'
    encoded = json.dumps([{
        'name': 'Local development signer', 'namespace': signer.namespace,
        'public_key': list(signer.public_key), 'development': True,
    }], indent=2) + '\n'
    descriptor, temporary = tempfile.mkstemp(prefix='.development-profile-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path
