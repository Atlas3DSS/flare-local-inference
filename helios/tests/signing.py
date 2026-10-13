"""Public test credentials shared by Python exporters and native fixture readers."""

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ignition.preparation.protection import Publisher


def publisher():
    return Publisher(Ed25519PrivateKey.from_private_bytes(bytes([42]) * 32),
                     'flare-tests', 'https://huggingface.co/flare-tests/fixture')
