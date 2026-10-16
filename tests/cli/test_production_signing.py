"""Production keys stay encrypted on disk and are unlocked through a terminal."""

import json
import os

from cryptography.hazmat.primitives import serialization
import pytest

from ignition import signing
from ignition.preparation.protection import Publisher


def test_encrypted_key_and_public_profile(tmp_path, monkeypatch):
    directory = tmp_path / 'keys'
    password = b'fixture-passphrase-not-a-production-secret'
    fingerprint = signing.generate(directory, 'Example publisher', 'example', password)
    path = directory / 'signing-key.pem'
    encoded = path.read_bytes()
    assert encoded.startswith(b'-----BEGIN ENCRYPTED PRIVATE KEY-----')
    with pytest.raises(TypeError):
        serialization.load_pem_private_key(encoded, None)
    with pytest.raises(ValueError):
        serialization.load_pem_private_key(encoded, b'incorrect')
    monkeypatch.setattr(signing, 'passphrase', lambda prompt: password)
    publisher = Publisher.load(path, 'example', 'https://huggingface.co/example/model')
    assert publisher.fingerprint.hex() == fingerprint
    profile = json.loads((directory / 'publisher.json').read_text())[0]
    assert bytes(profile['public_key']) == publisher.public_key
    assert not profile['development']
    publisher.key.public_key().verify(publisher.key.sign(b'check'), b'check')
    with pytest.raises(FileExistsError):
        signing.generate(directory, 'Example publisher', 'example', password)
    assert path.read_bytes() == encoded
    if os.name != 'nt':
        assert directory.stat().st_mode & 0o777 == 0o700
        assert path.stat().st_mode & 0o777 == 0o600
        path.chmod(0o644)
        with pytest.raises(ValueError, match='owner-only'):
            signing.load_key(path)


def test_no_passphrase_fallback_to_piped_input(monkeypatch):
    monkeypatch.setattr(signing.sys.stdin, 'isatty', lambda: False)
    monkeypatch.setattr(signing.getpass, 'getpass', lambda prompt: pytest.fail('must not prompt'))
    with pytest.raises(ValueError, match='interactive terminal'):
        signing.passphrase('secret')


def test_invalid_generation_leaves_no_directory(tmp_path):
    directory = tmp_path / 'keys'
    with pytest.raises(ValueError, match='at least 16'):
        signing.generate(directory, 'Example publisher', 'example', b'short')
    assert not directory.exists()


def test_terminal_echo_failure_cannot_expose_passphrase(monkeypatch):
    import warnings

    monkeypatch.setattr(signing.sys.stdin, 'isatty', lambda: True)

    def no_echo(prompt):
        warnings.warn('echo unavailable', signing.getpass.GetPassWarning)
        pytest.fail('must not read an echoing passphrase')

    monkeypatch.setattr(signing.getpass, 'getpass', no_echo)
    with pytest.raises(ValueError, match='cannot disable terminal echo'):
        signing.passphrase('secret')
