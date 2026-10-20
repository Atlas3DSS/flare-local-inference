"""Publisher key creation and promotion of prepared development packages."""

from ignition.errors import UsageError


def validate_repository(repository):
    """Accept a portable repository namespace or local recipe name."""
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?', repository) or '..' in repository:
        raise UsageError('export repository must be a portable namespace/repository identifier')


def snapshot_destination(configuration, repository):
    """Choose an unused local snapshot under a portable repository identifier."""
    import uuid
    validate_repository(repository)
    return (configuration.directory('package_cache') / ('models--' + repository.replace('/', '--'))
            / 'snapshots' / uuid.uuid4().hex / 'model.safetensors')


def convert(args):
    import json
    from pathlib import Path
    from google.protobuf.message import DecodeError
    from ignition.cli.config import Configuration
    from ignition.preparation.package_snapshot import convert as convert_package

    configuration = Configuration.load(args.config)
    output = args.output or snapshot_destination(configuration, args.package.stem)
    try:
        embedded = Path(__file__).parents[2] / 'helios/src/runtime_package/publishers.json'
        profiles = json.loads(embedded.read_text()) if embedded.is_file() else []
        paths = list(args.publisher_keys)
        development = configuration.path.parent / 'development-publisher.json'
        if development.is_file():
            paths.append(development)
        for path in paths:
            profiles.extend(json.loads(path.expanduser().read_text()))
        store = convert_package(args.package, output, profiles=profiles,
                                cache=configuration.directory('package_cache'),
                                shard_size=configuration.shard_size(args.shard_size))
    except (ValueError, OSError, KeyError, TypeError, DecodeError) as error:
        raise UsageError(str(error)) from error
    print(f'Converted package: {output}')
    print(f'Payload bytes written: {store.written}; reused: {store.reused}')


def keygen(args):
    from ignition.signing import generate, passphrase

    try:
        password = passphrase('New signing-key passphrase: ')
        if password != passphrase('Confirm signing-key passphrase: '):
            raise ValueError('passphrases do not match')
        fingerprint = generate(args.directory, args.name, args.namespace, password)
    except (ValueError, OSError, EOFError) as error:
        raise UsageError(str(error)) from error
    directory = args.directory.expanduser()
    print(f'Encrypted key: {directory / "signing-key.pem"}')
    print(f'Public profile: {directory / "publisher.json"}')
    print(f'SHA-256 fingerprint: {fingerprint}')


def key_password(args):
    from ignition.signing import change_password

    try:
        fingerprint = change_password(args.signing_key)
    except (ValueError, OSError, EOFError) as error:
        raise UsageError(str(error)) from error
    print(f'Updated encrypted key: {args.signing_key.expanduser()}')
    print(f'SHA-256 fingerprint: {fingerprint}')


def sign(args):
    import json
    from google.protobuf.message import DecodeError
    from ignition.cli.config import Configuration
    from ignition.preparation.protection import Publisher
    from ignition.preparation.resign import resign as promote
    from ignition.progress import StartupProgress

    configuration = Configuration.load(args.config)
    values = configuration.values.get('publishing', {})
    try:
        if args.output.exists():
            raise FileExistsError(args.output)
        key = args.signing_key or values.get('signing_key')
        namespace = args.namespace or values.get('namespace')
        source = args.official_source or values.get('official_source')
        if not all((key, namespace, source)):
            raise ValueError('signing requires signing_key, namespace, and official_source')
        publisher = Publisher.load(key, namespace, source)
        profile = args.development_publisher or configuration.path.parent / 'development-publisher.json'
        profiles = json.loads(profile.expanduser().read_text())
        if not isinstance(profiles, list) or len(profiles) != 1:
            raise ValueError('development profile must contain exactly one publisher')
        from ignition.preparation.package_snapshot import export_options
        with export_options(configuration.directory('package_cache')):
            promote(args.package, args.output, publisher=publisher, development_profile=profiles[0],
                    progress=StartupProgress(), shard_size=configuration.shard_size(getattr(args, "shard_size", None)))
    except (ValueError, OSError, EOFError, DecodeError) as error:
        raise UsageError(str(error)) from error
    print(f'Signed package: {args.output}')
    print(f'Publisher: {namespace}; SHA-256 fingerprint: {publisher.fingerprint.hex()}')
