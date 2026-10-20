# Production signing

Ignition signs runtime packages with Ed25519. Helios embeds the public identity
**Local Inference Lab, Inc.**, namespace `local-inference-lab`, from
[`publishers.json`](../helios/src/runtime_package/publishers.json). Its SHA-256
public-key fingerprint is:

```text
937c0143f6856d3ee8ad164f62e3d804bd825d7a8750d0accb9a5849f703a8c1
```

The embedded profile lets Flare verify signatures without consulting the network;
it does not verify identity. Every publisher, including this one, is verified the
same way: by the `publisher.json` at the root of the package's official source
repository. Execution still requires first-use acceptance. Private keys never
belong in source control, packages, or publisher profiles.

## Publish the verification key

Upload the `publisher.json` that key creation wrote, unchanged, to the root of
every repository named as an official source. Flare fetches it from the default
branch on every online load and requires it to list the key under the publisher
namespace. The file is a list, so a rotated key stays valid
by keeping the previous profile in it, and removing a profile revokes that key
for every package it signed. A repository whose `publisher.json` omits the
signing key makes the package fail to load everywhere; a repository without the
file leaves the identity unverified, and users must then supply the profile with
`--publisher-keys`.

Each repository should also hold the license file its model card names. The
exporter records the license name and a link to that file from the model card
front matter, Flare shows both before acceptance, and typing the namespace
agrees to the terms at the link.

## Create and protect a key

Run key creation in your own interactive terminal:

```sh
.venv/bin/ignition keygen \
  --directory ~/.config/flare/signing/local-inference-lab \
  --name 'Local Inference Lab, Inc.' --namespace local-inference-lab
```

This creates a random Ed25519 key encrypted as PKCS#8 PEM using the entered
passphrase, and a separate `publisher.json` containing only its public identity.
The directory must not already exist; key creation never overwrites a key.
On POSIX, the directory is mode `0700` and both files are mode `0600`.
Generating another key creates a different identity and does not match the
fingerprint above.

Use a unique, randomly generated passphrase stored in a password manager.
Ignition requires at least 16 UTF-8 bytes and confirms the passphrase. Enter it
only at the terminal prompt, never in chat, shell arguments, configuration, or
an environment variable. Signing also prompts in the terminal and refuses a
prompt whose input cannot be hidden.

Keep an encrypted backup of `signing-key.pem` and its public profile on separate
storage, with the passphrase recoverable separately. Verify that the backup can
be decrypted before depending on it. Losing either the key or its passphrase
prevents further signatures under that identity. File permissions and encryption
protect the stored key; they do not protect an unlocked signing process from a
compromised account or root. Sign only on a trusted host and review the package
being promoted.

For rotation, generate into a different directory and add the replacement public
profile to Helios. Previously signed packages require their original public key;
execution acceptance is specific to the key and namespace. If a private key is
compromised, remove its identity from distributed builds and revoke its acceptance
in affected trust stores with `flare trust revoke FINGERPRINT`.

## Change the passphrase

```sh
.venv/bin/ignition key-password \
  --signing-key ~/.config/flare/signing/local-inference-lab/signing-key.pem
```

The command prompts for the existing passphrase, then the replacement twice.
It writes an encrypted temporary file with owner-only permissions beside the key,
flushes it to disk, and atomically replaces the key file. The key and fingerprint
stay the same; publisher profiles, Helios builds, and signed packages need no
changes. A wrong existing passphrase or mismatched confirmation leaves the file
untouched. An interruption before replacement can leave an encrypted hidden
temporary file beside the key.

Update the encrypted backup after changing the passphrase. Existing backups
still use their previous passphrase. A passphrase change does not revoke copies
of the private key; use key rotation if a copy may be compromised.

## Promote a development package

Use a distinct destination and the actual repository URL where the production
package will be published:

```sh
.venv/bin/ignition sign /data/flare-runtime/development.safetensors \
  --output /data/flare-runtime/production.safetensors \
  --signing-key ~/.config/flare/signing/local-inference-lab/signing-key.pem \
  --namespace local-inference-lab \
  --official-source https://huggingface.co/local-inference-lab/REPOSITORY
```

Replace `REPOSITORY` with the intended repository name. The command does not
upload anything. The three publishing values can instead come from `[publishing]`
in the [configuration file](cli-design.md#signing-and-local-development); explicit
flags take precedence. It locates `development-publisher.json` beside that config
file. For a package prepared elsewhere, supply the original trusted public profile
with `--development-publisher PATH`. The development private key is unnecessary.

Re-signing verifies the development signatures and every executable and supporting
resource, decrypts executable resources, and encrypts them under the production
identity with a fresh package ID. Weight bytes and prepared execution contracts
are preserved. Stored-data hashes are checked against the development signatures
while resolving payloads, before the metadata-only primary is published.
`--shard-size SIZE` overrides `[publishing] shard_size` (default `64G`), a hard
payload-file limit of at most 64 GiB. Re-signing accepts monolithic, numbered-shard,
and snapshot sources and emits a snapshot with content-addressed payload blobs.
The signed logical resources remain whole. Adjacent executable and supporting
resources for one rank share bounded payload blobs; different ranks retain
separate blobs for selective distribution. Weight resources keep independent
blob identities so graph changes do not duplicate weights. Stored-data blobs
already in the package cache are hardlinked; executable blobs change with the
signing identity. An unchanged snapshot weight already linked into the cache
is hashed once for source verification and blob identity, without redundant
cache or destination-link reads.
Invalid packages fail without publishing a primary, and existing destinations
are never overwritten.

No checkpoint reload, compilation, graph capture, or GPU access is required.
The operation reads all source payloads to verify them and reports GiB read and
reused during stored-data verification. Large packages still require a complete
weight read. Promoting a monolithic or numbered-shard package also writes weight
blobs that are absent from the cache. Signing needs disk space for
uncached blobs plus temporary executable resources. Temporary plaintext executable
files reside in a private directory beside the destination and are removed when
the command exits normally or raises an error. A forced kill can leave that
directory; remove it only after confirming no signing process is using it.
Cached blobs and payload links can remain after failed publication and are reusable.
The source package remains intact. See [package snapshots](cli-design.md#package-snapshots)
for conversion that preserves an existing signature without a private key.

Rebuild Flare after changing embedded public profiles. A separately built or
installed executable retains the profiles it was built with.
