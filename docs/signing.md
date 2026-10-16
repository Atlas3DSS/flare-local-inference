# Production signing

Ignition signs runtime packages with Ed25519. Helios embeds the public identity
**Local Inference Lab, Inc.**, namespace `local-inference-lab`, from
[`publishers.json`](../helios/src/runtime_package/publishers.json). Its SHA-256
public-key fingerprint is:

```text
937c0143f6856d3ee8ad164f62e3d804bd825d7a8750d0accb9a5849f703a8c1
```

Embedding a public identity authenticates the publisher; execution still requires
first-use acceptance. Private keys never belong in source control, packages, or
publisher profiles.

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

## Promote a development package

Use a distinct destination and the actual repository URL where the production
package will be published:

```sh
.venv/bin/ignition resign /data/flare-runtime/development.safetensors \
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
while copying, before the final file is published. Invalid packages fail without
publishing an output; existing destinations are never overwritten.

No checkpoint reload, compilation, graph capture, or GPU access is required.
The operation reads and copies the entire package, so allow space for another
package plus temporary executable resources. Temporary plaintext executable files
reside in a private directory beside the destination and are removed when the
command exits normally or raises an error. A forced kill can leave that directory
or an incomplete hidden output file; remove those after confirming no signing
process is using them. The source package remains intact.

Rebuild Flare after changing embedded public profiles. A separately built or
installed executable retains the profiles it was built with.
