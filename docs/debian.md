# Debian packages

`flare-inference` installs the native Flare executable at `/usr/bin/flare` on
Ubuntu 24.04, with separate `arm64` and `amd64` packages. APT selects the package
for the installed system's architecture. The ARM64 build includes RoCE support
for DGX Spark. The package contains serving, host-agent, and Orbit commands.
Ignition, Python, model weights,
and the CUDA toolkit are separate preparation inputs. Serving uses the NVIDIA
driver installed by DGX OS and a model package matching the GPU architecture and
tensor-parallel size.

The Debian package name distinguishes the inference server from Ubuntu's `flare`
game package. The command remains `flare`.

## Install

Configure the Local Inference Lab repository with its scoped signing key:

```sh
sudo install -d -m 0755 /etc/apt/keyrings
sudo curl -fsSLo /etc/apt/keyrings/lil-flare-archive.asc \
  https://ppa.local-inference-lab.ai/lil-flare-archive.asc
sudo curl -fsSLo /etc/apt/sources.list.d/flare.sources \
  https://ppa.local-inference-lab.ai/flare.sources
sudo apt update
sudo apt install flare-inference
flare --help
```

The repository uses the `noble` suite and `main` component. APT authenticates
repository metadata and checks the package hashes listed in that metadata.
The APT OpenPGP key is separate from the Ed25519 PEM used to sign model packages.
See [APT's authentication specification](https://manpages.debian.org/bookworm/apt/apt-secure.8.en.html).

Installation and upgrades do not create or restart services. To install an
agent, run `/usr/bin/flare agent install` as the user who owns its models. The
command prints a pairing code and enables a systemd user service; see
[Orbit](orbit.md). The installer copies the executable into the user's home
directory. After a Debian package upgrade, rerun that command during a maintenance
window to update and restart the agent. Existing controller identities and caches
remain in place. Check `command -v flare` when a locally installed executable
might take precedence over `/usr/bin/flare`.

## Build

From a committed checkout, with Docker Buildx available:

```sh
./scripts/build_deb.sh --maintainer 'Release Maintainer <maintainer@example.org>'
```

The builder compiles the locked Cargo dependencies in Ubuntu 24.04 using Rust
1.94.1. It builds for Docker's native architecture and writes the `.deb` under
`dist/debian/`. `--output DIR` changes the destination; `--version VERSION`
sets an explicit Debian version. The default version contains the Cargo version,
UTC commit timestamp, and abbreviated commit ID. The installed
`/usr/share/doc/flare-inference/build-info` records the complete revision.
Commit native source changes before building; uncommitted changes cannot be
identified by that revision. Run the same build
on each architecture and collect both artifacts before signing the repository.
Docker's `DOCKER_HOST=ssh://HOST` option can select a remote native builder while
returning the `.deb` to the local output directory.

`dpkg-shlibdeps` derives shared-library requirements from Ubuntu's symbol tables.
ARM64 additionally depends on the verbs providers needed for RoCE. Compilation
and dependency discovery take place inside the build image so an NVIDIA OFED
installation on the builder cannot raise the package's minimum library versions.
Root ownership is set in the archive without requiring a privileged build.
The package includes the project license and dependency license notices.

## Sign and publish

Build a complete repository staging tree with all versions that should remain
installable. The output directory must not already exist:

```sh
python3 scripts/build_apt_repository.py \
  --output dist/apt \
  --gnupg-home /absolute/path/to/private-apt-keyring \
  --key FULL_OPENPGP_FINGERPRINT \
  dist/debian/*.deb
```

The builder writes packages under `pool/`, indices under `dists/noble/`, a public
key, and a Deb822 source configuration. It signs both `InRelease` and
`Release.gpg` and verifies the signatures before completing. Index files also
have SHA-256 and SHA-512 `by-hash` paths to keep updates consistent during
publication. The private key remains on the signing host.

Publish immutable package files and `by-hash` indices first, ordinary indices
second, and signed release metadata last. Use atomic file replacement for each
metadata file. Retain existing package files and `by-hash` paths so clients can
finish downloads using metadata fetched before the update. Never replace a
published `.deb` with different bytes under the same filename; increment the
version instead. Repository metadata has no automatic expiration; republish it
when the package set or signing key changes and renew the signing key before
its own expiration.

Before publication, install the `.deb` in a clean Ubuntu 24.04 environment and
exercise `--version`, `run --help`, `agent --help`, and `orbit --help`. After
publication, test an APT installation from the HTTPS endpoint using only the
published public key. These checks validate package dependencies and repository
authentication; GPU model serving requires a separate compatible-model test.
