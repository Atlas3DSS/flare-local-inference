# Serving container

The root `Dockerfile` builds the Rust Flare executable with Helios and copies it
into a Debian slim image. Neither stage requires the CUDA toolkit or Python.
The runtime contains no Ignition, Heliopause, model weights, or compiler.
Helios loads the host's `libcuda.so.1` and replays kernels and graphs from a
prepared runtime package.

The host needs a compatible NVIDIA driver, Docker, and
[NVIDIA Container Toolkit configured for Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker).
The package must match the GPU architecture and selected tensor-parallel size.
Prepare it outside the container; see [runtime packages](model-preparation.md#runtime-packages).

## Build

From the repository root:

```sh
docker build -t flare:local .
docker run --rm flare:local --version
docker run --rm flare:local run --help
```

The build context includes only the Rust serving sources and Protobuf schema.
Cargo uses `flare/Cargo.lock`. The build installs a C/C++ toolchain, CMake,
NASM, and `protoc`; these remain in the build stage. BuildKit caches Cargo
downloads and compilation output across builds.

Linux AMD64 and ARM64 use the same Dockerfile. Build on the target architecture
or select it with Docker Buildx. ARM64 also includes libibverbs and its providers
because Flare enables the RoCE transport on that architecture. Multi-host RoCE
deployment additionally requires the host's RDMA devices and network setup.

To build and publish both Linux architectures under one tag, log in to the
destination registry, then run:

```sh
./scripts/build_container.sh
```

The script creates a `docker-container` builder if the selected name is missing,
preserves existing builder configuration, and starts the builder before building.
The script uses builder `flare-multi`, pushes
`ghcr.io/local-inference-lab/flare:latest`, and displays the published manifest.
Use `--beta` or `--release` to publish `ghcr.io/local-inference-lab/flare:beta`
or `ghcr.io/local-inference-lab/flare:release`, respectively:

```sh
./scripts/build_container.sh --beta
./scripts/build_container.sh --release
```

Use `--tag IMAGE:TAG` for a custom image reference. The three tag options are
mutually exclusive. Override the builder with `--builder NAME`. Build output
uses plain progress logs, including when redirected. The script resolves the
repository root from its own location, so it can run from any working directory.
Docker selects the matching architecture when pulling the published tag.

## Continuous publication

The [serving-container workflow](../.github/workflows/container.yml) builds Linux
AMD64 and ARM64 images on pushes to `main`. It can also be started manually with
GitHub Actions' **Run workflow** control on `main`. A push containing several
commits builds its tip commit, rather than a separate image for every intermediate
commit. There is no path filter or pull-request trigger.

The workflow first publishes a staging tag of the form
`ghcr.io/local-inference-lab/flare:sha-COMMIT-RUN-ATTEMPT`. It checks the image
index and runs `--version`, `run --help`, `agent --help`, and `orbit --help` for
both architectures, without GPUs, networking, ports, or host mounts. These CPU
checks validate executable startup and CLI parsing; they do not validate model
serving. Only a successful build whose commit still matches remote `main` is
promoted to `ghcr.io/local-inference-lab/flare:latest`. Promotion uses the tested
index digest and verifies the resulting digest. A superseded build retains its
staging tag. The workflow does not change `beta` or `release`.

Register a dedicated, trusted Linux X64 self-hosted runner for the Flare
repository with label `lil-flare-builder`. An existing runner registered only
for another repository cannot accept this workflow. The runner needs `gh`,
`jq`, `flock`, Docker, and a system-installed Buildx plugin. Its rootless Docker
daemon must be reachable at
`unix:///run/lil-flashinfer-docker/docker.sock`, with the containerd image store
and ARM64 emulation configured. The workflow uses builder `default` to retain
that daemon's BuildKit caches; it does not install emulation or create a builder.

The runner user must be able to open
`/var/lib/github-flashinfer/locks/cu134-sm120-build.lock`. The workflow holds this
shared lock through building, smoke tests, and promotion, waiting up to six
hours for other builds. GitHub Actions queues runs without canceling the active
build; a remote `main` check prevents queued, superseded commits from replacing
`latest`. Docker credentials live in a temporary per-run configuration directory
and are removed when the build step exits.

The workflow grants its `GITHUB_TOKEN` repository-content read access and package
write access. The existing GHCR `flare` package must also grant the Flare
repository Actions write access through inherited repository permissions or
the package's **Manage Actions access** settings. Public pull access and an OCI
source label alone do not establish that write permission.

## Discord changelog

The [published-changes workflow](../.github/workflows/changelog.yml) checks for
unannounced changes after a successful container workflow and hourly at minute
17. It resolves `latest` through GHCR and requires both image architectures to
identify the same source commit. A private repository token reads commit
metadata; source files, patches, and commit bodies are not sent to the model.
The model receives sanitized commit subjects and produces short English notes.
The notes describe changes in `main`, including preparation changes that are
not part of the serving container.

At most one announcement is sent in six hours. Commits accumulate while that
interval is active, and the hourly check flushes them even without another push.
State is stored outside the checkout at
`/var/lib/github-flashinfer/state/flare-announcements`. Its runner-owned directory
must survive job cleanup and host restarts. A separate file lock serializes
updates. A successful Discord acknowledgment advances the commit cursor; a
transport failure triggers message-history reconciliation before another send.

Configure the repository secret `FLARE_DISCORD_BOT_TOKEN` and these repository
variables before enabling publication:

| Variable | Meaning |
| --- | --- |
| `FLARE_DISCORD_CHANNEL_ID` | Destination channel the bot can view and send to |
| `FLARE_ANNOUNCE_BASE_SHA` | Complete source SHA covered by the preceding announcement |
| `FLARE_ANNOUNCE_BASE_TIME` | That announcement's timestamp with a UTC offset |
| `FLARE_CHANGELOG_MODEL_URL` | OpenAI-compatible API base URL reachable from the runner |
| `FLARE_CHANGELOG_MODEL` | Model name served by that API |

The baseline variables initialize an absent state file. They do not reset an
existing cursor. The bot does not automatically mention users or roles. Preview
an explicitly selected range without sending a Discord message or changing state:

```sh
GH_TOKEN="$(gh auth token)" python3 scripts/announce_changes.py \
  --repository local-inference-lab/flare \
  --channel-id "$FLARE_DISCORD_CHANNEL_ID" \
  --preview --base "$FLARE_ANNOUNCE_BASE_SHA" \
  --image ghcr.io/local-inference-lab/flare:latest \
  --model-url "$FLARE_CHANGELOG_MODEL_URL" --model "$FLARE_CHANGELOG_MODEL"
```

If an uncertain send cannot be found in channel history, the job stops rather
than risking a duplicate. An operator can retry through **Run workflow**, using
the exact `pending.nonce` and `pending.head` from the state file as `retry_nonce`
and `retry_head`. This rechecks the channel history and still requires six hours
since the uncertain attempt; it preserves the frozen message and commit range.

## Run

This example exposes host GPU 0 and mounts a directory containing a prepared
package, its public publisher profile, and an already approved trust store:

```sh
docker run --rm --name flare \
  --gpus 'device=0,capabilities=compute' \
  --ulimit memlock=-1:-1 \
  --stop-timeout 60 \
  -p 127.0.0.1:8000:8000 \
  --mount type=bind,src=/absolute/path/to/packages,dst=/models,readonly \
  flare:local run /models/model.safetensors \
  --publisher-keys /models/publisher.json \
  --trust-store /models/trust.toml \
  --gpus 0 --host 0.0.0.0 --port 8000
```

The image runs as UID/GID `10001:10001`; mounted files and parent directories
must be readable by that user. Alternatively, pass Docker
`--user "$(id -u):$(id -g)"` and an explicit `--trust-store` path to use the host
file owner's identity. Mount only runtime files, including public keys;
private signing keys are preparation inputs. For first-use approval, add `-it`
and mount a writable trust directory separately. Pass its `trust.toml` path with
`--trust-store` so acceptance survives container removal. Unattended approval
uses the exact fingerprint and namespace described in
[signing and local development](cli-design.md#signing-and-local-development).

Docker's `--gpus` selects host devices. Flare's `--gpus` selects devices from the
container's visible list. For a two-rank package on host GPUs 4 and 5, use
`--gpus '"device=4,5",capabilities=compute'` on Docker and `--gpus 0,1` on Flare.
The NVIDIA runtime's `compute` capability mounts the driver libraries;
Flare does not require CUDA runtime, cuBLAS, or NCCL libraries in the image.
See [NVIDIA driver capabilities](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#driver-capabilities).

Bind Flare to `0.0.0.0` inside the container so Docker's published port reaches
it. Add `--api-key-file` with a mounted secret when authentication is required.
The executable is PID 1 and handles SIGTERM directly; `docker stop` allows
native replay to drain within the configured stop timeout.

## Downloading packages

Mount a writable directory as the Hub cache to reuse downloads across containers.
The directory must be owned by UID/GID `10001:10001`, or run with its owner's
identity through Docker's `--user` flag. For example:

```sh
mkdir -p "$HOME/.cache/huggingface/hub"
docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$HOME/.cache/huggingface/hub,dst=/cache" \
  flare:local download local-inference-lab/MODEL \
  --revision gb202 --cache-dir /cache
```

`MODEL` is a repository-name placeholder. An explicit revision permits downloading
without exposing GPUs. For automatic selection, expose the intended host GPUs
with Docker's `--gpus` and select their container-local indices with Flare's
`--gpus`. `flare run ORG/NAME` accepts the same download options, then loads the
cached package. Mount a persistent trust directory and pass `--trust-store` when
running with a host UID absent from the image's user database. Private repositories
add `--token-file` pointing to a separately mounted token file. See
[published packages](cli-design.md#downloading-published-packages) for revision,
filename selection, and offline behavior.

Disk-backed embedding tables and disk prefix-cache offload use Linux `io_uring`.
The container's seccomp policy must permit `io_uring_setup`, `io_uring_enter`,
and `io_uring_register` for those modes. Docker's default policy can block them;
use a policy allowing those calls. Prefix-cache disk directories also need a
writable bind mount owned by the serving user. `--ulimit memlock=-1:-1` permits
the pinned host storage used for transfers.
