# Multi-host serving with the Flare agent

Flare serves one model across two or four DGX Sparks using tensor parallelism.
Each host runs a process, called a rank, that executes its share of the model.
Rank zero coordinates the ranks and serves the HTTP API. The other hosts run
peer ranks.

The Flare agent is the service that lets rank zero start and supervise a peer
rank remotely. With `flare run PACKAGE --hosts HOSTS`, rank zero sends each agent
its Flare executable and the package data that peer needs. The agent caches
those files, launches the rank, forwards its output, and sends it SIGTERM if
rank zero's connection closes. The rank process executes the model through
Helios; the agent manages its files and process lifetime.

Model preparation runs through `ignition export` on rank zero and `ignition join`
on each peer. Ignition loads weights, tunes kernels, captures graphs, and writes
a signed package on rank zero. These preparation processes run directly on the
hosts and do not use the agent. Once a package exists, serving uses the Rust
Flare executable and its prepared graphs without Python.

## Requirements

- Two or four DGX Sparks, one GPU per host, with the same OS image and NVIDIA
  driver. GLM-5.3 Flash and Qwen3.8 Flash support this preparation path. Qwen TP4
  requires full-model validation; see [model preparation](model-preparation.md).
- The same Flare, Ignition, and FlashInfer sources on every preparation host.
  Preparation rejects peers whose source identities differ. Follow the
  [build instructions](../AGENTS.md#build-and-run-model-checkpoints) on each host.
- A source checkpoint in rank zero's Hugging Face hub cache, configured with
  `[paths] hub_cache`. Multi-host export requires the checkpoint to be cached
  before preparation starts.
- Active RoCE ports with IPv4 RoCEv2 GIDs on every host. The ranks exchange
  tensors through RoCEnante, b12x's RoCE collectives, whose kernels are captured
  in the model graphs. This transport uses the Sparks' ConnectX-7 ports.
- Enough disk for the checkpoint, temporary exports, and serving package, as
  described below. Also budget Spark's shared CPU/GPU memory according to the
  [DGX Spark preparation guidance](model-preparation.md#dgx-spark).

### Network connections

| Connection | Destination | Port |
| --- | --- | --- |
| Ignition preparation rendezvous | Rank zero | Selected by `--rendezvous`; examples use TCP 29630 |
| Ignition preparation coordination | Participating hosts | Additional dynamically allocated TCPStore and Gloo ports |
| Flare agent control and file transfer | Each peer | TCP 29900 by default |
| Flare serving rendezvous | Rank zero | TCP 29800 by default |
| HTTP API | Rank zero | TCP 8000 in these examples |

Preparation and serving also require a working RoCE fabric between the ranks.
Each host discovers its active RDMA devices at startup. The ranks select up to
two rails per peer on a switch or supported point-to-point topology; the package
does not pin the fabric used during preparation. On Spark, cabled port 0 commonly
appears as `rocep1s0f0` and `roceP2p1s0f0`.

Agent file transfers use ordinary TCP to the addresses named by `--hosts`.
Choose hostnames or addresses that route over the intended transfer network.
The RoCE device selection for model collectives does not select that TCP route.

### Checkpoint and package storage

A source checkpoint supplies weights to Ignition. A prepared package contains
the weights, graphs, and other resources that Flare loads for serving.

| Artifact | Rank zero | Each peer |
| --- | --- | --- |
| Source checkpoint | Retained in its hub cache | Missing files are copied into its hub cache during preparation and retained |
| Temporary export | Stages all ranks before publication | Stages its own rank in the system temporary directory, then sends it to rank zero |
| Signed serving package | Retains the complete package, including companion files | The agent caches its rank's resources and required shared data |
| Serving executable | Uses the executable invoked by the operator | The agent caches and runs the executable sent by rank zero |

On rank zero, allow roughly twice the final package size for staging and
publication, in addition to checkpoint storage. On a preparation peer, allow
space for the checkpoint and roughly twice its own exported rank size while
that export is assembled. Temporary exports are removed after use.

The agent stores packages under `~/.cache/flare/packages` and executables under
`~/.cache/flare/bin`, in the account running the service. After a successful
upload, the agent removes unused entries of the same kind; active connections
keep their entries pinned. This cleanup does not remove source checkpoints from
the hub cache.

## One-time setup: the Flare agent

Only serving peers need the agent service. Rank zero and all peers need the
same cluster token. The agent authenticates rank zero through an HMAC challenge;
the token itself is never transmitted. The agent connection uses TCP without
transport encryption. Anyone holding the token can run code as the service user
on the peer hosts, so keep the token private and use a trusted cluster network.

### Install the shared token

Create the token once on rank zero. This command refuses to overwrite an
existing token; reuse that token when adding hosts to the same cluster.

```sh
install -d -m 700 ~/.config/flare
(
  umask 077
  set -C
  head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n' > ~/.config/flare/agent.token
)
```

Copy the same file to each peer account. For a peer named `hertz`:

```sh
ssh hertz 'install -d -m 700 ~/.config/flare'
scp -p ~/.config/flare/agent.token hertz:.config/flare/agent.token
ssh hertz 'chmod 600 ~/.config/flare/agent.token'
```

The token must contain at least 32 bytes after trimming whitespace, with no
group or other permission bits. Its default path is
`~/.config/flare/agent.token`. For another path, use `flare agent --token-file PATH`
on the peer and `flare run --agent-token-file PATH` on rank zero.

### Install the peer service

On each peer, run these commands from the repository root after building Flare:

```sh
install -d ~/.local/lib/flare-agent ~/.config/systemd/user
install -m 755 flare/target/release/flare ~/.local/lib/flare-agent/flare
cat > ~/.config/systemd/user/flare-agent.service <<'UNIT'
[Unit]
Description=Flare agent for tensor-parallel peer ranks
After=network-online.target

[Service]
ExecStart=%h/.local/lib/flare-agent/flare agent --listen 0.0.0.0:29900
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
UNIT
systemctl --user daemon-reload
systemctl --user enable --now flare-agent
loginctl enable-linger "$USER"
journalctl --user -u flare-agent -n 20
```

Lingering keeps the user service running without an active login session.
The journal should report the agent's listening address and cache directory.
For a different listening port, change `--listen` and include that port in
rank zero's host list, for example `--hosts hertz:29901`.

The installed agent manages transfers and processes; serving ranks use the
executable sent by rank zero. Keep the agent compatible with rank zero's
transfer protocol. Multi-file packages require agent protocol 3. Protocol 2
supports rank byte ranges in a single-file package; protocol 1 receives the
whole single-file package. To update an agent, stop serving, replace the
installed executable, and restart `flare-agent`. Restart the agent after
replacing its token as well, because it reads the token at startup.

## Prepare the package

The following example prepares GLM-5.3 Flash across two Sparks. Rank zero uses
`10.200.0.12` on the cluster network, and `hertz` is rank one. Run preparation
commands from the repository root on each host.

On rank zero, select an unused output filename and start export:

```sh
.venv/bin/ignition export glm-5.3-flash --ranks 2 \
  --rendezvous 10.200.0.12:29630 --gpus 0 --max-concurrency 8 \
  --output /data/flare-runtime/glm53-spark-tp2.safetensors --verbose
```

On `hertz`, join the same rendezvous:

```sh
.venv/bin/ignition join 10.200.0.12:29630 --rank 1 --ranks 2
```

For four hosts, use `--ranks 4` everywhere and run one join process per peer,
with distinct ranks 1, 2, and 3.

Rank zero admits each peer once and plans from all participating GPUs. It sends
each peer the checkpoint files missing from that peer's hub cache. A peer with
an empty cache receives the complete checkpoint; files already present with the
expected content are reused. Each rank then loads its own weights and prepares
its graphs. Forward compilation is divided across ranks, and compiled artifacts
are shared with ranks that need them.

Each peer exports its rank to a temporary file and streams that file to rank
zero. Rank zero assembles all ranks, signs the package, encrypts its executable
resources, and removes staging files. The completed package records its rank
count and export identity. Keep its primary file and any companion files
together with their original names.

For GLM-5.3 Flash, context parallelism shards each attention page across ranks,
increasing total KV capacity compared with replicating complete pages. Use
`--no-context-parallel` during export to replicate pages instead. Use
`--no-stream-overlap` during export to serialize concurrent branches when
diagnosing scheduling-dependent faults. These flags change the prepared graphs.

## Serve

On rank zero, start the package with peers listed in rank order:

```sh
./flare/target/release/flare run /data/flare-runtime/glm53-spark-tp2.safetensors \
  --hosts hertz --gpus 0 --host 0.0.0.0 --port 8000
```

A four-rank package needs three peers, for example
`--hosts hertz,faraday,ampere`: Hertz is rank one, Faraday rank two, and Ampere
rank three. The host count must match the package's prepared rank count.

Rank zero validates the package and obtains publisher acceptance before
launching peers. It passes the accepted publisher profile and signing-key
approval to each agent. Each peer independently verifies the package for its
rank; `--verify-weights` also verifies the stored weight bytes.

Startup then proceeds as follows:

1. Rank zero connects to the agents and sends missing executable and package
   data to peers in parallel. Each peer receives the signed index, its own rank's
   resources, and the shared tables it needs.
2. The agents launch their ranks with rank zero's serving options, including
   concurrency, speculation, and embedding-table storage.
3. All ranks join the serving rendezvous, agree on capacity, and load their
   assigned resources.
4. The ranks exchange RoCE connection records and connect the transport. Rank
   zero serves HTTP once the rank group is ready.

The serving rendezvous listens on `0.0.0.0:29800` by default. Change its port
with `--rendezvous 0.0.0.0:PORT`. Rank zero advertises the local address selected
by its route to the first peer in `--hosts`; every peer must be able to reach
that address. A more restrictive bind address must include that interface.

Peer package files preserve the source files' logical lengths, with unwritten
regions represented as sparse holes. Only transferred ranges occupy disk on a
filesystem supporting sparse files. Use `du` to inspect allocated space;
`ls -l` reports logical file lengths. A multi-file package's primary and
companions stay together in one cache directory. Later starts reuse cached
files whose cache names and sizes match, avoiding another transfer.

Peer output appears on rank zero's terminal with a `[rank R host]` prefix.
Stop serving with Ctrl-C or SIGTERM on rank zero so native work drains before
resources are released. If the agent's connection to rank zero closes, the
agent sends SIGTERM to its peer process. The agent service stays available for
another launch. It does not restart a failed rank automatically; start the model
again from rank zero after resolving the failure.

### Start ranks manually

Manual startup requires the compatible executable and package files to be
available on every host, with publisher trust configured on each. Start rank
zero with `--rank 0 --rendezvous 0.0.0.0:29800` and no `--hosts`. Start each peer
with its assigned `--rank` and rank zero's reachable address:

```sh
./flare/target/release/flare run /data/flare-runtime/glm53-spark-tp2.safetensors \
  --rank 1 --rendezvous 10.200.0.12:29800 --gpus 0
```

Use identical serving options on every rank. Manual startup leaves file
distribution and process management to the operator.

## Failures

| Error or symptom | Check |
| --- | --- |
| `cannot reach flare agent` | The peer service is running; the hostname, listening port, route, and firewall allow rank zero to connect. |
| `agent token mismatch` | Rank zero and the agent read the same token. Restart an agent after replacing its token. |
| Token permission error | The token is private to its owner, normally mode `600`, and readable by the account running Flare. |
| `multi-file packages require flare agent protocol 3` | Install a compatible agent build and restart the service. |
| `rank R cannot hold the checkpoint` | The peer has space for missing checkpoint files, and received files pass content verification. |
| `rank zero refused rank R` | Every serving rank uses the same package export and world size. |
| `tensor-parallel rank identity or graph I/O schema differs` | All ranks use matching serving options and compatible executables. |
| `package RoCE kernels use proxy ABI ...` | The package's FlashInfer RoCEnante ABI matches the one vendored by the Flare executable; rebuild Flare or re-export as needed. |
| Missing IPv4 GID, rail link error, transport setup failure, or queue-pair connection failure | RoCE ports are active, have IPv4 addresses, and provide the required connectivity between hosts. |

A collective timeout poisons the rank group's transport and shuts it down.
The failed frame is not published. Check the rank-prefixed output on rank zero
and the peer agent journals before restarting the group.

## Experimental rings

`--roce-ring` connects neighbors in rank order and uses native CPU relays for
other destinations. The transport supports 2 through 16 ranks, subject to each
model recipe's supported tensor-parallel sizes. Four-Spark transport checks
pass; physical ring cabling and full-model ring serving remain unqualified.

See [RoCE ring transport](roce-ring.md) for topology, commands, validation, and
failure handling.
