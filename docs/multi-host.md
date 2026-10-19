# Multi-host serving with the Flare agent

Flare serves one model across two or four DGX Sparks using tensor parallelism.
Each host runs a process, called a rank, that executes its share of the model.
Rank zero coordinates the ranks and serves the HTTP API. The other hosts run
peer ranks.

Every host of a serving group runs the Flare agent, a user service that pairs
with the operator's controller machine, caches the Flare executable and package
data, starts the host's rank, supervises it, and streams its output. The
operator starts and stops groups with `flare fleet` commands from any machine
with the Flare executable; that machine is not a rank. [Fleet control](fleet.md)
specifies the agent protocol, pairing, grants, and process ownership.

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
| Flare agent control and file transfer | Every host, from the controller and from other hosts | TCP 29900 by default |
| Flare serving rendezvous | Rank zero | TCP 29800 |
| HTTP API | Rank zero | TCP 8000 in these examples |

Preparation and serving also require a working RoCE fabric between the ranks.
Each host discovers its active RDMA devices at startup. The ranks select up to
two rails per peer on a switch or supported point-to-point topology; the package
does not pin the fabric used during preparation. On Spark, cabled port 0 commonly
appears as `rocep1s0f0` and `roceP2p1s0f0`.

Agent file transfers use ordinary TCP to the addresses recorded in the fleet
file at pairing. Choose hostnames or addresses that route over the intended
transfer network. The RoCE device selection for model collectives does not
select that TCP route.

### Checkpoint and package storage

A source checkpoint supplies weights to Ignition. A prepared package contains
the weights, graphs, and other resources that Flare loads for serving.

| Artifact | Rank zero | Each peer |
| --- | --- | --- |
| Source checkpoint | Retained in its hub cache | Missing files are copied into its hub cache during preparation and retained |
| Temporary export | Stages all ranks before publication | Stages its own rank in the system temporary directory, then sends it to rank zero |
| Signed serving package | Retains the complete package, including companion files | The agent caches its rank's resources and required shared data, fetched from rank zero's host and digested once on arrival |
| Serving executable | The agent caches and runs the executable sent by the controller | The agent caches and runs the executable sent by the controller |

On rank zero, allow roughly twice the final package size for staging and
publication, in addition to checkpoint storage. On a preparation peer, allow
space for the checkpoint and roughly twice its own exported rank size while
that export is assembled. Temporary exports are removed after use.

The agent stores packages under `~/.cache/flare/packages` and executables under
`~/.cache/flare/bin`, in the account running the service. After a successful
transfer, the agent removes entries of the same kind that no running group and
no open connection uses. This cleanup does not remove source checkpoints from
the hub cache or packages outside the cache.

## One-time setup: the Flare agent

Every host of a group runs the agent service. The agent holds a node key and a
list of authorized controllers; a controller holds its own key and a fleet file
listing the nodes it paired with and named clusters in rank order. Anyone
holding an authorized controller key can run code as the service user on every
node, so keep `~/.config/flare/fleet/controller.key` private.

### Install the agent on each host

From the repository root after building Flare:

```sh
./flare/target/release/flare agent install
```

This copies the executable to `~/.local/lib/flare-agent/flare`, writes and
enables the `flare-agent` user service listening on `0.0.0.0:29900`, enables
lingering so the service runs without a login session, creates the node key,
and prints the node's name, fingerprint, and a one-time join code valid for 24
hours. `--listen IP:PORT` changes the listening address; `--join-code CODE`
installs a code of your own so one code covers several hosts;
`--accept-first` admits the first controller that connects without a code, for
a network you already trust. `journalctl --user -u flare-agent` shows the
agent's log.

To update an agent, stop the group it serves, run `flare agent install` again
with the new executable, and pair again only if the node was reset. Restarting
the service stops any rank it runs.

### Pair from the controller

On the machine that will control the fleet, pair with each host using the code
its installation printed:

```sh
flare fleet pair maxwell hertz faraday ampere --code ABCD-EFGH-IJKL-MNOP-QRST-UVWX-YZ23-4567
```

Pairing creates the controller key on first use, proves the join code to each
node inside the encrypted session, records each node's name, fingerprint,
public key, and address in `~/.config/flare/fleet.toml`, and installs the
controller as the node's authorized controller. Compare the fingerprints it
prints with what each host printed at installation. Hosts installed with
`--accept-first` pair without `--code`.

Then name a cluster in the fleet file, listing nodes in rank order:

```toml
[cluster.sparks]
nodes = ["maxwell", "hertz", "faraday", "ampere"]
```

The first node is rank zero. With `--roce-ring`, the order is also the ring.
`flare fleet status --cluster sparks` shows every node's free disk, controller
revision, and running group. A fleet with one cluster needs no `--cluster`.

To let another machine control the same fleet, run
`flare fleet controllers add NAME PUBLIC_KEY` with the other controller's
public key (the hex in its own fleet file); `flare fleet controllers remove KEY`
revokes it. `flare agent reset` on a host forgets its controllers so it can be
paired again.

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

From the controller, start the package on the cluster:

```sh
flare fleet run maxwell:/data/flare-runtime/glm53-spark-tp2.safetensors \
  --cluster sparks --gpus 0 --host 0.0.0.0 --port 8000
```

`NODE:PATH` names a package on a node's disk, normally rank zero's host where
preparation wrote it. A plain path names a package on the controller, which is
then sent to rank zero's host. The cluster's node count must match the
package's prepared rank count.

The controller verifies the package's signed index and obtains publisher
acceptance at its own terminal, then passes the accepted publisher profile and
signing-key approval to each agent. An agent digests bytes it fetched from
another host against the signed index before running them and remembers the
result with its cached copy, so a repeated launch of the same package skips
that step. Each rank independently verifies the package's signatures for its
rank; `--verify-weights` makes every rank digest its stored data at load.

Startup then proceeds as follows:

1. The controller connects to every node's agent, checks that no group is
   running and that disk suffices, and sends the Flare executable to nodes that
   lack it.
2. Rank zero's node holds the complete package. Each peer fetches the signed
   index, its own rank's resources, and the shared tables it needs from rank
   zero's host, in parallel, over the hosts' own network.
3. The controller signs a grant naming the group, the package, every member in
   rank order, the rendezvous address, and the serving options, and asks each
   agent to run its rank. Agents accept a grant only from their authorized
   controllers.
4. All ranks join the serving rendezvous, agree on capacity, and load their
   assigned resources.
5. The ranks exchange RoCE connection records and connect the transport. Rank
   zero serves HTTP once the rank group is ready, and the controller prints
   the address.

The serving rendezvous listens on rank zero's port 29800 at the address rank
zero's host uses to reach the first peer; every peer must be able to reach that
address.

Peer package files preserve the source files' logical lengths, with unwritten
regions represented as sparse holes. Only transferred ranges occupy disk on a
filesystem supporting sparse files. Use `du` to inspect allocated space;
`ls -l` reports logical file lengths. A multi-file package's primary and
companions stay together in one cache directory. Later starts reuse cached
files whose cache names and sizes match, avoiding another transfer.

Every rank's output appears on the controller's terminal with a `[rank R name]`
prefix. Ctrl-C stops the group: rank zero receives SIGTERM first and drains
native work, then peers still running after thirty seconds are stopped.
`--detach` returns once every rank has started; `flare fleet attach` shows a
running group's output, and `flare fleet stop` stops it. A controller that
disconnects, sleeps, or crashes does not stop the group; agents keep the last
64 KiB of each rank's output for a later attach. An agent does not restart a
failed rank; start the group again after resolving the failure.

### Start ranks manually

Manual startup requires the compatible executable and package files to be
available on every host, with publisher trust configured on each. Start rank
zero with `--rank 0 --rendezvous 0.0.0.0:29800`. Start each peer with its
assigned `--rank` and rank zero's reachable address:

```sh
./flare/target/release/flare run /data/flare-runtime/glm53-spark-tp2.safetensors \
  --rank 1 --rendezvous 10.200.0.12:29800 --gpus 0
```

Use identical serving options on every rank. Manual startup leaves file
distribution and process management to the operator.

## Failures

| Error or symptom | Check |
| --- | --- |
| `cannot reach flare agent` | The service is running on the host; the address in the fleet file, route, and firewall allow the controller and the other hosts to connect. |
| `is not an authorized controller of this node` | The node was paired with a different controller key; pair it, or add this controller with `flare fleet controllers add` from a controller it knows. |
| `join code proof rejected` or `this node requires its join code` | The code matches what `flare agent install` printed and has not expired or been used. |
| `peer speaks agent protocol N` | Every node and the controller run the same Flare build; run `flare agent install` with the matching executable. |
| `group ... is running on this node` | Stop the running group with `flare fleet stop` before starting another. |
| `rank R cannot hold the checkpoint` | The peer has space for missing checkpoint files, and received files pass content verification. |
| `rank zero refused rank R` | Every serving rank uses the same package export and world size. |
| `tensor-parallel rank identity or graph I/O schema differs` | All ranks use matching serving options and compatible executables. |
| `package RoCE kernels use proxy ABI ...` | The package's FlashInfer RoCEnante ABI matches the one vendored by the Flare executable; rebuild Flare or re-export as needed. |
| Missing IPv4 GID, rail link error, transport setup failure, or queue-pair connection failure | RoCE ports are active, have IPv4 addresses, and provide the required connectivity between hosts. |

A collective timeout poisons the rank group's transport and shuts it down.
The failed frame is not published. Check the rank-prefixed output on the
controller and the agents' journals before restarting the group.

## Experimental rings

`--roce-ring` connects neighbors in rank order and uses native CPU relays for
other destinations. The transport supports 2 through 16 ranks, subject to each
model recipe's supported tensor-parallel sizes. Four-Spark transport checks
pass; physical ring cabling and full-model ring serving remain unqualified.

See [RoCE ring transport](roce-ring.md) for topology, commands, validation, and
failure handling.
