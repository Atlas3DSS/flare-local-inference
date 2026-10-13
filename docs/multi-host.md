# Serving one model across DGX Sparks

A model too large for one GB10 runs tensor-parallel across two or four DGX
Sparks, one rank per host. GLM-5.3 Flash and Qwen3.8 Flash serve this way. The
ranks reduce and gather through RoCEnante, b12x's RoCE collectives over the
Sparks' ConnectX-7 ports, whose kernels are part of every captured graph; no
NCCL is involved. Preparation runs once, across all hosts, and writes one
package on rank zero. Serving starts on rank zero alone: Flare sends the package
and its own executable to each peer host's agent where missing, starts the peer
ranks, and serves HTTP.

## Requirements

- Two or four DGX Sparks on one RoCE fabric (a switch, or point-to-point cables),
  with the same OS image and NVIDIA driver. Each host uses one GPU.
- Each host's RoCE ports up, with an IPv4 RoCEv2 GID at index 3. RoCEnante uses
  every active RDMA device, at most two rails per peer. On a DGX Spark these are
  `rocep1s0f0` and `roceP2p1s0f0` for a cabled port 0.
- The checkpoint in rank zero's `[paths] hub_cache`. Preparation sends each peer
  the files its own `hub_cache` lacks, checking each against its content hash,
  and the peer keeps them there; every rank prepares its own shard.
- The same Flare, Ignition and FlashInfer sources on every host during
  preparation. Ranks with different sources refuse to join.
- On rank zero, free disk for both rank staging and the signed package during
  publication: about twice the whole model (GLM-5.3 Flash NVFP4: about 360 GB).
  Staging is removed after publication. Each peer keeps only its own rank's part
  of the newest package it received, in `~/.cache/flare/packages`, or the whole
  package when serving with `--verify-weights`.

## One-time setup: the Flare agent

Every peer host runs `flare agent`, which receives rank zero's executable and
package and runs the host's rank. It listens on port 29900 and admits only
holders of the cluster's shared token, through an HMAC challenge, so the token
never crosses the network. It runs only executables rank zero sent, verified
against their SHA-256 names, and passes their arguments without a shell. Keep
the token private: anyone holding it can run code on every host.

Create one token and copy it to every host, rank zero included:

```sh
mkdir -p ~/.config/flare && chmod 700 ~/.config/flare
head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n' > ~/.config/flare/agent.token
chmod 600 ~/.config/flare/agent.token
```

On each peer host, install the executable and a systemd user service:

```sh
mkdir -p ~/.local/lib/flare-agent ~/.config/systemd/user
cp flare/target/release/flare ~/.local/lib/flare-agent/flare
cat > ~/.config/systemd/user/flare-agent.service <<'EOF'
[Unit]
Description=Flare agent: runs a multi-host package's peer ranks for rank zero
After=network-online.target

[Service]
ExecStart=%h/.local/lib/flare-agent/flare agent --listen 0.0.0.0:29900
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now flare-agent
loginctl enable-linger "$USER"   # keep the agent running without a login session
journalctl --user -u flare-agent -n 5
```

The agent's own executable changes only with its protocol; the ranks it runs use
whatever executable rank zero sends. An agent built before peers kept only their
rank receives whole packages; reinstall it to keep only the rank's bytes.

Rank zero authenticates the package and obtains publisher acceptance before
starting peers. Its authenticated agent request carries that accepted publisher's
profile and exact signing-key approval. Each peer verifies the package with that
profile before loading its rank; `--verify-weights` also applies to peers.

## Prepare the package

Choose rank zero's address on the cluster network and a free port. On rank zero:

```sh
.venv/bin/ignition export glm-5.3-flash --ranks 2 --rendezvous 10.200.0.12:29630 \
  --gpus 0 --max-concurrency 8 --output /data/flare-runtime/glm53-spark-tp2.safetensors --verbose
```

On each peer host, with ranks numbered 1 through N-1:

```sh
.venv/bin/ignition join 10.200.0.12:29630 --rank 1 --ranks 2
```

Rank zero admits each rank once and plans from every rank's GPU. It sends each
peer the checkpoint files the peer lacks, all peers at once; the first
preparation on a new peer copies the whole checkpoint, and later ones copy
nothing. All ranks prepare together. Rank zero compiles forwards and sends peers
those they lack. Each peer then exports its rank and streams it over its
connection to rank zero, which appends it to the package as it arrives. The
package exists only on rank zero; it holds every rank and records the cluster's
size and an export id.

## Serve

On rank zero:

```sh
./flare/target/release/flare run /data/flare-runtime/glm53-spark-tp2.safetensors \
  --hosts hertz --gpus 0 --bind 0.0.0.0:8000
```

Name the peer hosts in rank order (`--hosts faraday,hertz,ampere` for four
ranks); they must be reachable at port 29900. Rank zero then:

1. sends each agent its own executable and the parts of the package that host's
   rank reads, unless the host already holds them, into `~/.cache/flare`. The
   copy is a sparse file of the package's length: the signed index, its rank's
   bundle and the tables every rank shares are written, and other ranks' storage
   takes no disk. With `--verify-weights` every peer receives the whole package,
   since verifying stored weights reads all of them. A received package or
   executable replaces older ones no running rank uses;
2. starts each peer's rank with the same serving options (concurrency,
   speculation, embedding-table storage), joining rank zero's rendezvous at port
   29800 (`--rendezvous 0.0.0.0:PORT` changes it);
3. loads its own rank while the peers load theirs;
4. exchanges RoCE queue-pair records with every peer, connects the transport,
   and serves once every rank is connected.

Peer output appears on rank zero's terminal prefixed with `[rank R host]`. The
first start copies the package; later starts with the same package and
executable copy nothing. Stop serving with Ctrl-C or SIGTERM on rank zero: the
ranks drain together, and each peer stops when rank zero's connection closes.

Without an agent, start each peer's rank yourself with identical serving options:

```sh
./flare/target/release/flare run glm53-spark-tp2.safetensors --rank 1 \
  --rendezvous 10.200.0.12:29800 --gpus 0
```

## Failures

- `agent token mismatch` or `cannot reach flare agent`: install the agent and
  the same token on that host, and open port 29900.
- `rank zero refused rank R`: the peer runs a different export or world size.
- `rank R cannot hold the checkpoint`: that peer lacks disk space for the
  checkpoint files it was missing, or a file arrived damaged.
- `tensor-parallel rank identity or graph I/O schema differs`: serving options
  differ between ranks.
- `package RoCE kernels use proxy ABI ...`: the package was prepared with a
  different FlashInfer RoCEnante than this Flare executable vendors; rebuild
  Flare or re-export.
- `RoCE transport setup failed` or `queue-pair connect failed`: an HCA the
  package names is missing or down on a host, or its GID index differs.
- A collective that times out on a peer poisons every rank's transport. The
  frame is not published, and the ranks shut down rather than serve wrong
  results.
