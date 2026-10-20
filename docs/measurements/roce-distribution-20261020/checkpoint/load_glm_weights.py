"""Four-host GLM TP4 loading benchmark using the real checkpoint and agent paths.

Run from each experiment checkout with its Python environment, rank, stream
count, experiment root, and rank-zero address. This loads all text/MTP weights
but does not tune, capture, export, or establish inference correctness.
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import socket
import sys
import time


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--rank", type=int, choices=range(4), required=True)
parser.add_argument("--streams", type=int, choices=range(1, 65), required=True)
parser.add_argument("--rendezvous", required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.root / "flare"))

from ignition.cli import environment
from ignition.cli.config import Configuration
from ignition.cli.preparation import Host

configuration = Configuration.load(args.root / "setup/config.toml")
gpu = Host.discover().gpus[0]
# Match prepare_rank's A16 expert loading with calibrated A4 prefill support.
environment.apply(configuration, (gpu,), (("B12X_W4A16_A4_PREFILL", "1"),))

import torch
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.checkpoint_ring import RingSource
from ignition.preparation.checkpoint_source import RootSource, RemoteSource, SourceServer, token
from ignition.preparation.host_ranks import (
    HostRendezvous, knock, local_topology, receive_object, send_object, source_identity,
)
from ignition.recipes.glm53.recipe import RECIPE
from ignition.recipes.glm53.weights import ModelWeights

torch.cuda.set_device(0)
identity, topology = source_identity(), local_topology(gpu)
key = token(configuration.get("preparation.checkpoint_token_file").value)
checkpoint = RECIPE.checkpoints[0]


def load(snapshot):
    started = time.monotonic()
    with torch.no_grad():
        weights = ModelWeights.load(inspect_indexed_checkpoint(snapshot), "cuda:0",
            shard=TensorParallelShard(args.rank, 4), mtp=True, expert_activation="a16",
            expand_expert_scales=False)
    return weights, dict(rank=args.rank, streams=args.streams,
                         load_seconds=time.monotonic() - started, allocated_bytes=weights.allocated_bytes)


def close_reader(reader):
    try:
        reader.connection.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    reader.thread.join()


if args.rank == 0:
    rendezvous = HostRendezvous(args.rendezvous, 4, identity=identity, topology=topology)
    topologies = rendezvous.accept(timeout=120)
    snapshot = (configuration.directory("hub_cache") /
                ("models--" + checkpoint.repository.replace("/", "--")) / "snapshots" / checkpoint.revision)
    with ExitStack() as owners:
        source = RootSource(snapshot, checkpoint)
        owners.callback(source.close)
        ring = RingSource(source, key, identity, topologies, streams=args.streams,
                          orbit_file=args.root / "setup/orbit.toml",
                          key_file=args.root / "setup/controller.key", cluster="ring")
        owners.callback(ring.close)
        readers, controls = [], []
        for rank in range(1, 4):
            peer = rendezvous.peers[rank]
            owners.callback(peer.executor.close)
            owners.callback(peer.control.close)
            controls.append(peer.control)
            send_object(peer.control, checkpoint)
            reader = SourceServer(source, peer.control, key, rank, identity, transport=ring.grant)
            readers.append(reader)
            reader.thread.start()
            owners.callback(close_reader, reader)
        for reader in readers:
            reader.wait_ready()
        for control in controls:
            send_object(control, "load")
        weights, result = load(snapshot)
        for reader in readers:
            reader.join()
        ring.join()
        results = [result, *(receive_object(control) for control in controls)]
        for control in controls:
            send_object(control, "close")
        print(json.dumps(dict(repository=checkpoint.repository, revision=checkpoint.revision,
                              tensor_parallel=4, mtp=True, expert_activation="a16", results=results)), flush=True)
else:
    control, executor = knock(args.rendezvous, args.rank, 4, identity=identity, topology=topology, timeout=120)
    with ExitStack() as owners:
        owners.callback(control.close)
        owners.callback(executor.close)
        admitted = receive_object(control)
        assert admitted == checkpoint
        source = RemoteSource(control, key, args.rank, identity, admitted,
                              directory=configuration.directory("export_staging"))
        owners.callback(source.close)
        with source.activate() as snapshot:
            assert receive_object(control) == "load"
            weights, result = load(snapshot)
        source.finish()
        result["checkpoint_source"] = source.report()
        print(json.dumps(result), flush=True)
        send_object(control, result)
        assert receive_object(control) == "close"
