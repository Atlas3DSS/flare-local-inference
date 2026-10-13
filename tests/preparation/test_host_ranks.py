"""Rank zero admits each peer rank once, from its own host, running rank zero's sources."""

import threading

import pytest

from ignition.preparation import host_ranks
from ignition.preparation.host_ranks import HostRendezvous, knock


def topology(boot):
    return {"host": boot, "boot_id": boot, "rdma": [],
            "gpu": {"name": "NVIDIA GB10", "uuid": f"GPU-{boot}", "total_bytes": 1 << 37,
                    "free_bytes": 1 << 36, "capability": [12, 1]}}


def rendezvous(world=2):
    owner = HostRendezvous("127.0.0.1:0", world, identity="same", topology=topology("zero"))
    return owner, f"127.0.0.1:{owner.listener.getsockname()[1]}"


def join(address, rank, world=2, *, identity="same", boot=None, results=None):
    try:
        sockets = knock(address, rank, world, identity=identity, topology=topology(boot or f"peer{rank}"),
                        timeout=10)
        if results is not None:
            results.append(sockets)
    except RuntimeError as error:
        if results is not None:
            results.append(error)


def test_admits_every_peer_rank_with_its_executor_connection():
    owner, address = rendezvous(4)
    results = []
    peers = [threading.Thread(target=join, args=(address, rank, 4), kwargs={"results": results})
             for rank in (3, 1, 2)]
    for peer in peers:
        peer.start()
    topologies = owner.accept(timeout=10)
    for peer in peers:
        peer.join()
    assert [item["boot_id"] for item in topologies] == ["zero", "peer1", "peer2", "peer3"]
    assert owner.hosts == ("zero", "peer1", "peer2", "peer3")
    assert all(isinstance(result, tuple) for result in results)
    group = host_ranks.HostRankGroup.__new__(host_ranks.HostRankGroup)
    group.rendezvous = owner
    assert len(group.connections) == len(group.controls) == 3


@pytest.mark.parametrize("arguments, reason", [
    ({"identity": "other"}, "source identity"),
    ({"world": 4}, "world size"),
])
def test_rejects_peers_that_differ_from_rank_zero(arguments, reason):
    owner, address = rendezvous()
    results = []
    peer = threading.Thread(target=join, args=(address, 1), kwargs={**arguments, "results": results})
    peer.start()
    with pytest.raises(TimeoutError):
        owner.accept(timeout=2)
    peer.join()
    assert isinstance(results[0], RuntimeError) and reason in str(results[0])


def test_rejects_a_second_process_for_an_admitted_rank():
    owner, address = rendezvous(4)
    failures = []

    def admit():
        try:
            owner.accept(timeout=3)
        except TimeoutError as error:
            failures.append(error)

    accepting = threading.Thread(target=admit)
    accepting.start()
    results = []
    join(address, 1, 4, results=results)
    join(address, 1, 4, results=results)
    accepting.join()
    assert isinstance(results[0], tuple) and failures
    assert isinstance(results[1], RuntimeError) and "already joined" in str(results[1])


def test_requires_one_rank_per_host():
    owner, address = rendezvous()
    peer = threading.Thread(target=join, args=(address, 1), kwargs={"boot": "zero"})
    peer.start()
    with pytest.raises(ValueError, match="different host"):
        owner.accept(timeout=10)
    peer.join()


def test_finds_the_interface_holding_an_address():
    assert host_ranks.interface_of("127.0.0.1") == "lo"
    with pytest.raises(RuntimeError):
        host_ranks.interface_of("203.0.113.7")


def test_peer_rank_file_streams_over_its_control_connection_into_the_package(tmp_path):
    import socket
    import threading

    from ignition.preparation import runtime_package_pb2 as proto
    from ignition.preparation.host_ranks import _Received
    from ignition.preparation.package_file import extend_cluster, publish, read_index

    files = []
    for rank, data in enumerate((b"rank-zero" * 300, b"rank-one" * 700)):
        source = tmp_path / f"source-{rank}"
        source.write_bytes(data)
        description = proto.ModelIndex(version=1, ranks=[proto.BundleIndex()])
        description.ranks[0].resources["storage.bin"].CopyFrom(proto.Span(offset=0, nbytes=len(data)))
        files.append(tmp_path / f"rank-{rank}.safetensors")
        publish(files[-1], description, [(0, source, 0, len(data))])
    receiver, sender = socket.socketpair()
    size = files[1].stat().st_size
    thread = threading.Thread(target=lambda: sender.sendfile(files[1].open("rb")))
    thread.start()
    stream = _Received(receiver, size)
    extend_cluster(files[0], [stream], (2, "export-b"))
    thread.join()
    assert stream.remaining == 0 and stream.read(8) == b""
    index, start = read_index(files[0])
    span = index.ranks[1].resources["storage.bin"]
    with files[0].open("rb") as package:
        package.seek(start + span.offset)
        assert package.read(span.nbytes) == b"rank-one" * 700
