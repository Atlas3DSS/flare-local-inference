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
        description = proto.ModelIndex(version=4, ranks=[proto.BundleIndex()])
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


def hub_snapshot(hub, files, *, revision="rev"):
    """A Hugging Face cache snapshot: LFS files link to SHA-256 blobs, others to Git object IDs."""
    import hashlib
    import os

    model = hub / "models--org--model"
    snapshot = model / "snapshots" / revision
    for name, data, kind in files:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if kind == "copy":
            target.write_bytes(data)
            continue
        blob = (hashlib.sha256(data).hexdigest() if kind == "lfs"
                else hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest())
        (model / "blobs").mkdir(parents=True, exist_ok=True)
        (model / "blobs" / blob).write_bytes(data)
        target.symlink_to(os.path.relpath(model / "blobs" / blob, target.parent))
    return snapshot


def share(snapshot, peer_snapshot, *, files=None):
    """Run rank zero's side and one peer's side of checkpoint sharing; returns (bytes sent, peer error)."""
    import socket
    from types import SimpleNamespace

    zero, peer = socket.socketpair()
    outcome = {}

    def hold():
        try:
            _, _, listed = host_ranks.receive_object(peer)
            host_ranks.hold_checkpoint(peer, peer_snapshot, listed if files is None else files)
        except BaseException as error:  # noqa: BLE001 - The test inspects the peer's failure.
            outcome["peer"] = error
            peer.close()

    thread = threading.Thread(target=hold)
    thread.start()
    try:
        outcome["sent"] = host_ranks.share_checkpoint([zero], snapshot, SimpleNamespace(revision="rev"))
    except (RuntimeError, ConnectionError) as error:
        outcome["zero"] = error
    thread.join()
    return outcome


def test_peers_receive_only_the_checkpoint_files_they_lack(tmp_path):
    import os

    config, licence = b'{"architectures": []}', b"terms"
    first, second = os.urandom(3 << 20), os.urandom((1 << 20) + 7)
    snapshot = hub_snapshot(tmp_path / "zero", [("config.json", config, "git"), ("tensors/a.safetensors", first, "lfs"),
                                                ("tensors/b.safetensors", second, "lfs"), ("LICENSES/x.txt", licence, "copy")])
    # The peer links one file already; another's blob is cached for a different revision.
    peer = hub_snapshot(tmp_path / "peer", [("tensors/a.safetensors", first, "lfs"), ("unrelated", second, "lfs")])
    (peer / "unrelated").unlink()
    outcome = share(snapshot, peer)
    assert "peer" not in outcome and "zero" not in outcome
    assert outcome["sent"] == len(config) + len(licence)
    for name, data in (("config.json", config), ("tensors/a.safetensors", first),
                       ("tensors/b.safetensors", second), ("LICENSES/x.txt", licence)):
        assert (peer / name).read_bytes() == data
    assert (peer / "tensors/b.safetensors").is_symlink() and (peer / "config.json").is_symlink()
    assert not (peer / "LICENSES/x.txt").is_symlink()
    assert not list(peer.parents[1].rglob("*.flare-*"))
    # A second share sends nothing.
    assert share(snapshot, peer)["sent"] == 0


def test_transferred_files_leave_the_page_cache_on_both_ends(tmp_path, monkeypatch):
    import os

    released = []
    monkeypatch.setattr(host_ranks, "release_page_cache",
                        lambda paths: released.append(sorted(path.name for path in paths)))
    data = os.urandom(1 << 16)
    snapshot = hub_snapshot(tmp_path / "zero", [("model.safetensors", data, "lfs"), ("config.json", b"{}", "git")])
    peer = hub_snapshot(tmp_path / "peer", [("config.json", b"{}", "git")])
    assert share(snapshot, peer)["sent"] == len(data)
    # The peer drops what it received; rank zero drops every file it may have read.
    assert sorted(released) == [["config.json", "model.safetensors"], ["model.safetensors"]]
    released.clear()
    # Nothing moved, so rank zero drops nothing.
    assert share(snapshot, peer)["sent"] == 0
    assert released == [[]]


def test_a_copied_file_counts_only_when_its_contents_match_its_blob(tmp_path):
    import os

    data = os.urandom(1 << 16)
    snapshot = hub_snapshot(tmp_path / "zero", [("model.safetensors", data, "lfs")])
    peer = hub_snapshot(tmp_path / "peer", [("model.safetensors", data, "copy")])
    assert share(snapshot, peer)["sent"] == 0 and not (peer / "model.safetensors").is_symlink()
    (peer / "model.safetensors").write_bytes(bytes(len(data)))
    assert share(snapshot, peer)["sent"] == len(data)
    assert (peer / "model.safetensors").is_symlink() and (peer / "model.safetensors").read_bytes() == data


def test_a_file_whose_contents_differ_from_its_blob_name_is_refused(tmp_path):
    import os

    snapshot = hub_snapshot(tmp_path / "zero", [("model.safetensors", os.urandom(4096), "lfs")])
    blob = (snapshot / "model.safetensors").resolve()
    blob.write_bytes(os.urandom(4096))
    peer = tmp_path / "peer" / "models--org--model" / "snapshots" / "rev"
    outcome = share(snapshot, peer)
    assert isinstance(outcome["peer"], ValueError) and "do not match" in str(outcome["peer"])
    assert "zero" in outcome
    assert not (peer.parents[1] / "blobs" / blob.name).exists()
    assert not list(peer.parents[1].rglob("*.flare-*")) and not (peer / "model.safetensors").exists()


def test_a_peer_without_space_or_given_an_escaping_path_fails_preparation(tmp_path, monkeypatch):
    import shutil
    from types import SimpleNamespace

    snapshot = hub_snapshot(tmp_path / "zero", [("model.safetensors", b"x" * 4096, "lfs")])
    peer = tmp_path / "peer" / "models--org--model" / "snapshots" / "rev"
    with monkeypatch.context() as patch:
        patch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=1 << 30))
        outcome = share(snapshot, peer)
    assert "GiB is free" in str(outcome["zero"]) and isinstance(outcome["peer"], OSError)
    outcome = share(snapshot, peer, files=(("../escape", 4096, None),))
    assert "outside its snapshot" in str(outcome["zero"]) and not (peer.parent / "escape").exists()


def test_multi_host_preparation_requires_a_hub_checkpoint(tmp_path):
    from types import SimpleNamespace

    owner, _ = rendezvous()
    owner.listener.close()
    with pytest.raises(ValueError, match="hub cache"):
        owner.prepare(tmp_path / "local-model", None, None, checkpoint=SimpleNamespace(revision="rev"))
