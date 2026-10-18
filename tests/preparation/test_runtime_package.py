"""Resource binding validation and cross-process native CUDA graph replay."""

import ctypes as C
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ignition.preparation.runtime_package import (
    Allocation, Argument, FileSource, Kernel, Reference, Relocation, RuntimePackage,
    _external_allocations, _live_pieces, _merge_ranges, _without_zero_ranges, _write_allocation,
    _KernelParams, _kernel_argument_data,
)


def package(*, allocations=None, arguments=None):
    return RuntimePackage(
        allocations=[Allocation("weight", 0x1000, 256)] if allocations is None else allocations,
        modules={"kernel": b"\x7fELFfixture"},
        kernels=[Kernel(1, "kernel", (Argument(8, (Relocation(0, Reference("weight")),)),)
                        if arguments is None else arguments)],
    )


def test_kernel_arguments_preserve_abi_padding_and_allow_parameterless_kernels():
    packed = C.create_string_buffer(b"ab______12345678")
    size = C.c_size_t(16)
    extra = (C.c_void_p * 5)(1, C.addressof(packed), 2, C.addressof(size), None)
    layout = [(0, 2), (8, 8)]
    assert _kernel_argument_data(_KernelParams(extra=extra), layout) == [b"ab", b"12345678"]
    pointers = (C.c_void_p * 2)(C.addressof(packed), C.addressof(packed) + 8)
    assert _kernel_argument_data(_KernelParams(params=pointers), layout) == [b"ab", b"12345678"]
    assert _kernel_argument_data(_KernelParams(), []) == []
    with pytest.raises(ValueError, match="both"):
        _kernel_argument_data(_KernelParams(params=pointers, extra=extra), layout)
    with pytest.raises(ValueError, match="no launch arguments"):
        _kernel_argument_data(_KernelParams(), layout)
    size.value = 15
    with pytest.raises(ValueError, match="beyond"):
        _kernel_argument_data(_KernelParams(extra=extra), layout)


@pytest.mark.parametrize("tag,buffer,size_pointer", [(3, True, True), (1, False, True), (1, True, False)])
def test_invalid_packed_kernel_arguments_are_rejected(tag, buffer, size_pointer):
    packed = C.create_string_buffer(8)
    size = C.c_size_t(8)
    extra = (C.c_void_p * 5)(tag, C.addressof(packed) if buffer else None,
                            2, C.addressof(size) if size_pointer else None, None)
    with pytest.raises(ValueError):
        _kernel_argument_data(_KernelParams(extra=extra), [(0, 8)])


def test_views_must_share_one_registered_backing_storage():
    with pytest.raises(ValueError, match="shared backing storage"):
        package(allocations=[Allocation("weight", 0x1000, 512), Allocation("view", 0x1100, 256)])
    owner = package(arguments=(Argument(8, (Relocation(0, Reference("weight", 128)),)),))
    assert owner._reference(0x1080, 128) == {"allocation": "weight", "offset": 128}
    with pytest.raises(ValueError, match="unregistered CUDA span"):
        owner._reference(0x1080, 129)


def test_peer_allocations_name_recorded_storage_without_payload():
    peer = Allocation("peer-0", 0x10000, 4096, alignment=4096, initialize="zero", preserve_address=True, memory="peer")
    owner = package(allocations=[Allocation("weight", 0x1000, 256), peer])
    assert owner._reference(0x10800, 64) == {"allocation": "peer-0", "offset": 0x800}
    for invalid in ({"initialize": "snapshot"}, {"preserve_address": False}):
        with pytest.raises(ValueError, match="another rank's storage"):
            package(allocations=[Allocation("weight", 0x1000, 256), Allocation(
                "peer-0", 0x10000, 4096, alignment=4096, memory="peer",
                **{"initialize": "zero", "preserve_address": True, **invalid})])


@pytest.mark.parametrize("relocations", [
    (Relocation(0, Reference("absent")),),
    (Relocation(0, Reference("weight", 256)),),
    (Relocation(1, Reference("weight")),),
    (Relocation(0, Reference("weight")), Relocation(0, Reference("weight"))),
    (Relocation(0, Reference("weight"), "tensor_map"),),
])
def test_invalid_reference_or_relocation_extent_is_rejected(relocations):
    with pytest.raises(ValueError):
        package(arguments=(Argument(8, relocations),))


def test_declared_pointer_must_match_live_binding():
    owner = package()
    relocation = Relocation(0, Reference("weight"))
    owner._check_bindings(None, (0x1000).to_bytes(8, "little"), [relocation])
    with pytest.raises(ValueError, match="pointer binding does not match"):
        owner._check_bindings(None, (0x1080).to_bytes(8, "little"), [relocation])


def test_export_never_overwrites_an_existing_directory(tmp_path):
    with pytest.raises(FileExistsError):
        package().write(tmp_path, {})


@pytest.mark.parametrize("offset,nbytes,closed,accepted", [
    (0, 2 << 20, False, True),
    (4096, 4096, False, True),
    (0, 2 << 20, True, False),
    (0, (2 << 20) + 4096, False, False),
    (-4096, 8192, False, False),
    (2 << 20, 4096, False, False),
])
def test_export_classifies_recorded_host_vmm_by_declared_ownership(
        tmp_path, monkeypatch, offset, nbytes, closed, accepted):
    from types import SimpleNamespace

    from ignition.preparation import runtime_package, tensor_metadata

    base = 0xc0000000000
    calls = []

    class Cuda:
        def call(self, name, *args):
            if name == "cuCtxGetCurrent":
                args[0]._obj.value = 17
            elif name == "cuPointerGetAttribute":
                calls.append(args[2].value)
                # Host-backed VMM has host memory type and no owning context.
                args[0]._obj.value = 1 if args[1].value == 2 else None

    class Classified(Exception):
        pass

    def classified(allocations, *args):
        host, = allocations
        assert (host.address, host.nbytes, host.memory) == (base, 2 << 20, "mapped_host")
        raise Classified

    monkeypatch.setattr(runtime_package, "_Cuda", Cuda)
    monkeypatch.setattr(tensor_metadata, "live_metadata", classified)
    prepared = SimpleNamespace(_consumed=False, signatures={}, export_zero_ranges=[], buffers={})
    storage = SimpleNamespace(error=None, dynamic_regions=[], transports=[], mappings={}, peer_storage=None,
        host_regions=[SimpleNamespace(address=base, extent=2 << 20, closed=closed)], granularity=2 << 20,
        unused_ranges=lambda: [],
        graph_pool=SimpleNamespace(snapshot=lambda **kwargs: []), numeric_storage=None)
    images = SimpleNamespace(allocations={base + offset: (nbytes, "device")},
                             allocation_contexts={base + offset: 17})
    if accepted:
        with pytest.raises(Classified):
            runtime_package.export_model(prepared, images, storage, tmp_path / "package", options={})
        assert calls == []
    else:
        with pytest.raises(ValueError, match="not device memory in the serving context"):
            runtime_package.export_model(prepared, images, storage, tmp_path / "package", options={})
        assert calls == [base + offset, base + offset]


def test_peer_export_keeps_shared_frontend_only_on_rank_zero(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace

    from ignition.preparation import memory_inventory, runtime_package

    monkeypatch.setattr(memory_inventory, "collect_inventory", lambda *args, **kwargs: {})
    from ignition.preparation.package_file import extend_cluster, read_index
    from ignition.recipes.declaration import SamplingDefaults

    class Cuda:
        lib = SimpleNamespace(cuCtxGetLimit=lambda *args: 215, cuFuncGetParamInfo=lambda *args: 1)

        def call(self, name, *args):
            if name == "cuGraphGetNodes":
                args[2]._obj.value = 1
                if args[1] is not None:
                    args[1][0] = 1
            elif name == "cuGraphKernelNodeGetParams_v2":
                args[1]._obj.func = 1
            elif name == "cuFuncGetName":
                args[0]._obj.value = b"fixture"

    cuda = Cuda()
    monkeypatch.setattr(runtime_package, "_Cuda", lambda: cuda)
    monkeypatch.setattr(RuntimePackage, "_graph", lambda self, cuda, name, graph:
                        {"name": name, "nodes": [], "edges": []})
    from ignition.preparation import cubin, module_images

    monkeypatch.setattr(module_images, "strip_debug", lambda image: image)
    monkeypatch.setattr(cubin, "redact_symbols", lambda image: (image, {"fixture": "_f0"}))
    monkeypatch.setattr(module_images, "global_symbols", lambda image: [])
    assets = {}
    for name in ("tokenizer", "chat_template"):
        path = tmp_path / name
        path.write_text(name)
        assets[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    graph = SimpleNamespace(raw_cuda_graph=lambda: 1, raw_cuda_graph_exec=lambda: 1)
    address = 0x100000000000
    prepared = SimpleNamespace(_consumed=False, signatures={"decode": SimpleNamespace(kind="decode")},
        export_zero_ranges=[(address, 4096)], graphs={"decode": graph},
        buffers={}, export_events={}, package_data=None,
        server_config={"model_id": "fixture", "assets": assets,
                       "sampling_defaults": SamplingDefaults(temperature=1., top_p=.95, top_k=-1).as_dict(),
                       "frontend": {"eos_token_ids": [17]}},
        manifest=lambda **kwargs: {"context": {}, "device": 0, "buffers": [], "graphs": []})
    storage = SimpleNamespace(error=None, dynamic_regions=[], transports=[], integrated=False,
        mappings={address: SimpleNamespace(address=address, nbytes=4096)},
        peer_storage=None, host_regions=[], granularity=4096, DEVICE_BASE=0x100000000000,
        RESERVATION_BYTES=1 << 20, unused_ranges=lambda: [],
        graph_pool=SimpleNamespace(snapshot=lambda **kwargs: []),
        numeric_storage=SimpleNamespace(ranges=lambda: []))
    images = SimpleNamespace(allocations={}, image_for=lambda cuda, function: b"\x7fELFfixture")
    paths = [tmp_path / f"rank-{rank}.safetensors" for rank in range(2)]
    for rank, path in enumerate(paths):
        runtime_package.export_model(prepared, images, storage, path, options={}, bundle_only=rank > 0)
    peer, _ = read_index(paths[1])
    assert not peer.assets and not peer.HasField("contract")
    assert not peer.HasField("frontend") and not peer.HasField("sampling_defaults")
    assert [graph.name for graph in peer.ranks[0].graphs] == ["decode"]
    with paths[1].open("rb") as stream:
        extend_cluster(paths[0], [stream], (2, "fixture"))
    combined, _ = read_index(paths[0])
    assert len(combined.ranks) == 2
    assert {asset.name for asset in combined.assets} == {"tokenizer", "chat_template"}
    assert combined.contract.model_id == "fixture"
    assert list(combined.frontend.eos_token_ids) == [17]
    assert combined.sampling_defaults.temperature == 1.
    assert prepared.server_config["assets"] == assets


def test_external_allocations_share_pages_without_reading_unallocated_bytes():
    allocations = _external_allocations([(0x1300, 512), (0x1F00, 512), (0x5800, 256)], 4096)
    assert [(a.address, a.nbytes) for a in allocations] == [(0x1000, 8192), (0x5000, 4096)]
    assert allocations[0].snapshot_spans == ((0x300, 512), (0xF00, 512))
    owner = package(allocations=allocations, arguments=(Argument(8),))
    assert owner._reference(0x1F00, 512) == {"allocation": "external-0", "offset": 0xF00}


@pytest.mark.parametrize("dependency", ["unused", "direct", "indirect"])
def test_export_rejects_reachable_storage_below_the_address_floor(tmp_path, monkeypatch, dependency):
    from types import SimpleNamespace
    from ignition.preparation import memory_inventory, module_images, runtime_package, tensor_metadata
    monkeypatch.setattr(memory_inventory, "collect_inventory", lambda *args, **kwargs: {})
    from ignition.preparation.package_file import read_index

    low, high, floor = 0x1208400000, 0x40000000000, 0x4000000000

    class Cuda:
        lib = SimpleNamespace(cuCtxGetLimit=lambda *args: 215, cuFuncGetParamInfo=lambda *args: 1)

        def call(self, name, *args):
            if name == "cuCtxGetCurrent":
                args[0]._obj.value = 17
            elif name == "cuPointerGetAttribute":
                args[0]._obj.value = 2 if args[1].value == 2 else 17
            elif name == "cuGraphGetNodes":
                args[2]._obj.value = 1
                if args[1] is not None:
                    args[1][0] = 1
            elif name == "cuGraphKernelNodeGetParams_v2":
                args[1]._obj.func = 1
            elif name == "cuFuncGetName":
                args[0]._obj.value = b"fixture"

        def download(self, address, size):
            assert address == high and size == 256
            return (low if dependency == "indirect" else 0).to_bytes(8, "little") + bytes(248)

    monkeypatch.setattr(runtime_package, "_Cuda", Cuda)
    monkeypatch.setattr(RuntimePackage, "_graph", lambda self, cuda, name, graph:
                        {"name": name, "nodes": [], "edges": []})
    monkeypatch.setattr(module_images, "strip_debug", lambda image: image)
    from ignition.preparation import cubin
    monkeypatch.setattr(cubin, "redact_symbols", lambda image: (image, {"fixture": "_f0"}))
    monkeypatch.setattr(module_images, "global_symbols", lambda image: [])
    monkeypatch.setattr(tensor_metadata, "live_metadata", lambda *args: ([], {}))
    graph = SimpleNamespace(raw_cuda_graph=lambda: 1, raw_cuda_graph_exec=lambda: 1)
    prepared = SimpleNamespace(_consumed=False, signatures={"decode": SimpleNamespace(kind="decode")},
        export_zero_ranges=[], graphs={"decode": graph},
        buffers={}, export_events={}, package_data=None, server_config={},
        manifest=lambda **kwargs: {"context": {}, "device": 0, "graphs": [], "buffers": [
            {"name": "root", "address": low if dependency == "direct" else high, "nbytes": 8}]})
    storage = SimpleNamespace(error=None, dynamic_regions=[], transports=[], integrated=True,
        mappings={high: SimpleNamespace(address=high, nbytes=256)}, peer_storage=None,
        host_regions=[], granularity=256, EXTERNAL_FLOOR=floor,
        unused_ranges=lambda: [],
        graph_pool=SimpleNamespace(snapshot=lambda **kwargs: []), numeric_storage=None)
    images = SimpleNamespace(allocations={low: (256, "device")}, allocation_contexts={low: 17},
                             image_for=lambda cuda, function: b"\x7fELFfixture")
    path = tmp_path / "package.safetensors"

    def export():
        return runtime_package.export_model(prepared, images, storage, path, options={}, bundle_only=True)

    if dependency == "unused":
        export()
        index, _ = read_index(path)
        assert [a.name for a in index.ranks[0].allocations] == ["device-0"]
        assert all(reservation.address >= floor for reservation in index.ranks[0].reservations)
    else:
        with pytest.raises(ValueError, match="below the restorable address floor"):
            export()
        assert not path.exists()
        assert not list(tmp_path.iterdir())


def test_disposable_spans_preserve_neighbors_and_external_page_gaps():
    zero = _merge_ranges(((0x1080, 32), (0x1090, 64), (0x1180, 128)))
    allocation = _without_zero_ranges(Allocation("mixed", 0x1000, 512), zero)
    assert allocation.snapshot_spans == ((0, 128), (208, 176))
    external = Allocation("external", 0x1000, 512, snapshot_spans=((64, 128), (384, 64)))
    assert _without_zero_ranges(external, zero).snapshot_spans == ((64, 64),)
    assert _without_zero_ranges(external, ((0x1000, 512),)).initialize == "zero"


@pytest.mark.parametrize("nonzero", [False, True])
def test_packed_payload_elides_zero_blocks_without_changing_contents(tmp_path, nonzero):
    block = 64 << 10
    source = bytearray(4 * block + 17)
    if nonzero:
        source[block + 3] = 7
        source[3 * block + 8] = 11
        source[-1] = 19
    class Cuda:
        def download(self, address, nbytes):
            return bytes(source[address - 0x1000:address - 0x1000 + nbytes])
    file = tmp_path / "data.bin"
    name, spans = _write_allocation(Cuda(), Allocation("data", 0x1000, len(source)), file)
    if not nonzero:
        assert name is None and spans == [] and not file.exists()
        return
    assert spans == [[block, block], [3 * block, block + 17]]
    assert file.stat().st_size == 2 * block + 17
    rebuilt = bytearray(len(source))
    with file.open("rb") as payload:
        for offset, size in spans:
            rebuilt[offset:offset + size] = payload.read(size)
    assert rebuilt == source


def test_payload_writes_flush_in_bounded_batches(tmp_path, monkeypatch):
    from ignition.preparation import runtime_package

    synced, dropped = [], []
    monkeypatch.setattr(runtime_package, "_WRITE_BACK_BYTES", 16 << 20)
    monkeypatch.setattr(runtime_package.os, "fsync" if os.name == "nt" else "fdatasync",
                        lambda fd: synced.append(fd))
    if os.name != "nt":
        monkeypatch.setattr(runtime_package.os, "posix_fadvise",
                            lambda fd, offset, length, advice: dropped.append((offset, length, advice)))

    class Cuda:
        def download(self, address, nbytes):
            return bytes([5]) * nbytes
    file = tmp_path / "weights.bin"
    _, spans = _write_allocation(Cuda(), Allocation("weights", 0x1000, 40 << 20), file)
    # Flush after 16 and 32 MiB and at the end.
    assert spans == [[0, 40 << 20]] and file.stat().st_size == 40 << 20
    assert len(synced) == 3
    if os.name != "nt":
        assert dropped == [(0, 0, os.POSIX_FADV_DONTNEED)] * 3


def test_declared_zero_ranges_are_not_downloaded(tmp_path):
    class Cuda:
        def download(self, address, nbytes):
            assert (address, nbytes) in ((0x1000, 64), (0x1100, 64))
            return bytes([13]) * nbytes
    allocation = _without_zero_ranges(Allocation("mixed", 0x1000, 320), ((0x1040, 192),))
    file = tmp_path / "mixed.bin"
    _, spans = _write_allocation(Cuda(), allocation, file)
    assert spans == [[0, 64], [256, 64]] and file.read_bytes() == bytes([13]) * 128


def test_checkpoint_source_requires_a_complete_nonempty_file_range(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"header" + bytes(128))
    assert FileSource(checkpoint, 6, 128).nbytes == 128
    for offset, size in ((-1, 1), (6, 0), (6, 129)):
        with pytest.raises(ValueError, match="within its checkpoint"):
            FileSource(checkpoint, offset, size)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires explicit CUDA test opt-in")
def test_graph_export_preserves_construction_order_across_branches():
    import ctypes as C
    import torch
    from ignition.preparation.runtime_package import _Cuda, _MemsetParams

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign CUDA_VISIBLE_DEVICES before running GPU tests"
    data = torch.empty(256, dtype=torch.uint8, device="cuda:0")
    cuda = _Cuda()
    graph, context = C.c_void_p(), C.c_void_p()
    cuda.call("cuCtxGetCurrent", C.byref(context))
    cuda.call("cuGraphCreate", C.byref(graph), C.c_uint(0))
    try:
        nodes = []
        for value in (17, 29, 41):
            params = _MemsetParams(dst=data.data_ptr() + len(nodes) * 64, pitch=0,
                                   value=value, element_size=1, width=64, height=1)
            node = C.c_void_p()
            dependencies = (C.c_void_p * 1)(nodes[0].value) if nodes else None
            cuda.call("cuGraphAddMemsetNode", C.byref(node), graph, dependencies,
                      C.c_size_t(int(bool(nodes))), C.byref(params), context)
            nodes.append(node)
        owner = package(allocations=[Allocation("weight", data.data_ptr(), data.nbytes)])
        description = owner._graph(cuda, "branches", graph.value)
        assert [node["value"] for node in description["nodes"]] == [17, 29, 41]
        assert {(edge["source"], edge["destination"]) for edge in description["edges"]} == {(0, 1), (0, 2)}
    finally:
        cuda.call("cuGraphDestroy", graph)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires explicit CUDA test opt-in")
@pytest.mark.parametrize("fixture,binary,expected,exec_handover", [
    ("runtime_package_fixture.py", "test_runtime_package", "192 graph launches", False),
    ("fixed_package_fixture.py", "test_fixed_runtime_package", "native fixed-address replay passed", False),
    ("fixed_package_fixture.py", "test_fixed_runtime_package", "native fixed-address replay passed", True),
])
def test_package_replays_without_python(tmp_path, fixture, binary, expected, exec_handover):
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign CUDA_VISIBLE_DEVICES before running GPU tests"
    root = Path(__file__).resolve().parents[2]
    subprocess.run([
        "cargo", "build", "--manifest-path", str(root / "helios/Cargo.toml"),
        "--no-default-features", "--features", "runtime-package,test-tools", "--bin", binary,
    ], check=True, cwd=root, timeout=180)
    executable = str(root / "helios/target/debug" / binary)
    arguments = [sys.executable, str(root / "helios/tests" / fixture), str(tmp_path)]
    if exec_handover:
        arguments.append(executable)
    result = subprocess.run(arguments, cwd=root, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    package_path = tmp_path / "package"
    from ignition.preparation.package_file import read_index

    index, start = read_index(package_path)
    from safetensors import safe_open

    with safe_open(package_path, framework="numpy") as tensors:
        assert tensors.keys() == ["flare.metadata", "flare.payload"]
        from ignition.preparation import runtime_package_pb2 as proto
        protected = proto.ProtectedIndex.FromString(tensors.get_tensor("flare.metadata").tobytes())
        shared = proto.PublisherMetadata.FromString(protected.groups[0].body).index
        signed = proto.RankProtection.FromString(protected.ranks[0].body)
        shared.ranks.add().CopyFrom(signed.bundle)
        assert shared == index
    document = index.ranks[0]
    storage = {allocation.file for allocation in document.allocations if allocation.HasField("file")}
    storage_end = max((document.resources[key].offset + document.resources[key].nbytes for key in storage), default=0)
    assert all(span.offset >= storage_end for key, span in document.resources.items() if key not in storage)
    def contents(key):
        span = document.resources[key]
        with package_path.open("rb") as source:
            source.seek(start + span.offset)
            stored = source.read(span.nbytes)
        resource = next((item for item in signed.executables.resources if item.offset == span.offset), None)
        if resource is None:
            return stored
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from helios.tests.signing import publisher
        from ignition.preparation.protection import CHUNK, chunk_aad, resource_key
        signer = publisher()
        cipher = AESGCM(resource_key(signer.public_key, protected.package_id, span.offset, signer.namespace))
        result, cursor = bytearray(), 0
        for chunk in range(len(resource.sha256)):
            size = min(CHUNK, resource.plain_bytes - len(result)) + 16
            result.extend(cipher.decrypt(chunk.to_bytes(12, 'big'), stored[cursor:cursor + size],
                chunk_aad(protected.package_id, span.offset, resource.plain_bytes, chunk)))
            cursor += size
        return bytes(result)
    for module in document.modules:
        assert b".py\0" not in contents(module.file)
        for symbol in module.globals:
            assert b"/preparation/internal/kernel.py" not in contents(symbol.file)
    assert all(function.symbol.startswith("_f") for function in document.functions)
    if fixture == "fixed_package_fixture.py":
        assert any(contents(symbol.file).rstrip(b"\0") == b"kernel"
                   for module in document.modules for symbol in module.globals)
        for name in (b"numeric_argument", b"integer_pointer_argument", b"pointer_table_argument",
                     b"transform", b"no_arguments", b"assertFile_0"):
            assert name not in document.SerializeToString()
            assert all(name + b"\0" not in contents(module.file) for module in document.modules)
    if not exec_handover:
        result = subprocess.run([executable, str(tmp_path)],
                                cwd=root, text=True, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    assert expected in result.stdout


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires explicit CUDA test opt-in")
def test_cooperative_package_clamps_to_device_capacity(tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign CUDA_VISIBLE_DEVICES before running GPU tests"
    root = Path(__file__).resolve().parents[2]
    subprocess.run([
        sys.executable, str(root / "helios/tests/cooperative_package_fixture.py"), str(tmp_path),
    ], check=True, cwd=root, timeout=120)
    subprocess.run([
        "cargo", "test", "--manifest-path", str(root / "helios/Cargo.toml"),
        "--no-default-features", "--features", "runtime-package", "--lib",
        "oversized_cooperative_grids_replay_at_device_capacity", "--", "--ignored",
    ], check=True, cwd=root, timeout=180,
        env={**os.environ, "FLARE_COOPERATIVE_TEST_PACKAGE": str(tmp_path)})


def test_only_allocator_inactive_granules_are_released():
    allocation = Allocation("weights", 0x80000000000, 66 << 20, alignment=2 << 20, preserve_address=True)
    begin, end = allocation.address, allocation.address + allocation.nbytes
    live = begin + (51 << 20) + (128 << 10)
    trimmed, = _live_pieces(allocation, [(live, end)])
    assert (trimmed.name, trimmed.address, trimmed.nbytes) == ("weights", begin, 52 << 20)
    # Freed blocks between live ones release their whole granules; partial granules stay backed.
    pieces = _live_pieces(allocation, [(begin + (3 << 20), begin + (9 << 20)),
                                       (begin + (9 << 20) + 256, begin + (20 << 20)), (end - (1 << 20), end)])
    assert [(p.name, p.address - begin, p.nbytes) for p in pieces] == [
        ("weights.0", 0, 4 << 20), ("weights.1", 8 << 20, 2 << 20), ("weights.2", 20 << 20, 46 << 20)]
    leading, = _live_pieces(allocation, [(begin - (8 << 20), begin + (5 << 20))])
    assert (leading.address - begin, leading.nbytes) == (4 << 20, 62 << 20)
    assert _live_pieces(allocation, [(begin, end)]) == (allocation,)
    assert _live_pieces(allocation, []) == (allocation,)
    # Disposable live scratch has no initializer, but still needs physical backing.
    zero = _without_zero_ranges(allocation, [(allocation.address, allocation.nbytes)])
    assert zero.initialize == "zero"
    assert _live_pieces(zero, []) == (zero,)
