"""Resource binding validation and cross-process native CUDA graph replay."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from ignition.preparation.runtime_package import (
    Allocation, Argument, FileSource, Kernel, Reference, Relocation, RuntimePackage,
    _external_allocations, _merge_ranges, _without_zero_ranges, _write_allocation,
)


def package(*, allocations=None, arguments=None):
    return RuntimePackage(
        allocations=[Allocation("weight", 0x1000, 256)] if allocations is None else allocations,
        modules={"kernel": b"\x7fELFfixture"},
        kernels=[Kernel(1, "kernel", (Argument(8, (Relocation(0, Reference("weight")),)),)
                        if arguments is None else arguments)],
    )


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


def test_external_allocations_share_pages_without_reading_unallocated_bytes():
    allocations = _external_allocations([(0x1300, 512), (0x1F00, 512), (0x5800, 256)], 4096)
    assert [(a.address, a.nbytes) for a in allocations] == [(0x1000, 8192), (0x5000, 4096)]
    assert allocations[0].snapshot_spans == ((0x300, 512), (0xF00, 512))
    owner = package(allocations=allocations, arguments=(Argument(8),))
    assert owner._reference(0x1F00, 512) == {"allocation": "external-0", "offset": 0xF00}


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
    result = subprocess.run(arguments, check=True, cwd=root, text=True, capture_output=True, timeout=180)
    package_path = tmp_path / "package"
    from ignition.preparation.package_file import read_index

    index, start = read_index(package_path)
    document = index.ranks[0]
    def contents(key):
        span = document.resources[key]
        with package_path.open("rb") as source:
            source.seek(start + span.offset)
            return source.read(span.nbytes)
    for module in document.modules:
        assert b".py\0" not in contents(module.file)
        for symbol in module.globals:
            if symbol.symbol.startswith("assertFile_"):
                assert contents(symbol.file).rstrip(b"\0") == b"kernel"
    if not exec_handover:
        result = subprocess.run([executable, str(tmp_path)],
                                check=True, cwd=root, text=True, capture_output=True, timeout=60)
    assert expected in result.stdout
