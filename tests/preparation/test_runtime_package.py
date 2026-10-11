"""Resource binding validation and cross-process native CUDA graph replay."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ignition.preparation.runtime_package import Allocation, Argument, Kernel, Reference, Relocation, RuntimePackage, _external_allocations


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
    document = json.loads((package_path / "package.json").read_text())
    for module in document["modules"]:
        assert b".py\0" not in (package_path / module["file"]).read_bytes()
        for symbol in module["globals"]:
            if symbol["symbol"].startswith("assertFile_"):
                assert (package_path / symbol["file"]).read_bytes().rstrip(b"\0") == b"kernel"
    if not exec_handover:
        result = subprocess.run([executable, str(tmp_path)],
                                check=True, cwd=root, text=True, capture_output=True, timeout=60)
    assert expected in result.stdout
