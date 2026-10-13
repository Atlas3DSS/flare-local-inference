"""Safetensors interoperability, bounded metadata reads, and stable storage placement."""

import json
from pathlib import Path
import struct

import pytest
from safetensors import safe_open

from ignition.preparation import runtime_package_pb2 as proto
from ignition.preparation.package_file import (
    ALIGNMENT, MAX_HEADER_BYTES, MAX_INDEX_BYTES, align, combine, publish, read_index,
)


def index():
    return proto.ModelIndex(version=1, ranks=[proto.BundleIndex()])


def header(payload_size, index_size):
    return {
        "__metadata__": {"format": "flare"},
        "flare.payload": {"dtype": "U8", "shape": [payload_size], "data_offsets": [0, payload_size]},
        "flare.metadata": {"dtype": "U8", "shape": [index_size],
                           "data_offsets": [payload_size, payload_size + index_size]},
    }


def write_container(path, description, payload, metadata):
    encoded = json.dumps(description, separators=(",", ":")).encode()
    encoded = encoded.ljust(align(8 + len(encoded)) - 8, b" ")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload + metadata)


@pytest.mark.parametrize("data", [b"", b"weights" + bytes(17)])
def test_official_safetensors_reader_sees_both_byte_tensors(tmp_path, data):
    source = tmp_path / "storage"
    source.write_bytes(data)
    description = index()
    description.ranks[0].resources["storage"].CopyFrom(proto.Span(offset=0, nbytes=len(data)))
    path = tmp_path / "model.safetensors"
    publish(path, description, [(0, source, 0, len(data))])
    with safe_open(path, framework="numpy") as tensors:
        assert tensors.keys() == ["flare.metadata", "flare.payload"]
        assert tensors.metadata() == {"format": "flare"}
        assert tensors.get_tensor("flare.payload").tobytes() == data
        assert proto.ModelIndex.FromString(tensors.get_tensor("flare.metadata").tobytes()) == description
    assert read_index(path) == (description, ALIGNMENT)


def test_metadata_growth_preserves_payload_bytes_and_offsets(tmp_path):
    source = tmp_path / "storage"
    source.write_bytes(b"unchanged weights")
    description = index()
    description.ranks[0].resources["storage"].CopyFrom(proto.Span(offset=4096, nbytes=source.stat().st_size))
    packages = []
    for generation in range(2):
        path = tmp_path / f"model-{generation}.safetensors"
        description.server.update({"description": "x" * (20000 * generation)})
        publish(path, description, [(4096, source, 0, source.stat().st_size)])
        packages.append(path)
    assert [read_index(path)[1] for path in packages] == [ALIGNMENT, ALIGNMENT]
    with safe_open(packages[0], framework="numpy") as first, safe_open(packages[1], framework="numpy") as second:
        assert first.get_tensor("flare.payload").tobytes() == second.get_tensor("flare.payload").tobytes()
        assert first.get_slice("flare.metadata").get_shape() != second.get_slice("flare.metadata").get_shape()


def test_index_read_does_not_touch_the_payload(tmp_path, monkeypatch):
    description = index().SerializeToString()
    path = tmp_path / "model.safetensors"
    write_container(path, header(2 << 20, len(description)), bytes(2 << 20), description)
    reads = []
    original = Path.open

    class Reader:
        def __init__(self, source):
            self.source = source
        def __getattr__(self, name):
            return getattr(self.source, name)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.source.close()
        def read(self, size=-1):
            reads.append((self.source.tell(), size))
            return self.source.read(size)

    monkeypatch.setattr(Path, "open", lambda self, *args, **kwargs: Reader(original(self, *args, **kwargs)))
    assert read_index(path) == (index(), ALIGNMENT)
    assert reads == [(0, 8), (8, ALIGNMENT - 8), (ALIGNMENT + (2 << 20), len(description))]


@pytest.mark.parametrize("world", [1, 2, 4])
def test_tp_weights_and_tables_precede_all_execution_data(tmp_path, world):
    outputs = []
    for generation in range(2):
        ranks = []
        for rank in range(world):
            contents = [b"graph" * (1 + generation * 2000), f"weights-{rank}".encode(), b"embedding-table"]
            source = tmp_path / f"source-{generation}-{rank}"
            source.write_bytes(b"".join(contents))
            description = index()
            bundle = description.ranks[0]
            bundle.allocations.add(name="weights", file="storage.bin")
            bundle.graphs.add(name="decode", resource="graph.bin")
            cursor = 0
            for key, data in zip(("graph.bin", "storage.bin"), contents):
                bundle.resources[key].CopyFrom(proto.Span(offset=cursor, nbytes=len(data)))
                cursor += len(data)
            if rank == 0:
                table = description.engram.tables.add(layer=1)
                table.weights.CopyFrom(proto.Span(offset=cursor, nbytes=len(contents[2])))
            path = tmp_path / f"rank-{generation}-{rank}.safetensors"
            publish(path, description, [(0, source, 0, source.stat().st_size)])
            ranks.append(path)
        output = tmp_path / f"model-{generation}.safetensors"
        combine(output, ranks, [])
        description, start = read_index(output)
        storage = [bundle.resources["storage.bin"] for bundle in description.ranks]
        storage.append(description.engram.tables[0].weights)
        graphs = [bundle.resources["graph.bin"] for bundle in description.ranks]
        assert max(span.offset + span.nbytes for span in storage) <= min(span.offset for span in graphs)
        with output.open("rb") as source:
            for rank, bundle in enumerate(description.ranks):
                for name, expected in (("storage.bin", f"weights-{rank}".encode()),
                                       ("graph.bin", b"graph" * (1 + generation * 2000))):
                    span = bundle.resources[name]
                    source.seek(start + span.offset)
                    assert source.read(span.nbytes) == expected
            source.seek(start)
            outputs.append(source.read(min(span.offset for span in graphs)))
        with safe_open(output, framework="numpy") as tensors:
            assert tensors.get_slice("flare.payload").get_shape()[0] >= graphs[-1].offset + graphs[-1].nbytes
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("field,value", [
    (("__metadata__", "format"), "pt"),
    (("flare.payload", "dtype"), "F32"),
    (("flare.payload", "shape"), [2, 2]),
    (("flare.payload", "shape"), [True]),
    (("flare.payload", "data_offsets"), [1, 5]),
    (("flare.payload", "data_offsets"), [-1, 3]),
    (("flare.metadata", "data_offsets"), [3, 7]),
    (("flare.metadata", "data_offsets"), [4, 1 << 64]),
    (("flare.metadata", "shape"), [MAX_INDEX_BYTES + 1]),
])
def test_invalid_safetensors_header_is_rejected(tmp_path, field, value):
    description = index().SerializeToString()
    container = header(4, len(description))
    container[field[0]][field[1]] = value
    path = tmp_path / "invalid.safetensors"
    write_container(path, container, b"data", description)
    with pytest.raises(ValueError):
        read_index(path)


def test_resources_cannot_address_protobuf_bytes(tmp_path):
    description = index()
    description.ranks[0].resources["outside"].CopyFrom(proto.Span(offset=4, nbytes=1))
    encoded = description.SerializeToString()
    path = tmp_path / "invalid.safetensors"
    write_container(path, header(4, len(encoded)), b"data", encoded)
    with pytest.raises(ValueError, match="resource exceeds"):
        read_index(path)


@pytest.mark.parametrize("prefix", [b"", struct.pack("<Q", 0), struct.pack("<Q", MAX_HEADER_BYTES + 1)])
def test_header_size_is_bounded_before_reading(tmp_path, prefix):
    path = tmp_path / "invalid.safetensors"
    path.write_bytes(prefix)
    with pytest.raises(ValueError):
        read_index(path)


def test_publication_failure_and_overwrite_leave_files_untouched(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"data")
    output = tmp_path / "model.safetensors"
    with pytest.raises(ValueError, match="truncated package resource"):
        publish(output, index(), [(0, source, 0, 5)])
    assert list(tmp_path.iterdir()) == [source]
    publish(output, index(), [(0, source, 0, 4)])
    saved = output.read_bytes()
    with pytest.raises(FileExistsError):
        publish(output, index(), [])
    assert output.read_bytes() == saved
