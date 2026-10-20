"""Engram export and resident delivery use identical interleaved checkpoint rows."""

from types import SimpleNamespace

import pytest
from b12x.sequence import engram

from ignition.components.parallel import TensorParallelShard
from ignition.preparation.runtime_package import FileSource
from ignition.preparation import runtime_package_pb2 as proto
from ignition.recipes.deepseek41.engram import Tables


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_package_planes_and_resident_segments_preserve_head_bytes(tmp_path, world):
    geometry = engram.build_geometry(layer_ids=(1,), base_table_size=101, compressed_vocab_size=32)
    rows = geometry.num_embeddings[0]
    sources, contents = [], []
    for width in (256, 8):
        data = bytes((row * 13 + column) % 251 for row in range(rows) for column in range(width))
        path = tmp_path / str(width)
        path.write_bytes(bytes(37) + data)
        sources.append(FileSource(path, 37, len(data)))
        contents.append(data)
    tables = Tables(geometry, tuple(range(32)), {1: tuple(sources)}, {}, {}, None,
                    TensorParallelShard(0, world), {}, None)
    index, payload = SimpleNamespace(engram=proto.Engram()), []
    tables.append(index, payload, 0)
    table = index.engram.tables[0]
    assert list(table.hash.primes) == list(geometry.primes[0])
    assert list(table.hash.multipliers) == list(geometry.multipliers[0])
    global_row = 0
    for rank in range(world):
        partition = tables.partition(0, rank)
        stored = table if world == 1 else table.shards[rank]
        if world > 1:
            assert (stored.first_row, stored.rows) == (global_row, partition.rows)
        for head in partition.heads:
            assert table.hash.offsets[head] == global_row + partition.row_mapping[head][2]
        rank_tables = Tables(geometry, tables.token_map, tables.sources, {}, {}, None,
                             TensorParallelShard(rank, world), {}, None)
        resident = rank_tables.resident_catalog()["tables"][0]["planes"]
        for ordinal, (name, width, source) in enumerate(zip(("weights", "scales"), (256, 8), contents)):
            span = getattr(stored, name)
            chunks = [(offset, path, first, nbytes) for offset, path, first, nbytes in payload
                      if span.offset <= offset < span.offset + span.nbytes]
            actual = b"".join(path.read_bytes()[first:first + nbytes] for _, path, first, nbytes in chunks)
            expected = b"".join(source[first * width:last * width] for first, last in partition.source_rows)
            assert actual == expected and len(actual) == span.nbytes
            assert resident[ordinal]["segments"] == [
                {"path": str(path), "offset": first, "nbytes": nbytes}
                for _, path, first, nbytes in chunks]
        global_row += partition.rows
    assert global_row == rows
    assert sum(nbytes for _, _, _, nbytes in payload) == rows * 264
