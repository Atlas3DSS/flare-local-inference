"""Typed prepared values preserve graph dataflow independently of model identity."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.protobuf.json_format import ParseDict

from ignition.preparation import contract as c, runtime_package_pb2 as p
from ignition.preparation.execution import GraphSignature


@pytest.mark.parametrize("recipe,page,lag,rows", [
    ("qwen38", 64, 1, 1), ("glm53", 256, 1, 1),
    ("deepseek41", 128, 0, None), ("deepseek4", 128, 0, None), ("mimo_v2", 128, 0, 1),
])
def test_recipe_declares_sources_storage_cursors_and_frontend(recipe, page, lag, rows):
    module = importlib.import_module(f"ignition.recipes.{recipe}.contract")
    signatures = {kind: GraphSignature(kind, 1, 4 if kind == "speculative" else 1,
        tuple(c.text_inputs(kind)), speculation_enabled=kind == "speculative")
        for kind in ("init", "restore", "snapshot", "prefill", "decode", "resume", "speculative")}
    pool = SimpleNamespace(pages={}, layout=SimpleNamespace(row_bytes=32,
        region=lambda name: SimpleNamespace(offset=24, nbytes=8)))
    contract = module.prepare(signatures, {"page_tables": None, "draft_page_tables": None}, pool,
        page_tokens=page, draft_lag=lag, rows=rows)
    encoded = ParseDict(contract, p.ModelContract())
    assert encoded.version == 1 and encoded.input_vocabulary >= encoded.vocabulary
    assert "schema" not in contract
    decode = next(g for g in encoded.graphs if g.kind == p.DECODE)
    assert decode.HasField("pending_token") == (recipe == "qwen38")
    assert any(b.source == p.PENDING_TOKEN for b in decode.inputs) == (recipe != "qwen38")
    speculative = next(g for g in encoded.graphs if g.kind == p.SPECULATIVE)
    assert speculative.draft_cursor.lag_tokens == lag
    assert speculative.state_rows_per_request == (4 if rows is None else rows)
    assert speculative.draft_cursor.state.offset == 24
    init = next(g for g in encoded.graphs if g.kind == p.INIT)
    assert next(b for b in init.inputs if b.source == p.SAMPLING_SEED).indexing == p.ACTIVE_ROW
    assert encoded.frontend.tools.kind in (p.QWEN_CODER, p.GLM_ARGUMENT_TAGS, p.DSML)
    fixture = json.loads((Path(__file__).parents[2] / "helios/tests/support/contracts" / f"{recipe}.json").read_text())
    assert fixture["frontend"] == contract["frontend"]
    assert fixture["vision"] == contract["vision"]


def test_device_pending_token_has_read_and_producer_declarations():
    module = importlib.import_module("ignition.recipes.qwen38.contract")
    pool = SimpleNamespace(layout=SimpleNamespace(row_bytes=32, region=lambda _: SimpleNamespace(offset=0, nbytes=8)))
    signatures = {kind: GraphSignature(kind, 1, 1, tuple(c.text_inputs(kind))) for kind in ("init", "restore", "prefill", "decode")}
    value = module.prepare(signatures, {"page_tables": None}, pool, page_tokens=64, draft_lag=None)
    decode = next(g for g in value["graphs"] if g["kind"] == p.DECODE)
    pending = decode["pending_token"]
    assert pending["buffer"] == "request_pending_token"
    for graph in value["graphs"]:
        assert any(s["view"] == pending and s["access"] in (p.WRITE, p.READ_WRITE) for s in graph["state"])


def test_history_plane_registers_actual_row_geometry():
    import torch
    tensor = torch.empty((8, 64, 12), dtype=torch.bfloat16)
    buffers = {}
    region = c.history_region(buffers, "keys", tensor, p.PAGED_KEY_VALUE)
    assert buffers["keys"] is tensor
    assert region["storage"] == dict(buffer="keys", offset=0, stride=1536, nbytes=1536)


def test_unknown_packing_fields_do_not_acquire_sources_from_model_identity():
    pool = SimpleNamespace(layout=SimpleNamespace(row_bytes=32, region=lambda _: SimpleNamespace(offset=0, nbytes=8)))
    module = importlib.import_module("ignition.recipes.glm53.contract")
    value = module.prepare({"decode": GraphSignature("decode", 1, 1, ("token_ids", "unused_scratch"))},
        {}, pool, page_tokens=256, draft_lag=None)
    assert [v["buffer"] for v in value["graphs"][0]["inputs"]] == ["token_ids"]
    assert value["graphs"][0]["pending_token"] is None
