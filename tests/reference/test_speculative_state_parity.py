"""CPU command contracts for actual-checkpoint speculative replay qualification."""

import hashlib
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from ignition.components.sampling import AdmissionBuffers
from ignition.preparation.execution import GraphSignature
from ignition.recipes.qwen38.decoder_state import DecoderInputs
from ignition.recipes.qwen38.state import StateTransferMetadata
from qualify.speculative_state_parity import LEGAL_SHAPES, Commands, _mask, command_plan


def fixture():
    return command_plan(list(range(100, 165)), 620, 721)


def model():
    inputs = DecoderInputs.allocate("cpu", context=1024, tokens=128)
    admission = AdmissionBuffers.allocate("cpu", vocab_size=248320)
    transfer = StateTransferMetadata.allocate(4, "cpu")
    buffers = {**inputs.buffers(), **{f.name: getattr(admission, f.name) for f in fields(admission)},
        **{"state_" + f.name: getattr(transfer, f.name) for f in fields(transfer)},
        "sampling_grammar_mask": torch.zeros((4, 7760), dtype=torch.int32),
        "copy_token_ids": torch.full((4, 7), -1, dtype=torch.int64),
        "copy_token_count": torch.zeros(4, dtype=torch.int32),
        "copy_expected_consumed": torch.full((4,), -1, dtype=torch.int64),
        "copy_generation": torch.zeros(4, dtype=torch.int64),
        "copy_grammar_mask": torch.zeros((4, 8, 7760), dtype=torch.int32),
        "copy_grammar_mask_count": torch.zeros(4, dtype=torch.int32)}
    forward = tuple(inputs.buffers()) + ("sampling_grammar_mask",)
    copy = tuple(name for name in buffers if name.startswith("copy_"))
    state = tuple("state_" + f.name for f in fields(transfer) if f.name != "errors")
    admit = tuple(f.name for f in fields(admission))
    signatures = {f"speculative{b}x{w}": GraphSignature("speculative", b, w, (*forward, *copy)) for b, w in LEGAL_SHAPES}
    for capacity in (1, 2, 4):
        for kind in ("init", "restore", "snapshot"):
            extras = forward if kind == "snapshot" else admit + (("page_tables",) if kind == "restore" else ())
            signatures[f"{kind}{capacity}"] = GraphSignature(kind, capacity, 1, state + extras)
        signatures[f"resume{capacity}"] = GraphSignature("resume", capacity, 1, forward)
    for capacity in (64, 128):
        signatures[f"prefill{capacity}"] = GraphSignature("prefill", 1, capacity, forward)
    return SimpleNamespace(buffers=buffers, signatures=signatures, memory_report={"pages": 100})


def make_builder(tmp_path):
    (tmp_path / "uploads").mkdir()
    return Commands(model(), fixture(), tmp_path)


def test_compare_clones_native_upload_defaults_after_reset_before_handoff(tmp_path, monkeypatch):
    import qualify.graph_execution as graph_qualification
    import qualify.speculative_state_parity as parity

    runtime = model()
    runtime.device, runtime.graphs = "cpu", {}
    runtime.manifest = lambda: {"graphs": []}
    grammar = runtime.buffers["sampling_grammar_mask"]
    grammar.fill_(-1)
    initial = grammar.clone()
    owner = SimpleNamespace(owners=[], cache={}, extra_base=9,
        pool=SimpleNamespace(backing=torch.empty((1, 0), dtype=torch.uint8),
                             layout=SimpleNamespace(regions=[], row_bytes=0)))
    builders, handoffs = [], []
    executor = SimpleNamespace(drained=False)

    class Reset:
        storage = {}

        def __init__(self, model, state):
            assert model is runtime and state is owner

        def reset(self):
            grammar.copy_(initial)

        def hashes(self):
            return {"grammar": hashlib.sha256(grammar.numpy().tobytes()).hexdigest()}

    def commands(*args):
        result = Commands(*args)
        builders.append(result)
        return result

    def observe(evidence, side, stage, *args):
        assert stage == "initial"
        if side == "python":
            # Model execution leaves a request mask that the first init does not overwrite.
            _mask(grammar[1], [fixture()["token"]])
        else:
            assert torch.equal(grammar, initial)

    def start_executor():
        assert len(builders) == 2
        assert torch.equal(grammar, initial)
        handoffs.append(True)
        return executor

    executor.close = lambda: setattr(executor, "drained", True)
    runtime.start_executor = start_executor
    monkeypatch.setattr(parity, "MutableState", Reset)
    monkeypatch.setattr(parity, "Commands", commands)
    monkeypatch.setattr(parity, "observe", observe)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: object())
    monkeypatch.setattr(graph_qualification, "verify_prepared_hybrid_state",
                        lambda pool, evidence: evidence.check("pristine", True))
    (tmp_path / "uploads").mkdir()
    evidence = graph_qualification.Checks(tmp_path)
    before_cuda = torch.cuda.is_initialized()
    parity.compare(runtime, owner, {**fixture(), "commands": []}, evidence, tmp_path, lambda _: None)

    assert handoffs == [True] and executor.drained
    assert evidence.value["reset"]["python"] == evidence.value["reset"]["native"]
    for operation in fixture()["commands"][:2]:
        expected, actual = (builder.prepare(operation) for builder in builders)
        assert actual == expected
        for name, sha in expected["uploads"]:
            assert builders[0].payloads[sha] == builders[1].payloads[sha], name
    assert builders[1].host["sampling_grammar_mask"][1].eq(-1).all()
    assert torch.cuda.is_initialized() == before_cuda


def test_all_shapes_and_state_transition_contracts():
    plan = fixture()
    rounds = [c for c in plan["commands"] if c["kind"] == "speculative"]
    assert {(c["batch"], c["width"]) for c in rounds} == set(LEGAL_SHAPES)
    assert len(LEGAL_SHAPES) == 14
    i = next(i for i, c in enumerate(plan["commands"]) if c.get("tag") == "eight_to_one")
    assert plan["commands"][i - 1]["width"] == 8
    assert plan["commands"][i]["width"] == 1
    assert plan["commands"][i]["require_previous"] == 8
    assert next(c for c in rounds if c.get("tag") == "inactive_leading_lane")["slots"] == [-1, 1]
    assert next(c for c in rounds if c.get("tag") == "inactive_interior_permutation")["slots"] == [2, -1, 0, 3]
    assert plan["commands"][-1]["stochastic"] is True
    assert [c["kind"] for c in plan["commands"]].count("snapshot") == 1
    assert [c["kind"] for c in plan["commands"]].count("restore") == 1


def test_init_and_prefill_use_physical_slots_exact_tokens_and_normal_eos(tmp_path):
    builder = make_builder(tmp_path)
    plan = fixture()
    command = builder.prepare(plan["commands"][0])
    assert command["graph"] == "init4"
    h = builder.host
    assert h["admission_slots"].tolist() == [3, 0, 2, -1]
    assert h["admission_eos_count"][:3].tolist() == [2, 2, 2]
    assert h["admission_eos_token_ids"][0, :2].tolist() == [248046, 248044]
    assert h["admission_grammar_enabled"][:3].eq(1).all()
    command = builder.prepare(plan["commands"][1])
    assert command["graph"] == "prefill64"
    assert h["token_ids"][:63].tolist() == plan["prompt"][:63]
    assert h["token_ids"][63:].eq(248044).all()
    assert h["prefill_positions"][:63, 0].tolist() == list(range(63))
    assert h["page_tables"][3, :12].tolist() == list(range(12))


def test_inactive_leading_lane_uses_physical_copy_and_grammar_rows(tmp_path):
    builder = make_builder(tmp_path)
    builder.prepare({"kind": "init", "slots": [1], "generation": 51, "lengths": [64], "budget": 32})
    builder.requests[1]["consumed"] = 64
    command = builder.prepare({"kind": "speculative", "batch": 2, "width": 4, "slots": [-1, 1]})
    h = builder.host
    assert command["graph"] == "speculative2x4"
    assert h["live_rows"].item() == 2
    assert h["row_slot_ids"].tolist() == [-1, 1, -1, -1]
    assert h["row_generations"].tolist() == [0, 51, 0, 0]
    assert h["copy_generation"].tolist() == [0, 51, 0, 0]
    assert h["copy_expected_consumed"].tolist() == [-1, 64, -1, -1]
    assert h["copy_token_ids"][1].tolist() == [620, 620, 620, -1, -1, -1, -1]
    assert h["copy_grammar_mask_count"].tolist() == [0, 4, 0, 0]
    assert h["copy_grammar_mask"][0].count_nonzero() == 0
    assert all(int(h["copy_grammar_mask"][1, i, 620 // 32]) & (1 << (620 % 32)) for i in range(4))


def test_stochastic_copy_ids_are_all_grammar_valid(tmp_path):
    builder = make_builder(tmp_path)
    builder.prepare({"kind": "init", "slots": [0], "generation": 2, "lengths": [64], "budget": 32})
    builder.prepare(fixture()["commands"][-1])
    h = builder.host
    for i, token in enumerate(h["copy_token_ids"][0].tolist()):
        assert int(h["copy_grammar_mask"][0, i, token // 32]) & (1 << (token % 32))
    assert all(h["copy_grammar_mask"][0, i].count_nonzero() == 2 for i in range(8))
    assert torch.equal(h["sampling_grammar_mask"][0], h["copy_grammar_mask"][0, 0])


def test_snapshot_restore_shares_full_pages_and_allocates_private_continuation(tmp_path):
    builder = make_builder(tmp_path)
    builder.snapshots[0] = {"tokens": list(range(64)), "pages": [44]}
    builder.next_page = 45
    command = builder.prepare({"kind": "restore", "slots": [3], "generation": 70, "snapshot": 0, "budget": 32})
    assert command["graph"] == "restore1"
    assert builder.requests[3]["pages"] == [44, *range(45, 56)]
    assert builder.host["state_source_slots"][0] == 5
    assert builder.host["state_destination_slots"][0] == 4
    assert builder.host["state_qsa_slots"][0] == 4
    assert builder.host["state_expected_consumed"].tolist() == [64, -1, -1, -1]


def test_mask_word_31_preserves_uint32_bits():
    value = torch.zeros(2, dtype=torch.int32)
    _mask(value, [0, 31, 32, 63])
    assert value.tolist() == [-2147483647, -2147483647]


@pytest.mark.parametrize("endpoint", [1, 10, 15, 63, 64, 65, 66, 67, 127])
def test_exact_checkpoint_staging_preserves_endpoint_and_private_tail(tmp_path, endpoint):
    builder = make_builder(tmp_path)
    builder.prepare({"kind": "init", "slots": [2], "generation": 11,
                     "lengths": [63], "budget": 512})
    request = builder.requests[2]
    request["consumed"] = endpoint
    request["tokens"] = [620] * endpoint
    command = builder.prepare({"kind": "snapshot", "slots": [2], "snapshot": 3})
    h = builder.host
    assert command["graph"] == "snapshot1"
    assert h["state_expected_consumed"].tolist() == [endpoint, -1, -1, -1]
    assert h["state_source_slots"].tolist() == [3, 0, 0, 0]
    assert h["state_destination_slots"].tolist() == [8, 0, 0, 0]
    assert h["state_accepted_tokens"].eq(0).all()
    assert h["page_tables"][2, :12].tolist() == request["pages"]
    shared = request["pages"][:endpoint // 64]
    builder.snapshots[3] = {"tokens": request["tokens"].copy(), "pages": shared}
    command = builder.prepare({"kind": "restore", "slots": [0], "generation": 12,
                               "snapshot": 3, "budget": 512})
    restored = builder.requests[0]
    assert command["graph"] == "restore1"
    assert h["state_source_slots"].tolist() == [8, 0, 0, 0]
    assert h["state_expected_consumed"].tolist() == [endpoint, -1, -1, -1]
    assert h["state_accepted_tokens"].tolist() == [1, 0, 0, 0]
    assert restored["pages"][:endpoint // 64] == shared
    assert not set(restored["pages"][endpoint // 64:]) & set(request["pages"])
    uploads = dict(command["uploads"])
    assert "page_tables" in uploads
    assert builder.payloads[uploads["page_tables"]] == bytes(h["page_tables"].view(torch.uint8).numpy())


def test_complete_command_contract_preserves_wide_state_across_reuse(tmp_path):
    builder = make_builder(tmp_path)
    commands = []
    for operation in fixture()["commands"]:
        command = builder.prepare(operation)
        records = []
        for slot in range(4):
            r = [0] * 24
            r[4] = r[12] = -1
            r[16:] = [-1] * 8
            if slot not in operation["slots"]:
                records.append(r)
                continue
            request = builder.requests[slot]
            before = request["consumed"]
            width = operation.get("width", 1)
            if operation["kind"] == "speculative":
                proposed = 1 if operation.get("stochastic") else width
                emitted = min(proposed, request["budget"] - request["generated"])
                consumed = before + emitted
                enabled = request["generated"] < request["budget"]
                r[9] = width - 1 if enabled else 0
                r[10] = min(proposed - 1, emitted)
                r[11] = (1 << (width - 1)) - 1 if enabled else 0
                r[13] = -1
            else:
                emitted = int(operation["kind"] in ("prefill", "resume"))
                consumed = before + (operation["length"] if operation["kind"] == "prefill" else 0)
                r[13] = max(0, consumed - 1)
            generated = request["generated"] + emitted
            r[:8] = [request["generation"], command["sequence"], consumed, generated,
                     620 if emitted else -1, 2 if generated == request["budget"] else 1, 0, 0]
            r[8], r[15] = emitted, width
            r[16:16 + emitted] = [620] * emitted
            records.append(r)
        checks, _ = builder.complete(command, records)
        assert all(checks.values()), (command, [key for key, value in checks.items() if not value])
        commands.append(command)
    assert builder.next_page == 71
    assert builder.snapshots[0]["tokens"] == fixture()["prompt"][:63] + [620] * 9
    assert builder.previous[2] == 3  # Clipped inside an accepted seven-draft prefix.
    assert builder.requests[2]["generated"] == 4
    assert builder.requests[2]["consumed"] == 66
    reuse = next(c for c in commands if c.get("tag") == "epoch_reuse")
    assert reuse["previous_accepted_before"][3] == 1
    assert reuse["accepted_generation_before"][3] == 12
    assert builder.accepted_generation[3] == 12


@pytest.mark.parametrize("prompt,a,b", [([1] * 64, 2, 3), ([1] * 65, 2, 2), ([1] * 65, 248044, 3)])
def test_invalid_command_fixture_is_rejected(prompt, a, b):
    with pytest.raises(ValueError):
        command_plan(prompt, a, b)
