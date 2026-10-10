"""Actual-checkpoint copy verification state parity through Python and Rust replay.

Singleton conditional grammar makes accepted counts predictable without changing
model logits or replacing target computation. A separate two-token grammar case
records acceptance under the actual target probabilities. This qualification
does not exercise the MTP draft model or establish cross-width model equivalence.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from collections import Counter
from dataclasses import fields, is_dataclass
from pathlib import Path

from qualify.provenance import write_json

ROOT = Path(__file__).resolve().parents[1]
VOCAB = 248320
LEGAL_SHAPES = tuple((b, w) for b in (1, 2, 4) for w in range(1, 9) if b * w <= 8)


def command_plan(prompt, token, alternative):
    """Build a CPU command contract; token IDs come from the exact tokenizer."""
    if len(prompt) != 65 or any(type(x) is not int or not 0 <= x < VOCAB for x in prompt):
        raise ValueError("the fixture requires exactly 65 vocabulary token IDs")
    if token == alternative or any(type(x) is not int or not 0 <= x < VOCAB for x in (token, alternative)):
        raise ValueError("grammar tokens must be distinct vocabulary IDs")
    if {token, alternative} & {248044, 248046}:
        raise ValueError("fixture grammar tokens must not be EOS")
    commands = []

    def add(kind, **kwargs):
        commands.append({"kind": kind, **kwargs})

    def round_(b, w, slots, **kwargs):
        if (b, w) not in LEGAL_SHAPES or len(slots) > b:
            raise ValueError("invalid speculative graph geometry")
        active = [s for s in slots if s >= 0]
        if len(active) != len(set(active)) or any(s < -1 or s > 3 for s in slots):
            raise ValueError("packed slots must be unique physical IDs or -1")
        add("speculative", batch=b, width=w, slots=slots, **kwargs)

    add("init", slots=[3, 0, 2], generation=11, lengths=[63, 64, 65], budget=512)
    for slot, length in ((3, 63), (0, 64), (2, 65)):
        add("prefill", slots=[slot], length=length)
    round_(1, 8, [3], tag="all_eight_columns")
    round_(1, 1, [3], tag="eight_to_one", require_previous=8)
    add("snapshot", slots=[3], snapshot=0)
    add("restore", slots=[1], generation=12, snapshot=0, budget=512)
    add("resume", slots=[1])
    round_(2, 4, [-1, 1], tag="inactive_leading_lane")
    round_(4, 2, [2, -1, 0, 3], tag="inactive_interior_permutation")
    for width in range(2, 8):
        round_(1, width, [3])
    for width in range(1, 4):
        round_(2, width, [1, 0])
    round_(4, 1, [3, 2, 1, 0])
    round_(1, 8, [3], tag="before_epoch_reuse")
    add("init", slots=[3], generation=12, lengths=[64], budget=512)
    add("prefill", slots=[3], length=64)
    round_(1, 1, [3], tag="epoch_reuse", require_previous=1, require_gdn_previous=1)
    add("init", slots=[2], generation=12, lengths=[63], budget=4)
    add("prefill", slots=[2], length=63)
    round_(1, 8, [2], tag="budget_clip_inside_drafts")
    round_(1, 1, [2], tag="terminal_replay")
    round_(1, 8, [0], tag="observed_two_token_acceptance", stochastic=True)
    return {"prompt": list(prompt), "token": token, "alternative": alternative,
            "commands": commands, "shapes": [list(shape) for shape in LEGAL_SHAPES]}


def _mask(tensor, tokens):
    tensor.zero_()
    for token in tokens:
        word, bit = divmod(token, 32)
        unsigned = int(tensor[..., word].reshape(-1)[0]) & 0xFFFFFFFF
        value = unsigned | (1 << bit)
        tensor[..., word] = value if value < 2**31 else value - 2**32


class Commands:
    """Stage complete graph inputs and retain identical native upload bytes."""

    def __init__(self, model, fixture, artifacts):
        import torch

        from flare.components.sampling import AdmissionBuffers

        self.model, self.fixture, self.artifacts = model, fixture, artifacts
        self.host = {name: tensor.detach().cpu().clone() for name, tensor in model.buffers.items()}
        self.admission = AdmissionBuffers(**{f.name: self.host[f.name] for f in fields(AdmissionBuffers)})
        self.requests, self.snapshots, self.payloads = {}, {}, {}
        self.previous = [1] * 4
        self.accepted_generation = [0] * 4
        self.target_generation, self.target_offset = [0] * 4, [0] * 4
        self.next_page = 0
        self.sequence = 0
        self.token = fixture["token"]
        self.torch = torch

    def _rows(self, slots):
        h = self.host
        h["row_slot_ids"].fill_(-1)
        h["row_generations"].zero_()
        h["live_rows"].fill_(len(slots))
        h["query_lengths"].zero_()
        h["sample_enabled"].fill_(1)
        h["vision_token_indices"].fill_(-1)
        h["page_tables"].fill_(-1)
        h["copy_token_ids"].fill_(-1)
        h["copy_token_count"].zero_()
        h["copy_expected_consumed"].fill_(-1)
        h["copy_generation"].zero_()
        h["copy_grammar_mask_count"].zero_()
        h["copy_grammar_mask"].zero_()
        for row, slot in enumerate(slots):
            if slot < 0:
                continue
            request = self.requests[slot]
            h["row_slot_ids"][row] = slot
            h["row_generations"][row] = request["generation"]
            h["page_tables"][slot, :len(request["pages"])].copy_(self.torch.tensor(request["pages"]))
            _mask(h["sampling_grammar_mask"][slot], [self.token])

    def prepare(self, operation):
        from flare.recipes.qwen38.sampling import RequestAdmission

        h, kind, slots = self.host, operation["kind"], operation["slots"]
        active = [slot for slot in slots if slot >= 0]
        capacity = next(c for c in (1, 2, 4) if c >= len(slots))
        if kind in ("init", "restore"):
            h["admission_count"].fill_(len(slots))
            h["admission_slots"].fill_(-1)
            for row, slot in enumerate(slots):
                snapshot = self.snapshots[operation["snapshot"]] if kind == "restore" else None
                prompt = list(snapshot["tokens"]) if snapshot else self.fixture["prompt"][:operation["lengths"][row]]
                prefix = snapshot["pages"] if snapshot else []
                # Full immutable pages are shared; every continuation page is private.
                pages = [*prefix, *range(self.next_page, self.next_page + 12 - len(prefix))]
                self.next_page += 12 - len(prefix)
                generation = operation["generation"]
                seed = (2**64 - 31) + slot
                self.requests[slot] = {"generation": generation, "seed": seed, "prompt": prompt,
                    "tokens": list(prompt) if snapshot else [], "pages": pages,
                    "consumed": len(prompt) if snapshot else 0, "generated": 0,
                    "budget": operation["budget"], "histogram": dict(Counter(prompt)),
                    "pending": -1}
                admission = RequestAdmission(generation, seed, max_new_tokens=operation["budget"],
                    temperature=1, top_k=-1, top_p=1)
                self.admission.stage(row, slot, admission, prompt)
                h["admission_grammar_enabled"][row] = 1
                _mask(h["admission_grammar_mask"][row], [self.token])
            self._rows(slots)
        else:
            self._rows(slots)
        if kind in ("init", "restore", "snapshot"):
            h["state_num_ops"].fill_(len(slots))
            for name in ("source_slots", "destination_slots", "operation", "accepted_tokens"):
                h["state_" + name].zero_()
            h["state_qsa_slots"].fill_(-1)
            h["state_expected_consumed"].fill_(-1)
            for row, slot in enumerate(slots):
                source = 5 + operation["snapshot"] if kind == "restore" else slot + 1
                destination = 5 + operation["snapshot"] if kind == "snapshot" else slot + 1
                h["state_source_slots"][row] = source
                h["state_destination_slots"][row] = destination
                h["state_operation"][row] = int(kind == "init")
                h["state_accepted_tokens"][row] = int(kind != "snapshot")
                h["state_expected_consumed"][row] = self.requests[slot]["consumed"]
                if kind != "snapshot":
                    h["state_qsa_slots"][row] = slot + 1
            graph = f"{kind}{capacity}"
        elif kind == "prefill":
            request = self.requests[active[0]]
            length = operation["length"]
            h["query_lengths"][0] = length
            h["token_ids"].fill_(248044)
            h["token_ids"][:length] = self.torch.tensor(request["prompt"])
            h["prefill_positions"].zero_()
            h["prefill_positions"][:length] = self.torch.arange(length)[:, None].expand(-1, 3)
            graph = f"prefill{64 if length <= 64 else 128}"
        elif kind == "resume":
            graph = f"resume{capacity}"
        elif kind == "speculative":
            width = operation["width"]
            graph = f"speculative{operation['batch']}x{width}"
            for slot in active:
                request = self.requests[slot]
                allowed = [self.token, self.fixture["alternative"]] if operation.get("stochastic") else [self.token]
                _mask(h["sampling_grammar_mask"][slot], allowed)
                ids = ([self.fixture["alternative"], self.token] * 4)[:width - 1] if operation.get("stochastic") else [self.token] * (width - 1)
                h["copy_token_ids"][slot, :width - 1] = self.torch.tensor(ids, dtype=self.torch.int64)
                h["copy_token_count"][slot] = width - 1
                h["copy_expected_consumed"][slot] = request["consumed"]
                h["copy_generation"][slot] = request["generation"]
                h["copy_grammar_mask_count"][slot] = width
                for column in range(width):
                    _mask(h["copy_grammar_mask"][slot, column], allowed)
        else:
            raise ValueError(f"unsupported command {kind}")
        if graph not in self.model.signatures:
            raise ValueError(f"required production graph is missing: {graph}")
        if self.next_page >= self.model.memory_report["pages"]:
            raise ValueError("fixture KV reservation exceeds the prepared pool")
        uploads = []
        for name in self.model.signatures[graph].inputs:
            raw = bytes(self.host[name].contiguous().view(self.torch.uint8).numpy().tobytes())
            sha = hashlib.sha256(raw).hexdigest()
            if sha not in self.payloads:
                self.payloads[sha] = raw
                (self.artifacts / "uploads" / sha).write_bytes(raw)
            uploads.append((name, sha))
        self.sequence += 1
        return {**operation, "graph": graph, "sequence": self.sequence, "frame": 0,
                "uploads": uploads, "previous_accepted_before": list(self.previous),
                "accepted_generation_before": list(self.accepted_generation)}

    def complete(self, command, records):
        """Independent token/count oracle; no model-state values are synthesized."""
        kind, slots = command["kind"], command["slots"]
        checks = {}
        emitted_by_slot = {}
        for slot in range(4):
            r = records[slot]
            if slot not in slots:
                checks[f"slot{slot}.inactive"] = r[:8] == [0, 0, 0, 0, -1, 0, 0, 0] and r[8] == 0 and r[16:] == [-1] * 8
                continue
            request = self.requests[slot]
            before, generated = request["consumed"], request["generated"]
            count = r[8]
            if kind == "speculative":
                width = command["width"]
                if command.get("stochastic"):
                    expected_count = count
                    checks[f"slot{slot}.observed_count_range"] = 1 <= count <= width
                    checks[f"slot{slot}.grammar_support"] = all(x in (self.token, self.fixture["alternative"]) for x in r[16:16 + count])
                else:
                    expected_count = min(width, max(0, request["budget"] - generated))
                    checks[f"slot{slot}.tokens"] = r[16:16 + count] == [self.token] * expected_count
                checks[f"slot{slot}.emitted_count"] = count == expected_count
                checks[f"slot{slot}.accepted_drafts"] = r[10] == (min(width - 1, count) if not command.get("stochastic") else count - 1)
                checks[f"slot{slot}.offered"] = r[9] == (width - 1 if generated < request["budget"] else 0)
                checks[f"slot{slot}.copy_mask"] = r[11] == ((1 << (width - 1)) - 1 if generated < request["budget"] else 0)
                checks[f"slot{slot}.draft_unavailable"] = r[13] == -1
                checks[f"slot{slot}.width"] = r[15] == width
                if count:
                    inputs = [request["pending"], *self.host["copy_token_ids"][slot].tolist()]
                    request["tokens"].extend(inputs[:count])
                    request["consumed"] += count
                    self.previous[slot] = count
                    self.accepted_generation[slot] = request["generation"]
                    if self.target_generation[slot] != request["generation"]:
                        self.target_offset[slot] = 0
                    self.target_generation[slot] = request["generation"]
                    self.target_offset[slot] += 4 * ((width + 4) // 4)
            elif kind in ("prefill", "resume"):
                checks[f"slot{slot}.sample"] = count == 1 and r[16] == self.token
                if kind == "prefill":
                    request["tokens"].extend(request["prompt"])
                    request["consumed"] += command["length"]
            else:
                checks[f"slot{slot}.no_sample"] = count == 0
                if kind in ("init", "restore"):
                    self.previous[slot] = 1
                    self.accepted_generation[slot] = request["generation"]
            ids = r[16:16 + count]
            request["generated"] += count
            for token in ids:
                request["histogram"][token] = request["histogram"].get(token, 0) + 1
            if count:
                request["pending"] = ids[-1]
            checks[f"slot{slot}.model_reserved"] = r[12] == -1
            checks[f"slot{slot}.record"] = (r[:4] == [request["generation"], command["sequence"], request["consumed"], request["generated"]]
                and r[4] == (ids[-1] if ids else -1)
                and r[5:8] == [2 if request["generated"] == request["budget"] else 1, 0, 0]
                and r[14] == 0 and r[16 + count:] == [-1] * (8 - count))
            emitted_by_slot[slot] = count
        if kind == "snapshot":
            request = self.requests[slots[0]]
            length = request["consumed"]
            self.snapshots[command["snapshot"]] = {"tokens": request["tokens"][:length], "pages": request["pages"][:length // 64]}
        return checks, emitted_by_slot


class MutableState:
    """Explicit mutable recipe owners; packed weights and compiled plans stay intact."""

    def __init__(self, model, owner):
        import torch

        from flare.preparation.scratch import ScratchArena
        from flare.recipes.qwen38.decoder import DecoderScratch, ModelWeights
        from flare.recipes.qwen38.vision import VisionScratch
        from flare.recipes.qwen38.vision_state import VisionFeatureState

        self.model, self.owner, self.storage = model, owner, {}
        self.features = next(x for x in model._resources if isinstance(x, VisionFeatureState))
        self.excluded = {owner.pool.backing.untyped_storage().data_ptr(), self.features.arena.untyped_storage().data_ptr()}
        self.excluded.update(getattr(layer, f.name).untyped_storage().data_ptr() for layer in owner.cache.values() for f in fields(layer))
        seen = set()
        safe_modules = {"sampling", "speculation", "speculative_state", "speculative_model", "draft_state", "decoder_state", "maintenance", "vision_state"}
        skip = {"weights", "mtp_weights", "pool", "cache", "qsa_states", "rotary", "session", "table", "model", "owner", "decoder", "decoders", "layers", "plans", "plan", "binding", "projection", "operations"}

        def add(name, value):
            if isinstance(value, torch.Tensor):
                storage = value.untyped_storage()
                key = storage.data_ptr()
                if key in self.excluded or key in self.storage or not storage.nbytes():
                    return
                raw = torch.empty(0, dtype=torch.uint8, device=value.device).set_(storage, 0, (storage.nbytes(),), (1,))
                self.storage[key] = (name, raw, raw.cpu())
                return
            if id(value) in seen:
                return
            seen.add(id(value))
            if is_dataclass(value) and not isinstance(value, type):
                items = [(f.name, getattr(value, f.name)) for f in fields(value)]
            elif isinstance(value, dict):
                items = value.items()
            elif isinstance(value, (tuple, list)):
                items = enumerate(value)
            elif type(value).__module__.rsplit(".", 1)[-1] in safe_modules:
                items = vars(value).items()
            else:
                return
            for key, child in items:
                if key not in skip:
                    add(f"{name}.{key}", child)

        add("buffer", dict(model.buffers))
        add("speculation", owner)
        for index, operation in enumerate(owner.owners):
            if getattr(operation, "operations", None) is not None:
                add(f"round{index}.operations", operation.operations)
        for index, decoder in enumerate(next(x for x in model._resources if isinstance(x, list) and x and hasattr(x[0], "qsa_states"))):
            add(f"decoder{index}.state", decoder.state)
            add(f"decoder{index}.sampler", decoder.sampler)
            add(f"decoder{index}.injection", decoder.embedding_injection)
            for layer, (_, attention, _, _) in enumerate(decoder.layers):
                if hasattr(attention, "inactive_rows"):
                    add(f"decoder{index}.inactive{layer}", attention.inactive_rows)
        for index, value in enumerate(model._resources):
            if isinstance(value, (DecoderScratch, VisionScratch, VisionFeatureState)):
                add(f"resource{index}", value)
            elif isinstance(value, ScratchArena):
                add("scratch_arena", value.tensor)
            elif isinstance(value, ModelWeights):
                add("expert_scales", value.expert_scale_scratch)
                for layer, weights in enumerate(value.layers):
                    add(f"expert{layer}.active", weights.moe.experts._impl.nvfp4_csf.active)
            elif isinstance(value, list):
                add(f"resource{index}", value)
        self.request_initial = {f.name: getattr(owner.requests, f.name).cpu() for f in fields(owner.requests)}

    def reset(self):
        import torch

        if self.model.consumed:
            raise RuntimeError("reset is forbidden after native handoff")
        pool = self.owner.pool
        pool.backing.fill_(0x3D)
        pool.backing[0].zero_()
        for _, raw, initial in self.storage.values():
            raw.copy_(initial)
        for name, initial in self.request_initial.items():
            getattr(self.owner.requests, name).copy_(initial)
        self.owner.previous_accepted.fill_(1)
        self.owner.accepted_generation.zero_()
        self.owner.rng.generation.zero_()
        self.owner.rng.target_offset.zero_()
        self.owner.draft_rng_generation.zero_()
        self.owner.draft_rng_offset.zero_()
        for layer in self.owner.cache.values():
            for f in fields(layer):
                getattr(layer, f.name).fill_(-1 if "positions" in f.name else 0.125)
        self.features.arena.fill_(-0.125)
        torch.cuda.synchronize(self.model.device)

    def hashes(self):
        from qualify.graph_execution import byte_view, sha256

        result = {name: sha256(byte_view(raw.cpu())) for name, raw, _ in self.storage.values()}
        for row in range(self.owner.pool.backing.shape[0]):
            result[f"hybrid{row}"] = sha256(byte_view(self.owner.pool.backing[row].cpu()))
        for layer, cache in self.owner.cache.items():
            for f in fields(cache):
                tensor = getattr(cache, f.name)
                for start in range(0, tensor.shape[0], 16):
                    result[f"qsa{layer}.{f.name}.{start}"] = sha256(byte_view(tensor[start:start + 16].cpu()))
        result["features"] = sha256(byte_view(self.features.arena.cpu()))
        return result


def _allocated(device):
    import torch

    stats = torch.cuda.memory_stats(device)
    return {key: stats[key] for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed", "num_device_alloc", "num_device_free")}


def observe(evidence, side, stage, model, owner, pages, operation=None):
    """Copy persistent state after a completion fence, without GPU gather kernels."""
    from flare.recipes.qwen38.vision_state import VisionFeatureState

    pool = owner.pool
    for row in range(pool.backing.shape[0]):
        cpu = pool.backing[row].cpu()
        offset = 0
        for region in pool.layout.regions:
            if region.offset > offset:
                evidence.tensor(side, stage, f"hybrid.{row}.padding{offset}", cpu[offset:region.offset])
            evidence.tensor(side, stage, f"hybrid.{row}.{region.name}", cpu[region.offset:region.offset + region.nbytes])
            offset = region.offset + region.nbytes
        if offset < pool.layout.row_bytes:
            evidence.tensor(side, stage, f"hybrid.{row}.padding{offset}", cpu[offset:])
    for layer, cache in owner.cache.items():
        for f in fields(cache):
            tensor = getattr(cache, f.name)
            if f.name.endswith("cache"):
                for page in (*range(pages + 1), tensor.shape[0] - 1):
                    evidence.tensor(side, stage, f"qsa.{layer}.{f.name}.page{page}", tensor[page].cpu())
            else:
                for row in range(tensor.shape[0]):
                    evidence.tensor(side, stage, f"qsa.{layer}.{f.name}.slot{row}", tensor[row].cpu())
    request = {f.name: getattr(owner.requests, f.name).cpu() for f in fields(owner.requests)}
    for name, tensor in request.items():
        for slot in range(4):
            evidence.tensor(side, stage, f"request.{name}.slot{slot}", tensor[slot:slot + 1])
    for name in ("previous_accepted", "accepted_generation", "candidates", "draft_rng_generation", "draft_rng_offset"):
        evidence.tensor(side, stage, f"speculation.{name}", getattr(owner, name).cpu())
    for f in fields(owner.rng):
        evidence.tensor(side, stage, f"target_rng.{f.name}", getattr(owner.rng, f.name).cpu())
    for name, tensor in model.buffers.items():
        evidence.tensor(side, stage, f"buffer.{name}", tensor.cpu())
    features = next(x for x in model._resources if isinstance(x, VisionFeatureState))
    for name in ("generation", "count"):
        evidence.tensor(side, stage, f"features.{name}", getattr(features, name).cpu())
    for slot in range(4):
        evidence.tensor(side, stage, f"features.arena{slot}", features.arena[slot].cpu())
    if operation is not None and hasattr(operation, "decoder"):
        for name in ("logits", "sample_hidden"):
            evidence.tensor(side, stage, f"target.{name}", getattr(operation.decoder.scratch, name).cpu())
        for name in ("state_indices", "accepted_tokens", "query_start_loc", "convolution_slots"):
            evidence.tensor(side, stage, f"target.gdn.{name}", getattr(operation.state.gdn, name).cpu())
        for name in ("request_ids", "query_positions", "sequence_lengths", "slot_mapping"):
            evidence.tensor(side, stage, f"target.qsa.{name}", getattr(operation.state.qsa, name).cpu())
        target, _, verify, commit, _ = operation.operations
        for name, tensor in (("target_valid", target.valid), ("acceptance_ids", verify.output.token_ids),
                ("accepted_drafts", verify.output.accepted_drafts), ("emitted_count", commit.emitted_count)):
            evidence.tensor(side, stage, name, tensor.cpu())
    return request


def _require(evidence, name, passed, **details):
    evidence.check(name, passed, **details)
    if not passed:
        raise AssertionError(name)


def compare(model, owner, fixture, evidence, artifacts, progress):
    import torch

    from qualify.graph_execution import verify_prepared_hybrid_state

    verify_prepared_hybrid_state(owner.pool, evidence)
    _require(evidence, "prepared_hybrid_state_pristine_gate", evidence.value["checks"][-1]["passed"])
    reset = MutableState(model, owner)
    reset.reset()
    initial = reset.hashes()
    evidence.value["reset"] = {"python": initial, "scratch_storage_count": len(reset.storage),
        "scratch_storage_bytes": sum(raw.nbytes for _, raw, _ in reset.storage.values())}
    manifest = model.manifest()
    evidence.value["manifest"] = manifest
    graph_indices = {name: index for index, name in enumerate(model.graphs)}
    operations = {(x.batch, x.width): x for x in owner.owners if hasattr(x, "batch") and hasattr(x, "width")}
    evidence.value["state_contract"] = {"regions": [r.name for r in owner.pool.layout.regions],
        "region_layout": [{"name": r.name, "shape": list(r.shape), "dtype": str(r.dtype),
                           "offset_bytes": r.offset, "nbytes": r.nbytes} for r in owner.pool.layout.regions],
        "all_hybrid_rows": owner.pool.backing.shape[0], "extra_base": owner.extra_base,
        "row_bytes": owner.pool.layout.row_bytes, "qsa_layers": list(owner.cache),
        "readback_limit_bytes": 32 << 30, "executor_overlap": False,
        "snapshot_slot": 5, "draft_model_exercised": False}
    evidence.value["readback_contract"] = {
        "hybrid_bytes_per_observation": owner.pool.backing.nbytes,
        "hybrid_bytes_all_observations": 2 * (len(fixture["commands"]) + 1) * owner.pool.backing.nbytes,
        "fixture_pages": list(range(73)), "additional_page": "last allocated pool page",
        "raw_qsa": "all slots, payloads, logical/rotary tags and interval anchors",
        "vision_features": "all four feature arenas and generation/count arrays remain unchanged",
        "host_deduplication_limit_bytes": 32 << 30,
        "limits": "Full model state hashes at every stage; raw differences retain bounded byte windows. No timing comparison."}
    commands = []
    executor = None
    python_stream = torch.cuda.Stream(device=model.device)
    try:
        for side in ("python", "native"):
            if side == "native":
                reset.reset()
                evidence.value["reset"]["native"] = reset.hashes()
                _require(evidence, "identical_initial_storage", initial == evidence.value["reset"]["native"])
                _require(evidence, "identical_graphs_and_buffers", model.manifest() == manifest)
            builder = Commands(model, fixture, artifacts)
            if side == "native":
                executor = model.start_executor()
                model.runtime = executor
            # All future fixture pages and a far canary are observed from the start.
            observed_pages = 72
            _require(evidence, f"{side}.page_capacity", model.memory_report["pages"] > observed_pages + 1)
            observe(evidence, side, "initial", model, owner, observed_pages)
            previous_stage = "initial"
            for index, spec in enumerate(fixture["commands"]):
                command = builder.prepare(spec)
                if side == "python":
                    commands.append(command)
                else:
                    _require(evidence, f"upload.{index}.identical", command == {k: v for k, v in commands[index].items() if k != "python_records"})
                operation = operations.get((spec.get("batch"), spec.get("width")))
                active = [s for s in command["slots"] if s >= 0]
                if "require_previous" in spec:
                    previous = owner.previous_accepted.cpu().tolist()
                    _require(evidence, f"{side}.{index}.prior_full_column", all(previous[s] == spec["require_previous"] for s in active), actual=previous)
                uploads = [(name, torch.frombuffer(bytearray(builder.payloads[sha]), dtype=torch.uint8).pin_memory()) for name, sha in command["uploads"]] if side == "python" else []
                frame = torch.tensor([0], dtype=torch.int32, pin_memory=True) if side == "python" else None
                sequence = torch.tensor([command["sequence"]], dtype=torch.int64, pin_memory=True) if side == "python" else None
                before = _allocated(model.device)
                if side == "python":
                    with torch.cuda.stream(python_stream):
                        for name, cpu in uploads:
                            model.buffers[name].view(torch.uint8).reshape(-1).copy_(cpu, non_blocking=True)
                        model.buffers["frame_index"].copy_(frame, non_blocking=True)
                        model.buffers["submission_sequence"].copy_(sequence, non_blocking=True)
                        model.graphs[command["graph"]].replay()
                    python_stream.synchronize()
                    after = _allocated(model.device)
                    records = model.buffers["completion_records"][0].cpu().tolist()
                    command["python_records"] = records
                else:
                    fence = executor.submit(graph_indices[command["graph"]], [(name, 0, builder.payloads[sha]) for name, sha in command["uploads"]])
                    _require(evidence, f"native.{index}.fence", fence == command["sequence"])
                    deadline = time.monotonic() + 300
                    while (completion := executor.poll()) is None:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"native command {index} completion exceeded 300 seconds")
                        time.sleep(.001)
                    after = _allocated(model.device)
                    _require(evidence, f"native.{index}.completion_identity", completion.fence == fence and completion.graph_index == graph_indices[command["graph"]])
                    records = [list(row) for row in completion.records]
                    _require(evidence, f"native.{index}.completion_bitwise", records == commands[index]["python_records"])
                _require(evidence, f"{side}.{index}.replay_noalloc", before == after, before=before, after=after)
                checks, emitted = builder.complete(command, records)
                for name, passed in checks.items():
                    _require(evidence, f"{side}.{index}.{name}", passed, records=records if not passed else None)
                stage = f"{index:02d}.{command['graph']}"
                request = observe(evidence, side, stage, model, owner, observed_pages, operation)
                prior, current = evidence.value["observations"][previous_stage], evidence.value["observations"][stage]
                mutating = [slot for slot in active if spec["kind"] != "snapshot"
                            and (spec["kind"] != "speculative" or emitted.get(slot, 0))]
                writable_rows = {slot + 1 for slot in mutating}
                if spec["kind"] == "speculative":
                    writable_rows.update(row for slot in mutating
                        for row in range(owner.extra_base + slot * 7, owner.extra_base + (slot + 1) * 7))
                if spec["kind"] == "snapshot":
                    writable_rows.add(5 + spec["snapshot"])
                unchanged = [name for name in current if name.startswith("hybrid.") and int(name.split(".")[1]) not in writable_rows]
                unchanged += [name for name in current if name.startswith("request.") and int(name.rsplit("slot", 1)[1]) not in mutating]
                writable_pages = {page for slot in mutating for page in builder.requests[slot]["pages"]} if spec["kind"] in ("prefill", "speculative") else set()
                if spec["kind"] == "restore":
                    writable_pages.update(builder.requests[slot]["pages"][builder.requests[slot]["consumed"] // 64]
                        for slot in active if builder.requests[slot]["consumed"] % 64)
                unchanged += [name for name in current if name.startswith("qsa.") and (
                    (".page" in name and int(name.rsplit("page", 1)[1]) not in writable_pages)
                    or (".slot" in name and int(name.rsplit("slot", 1)[1]) not in {slot + 1 for slot in mutating}))]
                immutable_pages = {page for snapshot in builder.snapshots.values() for page in snapshot["pages"]}
                unchanged += [name for name in current if name.startswith("qsa.") and ".page" in name
                              and int(name.rsplit("page", 1)[1]) in immutable_pages]
                unchanged += [name for name in current if name.startswith("features.")]
                changed = [name for name in unchanged if prior[name][side]["sha256"] != current[name][side]["sha256"]]
                _require(evidence, f"{side}.{index}.inactive_state_unchanged", not changed, changed=changed, observed=len(unchanged))
                for slot, expected in builder.requests.items():
                    histogram = request["histogram"][slot]
                    actual_histogram = {token: int(histogram[token]) for token in histogram.nonzero().flatten().tolist()}
                    _require(evidence, f"{side}.{index}.histogram{slot}", actual_histogram == expected["histogram"])
                    _require(evidence, f"{side}.{index}.rng{slot}", request["rng_counter"][slot].item() == expected["generated"] and request["seed"][slot].item() == expected["seed"])
                    _require(evidence, f"{side}.{index}.history{slot}", request["ple_history"][slot].tolist() == ([248044, 248044] + expected["tokens"])[-2:])
                _require(evidence, f"{side}.{index}.previous_accepted", owner.previous_accepted.cpu().tolist() == builder.previous)
                _require(evidence, f"{side}.{index}.accepted_generation", owner.accepted_generation.cpu().tolist() == builder.accepted_generation)
                _require(evidence, f"{side}.{index}.target_rng", owner.rng.generation.cpu().tolist() == builder.target_generation and owner.rng.target_offset.cpu().tolist() == builder.target_offset)
                _require(evidence, f"{side}.{index}.no_draft_rng", not bool(owner.draft_rng_offset.cpu().any()) and not bool(owner.draft_rng_generation.cpu().any()))
                if operation is not None:
                    indices = operation.state.gdn.state_indices.cpu().tolist()
                    accepted = operation.state.gdn.accepted_tokens.cpu().tolist()
                    for row, slot in enumerate(spec["slots"]):
                        was_running = slot >= 0 and (emitted.get(slot, 0) > 0)
                        expected_indices = ([slot + 1, *range(owner.extra_base + slot * 7, owner.extra_base + (slot + 1) * 7)] if was_running else [0] * 8)
                        _require(evidence, f"{side}.{index}.indices{row}", indices[row] == expected_indices)
                        if was_running:
                            previous = (command["previous_accepted_before"][slot] if command["accepted_generation_before"][slot] == builder.requests[slot]["generation"] else 1)
                            _require(evidence, f"{side}.{index}.gdn_input_column{row}", accepted[row] == previous, actual=accepted[row], expected=previous)
                    _require(evidence, f"{side}.{index}.finite_logits", all(bool(torch.isfinite(operation.decoder.scratch.logits[row * spec['width']:(row + 1) * spec['width']].cpu()).all()) for row, slot in enumerate(spec["slots"]) if emitted.get(slot, 0)))
                if spec["kind"] in ("snapshot", "restore"):
                    from qualify.checkpoint_state import checkpoint_transfer_checks

                    slot = spec["slots"][0]
                    request_state = builder.requests[slot]
                    for key, passed in checkpoint_transfer_checks(owner.pool, owner.cache,
                            slot=slot, snapshot=spec["snapshot"], pages=request_state["pages"],
                            consumed=request_state["consumed"], accepted=builder.previous[slot],
                            extra_base=owner.extra_base).items():
                        _require(evidence, f"{side}.{index}.checkpoint.{key}", passed)
                previous_stage = stage
                evidence.value["commands"] = commands
                progress(f"{side} {index + 1}/{len(fixture['commands'])} {stage}; retained baseline {evidence.bytes / 2**30:.2f} GiB")
        _require(evidence, "all_observations_bitwise_equal", not evidence.value["mismatches"], mismatches=len(evidence.value["mismatches"]))
    finally:
        if executor is not None:
            executor.close()
            _require(evidence, "native_streams_drained", executor.drained)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--prefill-chunk-size", type=int, default=4096)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.prefill_chunk_size < 128 or args.prefill_chunk_size % 64:
        parser.error("prefill chunk size must be 64-aligned and at least 128")
    if args.artifacts.exists():
        parser.error("diagnostic tensor directory must not exist")
    import torch
    from transformers import AutoTokenizer

    from flare.preparation.qwen38 import MODEL_ID, REVISION

    snapshot = args.snapshot or Path.home() / ".cache/huggingface/hub" / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots" / REVISION
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    token, = tokenizer.encode(" apple", add_special_tokens=False)
    alternative, = tokenizer.encode(" orange", add_special_tokens=False)
    prompt = tokenizer.encode("Speculative state parity: " + "The capital of France is Paris. " * 20, add_special_tokens=False)[:65]
    fixture = command_plan(prompt, token, alternative)
    args.artifacts.mkdir(parents=True)
    (args.artifacts / "uploads").mkdir()
    write_json(args.artifacts / "fixture.json", fixture)
    if args.validate_only:
        if torch.cuda.is_initialized():
            raise AssertionError("CPU validation initialized CUDA")
        print(f"Validated {len(fixture['commands'])} state transitions")
        return 0
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("an explicitly assigned CUDA_VISIBLE_DEVICES is required")

    from flare.recipes.qwen38.bootstrap import PreparationConfig, prepare_model
    from flare.recipes.qwen38.speculative_model import SpeculativePreparation
    from qualify.graph_execution import Checks

    evidence = Checks(args.artifacts)
    evidence.value["fixture"] = fixture
    started = time.monotonic()
    progress = lambda message: print(f"{time.monotonic() - started:.1f}s {message}", flush=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    try:
        config = PreparationConfig(prefill_chunk_size=args.prefill_chunk_size, mtp=True,
            speculative_proposal_source="copy", history_drafts=True)
        with prepare_model(snapshot, config, progress=progress) as model:
            owner = next(x for x in model._resources if isinstance(x, SpeculativePreparation))
            evidence.value["memory_report"] = model.memory_report
            compare(model, owner, fixture, evidence, args.artifacts, progress)
    except BaseException as error:
        import traceback
        evidence.value["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        evidence.value["elapsed_seconds"] = time.monotonic() - started
        evidence.value["deduplicated_baseline_bytes"] = evidence.bytes
        evidence.value["passed"] = not evidence.value.get("error") and not evidence.value["mismatches"] and all(x["passed"] for x in evidence.value["checks"])
    return 0 if evidence.value["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
