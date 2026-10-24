"""Strict dense MTP qualification using retained production CUDA graphs (replay-commit copy of the
earlier MTP validator; same 47 checks, same byte comparisons, same 28 long-sequence identities).

Replay-commit layout: each request owns ONE committed continuation row (request_state_rows[slot, 0]); verification
writes no candidate rows, the round replays the accepted prefix into that row. What changed here and nothing else:
  - fixtures place one row per slot (reset) and the snapshot row right after the live rows;
  - the forced-prefix check restores the committed rows to their pre-round bytes before every accepted count and
    drives the round's replay commit (counts) instead of the earlier column copy (sources);
  - the commit-growth gate admits the last backed row instead of row 20 and has no descriptor bound to assert;
  - evidence goes to <root>/validation-attempts.jsonl.

The forced-prefix diagnostic invokes an already prepared target decoder and
its replay-commit operations eagerly; it is never a serving fallback. Ordinary
references, learned drafting, admission, snapshots and restores replay the
actual retained catalog graphs. No tolerance is applied to committed bytes.
"""

from collections.abc import Mapping
import json
from pathlib import Path
import time
import traceback

import torch


PROMPTS = (
    "Write a Python function that merges two sorted lists without modifying either input.\n```python\n",
    "Implement binary search in Rust and explain its invariant.\n```rust\n",
    "Write a TypeScript async retry helper with exponential backoff.\n```typescript\n",
    "Implement a stable topological sort in Python with cycle detection.\n```python\n",
)


def _bytes(tensor):
    return tensor.detach().cpu().contiguous().view(torch.uint8).reshape(-1)


def _compare(expected, actual):
    """Retain useful failure evidence while comparing every supplied byte."""
    result = {"passed": True, "fields": 0, "bytes": 0, "mismatches": []}
    for name in sorted(set(expected) | set(actual)):
        if name not in expected or name not in actual:
            result["mismatches"].append({"field": name, "error": "missing field"})
            continue
        left, right = _bytes(expected[name]), _bytes(actual[name])
        result["fields"] += 1
        result["bytes"] += left.numel()
        if left.shape != right.shape:
            result["mismatches"].append({"field": name, "expected_bytes": left.numel(), "actual_bytes": right.numel()})
        elif not torch.equal(left, right):
            unequal = left != right
            first = unequal.nonzero().flatten()[:8]
            mismatch = {"field": name, "different_bytes": int(unequal.sum()),
                "first_byte_offsets": first.tolist(), "expected": left[first].tolist(), "actual": right[first].tolist()}
            if expected[name].dtype == actual[name].dtype == torch.float32:
                elements = (first // 4).unique()
                mismatch.update(first_element_indices=elements.tolist(),
                    expected_float32=left.view(torch.float32)[elements].tolist(),
                    actual_float32=right.view(torch.float32)[elements].tolist())
            result["mismatches"].append(mismatch)
    result["passed"] = not result["mismatches"]
    return result


def _sampling_support(logits, probabilities, *, sampled=None):
    """Bounded JSON evidence for exact maximum ties and retained probability support."""
    logits, probabilities = logits.detach().cpu(), probabilities.detach().cpu()
    maximum = logits.max()
    tied = (logits == maximum).nonzero().flatten()
    supported = (probabilities > 0).nonzero().flatten()
    limit = 256
    result = {"maximum_logit": float(maximum), "maximum_tie_count": tied.numel(),
              "maximum_token_ids": tied[:limit].tolist(), "maximum_ids_truncated": tied.numel() > limit,
              "positive_probability_count": supported.numel(),
              "positive_probability_token_ids": supported[:limit].tolist(),
              "positive_probabilities": probabilities[supported[:limit]].tolist(),
              "probability_ids_truncated": supported.numel() > limit,
              "probability_mass": float(probabilities.double().sum()),
              "logits_exact_bf16": torch.equal(logits.float().view(torch.int32),
                                                logits.bfloat16().float().view(torch.int32))}
    if sampled is not None:
        result["sampled_token_id"] = int(sampled)
        result["sampled_token_probability"] = float(probabilities[int(sampled)])
    return result


class _Harness:
    def __init__(self, prepared, *, max_new_tokens=256):
        from ignition.recipes.qwen38_dense.speculative import PreparedRound

        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("qualification requires a positive output-token admission limit")
        self.prepared, self.max_new_tokens = prepared, max_new_tokens
        catalogs = [resource for resource in prepared.resources if isinstance(resource, Mapping)
                    and any(isinstance(value, PreparedRound) for value in resource.values())]
        if len(catalogs) != 1:
            raise ValueError("expected one retained dense MTP graph-owner catalog")
        self.catalog = catalogs[0]
        self.rounds = {(value.batch, value.width): (name, value) for name, value in self.catalog.items()
                       if isinstance(value, PreparedRound)}
        self.owner = next(iter(self.rounds.values()))[1].owner
        self.pool, self.inputs, self.requests = self.owner.pool, self.owner.inputs, self.owner.requests
        self.width = max(width for _, width in self.rounds)
        self.capacity = self.requests.capacity
        self.device = self.pool.backing.device
        self.commands = self.catalog[self.name("init", 1, True)].commands
        self.admission = self.catalog[self.name("init", 1, True)].state.admission.buffers
        self.page_tokens = int(prepared.server_config["page_tokens"])
        # Replay-commit: one committed row per slot (1..capacity); the snapshot row follows them.
        self.snapshot_row = 1 + self.capacity
        physical_rows, physical_pages = self.pool.memory_capacity.cpu().tolist()
        if (physical_rows <= self.snapshot_row or physical_pages < 3 * self.capacity + 2
                or self.inputs.page_tables.shape[1] < 2 or self.page_tokens != 128):
            raise ValueError("boundary qualification needs backed live/checkpoint/snapshot rows, three pages per request and two spares")
        self.spare_page = 3 * self.capacity
        if torch.cuda.get_device_capability(self.device) != (7, 0):
            raise ValueError("strict MTP qualification requires actual SM70 hardware")
        for name in self.catalog:
            if name not in prepared.graphs or not isinstance(prepared.graphs[name], torch.cuda.CUDAGraph):
                raise TypeError(f"{name}: actual retained CUDA graph required")
        self.epoch, self.sequence = 0, int(prepared.buffers["submission_sequence"].item())
        self.replays = {}
        self.startup = self._save_mutable()
        self.linear = [i for i, kind in enumerate(self.owner.weights.config.layers) if kind == "linear_attention"]
        self.full = [i for i, kind in enumerate(self.owner.weights.config.layers) if kind == "full_attention"]
        # SM70 prefill attention status accumulators of the retained prefill decoders; every prefill
        # replay with a live row asserts status == 0 (bit 0 non-finite K/V, bit 1 V range > 2^80, bit 2 inadmissible plan).
        self.prefill_status = {}
        for name, forward in self.catalog.items():
            decoder = getattr(forward, "decoder", None) or getattr(getattr(forward, "generation", None), "decoder", None)
            status = getattr(decoder, "prefill_status", None)
            if status is not None:
                self.prefill_status[name] = status
        self.prefill_status_checks = 0

    def name(self, kind, batch, mtp, *, width=None):
        choices = [(signature.token_capacity, name) for name, signature in self.prepared.signatures.items()
                   if signature.kind == kind and signature.batch_capacity == batch
                   and signature.speculation_enabled == mtp
                   and (width is None or signature.token_capacity == width)]
        if not choices:
            raise ValueError(f"missing captured {kind}: B={batch}, MTP={mtp}, width={width}")
        return (max(choices) if kind == "prefill" else min(choices))[1]

    def _mutable(self):
        tensors = list(self.prepared.buffers.items())
        for prefix, owner in (("request", self.requests), ("commands", self.commands),
                              ("admission", self.admission), ("mtp_rng", self.owner.rng)):
            tensors.extend((prefix + "." + key, value) for key, value in vars(owner).items()
                           if isinstance(value, torch.Tensor))
        tensors.extend(("mtp." + name, getattr(self.owner, name)) for name in ("generation", "offset"))
        pool_addresses = {value.untyped_storage().data_ptr()
                          for value in (self.pool.backing, self.pool.reserved_backing, *self.pool.pages.values())}
        seen = set()
        for name, tensor in tensors:
            if tensor.untyped_storage().data_ptr() in pool_addresses:
                continue
            key = (tensor.data_ptr(), tensor.shape, tensor.stride(), tensor.dtype)
            if key not in seen:
                seen.add(key)
                yield name, tensor

    def _save_mutable(self):
        return [(name, tensor, tensor.detach().cpu().clone()) for name, tensor in self._mutable()]

    @staticmethod
    def _load(saved):
        for _, tensor, value in saved:
            tensor.copy_(value)

    def reset(self):
        self.pool.clear_for_preparation()
        self._load(self.startup)
        self.inputs.request_state_rows.zero_()
        for slot in range(self.capacity):
            self.inputs.request_state_rows[slot, 0] = 1 + slot
        self.inputs.page_tables.fill_(-1)
        self.owner.draft_tables.fill_(-1)
        for slot in range(self.capacity):
            self.inputs.page_tables[slot, :2].copy_(torch.tensor([3 * slot, 3 * slot + 1], device=self.device))
            self.owner.draft_tables[slot, :2].copy_(torch.tensor([3 * slot + 1, 3 * slot + 2], device=self.device))

    def rows(self, slots, *, lengths=None, sample=False):
        self.inputs.row_slot_ids.fill_(-1)
        self.inputs.row_generations.zero_()
        self.inputs.query_lengths.zero_()
        self.inputs.sample_enabled.zero_()
        for row, slot in enumerate(slots):
            self.inputs.row_slot_ids[row] = slot
            if slot >= 0:
                self.inputs.row_generations[row] = self.requests.generation[slot]
                self.inputs.query_lengths[row] = 1 if lengths is None else lengths[row]
                self.inputs.sample_enabled[row] = int(sample)
        self.inputs.live_rows.fill_(len(slots))

    def replay(self, name, slots=(), *, rejected=()):
        """Replay one retained graph. ``rejected`` lists slots the batch validator must have refused this round
        (status ERROR, error code 32 = admission/row validation); every other live slot must have completed normally."""
        self.sequence += 1
        frame = self.sequence % 2
        self.prepared.buffers["frame_index"].fill_(frame)
        self.prepared.buffers["submission_sequence"].fill_(self.sequence)
        allocations = torch.cuda.memory_stats(self.device)["allocation.all.allocated"]
        self.prepared.graphs[name].replay()
        torch.cuda.synchronize(self.device)
        if torch.cuda.memory_stats(self.device)["allocation.all.allocated"] != allocations:
            raise RuntimeError(f"{name}: CUDA allocation occurred during graph replay")
        self.replays[name] = self.replays.get(name, 0) + 1
        status = self.prefill_status.get(name)
        if status is not None and any(slot >= 0 for slot in slots):
            self.prefill_status_checks += 1
            value = int(status.item())
            if value:
                raise RuntimeError(f"{name}: SM70 prefill attention status {value} after the chunk "
                                   "(bit 0 non-finite K/V, bit 1 V range > 2^80, bit 2 inadmissible plan)")
        records = self.prepared.buffers["completion_records"][frame].cpu().clone()
        for slot in slots:
            if slot < 0:
                continue
            record = records[slot].tolist()
            if record[0] != int(self.requests.generation[slot]) or record[1] != self.sequence:
                raise RuntimeError(f"{name}: stale completion for slot {slot}: {record}")
            if slot in rejected:
                if record[5] != 3 or record[6] != 32 or int(self.requests.error_code[slot]) != 32:
                    raise RuntimeError(f"{name}: slot {slot} was not rejected by the batch validator: {record}")
                continue
            if record[5] not in (1, 2) or record[6] or int(self.requests.error_code[slot]):
                raise RuntimeError(f"{name}: request rejected for slot {slot}: {record}")
            if record[2] != int(self.requests.consumed_tokens[slot]):
                raise RuntimeError(f"{name}: completion/request consumed endpoints disagree: {record}")
        if name.startswith(("init", "snapshot", "restore")):
            owner = self.catalog[name]
            if self.commands.errors.any() or owner.state.admission_errors.any():
                raise RuntimeError(f"{name}: maintenance validation failed: {self.commands.errors.cpu().tolist()}")
        return records

    def admission_row(self, row, slot, ids):
        from ignition.components.sampling import RequestAdmission

        self.admission.stage(row, slot, RequestAdmission(generation=self.epoch, seed=17 + slot,
            max_new_tokens=self.max_new_tokens, temperature=1., top_k=1, top_p=1., repetition_penalty=1., eos_token_ids=()), ids)

    def transfers(self, entries):
        c = self.commands
        c.request_slots.fill_(-1)
        c.source_slots.fill_(-1)
        c.destination_slots.fill_(-1)
        c.operation.zero_()
        c.expected_consumed.fill_(-1)
        for row, (slot, source, destination, endpoint, reset) in enumerate(entries):
            c.request_slots[row], c.source_slots[row], c.destination_slots[row] = slot, source, destination
            c.expected_consumed[row], c.operation[row] = endpoint, int(reset)
        c.num_ops.fill_(len(entries))

    def initialize(self, prompts, *, mtp=True):
        self.reset()
        self.epoch += 1
        batch = len(prompts)
        for slot, ids in enumerate(prompts):
            self.admission_row(slot, slot, ids)
        self.admission.admission_count.fill_(batch)
        # Replay-commit: one committed row per request (rows 1..capacity), whatever the verification width.
        self.transfers([(slot, -1, 1 + slot, 0, True) for slot in range(batch)])
        self.replay(self.name("init", batch, mtp), range(batch))
        self.admission.admission_count.zero_()
        self.commands.num_ops.zero_()
        name = self.name("prefill", 1, mtp)
        capacity = self.prepared.signatures[name].token_capacity
        for slot, ids in enumerate(prompts):
            for start in range(0, len(ids), capacity):
                part = ids[start:start + capacity]
                self.rows([slot], lengths=[len(part)])
                self.inputs.token_ids.zero_()
                self.inputs.token_ids[:len(part)].copy_(torch.tensor(part, device=self.device))
                self.replay(name, [slot])
            if int(self.requests.consumed_tokens[slot]) != len(ids):
                raise RuntimeError("captured prefill did not consume the complete prompt")
        self.rows(list(range(batch)))

    def model(self, slots, endpoints, *, draft=False):
        result = {}
        for slot, endpoint in zip(slots, endpoints, strict=True):
            row = int(self.inputs.request_state_rows[slot, 0])
            for layer in self.linear:
                state = self.pool.recurrent(layer)
                result[f"slot{slot}.layer{layer}.recurrent"] = state.recurrent[row].cpu().clone()
                result[f"slot{slot}.layer{layer}.convolution"] = state.convolution[row].cpu().clone()
            result[f"slot{slot}.last_hidden"] = self.pool.last_hidden[row].cpu().clone()
            for name, cache in self.pool.pages.items():
                is_draft = name.startswith("draft.")
                if is_draft and not draft:
                    continue
                length = max(0, endpoint - 1) if is_draft else endpoint
                tables = self.owner.draft_tables if is_draft else self.inputs.page_tables
                bytes_per_token = cache.shape[1] // self.page_tokens
                for logical in range((length + self.page_tokens - 1) // self.page_tokens):
                    count = min(self.page_tokens, length - logical * self.page_tokens)
                    page = int(tables[slot, logical])
                    if not 0 <= page < int(self.pool.memory_capacity[1]):
                        raise RuntimeError("attempt to compare an unbacked logical cache page")
                    result[f"slot{slot}.{name}.page{logical}"] = cache[page, :count * bytes_per_token].cpu().clone()
        return result

    def branch(self, batch):
        saved = self._save_mutable()
        for slot in range(batch):
            row = int(self.inputs.request_state_rows[slot, 0])
            saved.append((f"row{row}", self.pool.backing[row], self.pool.backing[row].cpu().clone()))
        for name, cache in self.pool.pages.items():
            pages = int(self.pool.memory_capacity[1])
            saved.append((name, cache[:pages], cache[:pages].cpu().clone()))
        return saved

    def decode_inputs(self, candidates, slots):
        self.rows(slots)
        for row, slot in enumerate(slots):
            if slot >= 0:
                self.requests.pending_token[slot] = candidates[row]
        self.replay(self.name("decode", len(slots), False), slots)

    def decode_probe_enabled(self):
        """The S1L decode attention is wired when the retained decode decoders carry a four-agent pack."""
        return any(getattr(getattr(f, "decoder", None), "decode_agents", None) is not None for f in self.catalog.values())

    def injected_status(self, ids):
        """Poison one visible K page of the first full-attention layer before the last prefill chunk. The chunk
        must report status bit 0 and the completion record must carry ERROR (3) / error code 34 (requests.status /
        error_code likewise); after restoring the page the same prompt must prefill cleanly again."""
        self.reset()
        self.epoch += 1
        slot = 0
        self.admission_row(slot, slot, ids)
        self.admission.admission_count.fill_(1)
        self.transfers([(slot, -1, 1 + slot, 0, True)])
        self.replay(self.name("init", 1, True), [slot])
        self.admission.admission_count.zero_()
        self.commands.num_ops.zero_()
        name = self.name("prefill", 1, True)
        if name not in self.prefill_status:
            raise RuntimeError(f"{name}: retained prefill graph exposes no SM70 prefill attention status")
        capacity = self.prepared.signatures[name].token_capacity
        # The poisoned page must hold COMMITTED context (a single-chunk prompt rewrites page 0 itself before attention
        # reads it): extend the prompt until it needs at least two chunks and poison before the last one.
        ids = list(ids)
        while len(ids) <= capacity:
            ids = ids + ids
        parts = [ids[start:start + capacity] for start in range(0, len(ids), capacity)]
        if len(parts) < 2:
            raise RuntimeError("injected-status probe needs a committed page before the poisoned chunk")

        def chunk(part):
            self.rows([slot], lengths=[len(part)])
            self.inputs.token_ids.zero_()
            self.inputs.token_ids[:len(part)].copy_(torch.tensor(part, device=self.device))

        for part in parts[:-1]:
            chunk(part)
            self.replay(name, [slot])
        page, layer = int(self.inputs.page_tables[slot, 0]), self.full[0]
        key = self.pool.pages[f"layer{layer}.key"]
        saved = key[page].clone()
        key[page].view(torch.int16)[:4] = 0x7FC0          # four BF16 NaN words in position 0 of the first visible page
        chunk(parts[-1])
        self.sequence += 1
        frame = self.sequence % 2
        self.prepared.buffers["frame_index"].fill_(frame)
        self.prepared.buffers["submission_sequence"].fill_(self.sequence)
        self.prepared.graphs[name].replay()
        torch.cuda.synchronize(self.device)
        self.replays[name] = self.replays.get(name, 0) + 1
        status = int(self.prefill_status[name].item())
        record = self.prepared.buffers["completion_records"][frame][slot].cpu().tolist()
        request_status, request_error = int(self.requests.status[slot]), int(self.requests.error_code[slot])
        key[page].copy_(saved)
        result = {"prefill_status": status, "record_status": record[5], "record_error_code": record[6],
                  "request_status": request_status, "request_error_code": request_error, "poisoned_layer": layer,
                  "poisoned_page": page, "prefill_chunks": len(parts), "record_sampled_token": record[4]}
        result["passed"] = bool(status & 1) and record[5] == 3 and record[6] == 34 and request_status == 3 and request_error == 34
        self.initialize([ids])                            # clean replay of the same prompt: status 0 on every chunk, no error
        result["recovered"] = True
        return result

    def replay_raw(self, name, slot):
        """Replay without the rejection checks; return (records row, requests.status, requests.error_code) for slot."""
        self.sequence += 1
        frame = self.sequence % 2
        self.prepared.buffers["frame_index"].fill_(frame)
        self.prepared.buffers["submission_sequence"].fill_(self.sequence)
        self.prepared.graphs[name].replay()
        torch.cuda.synchronize(self.device)
        self.replays[name] = self.replays.get(name, 0) + 1
        record = self.prepared.buffers["completion_records"][frame][slot].cpu().tolist()
        return record, int(self.requests.status[slot]), int(self.requests.error_code[slot])

    def decode_nonfinite_probe(self, ids, *, width):
        """S1L gate 3: NaN K words in a VISIBLE page must fail the request with the samplers' non-finite code 1
        (status 3) on an ordinary decode step and with speculation's code 6 on a speculative round; NaN in a physical page
        no table references must leave the emitted tokens unchanged.

        Every replay starts from the SAME complete state (``branch``: mutable buffers + pool backing rows + every pool
        page, as ``forced`` does) - an earlier ``_save_mutable`` left the pool out, so the speculative round's committed
        K/V and recurrent state leaked from the clean replay into the next one and the comparison was not like for like
        (prepare-32k.json attempt 1: speculative unreferenced_identical False with the kernel untouched). A second clean
        replay (``clean_repeat``) is recorded as the idempotence control; the full completion records are kept."""
        results = {}
        layer = self.full[0]
        key = self.pool.pages[f"layer{layer}.key"]
        for kind in ("decode", "speculative"):
            mtp = kind == "speculative"
            self.initialize([ids], mtp=mtp)
            self.rows([0], sample=True)
            self.replay(self.name("resume", 1, mtp), [0])
            name = self.name("speculative", 1, True, width=width) if mtp else self.name("decode", 1, False)
            self.rows([0], lengths=[width if mtp else 1], sample=True)
            base = self.branch(1)
            clean, _, _ = self.replay_raw(name, 0)
            # idempotence control: the same replay from the same complete state must emit the same record
            self._load(base)
            clean_repeat, _, _ = self.replay_raw(name, 0)
            # unreferenced page: a spare physical page outside every table must not matter
            self._load(base)
            spare = int(self.pool.memory_capacity[1]) - 1
            key[spare].view(torch.int16)[:64] = 0x7FC0
            unreferenced, _, _ = self.replay_raw(name, 0)
            # visible page: the request's first page, position 0 (restoring the base also restores the spare page)
            self._load(base)
            page = int(self.inputs.page_tables[0, 0])
            key[page].view(torch.int16)[:4] = 0x7FC0
            poisoned, status, code = self.replay_raw(name, 0)
            self._load(base)
            emitted = slice(16, 16 + width) if mtp else slice(4, 5)
            same = lambda a, b: a[emitted] == b[emitted] and a[5] == b[5] and a[6] == b[6]   # noqa: E731
            # speculative rounds report a non-finite target distribution with speculation's own code 6 (speculation.py), the
            # ordinary decode sampler with code 1; both are ERROR (3).
            expected_code = 6 if mtp else 1
            results[kind] = {"clean_token": clean[4], "unreferenced_token": unreferenced[4],
                             "clean_repeat_identical": same(clean_repeat, clean),
                             "unreferenced_identical": same(unreferenced, clean) and unreferenced[6] == 0,
                             "poisoned_record_status": poisoned[5], "poisoned_record_error_code": poisoned[6],
                             "poisoned_request_status": status, "poisoned_request_error_code": code, "poisoned_page": page,
                             "spare_page": spare, "layer": layer, "records": {"clean": clean, "clean_repeat": clean_repeat,
                                                                              "unreferenced": unreferenced, "poisoned": poisoned}}
            results[kind]["passed"] = (results[kind]["clean_repeat_identical"] and results[kind]["unreferenced_identical"]
                                       and poisoned[5] == 3 and poisoned[6] == expected_code and status == 3 and code == expected_code)
        results["passed"] = all(results[k]["passed"] for k in ("decode", "speculative"))
        return results

    def greedy(self, prompts, *, width=None, tokens=16):
        batch, mtp = len(prompts), width is not None
        self.initialize(prompts, mtp=mtp)
        self.rows(list(range(batch)), sample=True)
        records = self.replay(self.name("resume", batch, mtp), range(batch))
        result = [[int(records[slot, 4])] for slot in range(batch)]
        rounds = []
        while any(len(ids) < tokens for ids in result):
            lengths = [min(width or 1, tokens - len(ids)) for ids in result]
            slots = [slot if lengths[slot] else -1 for slot in range(batch)]
            self.rows(slots, lengths=lengths, sample=True)
            before = self.requests.consumed_tokens[:batch].cpu().clone()
            name = self.name("speculative", batch, True, width=width) if mtp else self.name("decode", batch, False)
            records = self.replay(name, slots)
            emitted = []
            for slot in range(batch):
                count = int(records[slot, 8]) if mtp and slots[slot] >= 0 else int(slots[slot] >= 0)
                if slots[slot] >= 0 and not 1 <= count <= lengths[slot]:
                    raise RuntimeError(f"{name}: invalid emitted count {count}")
                ids = records[slot, 16:16 + count].tolist() if mtp else ([int(records[slot, 4])] if count else [])
                if any(not 0 <= token < self.requests.vocab_size for token in ids):
                    raise RuntimeError(f"{name}: emitted token outside vocabulary")
                result[slot].extend(ids)
                if int(self.requests.consumed_tokens[slot]) != int(before[slot]) + count:
                    raise RuntimeError(f"{name}: emitted count and committed prefix disagree")
                emitted.append(count)
            rounds.append(emitted)
        return {"token_ids": result, "round_emitted_counts": rounds,
                "mean_emitted_per_active_request_round": sum(map(sum, rounds)) / sum(sum(c > 0 for c in row) for row in rounds)}

    def actual_prefix(self, prompts, width):
        """Compare a production graph's actual commit to captured ordinary steps."""
        batch = len(prompts)
        slots = list(range(batch))
        self.initialize(prompts, mtp=True)
        self.rows(slots, sample=True)
        self.replay(self.name("resume", batch, True), slots)
        base = self.branch(batch)
        before = self.requests.consumed_tokens[:batch].cpu().tolist()
        self.rows(slots, lengths=[width] * batch, sample=True)
        records = self.replay(self.name("speculative", batch, True, width=width), slots)
        counts = records[:batch, 8].tolist()
        if any(not 1 <= count <= width for count in counts):
            raise RuntimeError("production graph emitted an invalid accepted prefix")
        candidates = self.rounds[batch, width][1].candidates.cpu().clone()
        endpoints = [start + count for start, count in zip(before, counts, strict=True)]
        actual = self.model(slots, endpoints)
        self._load(base)
        for column in range(max(counts)):
            active = [slot if column < counts[slot] else -1 for slot in slots]
            self.decode_inputs(candidates[:, column], active)
        compared = _compare(self.model(slots, endpoints), actual)
        return {**compared, "batch": batch, "verification_width": width,
                "accepted_inputs": counts, "candidate_ids": candidates.tolist(),
                "execution": "captured production speculative commit versus captured ordinary teacher-forced decode"}

    def forced(self, prompts, *, inactive=False, width=None):
        """Every acceptance count from one prepared target verification."""
        from ignition.components.state import save_hidden

        batch, width = len(prompts), self.width if width is None else width
        self.initialize(prompts, mtp=True)
        slots = list(range(batch))
        if inactive:
            slots[1] = -1
        active_slots = [slot for slot in slots if slot >= 0]
        endpoint = len(prompts[0])
        base = self.branch(batch)
        # Fixed, explicit candidate inputs independent of target/learned sampling.
        candidates = torch.tensor([[prompts[row][-(column + 1)] for column in range(width)]
                                  for row in range(batch)], dtype=torch.int64, device=self.device)
        references = []
        untouched = self.model([1], [endpoint], draft=True) if inactive else None
        for column in range(width):
            self.decode_inputs(candidates[:, column], slots)
            references.append(self.model(active_slots, [endpoint + column + 1] * len(active_slots)))
        self._load(base)
        operation = self.rounds[batch, width][1]
        self.rows(slots, lengths=[width] * batch)
        states = self.inputs.request_state_rows[:batch, 0].contiguous()
        canonical_rows = states
        active = torch.tensor([slot >= 0 for slot in slots], device=self.device)
        lengths = active.to(torch.int32) * width
        starts = self.requests.consumed_tokens[:batch].clone()
        operation.candidates.copy_(candidates)
        operation.metadata.set(starts, lengths, states, self.inputs.page_tables[:batch], candidates,
                               self.inputs.live_rows)
        d, m = operation.delta, operation.metadata
        d.query_start_loc.copy_(m.query_start_loc)
        d.convolution_slots.copy_(torch.where(active, states, -1))
        d.fresh.copy_(starts == 0)
        d.accepted_tokens.fill_(1)
        d.state_indices[:, 0].copy_(torch.where(active, states, 0))
        d.num_seqs.copy_(m.live_counts[1:])
        d.num_tokens.copy_(m.live_counts[:1])
        # The committed rows' pre-round bytes (S0): verification must leave them untouched, and every accepted
        # count replays from them. Recurrent copies live on the CPU (4 rows x 144 MiB).
        recurrent_before = {layer: self.pool.recurrent(layer).recurrent[states].cpu().clone() for layer in self.linear}
        operation.decoder()
        torch.cuda.synchronize(self.device)
        # Bitwise (uint8 views): a +0.0/-0.0 difference must count as a write.
        untouched_rows = all(torch.equal(_bytes(self.pool.recurrent(layer).recurrent[states]), _bytes(recurrent_before[layer]))
                             for layer in self.linear)
        conv = {layer: self.pool.recurrent(layer).convolution[states].clone() for layer in self.linear}
        hidden = operation.decoder.scratch.hidden.view(batch, width, -1)
        checks = []
        for accepted in range(1, width + 1):
            for layer in self.linear:
                state = self.pool.recurrent(layer)
                state.convolution[states] = conv[layer]
                state.recurrent[states] = recurrent_before[layer].to(self.device)
            operation.commit_counts.copy_(active.to(torch.int32) * accepted)
            operation.replay_commit()
            operation.convolution_commit()
            save_hidden(hidden[:, accepted - 1], self.pool.last_hidden, canonical_rows, active)
            torch.cuda.synchronize(self.device)
            compared = _compare(references[accepted - 1], self.model(active_slots, [endpoint + accepted] * len(active_slots)))
            compared.update(accepted_inputs=accepted, verification_width=width, batch=batch,
                            prompt_tokens=endpoint, inactive=inactive, verification_left_committed_rows_untouched=untouched_rows)
            compared["passed"] &= untouched_rows
            if inactive:
                compared["inactive_state"] = _compare(untouched, self.model([1], [endpoint], draft=True))
                compared["passed"] &= compared["inactive_state"]["passed"]
            checks.append(compared)
        return {"passed": all(check["passed"] for check in checks), "checks": checks,
                "candidate_ids": candidates.cpu().tolist(),
                "execution": "captured ordinary decode versus eager invocation of already-prepared verification decoder and prepared replay-commit operations (committed rows restored to their pre-round bytes before every accepted count)"}

    def inactive(self, batch):
        rows, pages = self.pool.memory_capacity.cpu().tolist()
        values = [("rows", self.pool.backing[:rows]), *[(name, value[:pages]) for name, value in self.pool.pages.items()]]
        values.extend(("requests." + name, value) for name, value in vars(self.requests).items() if isinstance(value, torch.Tensor))
        values.extend(("rng." + name, value) for name, value in vars(self.owner.rng).items())
        values.extend(("mtp." + name, getattr(self.owner, name)) for name in ("generation", "offset"))
        before = {name: value.cpu().clone() for name, value in values}
        self.rows([])
        self.replay(self.name("speculative", batch, True, width=self.width))
        return _compare(before, {name: value for name, value in values})

    def alias_lane(self, prompt, *, stale=False):
        """An ignored packed lane must not invalidate its live slot alias."""
        self.initialize([prompt], mtp=True)
        self.rows([0], sample=True)
        self.replay(self.name("resume", 1, True), [0])
        base = self.branch(1)
        name = self.name("speculative", 4, True, width=self.width)

        def result():
            records = self.replay(name, [0])
            count = int(records[0, 8])
            if not 1 <= count <= self.width:
                raise RuntimeError(f"{name}: aliased-lane check did not make valid progress: {count}")
            record = records[0].clone()
            record[1] = 0  # Submission sequences intentionally differ between replays.
            values = self.model([0], [len(prompt) + count], draft=True)
            values["completion"] = record
            for prefix, owner in (("request", self.requests), ("rng", self.owner.rng)):
                values.update((prefix + "." + key, value.cpu().clone()) for key, value in vars(owner).items()
                              if isinstance(value, torch.Tensor))
            values.update(("mtp." + key, getattr(self.owner, key).cpu().clone())
                          for key in ("generation", "offset"))
            return values, records[0, 16:16 + count].tolist()

        self.rows([0], lengths=[self.width], sample=True)
        expected, expected_ids = result()
        self._load(base)
        self.rows([0], lengths=[self.width], sample=True)
        self.inputs.row_slot_ids[1] = 0
        self.inputs.row_generations[1] = self.requests.generation[0] - int(stale)
        self.inputs.live_rows.fill_(2 if stale else 1)
        # rows() leaves the ignored lane's query length at zero. Both the
        # padding case and stale-generation case would poison slot zero if
        # extra speculative validation reduced bad lanes without residency.
        actual, actual_ids = result()
        compared = _compare(expected, actual)
        return {**compared, "batch": 4, "live_rows": 2 if stale else 1,
                "slot_ids": [0, 0, -1, -1], "query_lengths": [self.width, 0, 0, 0],
                "stale_generation": stale, "expected_token_ids": expected_ids,
                "actual_token_ids": actual_ids}

    def snapshot_branch(self, prompts):
        batch = len(prompts)
        self.initialize(prompts, mtp=True)
        endpoint = len(prompts[0])
        original = self.model(range(batch), [endpoint] * batch, draft=True)
        row = int(self.inputs.request_state_rows[0, 0])
        self.rows([0])
        self.transfers([(0, row, self.snapshot_row, endpoint, False)])
        self.replay(self.name("snapshot", batch, True), [0])
        self.commands.num_ops.zero_()
        immutable = self.pool.backing[self.snapshot_row].cpu().clone()
        # Branch A updates the original trailing draft page before restoration.
        self.rows([0], sample=True)
        self.replay(self.name("resume", 1, True), [0])
        self.rows([0], lengths=[self.width], sample=True)
        records = self.replay(self.name("speculative", 1, True, width=self.width), [0])
        count = int(records[0, 8])
        first_ids = records[0, 16:16 + count].tolist()
        first_state = self.model([0], [endpoint + count], draft=True)
        self.epoch += 1
        self.admission_row(0, 0, prompts[0])
        self.admission.admission_count.fill_(1)
        # Native offset=1 shares target page zero, while both remaining physical
        # allocations are private. P=129 must save the now-full draft page zero;
        # P=130 must additionally save partial draft page one.
        self.inputs.page_tables[0, 1] = self.spare_page
        self.owner.draft_tables[0, 0] = self.spare_page
        self.owner.draft_tables[0, 1] = self.spare_page + 1
        for name, cache in self.pool.pages.items():
            cache[self.spare_page:self.spare_page + 2].fill_(173 if name.endswith("key") else 91)
        self.transfers([(0, self.snapshot_row, row, endpoint, False)])
        self.replay(self.name("restore", batch, True), [0])
        self.admission.admission_count.zero_()
        self.commands.num_ops.zero_()
        restored = _compare(original, self.model(range(batch), [endpoint] * batch, draft=True))
        self.rows([0], sample=True)
        self.replay(self.name("resume", 1, True), [0])
        self.rows([0], lengths=[self.width], sample=True)
        records = self.replay(self.name("speculative", 1, True, width=self.width), [0])
        second_count = int(records[0, 8])
        second_ids = records[0, 16:16 + second_count].tolist()
        branched = _compare(first_state, self.model([0], [endpoint + second_count], draft=True))
        snapshot = _compare({"immutable_snapshot": immutable}, {"immutable_snapshot": self.pool.backing[self.snapshot_row]})
        return {"passed": restored["passed"] and branched["passed"] and snapshot["passed"] and first_ids == second_ids,
                "endpoint": endpoint, "batch": batch, "restore": restored, "branch_state": branched,
                "immutable_snapshot": snapshot, "first_tokens": first_ids, "restored_tokens": second_ids,
                "target_full_page_shared": True, "draft_private_pages_relocated": 2}

    def cleanup(self):
        self.pool.clear_for_preparation()
        self._load(self.startup)
        torch.cuda.synchronize(self.device)
        result = _compare({name: value for name, _, value in self.startup},
                          {name: tensor for name, tensor, _ in self.startup})
        rows, pages = self.pool.memory_capacity.cpu().tolist()
        # CUDA count_nonzero can promote this multi-GiB uint8 pool to an
        # eight-times-larger temporary. Inspect bounded CPU copies instead.
        nonzero = {"continuations": sum(int(torch.count_nonzero(self.pool.backing[row].cpu()))
                                        for row in range(rows))}
        nonzero.update({name: sum(int(torch.count_nonzero(value[page].cpu())) for page in range(pages))
                        for name, value in self.pool.pages.items()})
        result["pool_nonzero_bytes"] = nonzero
        result["passed"] &= not any(nonzero.values())
        return result

    def trace_first_divergence(self, prompts, *, width=None, output_token_index=None):
        """Failure-only diagnosis; never participates in qualification."""
        batch, width = len(prompts), self.width if width is None else width
        if (batch, width) not in self.rounds:
            raise ValueError(f"missing diagnostic target graph for B={batch}, width={width}")
        slots = list(range(batch))
        sampling, common_prefix, prefill_comparison = None, None, None
        if output_token_index is None:
            self.initialize(prompts, mtp=True)
            candidates = torch.tensor([[ids[-(column + 1)] for column in range(width)] for ids in prompts],
                                      dtype=torch.int64, device=self.device)
            endpoints = [len(ids) + 1 for ids in prompts]
        else:
            if type(output_token_index) is not int or not 1 <= output_token_index < self.max_new_tokens:
                raise ValueError("sampling diagnosis requires a positive admitted output-token index")
            # Compare the complete target continuation from the two prefill
            # graphs before either mode has consumed a generated token.
            self.initialize(prompts, mtp=False)
            prefill_reference = self.model(slots, [len(ids) for ids in prompts])
            self.initialize(prompts, mtp=True)
            prefill_comparison = _compare(prefill_reference, self.model(slots, [len(ids) for ids in prompts]))
            del prefill_reference
            self.initialize(prompts, mtp=False)
            self.rows(slots, sample=True)
            records = self.replay(self.name("resume", batch, False), slots)
            common_prefix = [[int(records[slot, 4])] for slot in slots]
            for _ in range(1, output_token_index):
                self.rows(slots, sample=True)
                records = self.replay(self.name("decode", batch, False), slots)
                for slot in slots:
                    common_prefix[slot].append(int(records[slot, 4]))
            # Column zero consumes the ordinary pending token at the shared
            # prefix. Later diagnostic columns cannot affect its causal output.
            candidates = self.requests.pending_token[:batch, None].expand(batch, width).clone()
            endpoints = [len(ids) + output_token_index for ids in prompts]
            if (any(len(ids) != output_token_index for ids in common_prefix)
                    or self.requests.consumed_tokens[:batch].cpu().tolist() != [end - 1 for end in endpoints]):
                raise RuntimeError("sampling diagnostic did not reach the requested shared prefix")
        base = self.branch(batch)
        if output_token_index is None:
            self.decode_inputs(candidates[:, 0], slots)
        else:
            self.rows(slots, sample=True)
            self.replay(self.name("decode", batch, False), slots)
        reference = self.model(slots, endpoints)
        generation = self.catalog[self.name("decode", batch, False)]

        def final_outputs(decoder, indices):
            hidden = decoder.scratch.hidden[indices].cpu().clone()
            logits = decoder.scratch.logits[indices].cpu().clone()
            return {"hidden": hidden, "logits": logits, "greedy_tokens": logits.argmax(-1)}

        captured_outputs = final_outputs(generation.decoder, slots)
        if output_token_index is not None:
            completion = generation.completion
            ordinary_probabilities = completion.probabilities.cpu().clone()
            ordinary_scores = completion.scores.cpu().clone()
            sampling = {"output_token_index_0based": output_token_index,
                        "ordinary_common_prefix_token_ids": common_prefix,
                        "prefill_ordinary_vs_mtp_target_state": prefill_comparison,
                        "shared_prefix_consumed_tokens": [end - 1 for end in endpoints],
                        "ordinary": [_sampling_support(captured_outputs["logits"][row], ordinary_probabilities[row],
                                      sampled=completion.sampled_tokens[row]) for row in slots],
                        "ordinary_seed": completion.packed_seed.cpu().tolist(),
                        "ordinary_offset": completion.packed_offset.cpu().tolist(),
                        "ordinary_valid": completion.sample_valid.cpu().tolist(),
                        "target_sampling": "Conditional distribution only; no synthetic verifier RNG draw is substituted for the failing native run"}

        def observe(decoder, indices, invoke):
            from ignition.recipes.qwen38_dense import decoder as decoder_module
            from ignition.components.sm70_norm import PreparedRMSNorm
            outputs, handles = {}, []
            original_norm, norm_index = decoder_module.rms_norm, 0
            original_prepared_norm, prepared_norm_index = PreparedRMSNorm.__call__, 0
            def traced_norm(value, weight, epsilon):
                nonlocal norm_index
                prefix = f"norm{norm_index:03d}"
                norm_index += 1
                outputs[prefix + ".input"] = value[indices].cpu().clone()
                result = original_norm(value, weight, epsilon)
                outputs[prefix + ".output"] = result[indices].cpu().clone()
                return result

            def traced_prepared_norm(operation, value):
                nonlocal prepared_norm_index
                prefix = f"prepared_norm{prepared_norm_index:03d}"
                prepared_norm_index += 1
                outputs[prefix + ".input"] = value[indices].cpu().clone()
                result = original_prepared_norm(operation, value)
                outputs[prefix + ".output"] = result[indices].cpu().clone()
                return result

            for index, layer in enumerate(decoder.layers):
                def hook(module, args, output, index=index):
                    outputs[f"layer{index:02d}.hidden"] = output[indices].cpu().clone()
                handles.append(layer.register_forward_hook(hook))
            try:
                decoder_module.rms_norm = traced_norm
                PreparedRMSNorm.__call__ = traced_prepared_norm
                invoke()
                torch.cuda.synchronize(self.device)
            finally:
                decoder_module.rms_norm = original_norm
                PreparedRMSNorm.__call__ = original_prepared_norm
                for handle in handles:
                    handle.remove()
            return outputs

        self._load(base)
        self.rows(slots, sample=output_token_index is not None)
        for slot in slots:
            self.requests.pending_token[slot] = candidates[slot, 0]
        ordinary = observe(generation.decoder, list(range(batch)), generation)
        ordinary_outputs = final_outputs(generation.decoder, slots)
        captured_vs_eager = _compare(reference, self.model(slots, endpoints))
        self._load(base)
        operation = self.rounds[batch, width][1]
        self.rows(slots, lengths=[width] * batch)
        states = self.inputs.request_state_rows[:batch, 0].contiguous()
        starts = self.requests.consumed_tokens[:batch].clone()
        operation.candidates.copy_(candidates)
        if output_token_index is not None:
            operation.drafts.copy_(candidates[:, 1:])
        operation.metadata.set(starts, torch.full((batch,), width, device=self.device, dtype=torch.int32),
            states, self.inputs.page_tables[:batch], candidates, self.inputs.live_rows)
        d, m = operation.delta, operation.metadata
        d.query_start_loc.copy_(m.query_start_loc)
        d.convolution_slots.copy_(states)
        d.fresh.copy_(starts == 0)
        d.accepted_tokens.fill_(1)
        d.state_indices[:, 0].copy_(states)
        d.num_seqs.copy_(m.live_counts[1:])
        d.num_tokens.copy_(m.live_counts[:1])
        packed = observe(operation.decoder, [slot * width for slot in slots], operation.decoder)
        packed_outputs = final_outputs(operation.decoder, [slot * width for slot in slots])
        layer_comparison = _compare(ordinary, packed)
        if sampling is not None:
            from ignition.components.state import save_hidden
            target = operation.operations[0]
            target()
            target_probabilities = target.probabilities[:, 0].cpu().clone()
            sampling.update(target=[_sampling_support(packed_outputs["logits"][row], target_probabilities[row])
                                    for row in slots], target_valid=target.valid[:, 0].cpu().tolist(),
                            ordinary_vs_target_scores=_compare({"scores": ordinary_scores},
                                                               {"scores": target.scores[:, 0]}),
                            ordinary_vs_target_probabilities=_compare({"probabilities": ordinary_probabilities},
                                                                      {"probabilities": target_probabilities}))
            # Commit exactly the first target input before comparing all
            # recurrent, convolution, hidden and logical KV-prefix bytes.
            operation.commit_counts.fill_(1)
            operation.replay_commit()
            operation.convolution_commit()
            hidden = operation.decoder.scratch.hidden.view(batch, width, -1)
            save_hidden(hidden[:, 0], self.pool.last_hidden, states,
                        torch.ones(batch, dtype=torch.uint8, device=self.device))
            sampling["ordinary_vs_target_committed_state"] = _compare(reference, self.model(slots, endpoints))
        result = {"execution": "failure-only eager layer and native/Python norm hooks; first candidate of every request",
                "batch": batch, "verification_width": width, "prompt_tokens": [len(ids) for ids in prompts],
                "candidate_ids": candidates.cpu().tolist(),
                "captured_vs_eager_ordinary_state": captured_vs_eager,
                "captured_vs_eager_ordinary_outputs": _compare(captured_outputs, ordinary_outputs),
                "captured_vs_packed_outputs": _compare(captured_outputs, packed_outputs),
                "ordinary_vs_packed_layers": layer_comparison,
                "first_divergent_layer": next((item["field"] for item in layer_comparison["mismatches"]
                                               if item["field"].startswith("layer")), None),
                "first_divergent_norm": next((item["field"] for item in layer_comparison["mismatches"]
                                              if "norm" in item["field"]), None),
                "greedy_token_ids": {name: outputs["greedy_tokens"].tolist() for name, outputs in
                                     (("retained", captured_outputs), ("eager", ordinary_outputs),
                                      ("packed", packed_outputs))}}
        if sampling is not None:
            result["sampling_tie_diagnosis"] = sampling
        return result


class _CommitGrowthHarness(_Harness):
    """Admit the last backed row as a committed row after a logical bound change from N-1 to N.

    Replay-commit has no device descriptor bound (the round's batch validator checks rows against
    ``memory_capacity[0]`` on device before every round); the gate therefore proves that a row admitted by a logical
    capacity change inside the already-backed reservation is committed into by the retained graphs and by the direct
    replay commit exactly like any other row.
    """

    def __init__(self, prepared):
        super().__init__(prepared)
        if self.width != 5 or (4, 5) not in self.rounds or (4, 4) not in self.rounds:
            raise ValueError("commit growth qualification requires B4 widths 4 and 5")
        self.physical_rows = int(self.pool.memory_capacity[0])
        self.new_row = self.physical_rows - 1
        if self.new_row <= self.capacity or self.pool.backing.shape[0] < self.physical_rows:
            raise ValueError("commit growth qualification requires an already-backed row beyond the live rows")
        self.operation = self.rounds[4, 5][1]
        self.commit = self.operation.replay_commit
        self.growth_events = []

    def initialize(self, prompts, *, mtp=True):
        if len(prompts) != 4 or not mtp:
            raise ValueError("commit growth fixture requires four MTP requests")
        self.pool.memory_capacity[0] = self.physical_rows
        super().initialize(prompts, mtp=mtp)
        # No live commit occurs while the last backed row is logically excluded.
        self.pool.memory_capacity[0] = self.new_row
        address, shape = self.pool.reserved_backing.data_ptr(), tuple(self.pool.reserved_backing.shape)
        self.pool.memory_capacity[0] = self.new_row + 1
        # Make the newly admitted row request 3's committed row (its state moves with it).
        old_row = int(self.inputs.request_state_rows[3, 0])
        self.pool.backing[self.new_row].copy_(self.pool.backing[old_row])
        self.inputs.request_state_rows[3, 0] = self.new_row
        mapping = self.inputs.request_state_rows[:4, 0].cpu().tolist()
        if mapping[3] != self.new_row or len(set(mapping)) != 4:
            raise RuntimeError("growth fixture did not retain unique committed rows")
        if (address, shape) != (self.pool.reserved_backing.data_ptr(), tuple(self.pool.reserved_backing.shape)):
            raise RuntimeError("logical admission growth changed the physical reservation")
        self.growth_events.append({"old_capacity": self.new_row, "new_capacity": self.new_row + 1,
            "new_canonical_row": self.new_row, "previous_request3_row": old_row, "request_rows": mapping,
            "reservation_address_unchanged": True, "reservation_shape_unchanged": True})

    def excluded_round(self, prompts, width):
        """Negative capacity check: the retained round runs while request 3's committed row (the last backed row) is
        logically excluded again (memory_capacity[0] = new_row). Replay-commit has no bound of its own, so the round's
        batch validator must refuse request 3 (status ERROR, code 32), the destination row's bytes must be untouched and
        its request must not advance, while the three admitted requests complete a normal round. The row is then
        re-admitted and every mutable byte restored for the positive comparisons."""
        batch = len(prompts)
        slots = list(range(batch))
        self.initialize(prompts, mtp=True)
        self.rows(slots, sample=True)
        self.replay(self.name("resume", batch, True), slots)
        base = self.branch(batch)
        if int(self.inputs.request_state_rows[3, 0]) != self.new_row:
            raise RuntimeError("negative capacity fixture lost the newly admitted row")
        destination = _bytes(self.pool.backing[self.new_row])
        null_row = _bytes(self.pool.backing[0])
        consumed = self.requests.consumed_tokens[:batch].cpu().clone()
        generated = self.requests.generated_tokens[:batch].cpu().clone()
        name = self.name("speculative", batch, True, width=width)
        self.pool.memory_capacity[0] = self.new_row
        try:
            self.rows(slots, lengths=[width] * batch, sample=True)
            records = self.replay(name, slots, rejected=(3,))
        finally:
            self.pool.memory_capacity[0] = self.new_row + 1
        counts = records[:batch, 8].tolist()
        mismatches = []
        if not torch.equal(_bytes(self.pool.backing[self.new_row]), destination):
            unequal = (_bytes(self.pool.backing[self.new_row]) != destination)
            mismatches.append({"field": f"backing.row{self.new_row}", "different_bytes": int(unequal.sum()),
                               "first_byte_offsets": unequal.nonzero().flatten()[:8].tolist()})
        if not torch.equal(_bytes(self.pool.backing[0]), null_row):
            mismatches.append({"field": "backing.row0", "error": "null row written"})
        after = self.requests.consumed_tokens[:batch].cpu()
        if int(after[3]) != int(consumed[3]) or int(self.requests.generated_tokens[3]) != int(generated[3]) or counts[3] != 0:
            mismatches.append({"field": "request3", "error": "rejected request advanced",
                               "consumed": [int(consumed[3]), int(after[3])], "emitted": counts[3]})
        if any(not 1 <= counts[slot] <= width for slot in range(3)) or any(int(after[slot]) <= int(consumed[slot]) for slot in range(3)):
            mismatches.append({"field": "requests0-2", "error": "admitted requests did not complete a normal round",
                               "emitted": counts[:3]})
        self._load(base)
        return {"passed": not mismatches, "fields": 4, "bytes": int(destination.numel()) + int(null_row.numel()),
                "mismatches": mismatches, "batch": batch, "verification_width": width, "accepted_inputs": counts,
                "excluded_row": self.new_row, "memory_capacity_during_round": self.new_row,
                "request3_status_error_code": 32,
                "execution": "retained speculative round with the destination row logically excluded: validator rejection, destination bytes unchanged"}


def validate_commit_capacity_growth(prepared, root, progress=print):
    """Separate E4 gate; returns evidence without adding to the original 47 checks.

    This changes logical admission within an already-backed reservation. It
    validates the existing VMM memory_capacity binding without allocating or
    mapping new physical backing. All backed model state and changed metadata
    are restored from CPU copies even when a comparison fails. Replay-commit:
    the retained B4 width-5 and width-4 rounds and the direct replay commit at
    every accepted count must commit into the newly admitted row exactly as the
    ordinary teacher-forced reference (no descriptor bound exists to refresh).
    """
    from tokenizers import Tokenizer

    report = {"passed": False, "checks": [], "root": str(Path(root)),
        "scope": "Logical admission growth N-1→N inside already-backed preparation rows; no new VMM allocation or mapping",
        "native_runtime_tested": False, "batch": 4, "verification_width": 5, "growth_events": []}
    harness, saved, operation, original_commit = None, [], None, None
    started = time.monotonic()
    try:
        with torch.inference_mode(), torch.cuda.device(prepared.device):
            harness = _CommitGrowthHarness(prepared)
            h, operation, original_commit = harness, harness.operation, harness.commit
            report["growth_events"] = h.growth_events
            report["newly_admitted_destination_row"] = h.new_row
            capacity = h.pool.memory_capacity.cpu().tolist()
            report["original_memory_capacity"] = capacity
            saved = h._save_mutable()
            saved.extend((f"backed.row{row}", h.pool.backing[row], h.pool.backing[row].cpu().clone())
                         for row in range(capacity[0]))
            saved.extend((f"backed.{name}", value[:capacity[1]], value[:capacity[1]].cpu().clone())
                         for name, value in h.pool.pages.items())
            # Keep explicit commit/verification metadata pristine as well as
            # request, recurrent, convolution, hidden and KV state.
            seen = {(tensor.data_ptr(), tensor.shape, tensor.stride(), tensor.dtype) for _, tensor, _ in saved}
            for prefix, owner in (("round", operation), ("metadata", operation.metadata),
                                  ("delta", operation.delta), ("commit", original_commit)):
                for name, tensor in vars(owner).items():
                    if not isinstance(tensor, torch.Tensor) or tensor is h.pool.reserved_backing:
                        continue
                    key = (tensor.data_ptr(), tensor.shape, tensor.stride(), tensor.dtype)
                    if key not in seen:
                        seen.add(key)
                        saved.append((f"private.{prefix}.{name}", tensor, tensor.cpu().clone()))
            tokenizer = Tokenizer.from_file(prepared.server_config["assets"]["tokenizer"]["path"])
            tokens = [tokenizer.encode(text, add_special_tokens=False).ids for text in PROMPTS]
            # One page-boundary fixture exercises all five candidates and the
            # ordinary teacher-forced reference with the same initial history.
            prompts = [(ids * ((127 + len(ids) - 1) // len(ids)))[:127] for ids in tokens]
            committed_rows = []
            report["direct_commit_rows"] = committed_rows

            def direct_commit():
                original_commit()
                rows = operation.metadata.state_rows.cpu().tolist()
                if rows[3] != h.new_row:
                    raise RuntimeError("direct replay commit did not target the newly admitted row")
                committed_rows.append(rows)

            def run(name, function):
                if progress is not None:
                    progress("commit capacity growth: " + name)
                try:
                    result = function()
                except Exception as error:
                    report["checks"].append({"name": name, "passed": False,
                                             "error_type": type(error).__name__, "error": str(error)})
                    raise
                report["checks"].append({"name": name, **result})
                if progress is not None:
                    progress("commit capacity growth result: " + name + (" PASS" if result["passed"] else " FAIL"))

            def actual(width):
                result = h.actual_prefix(prompts, width)
                rows = h.inputs.request_state_rows[:4, 0].cpu().tolist()
                if rows[3] != h.new_row:
                    raise RuntimeError("retained round did not commit into the newly admitted row")
                result["committed_rows"] = rows
                return result

            run("retained_B4_width4_with_destination_excluded_rejected_unchanged", lambda: h.excluded_round(prompts, 4))
            run("retained_actual_B4_width5_after_growth_vs_ordinary", lambda: actual(5))
            run("retained_actual_B4_width4_after_growth_vs_ordinary", lambda: actual(4))
            operation.replay_commit = direct_commit
            run("forced_direct_acceptance1_to5_after_growth_vs_ordinary", lambda: h.forced(prompts, width=5))
            operation.replay_commit = original_commit
            report.update(growth_events=h.growth_events, direct_commit_rows=committed_rows, graph_replays=h.replays,
                          equivalence="With the destination row excluded the retained width-4 round is rejected and leaves its bytes unchanged; admitted, the retained width-5 and width-4 rounds and the direct replay commit at every accepted prefix match the same ordinary full-model reference in the newly admitted row")
            if len(committed_rows) != 5 or any(rows[3] != h.new_row for rows in committed_rows):
                raise RuntimeError("growth qualification did not exercise all five prefixes into the newly admitted row")
            report["passed"] = len(report["checks"]) == 4 and all(item["passed"] for item in report["checks"])
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        if operation is not None and original_commit is not None:
            operation.replay_commit = original_commit
        if harness is not None and saved:
            try:
                with torch.inference_mode(), torch.cuda.device(prepared.device):
                    harness._load(saved)
                    torch.cuda.synchronize(harness.device)
                    report["restore"] = _compare({name: value for name, _, value in saved},
                                                 {name: tensor for name, tensor, _ in saved})
                    report["restored_memory_capacity"] = harness.pool.memory_capacity.cpu().tolist()
                report["passed"] &= report["restore"]["passed"]
            except Exception as error:
                report["passed"] = False
                report["restore_error"] = {"type": type(error).__name__, "message": str(error)}
        report["elapsed_seconds"] = time.monotonic() - started
    return report


class _LongSequenceHarness(_Harness):
    """Fixed 128-token prefill and bounded private pages for long comparisons."""

    def __init__(self, prepared, *, tokens=512):
        self.page_counts, self.seeds = [], []
        super().__init__(prepared, max_new_tokens=tokens)

    def name(self, kind, batch, mtp, *, width=None):
        if kind != "prefill":
            return super().name(kind, batch, mtp, width=width)
        matches = [name for name, signature in self.prepared.signatures.items()
                   if signature.kind == "prefill" and signature.batch_capacity == 1
                   and signature.speculation_enabled == mtp and signature.token_capacity == 128]
        if len(matches) != 1:
            raise ValueError(f"long invariance requires one retained 128-token prefill graph, MTP={mtp}")
        return matches[0]

    def page_requirement(self, prompts):
        counts = [(len(ids) + self.max_new_tokens + self.page_tokens - 1) // self.page_tokens for ids in prompts]
        needed = sum(count + 1 for count in counts)
        backed = int(self.pool.memory_capacity[1])
        if (not counts or len(counts) > self.capacity or max(counts) > self.inputs.page_tables.shape[1]
                or needed > backed):
            raise ValueError(f"long invariance needs {needed} backed pages and {max(counts, default=0)} "
                             f"pages/request; available={backed}, table_width={self.inputs.page_tables.shape[1]}")
        return counts, needed

    def initialize(self, prompts, *, mtp=True):
        self.page_counts, _ = self.page_requirement(prompts)
        if len(self.seeds) != len(prompts):
            raise ValueError("long invariance requires one stable seed per prompt")
        super().initialize(prompts, mtp=mtp)

    def reset(self):
        super().reset()
        self.inputs.page_tables.fill_(-1)
        self.owner.draft_tables.fill_(-1)
        begin = 0
        for slot, count in enumerate(self.page_counts):
            pages = torch.arange(begin, begin + count, dtype=torch.int32, device=self.device)
            self.inputs.page_tables[slot, :count].copy_(pages)
            # The target and shifted draft tables stay identical across modes.
            # Each request owns an extra physical page for its private draft tail.
            self.owner.draft_tables[slot, :count].copy_(pages + 1)
            begin += count + 1

    def admission_row(self, row, slot, ids):
        from ignition.components.sampling import RequestAdmission

        self.admission.stage(row, slot, RequestAdmission(generation=self.epoch, seed=self.seeds[slot],
            max_new_tokens=self.max_new_tokens, temperature=1., top_k=1, top_p=1.,
            repetition_penalty=1., eos_token_ids=()), ids)

    def branch(self, batch):
        """Restore every candidate row and assigned target/draft page in long traces."""
        if batch != len(self.page_counts):
            raise ValueError("long branch snapshot must cover every initialized request")
        saved = self._save_mutable()
        rows = self.inputs.request_state_rows[:batch, 0].cpu().reshape(-1).unique().tolist()
        backed_rows, backed_pages = self.pool.memory_capacity.cpu().tolist()
        if any(not 0 <= row < backed_rows for row in rows):
            raise ValueError("long branch snapshot contains an unbacked continuation row")
        pages = sum(count + 1 for count in self.page_counts)
        if pages > backed_pages:
            raise ValueError("long branch snapshot exceeds backed page capacity")
        saved.extend((f"row{row}", self.pool.backing[row], self.pool.backing[row].cpu().clone()) for row in rows)
        saved.extend((name, value[:pages], value[:pages].cpu().clone()) for name, value in self.pool.pages.items())
        return saved


def diagnose_long_sequence(prepared, root, progress=print, *, checkpoint, prompts_path, widths=None):
    """Trace long coding-prompt baselines independently of qualification or export.

    The retained ordinary graph supplies state/output goldens. Eager ordinary
    and packed target decoders expose complete layer and norm samples for the
    first candidate of each request, with the same restored prompt history.
    """
    import sys

    root, harness, started = Path(root), None, time.monotonic()
    from benchmark_mtp import load_prompts
    from measure_mtp import _tokenize

    report = {"status": "started", "diagnostic_only": True, "cases": [], "sampling_cases": [],
              "scope": "Long-prompt baselines plus shared ordinary prefixes at reported divergence positions; distribution support is inspected without recreating speculative RNG history",
              "prefill_chunk_tokens": 128, "source": str(Path(prompts_path)),
              "seed_policy": "17 + original code prompt index", "failures": []}
    try:
        with torch.inference_mode(), torch.cuda.device(prepared.device):
            harness = _LongSequenceHarness(prepared)
            selected = load_prompts(Path(prompts_path), ("code-03", "code-00", "code-01", "code-02"))
            prompts = _tokenize(Path(checkpoint), selected)
            counts, pages = harness.page_requirement(prompts)
            report.update(prompts=[{"id": item["id"], "prompt_tokens": len(ids), "page_count": count}
                                   for item, ids, count in zip(selected, prompts, counts, strict=True)],
                          required_backed_pages=pages, backed_pages=int(harness.pool.memory_capacity[1]))
            widths = tuple(widths) if widths is not None else tuple(
                width for width in (4, 5) if all((batch, width) in harness.rounds for batch in (1, 4)))
            if not widths or any(width not in (4, 5) or any((batch, width) not in harness.rounds
                                  for batch in (1, 4)) for width in widths):
                raise ValueError("long diagnosis requires retained B1/B4 target widths 4 or 5")
            report["verification_widths"] = list(widths)
            # The reported depth3 and depth4 divergences have different
            # code-00 positions. Inspect the small code-03 prefix first.
            sampling_cases = [(0, 5, width) for width in widths]
            sampling_cases += [(1, index, width) for index, width in ((278, 4), (197, 5)) if width in widths]
            for index, output_index, width in sampling_cases:
                prompt_id = selected[index]["id"]
                label = f"{prompt_id}_B1_width{width}_output{output_index}"
                record = {"name": label, "prompt_id": prompt_id, "output_token_index_0based": output_index}
                report["sampling_cases"].append(record)
                if progress is not None:
                    progress("long sampling diagnosis: " + label)
                step = time.monotonic()
                try:
                    harness.seeds = [17 + int(prompt_id.split("-")[-1])]
                    record.update(harness.trace_first_divergence([prompts[index]], width=width,
                                                                 output_token_index=output_index), status="complete")
                    sampling = record["sampling_tie_diagnosis"]
                    if progress is not None:
                        progress("long sampling evidence: " + json.dumps({"name": label,
                            "prefill_state_equal": sampling["prefill_ordinary_vs_mtp_target_state"]["passed"],
                            "ordinary_target_state_equal": sampling["ordinary_vs_target_committed_state"]["passed"],
                            "ordinary_target_outputs_equal": record["captured_vs_packed_outputs"]["passed"],
                            "probabilities_equal": sampling["ordinary_vs_target_probabilities"]["passed"],
                            "ordinary": sampling["ordinary"], "target": sampling["target"]}))
                except Exception as error:
                    record.update(status="failed", error_type=type(error).__name__, error=str(error),
                                  traceback=traceback.format_exc())
                    report["failures"].append(label)
                record["elapsed_seconds"] = time.monotonic() - step
            for indices in ((0,), (1,), (2,), (3,), (0, 1, 2, 3)):
                for width in widths:
                    prompt_ids = [selected[index]["id"] for index in indices]
                    label = f"B{len(indices)}_width{width}_" + "_".join(prompt_ids)
                    record = {"name": label, "prompt_ids": prompt_ids}
                    report["cases"].append(record)
                    if progress is not None:
                        progress("long arithmetic diagnosis: " + label)
                    step = time.monotonic()
                    try:
                        harness.seeds = [17 + int(selected[index]["id"].split("-")[-1]) for index in indices]
                        record.update(harness.trace_first_divergence([prompts[index] for index in indices], width=width),
                                      status="complete")
                    except Exception as error:
                        record.update(status="failed", error_type=type(error).__name__, error=str(error),
                                      traceback=traceback.format_exc())
                        report["failures"].append(label)
                    record["elapsed_seconds"] = time.monotonic() - step
                    if progress is not None:
                        progress("long arithmetic diagnosis result: " + label + " " + record["status"]
                                 + "; first layer=" + str(record.get("first_divergent_layer"))
                                 + "; first norm=" + str(record.get("first_divergent_norm")))
            report["graph_replays"] = harness.replays
            report["status"] = "complete" if not report["failures"] else "failed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        if harness is not None:
            try:
                with torch.inference_mode(), torch.cuda.device(prepared.device):
                    report["cleanup"] = harness.cleanup()
                    if not report["cleanup"]["passed"]:
                        report["status"] = "failed"
            except Exception as error:
                report.update(status="failed", cleanup_error=str(error))
        report["elapsed_seconds"] = time.monotonic() - started
    return report


def _sequence_comparison(expected, actual, *, tokens=512):
    """Exact sequence identity with a bounded first-divergence explanation."""
    common = next((index for index, pair in enumerate(zip(expected, actual)) if pair[0] != pair[1]),
                  min(len(expected), len(actual)))
    passed = len(expected) == len(actual) == tokens and expected == actual
    result = {"passed": passed, "expected_tokens": len(expected), "actual_tokens": len(actual),
              "required_tokens": tokens, "common_prefix_tokens": common}
    if not passed:
        begin, end = max(0, common - 4), common + 5
        result["first_difference"] = {"token_index_0based": common, "token_number_1based": common + 1,
            "expected_token": expected[common] if common < len(expected) else None,
            "actual_token": actual[common] if common < len(actual) else None,
            "context_start_index": begin, "expected_context": expected[begin:end], "actual_context": actual[begin:end]}
    return result


def validate_long_sequence(prepared, root, progress=print, *, checkpoint, prompts_path):
    """Separate 512-token gate: ordinary/MTP and alone/co-scheduled identity.

    Uses code-00 through code-03 from the native benchmark, their actual chat
    template, retained graphs only, and exactly 128-token prefill chunks. The
    original short validator and its default two-page fixture are unaffected.
    """
    import sys

    root, harness, started = Path(root), None, time.monotonic()
    from benchmark_mtp import load_prompts
    from measure_mtp import _tokenize

    report = {"passed": False, "checks": [], "runs": [], "failures": [],
        "native_runtime_tested": False, "execution": "retained production CUDA graphs",
        "output_tokens": 512, "prefill_chunk_tokens": 128, "draft_depths": [3, 4],
        "sampling": {"temperature": 1.0, "top_k": 1, "top_p": 1.0, "repetition_penalty": 1.0,
                     "eos_disabled": True, "seed_policy": "17 + stable prompt index"},
        "source": str(Path(prompts_path))}
    evidence, run_id = root / "validation-attempts.jsonl", time.time_ns()

    def persist(event):
        with evidence.open("a") as file:
            file.write(json.dumps({"run_id": run_id, "suite": "long_sequence", **event}, default=str) + "\n")

    try:
        with torch.inference_mode(), torch.cuda.device(prepared.device):
            harness = _LongSequenceHarness(prepared)
            selected = load_prompts(Path(prompts_path), ("code-00", "code-01", "code-02", "code-03"))
            prompts = _tokenize(Path(checkpoint), selected)
            counts, required = harness.page_requirement(prompts)
            report.update(prompts=[{"id": item["id"], "prompt_tokens": len(ids), "page_count": count,
                                    "token_ids": ids} for item, ids, count in zip(selected, prompts, counts, strict=True)],
                          required_backed_pages=required, backed_pages=int(harness.pool.memory_capacity[1]))
            # Require all requested capacities before spending time on a partial suite.
            for mtp in (False, True):
                harness.name("prefill", 1, mtp)
                for batch in (1, 4):
                    harness.name("init", batch, mtp)
                    harness.name("resume", batch, mtp)
                    if not mtp:
                        harness.name("decode", batch, False)
            for batch in (1, 4):
                for width in (4, 5):
                    harness.name("speculative", batch, True, width=width)
            runs = {}
            for mode, width in (("ordinary", None), ("depth3", 4), ("depth4", 5)):
                for indices in ((0,), (1,), (2,), (3,), (0, 1, 2, 3)):
                    batch = len(indices)
                    label = f"{mode}_B{batch}_" + "_".join(selected[index]["id"] for index in indices)
                    record = {"name": label, "mode": mode, "batch": batch, "prompt_indices": list(indices),
                              "prompt_ids": [selected[index]["id"] for index in indices], "passed": False}
                    report["runs"].append(record)
                    if progress is not None:
                        progress("long MTP invariance: " + label + " (512 tokens, 128-token prefill chunks)")
                    step = time.monotonic()
                    try:
                        harness.seeds = [17 + index for index in indices]
                        actual = harness.greedy([prompts[index] for index in indices], width=width, tokens=512)
                        record.update(actual, passed=all(len(ids) == 512 for ids in actual["token_ids"]))
                        runs[mode, indices] = record
                        if not record["passed"]:
                            report["failures"].append(label)
                    except Exception as error:
                        record.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
                        report["failures"].append(label)
                    record["elapsed_seconds"] = time.monotonic() - step
                    persist({"event": "run", **record})
                    if progress is not None:
                        progress("long MTP invariance result: " + label + (" PASS" if record["passed"] else " FAIL"))

            def compare(name, expected_key, actual_key, prompt_index):
                result = {"name": name, "prompt_id": selected[prompt_index]["id"], "passed": False}
                expected, actual = runs.get(expected_key), runs.get(actual_key)
                if expected is None or actual is None:
                    result["error"] = "required generation run failed; identity cannot qualify"
                else:
                    left = expected["token_ids"][expected_key[1].index(prompt_index)]
                    right = actual["token_ids"][actual_key[1].index(prompt_index)]
                    result.update(_sequence_comparison(left, right), expected_run=expected["name"], actual_run=actual["name"])
                report["checks"].append(result)
                if not result["passed"]:
                    report["failures"].append(name)
                persist({"event": "check", **result})
                if progress is not None:
                    progress("long MTP invariance comparison: " + name + (" PASS" if result["passed"] else " FAIL"))

            cohort = (0, 1, 2, 3)
            for index, item in enumerate(selected):
                for mode in ("ordinary", "depth3", "depth4"):
                    compare(f"alone_vs_coscheduled_{mode}_{item['id']}",
                            (mode, (index,)), (mode, cohort), index)
                for mode in ("depth3", "depth4"):
                    for indices in ((index,), cohort):
                        compare(f"speculative_vs_no_mtp_{mode}_B{len(indices)}_{item['id']}",
                                ("ordinary", indices), (mode, indices), index)
            report["graph_replays"] = harness.replays
            report["passed"] = len(report["checks"]) == 28 and not report["failures"]
    except Exception as error:
        report["failures"].append("long_sequence setup")
        report["setup_error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        if harness is not None:
            try:
                with torch.inference_mode(), torch.cuda.device(prepared.device):
                    report["cleanup"] = harness.cleanup()
                if not report["cleanup"]["passed"]:
                    report["failures"].append("long_sequence cleanup")
            except Exception as error:
                report["failures"].append("long_sequence cleanup")
                report["cleanup_error"] = {"type": type(error).__name__, "message": str(error)}
        report["elapsed_seconds"] = time.monotonic() - started
        report["passed"] &= not report["failures"]
        persist({"event": "summary", "passed": report["passed"], "checks": len(report["checks"]),
                 "failures": report["failures"], "elapsed_seconds": report["elapsed_seconds"]})
    return report


def validate(prepared, root, progress=print, *, long_sequence=False, checkpoint=None, prompts_path=None):
    """Return strict pass/fail and retained evidence; never widen byte checks."""
    from tokenizers import Tokenizer

    if long_sequence and (checkpoint is None or prompts_path is None):
        raise ValueError('long-sequence validation requires checkpoint and prompts_path')
    started, harness = time.monotonic(), None
    report = {"passed": False, "checks": [], "failures": [], "native_runtime_tested": False,
              "byte_comparison": "exact uint8 equality; logical committed KV prefix only; full canonical recurrent/convolution fields",
              "forced_prefix_execution": "diagnostic eager prepared decoder; captured ordinary references",
              "root": str(Path(root)), "torch": torch.__version__}
    evidence = Path(root) / "validation-attempts.jsonl"
    run_id = time.time_ns()

    def persist(event):
        # Preserve completed checks even if an external lease watcher must
        # terminate preparation before its final report can be written.
        with evidence.open("a") as file:
            file.write(json.dumps({"run_id": run_id, **event}, default=str) + "\n")

    def check(name, function):
        if progress is not None:
            progress("strict MTP validation: " + name)
        try:
            result = function()
            report["checks"].append({"name": name, **result})
            if not result["passed"]:
                report["failures"].append(name)
            if progress is not None:
                progress("strict MTP validation result: " + name + (" PASS" if result["passed"] else " FAIL"))
        except Exception as error:
            report["checks"].append({"name": name, "passed": False, "error_type": type(error).__name__,
                "error": str(error), "traceback": traceback.format_exc()})
            report["failures"].append(name)
            if progress is not None:
                progress(f"strict MTP validation result: {name} ERROR {type(error).__name__}: {error}")
        persist({"event": "check", **report["checks"][-1]})

    try:
        with torch.inference_mode(), torch.cuda.device(prepared.device):
            harness = _Harness(prepared)
            tokenizer = Tokenizer.from_file(prepared.server_config["assets"]["tokenizer"]["path"])
            prompts = [tokenizer.encode(text, add_special_tokens=False).ids for text in PROMPTS]
            if any(len(ids) < harness.width for ids in prompts):
                raise ValueError("fixed prompts do not contain enough independent candidate inputs")
            report.update(prompt_texts=list(PROMPTS), prompt_ids=prompts, maximum_verification_width=harness.width,
                          accepted_counts=list(range(1, harness.width + 1)), device=torch.cuda.get_device_name(prepared.device))
            for batch in (1, 4):
                if (batch, harness.width) not in harness.rounds:
                    report["failures"].append(f"missing B={batch} prepared catalog")
                    continue
                selected = prompts[:batch]
                baseline = {}

                def ordinary():
                    baseline.update(harness.greedy(selected))
                    return {"passed": True, **baseline}

                check(f"ordinary_greedy_B{batch}", ordinary)
                for width in sorted(w for b, w in harness.rounds if b == batch):
                    def learned(width=width):
                        actual = harness.greedy(selected, width=width)
                        if "token_ids" not in baseline:
                            raise RuntimeError("ordinary reference failed; learned greedy cannot qualify")
                        return {"passed": actual["token_ids"] == baseline["token_ids"],
                                "ordinary_token_ids": baseline["token_ids"], **actual}
                    check(f"learned_greedy_B{batch}_width{width}", learned)
                    check(f"actual_prefix_B{batch}_width{width}",
                          lambda width=width: harness.actual_prefix(selected, width))
                for endpoint in (127, 128):
                    boundary = [(ids * ((endpoint + len(ids) - 1) // len(ids)))[:endpoint] for ids in selected]
                    check(f"forced_prefix_B{batch}_P{endpoint}", lambda boundary=boundary: harness.forced(boundary, width=4))
                    if batch == 4:
                        check(f"forced_prefix_B4_inactive_P{endpoint}",
                              lambda boundary=boundary: harness.forced(boundary, inactive=True, width=4))
                if harness.width > 4:
                    for endpoint in (127, 128):
                        boundary = [(ids * ((endpoint + len(ids) - 1) // len(ids)))[:endpoint] for ids in selected]
                        check(f"forced_prefix_B{batch}_P{endpoint}_width{harness.width}",
                              lambda boundary=boundary: harness.forced(boundary))
                        if batch == 4:
                            check(f"forced_prefix_B4_inactive_P{endpoint}_width{harness.width}",
                                  lambda boundary=boundary: harness.forced(boundary, inactive=True))
                for endpoint in (128, 129, 130):
                    boundary = [(ids * ((endpoint + len(ids) - 1) // len(ids)))[:endpoint] for ids in selected]
                    check(f"snapshot_restore_branch_B{batch}_P{endpoint}",
                          lambda boundary=boundary: harness.snapshot_branch(boundary))
                check(f"all_inactive_B{batch}", lambda batch=batch: harness.inactive(batch))
                if batch == 4:
                    check("padded_slot_alias_B4", lambda: harness.alias_lane(selected[0]))
                    check("stale_generation_slot_alias_B4", lambda: harness.alias_lane(selected[0], stale=True))
            check("sm70_prefill_attention_injected_status", lambda: harness.injected_status(prompts[0]))
            if harness.decode_probe_enabled():
                check("sm70_decode_attention_nonfinite_probe", lambda: harness.decode_nonfinite_probe(prompts[0], width=harness.width))
            report["graph_replays"] = harness.replays
            report["prefill_status_checks"] = harness.prefill_status_checks
            if report["failures"]:
                try:
                    report["diagnostics"] = harness.trace_first_divergence(prompts)
                except Exception as error:
                    report["diagnostics"] = {"error": str(error), "traceback": traceback.format_exc()}
            if long_sequence:
                report["long_sequence"] = validate_long_sequence(prepared, root, progress=progress,
                    checkpoint=checkpoint, prompts_path=prompts_path)
                if not report["long_sequence"]["passed"]:
                    report["failures"].append("long_sequence invariance")
    except Exception as error:
        report["failures"].append("validator setup")
        report["setup_error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        if harness is not None:
            with torch.inference_mode(), torch.cuda.device(prepared.device):
                check("restore_pre_export_state", harness.cleanup)
        report["elapsed_seconds"] = time.monotonic() - started
        report["passed"] = bool(report["checks"]) and not report["failures"]
        persist({"event": "summary", "passed": report["passed"], "failures": report["failures"],
                 "elapsed_seconds": report["elapsed_seconds"]})
    return report
