"""File-backed FP8 n-gram tables, compact GPU inputs, and residual injection."""

from dataclasses import dataclass

import torch
from b12x.norm import hyperconnection
from b12x.preparation import PreparedCall
from b12x.sequence import engram

from ignition.components.engram import EngramInputs, PreparedEngramLookup, PreparedEngramRequests
from ignition.components.linear import QuantizedProjection
from ignition.components.state import gather_rows, save_hidden
from ignition.preparation.runtime_package import FileSource
from ignition.preparation.tuning import declare
from ignition.progress import phase


@dataclass(frozen=True)
class Tables:
    geometry: object
    token_map: tuple[int, ...]
    sources: dict
    inputs: dict
    mailboxes: dict
    sequence: torch.Tensor

    @classmethod
    def prepare(cls, manifest, config, *, tokens, device, sequence, host_allocator, progress=None):
        from transformers import PreTrainedTokenizerFast

        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(manifest.snapshot / "tokenizer.json"))
        token_map, size = engram.build_compressed_token_map(tokenizer)
        text = config.text
        if size != text["engram_compressed_vocab_size"]:
            raise ValueError(f"Engram compressed vocabulary differs: {size}")
        geometry = engram.build_geometry(layer_ids=tuple(text["engram_layer_ids"]),
            base_table_size=text["engram_vocab_size"], compressed_vocab_size=size)
        if geometry.num_embeddings != tuple(text["engram_num_embeddings"]):
            raise ValueError("Engram hash geometry differs from checkpoint tables")
        sources, inputs, mailboxes = {}, {}, {}
        with phase(progress, "Prepare embedding tables") as stage:
            for index, layer in enumerate(geometry.layer_ids):
                rows = geometry.num_embeddings[index]
                pair = []
                for suffix, columns, dtype in (("weight", 256, "F8_E4M3"), ("scale", 8, "F8_E8M0")):
                    name = f"layers.{layer}.engram.embed.{suffix}"
                    metadata = manifest.tensors[name]
                    if metadata.shape != (rows, columns) or metadata.dtype != dtype:
                        raise ValueError(f"{name}: invalid Engram table geometry or dtype")
                    pair.append(FileSource(manifest.snapshot / metadata.shard, metadata.offset, rows * columns))
                sources[layer] = tuple(pair)
                inputs[layer] = EngramInputs.allocate(tokens, device)
                mailboxes[layer] = host_allocator(f"engram_{layer}_requests", (tokens * 24 + 1,), torch.int64)
                mailboxes[layer].host_view.zero_()
            stage.update(f"{sum(source.nbytes for pair in sources.values() for source in pair) / (1 << 30):.1f} GiB in checkpoint spans")
        return cls(geometry, tuple(token_map), sources, inputs, mailboxes, sequence)

    def buffers(self):
        return {**{name: tensor for layer, inputs in self.inputs.items() for name, tensor in inputs.buffers(layer).items()},
                **{f"engram_{layer}_requests": mailbox.device_view for layer, mailbox in self.mailboxes.items()}}

    def events(self):
        return {f"engram_{layer}_ready": inputs.ready.cuda_event for layer, inputs in self.inputs.items()}

    def resident_catalog(self):
        """Source spans and native bindings for shared RAM delivery without package export."""
        return {"vocabulary": len(self.token_map), "tables": [
            {"planes": [{"path": str(source.path), "offset": source.offset,
                         "nbytes": source.nbytes, "row_bytes": row_bytes}
                        for source, row_bytes in zip(self.sources[layer], (256, 8), strict=True)],
             "heads": 24, "weights": f"engram_{layer}_weights", "scales": f"engram_{layer}_scales",
             "hashes": f"engram_{layer}_hashes", "event": f"engram_{layer}_ready",
             "request": f"engram_{layer}_requests"}
            for layer in self.geometry.layer_ids]}

    def append(self, index, payload, cursor):
        from ignition.preparation.package_file import align
        from ignition.preparation import runtime_package_pb2 as proto

        target = index.engram
        target.token_map.extend(self.token_map)
        target.pad_token_id = 2
        target.compressed_vocabulary = self.geometry.compressed_vocab_size
        for ordinal, layer in enumerate(self.geometry.layer_ids):
            table = target.tables.add(layer=layer, rows=self.geometry.num_embeddings[ordinal],
                embedding_width=256, scale_block=32, encoding=proto.EngramTable.FP8_E4M3_E8M0,
                weight_buffer=f"engram_{layer}_weights", scale_buffer=f"engram_{layer}_scales",
                hash_buffer=f"engram_{layer}_hashes", ready_event=f"engram_{layer}_ready",
                request_buffer=f"engram_{layer}_requests")
            table.hash.method = proto.EngramHash.XOR_MODULO
            table.hash.orders.extend(order for order in (2, 3, 4) for _ in range(8))
            table.hash.multipliers.extend(self.geometry.multipliers[ordinal])
            table.hash.primes.extend(self.geometry.primes[ordinal])
            table.hash.offsets.extend(self.geometry.offsets[ordinal])
            for name, source in zip(("weights", "scales"), self.sources[layer], strict=True):
                cursor = align(cursor)
                span = getattr(table, name)
                span.offset, span.nbytes = cursor, source.nbytes
                payload.append((cursor, source.path, source.offset, source.nbytes))
                cursor += source.nbytes
        return cursor


class PreparedTableRequests:
    def __init__(self, tables, metadata, pool, *, session, name):
        self.metadata, self.pool = metadata, pool
        rows, device = metadata.starts.numel(), metadata.starts.device
        self.raw_history = torch.zeros((rows, 3), dtype=torch.int64, device=device)
        self.history = torch.zeros_like(self.raw_history)
        self.columns = torch.arange(3, device=device)
        self.token_map = torch.tensor(tables.token_map, dtype=torch.int64, device=device)
        self.operations = tuple(PreparedEngramRequests(inputs, metadata.token_ids, metadata.query_start_loc,
            self.history, metadata.live_counts, tables.mailboxes[layer].device_view, tables.sequence,
            layer=layer, token_map=tables.token_map, geometry=tables.geometry, session=session,
            name=f"{name}.{layer}") for layer, inputs in tables.inputs.items())

    def __call__(self):
        m = self.metadata
        gather_rows(self.pool.region("engram_history"), m.state_rows, self.raw_history)
        raw = torch.where(self.columns >= 3 - m.starts[:, None], self.raw_history, 2)
        self.history.copy_(self.token_map[raw])
        for operation in self.operations:
            operation()

    def commit(self, counts):
        m = self.metadata
        relative = counts[:, None].long() + self.columns - 3
        indices = (m.query_start_loc[:-1, None] + relative).clamp(0, m.token_ids.numel() - 1)
        history = torch.where(relative >= 0, m.token_ids[indices],
                              self.raw_history.gather(1, (relative + 3).clamp(0, 2)))
        save_hidden(history, self.pool.region("engram_history"), m.state_rows, counts > 0)


class PreparedEngram:
    def __init__(self, layer, weights, tables, source, output, metadata, pool, *, session, name, workspace):
        self.source, self.output, self.metadata, self.pool = source, output, metadata, pool
        tokens, device = metadata.token_ids.numel(), source.device
        self.mask = torch.zeros(tokens, dtype=torch.bool, device=device)
        self.gathered = torch.zeros((tokens, 6144), dtype=torch.bfloat16, device=device)
        self.projected = torch.zeros((tokens, 25600), dtype=torch.bfloat16, device=device)
        self.lookup = PreparedEngramLookup(tables.inputs[layer], self.gathered, metadata.live_counts[:1],
            layer=layer, token_map=tables.token_map, geometry=tables.geometry,
            session=session, name=name + ".lookup")
        self.projection = QuantizedProjection(self.gathered, weights["projection"], self.projected,
            session=session, name=name + ".projection", workspace=workspace, activation_mode="auto")
        self.norm_weight = weights["weight"].view(-1)
        self.eps = pool.config.text["rms_norm_eps"]
        self.mix_plan = hyperconnection.plan(hyperconnection.Caps(device=device, max_tokens=tokens, hidden_size=5120),
            invocation={"operation": "engram_mix", "eps": self.eps, "token_mask": True})
        declare(session, (self.mix_plan.request(name=name + ".mix", prepare_call=lambda state: PreparedCall(
            run=lambda: state.launch(source.view(tokens, -1), self.projected, self.norm_weight,
                                      self.mask, output.view(tokens, -1), eps=self.eps))),))

    def __call__(self, source):
        m = self.metadata
        self.mask.copy_(m.positions >= 0)
        self.lookup(source)
        self.projection(self.gathered)
        hyperconnection.run_engram_mix(source.view(source.shape[0], -1), self.projected, self.norm_weight,
            eps=self.eps, plan=self.mix_plan, out=self.output.view(source.shape[0], -1), token_mask=self.mask)
        return self.output
