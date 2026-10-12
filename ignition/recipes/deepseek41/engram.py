"""File-backed FP8 n-gram tables, compact GPU inputs, and residual injection."""

from dataclasses import dataclass

import torch
from b12x.norm import hyperconnection
from b12x.preparation import PreparedCall
from b12x.sequence import engram

from ignition.components.engram import EngramInputs, PreparedEngramLookup
from ignition.components.linear import QuantizedProjection
from ignition.preparation.runtime_package import FileSource
from ignition.preparation.tuning import declare
from ignition.progress import phase


@dataclass(frozen=True)
class Tables:
    geometry: object
    token_map: tuple[int, ...]
    sources: dict
    inputs: dict

    @classmethod
    def prepare(cls, manifest, config, *, tokens, device, progress=None):
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
        sources, inputs = {}, {}
        with phase(progress, "Prepare Engram tables") as stage:
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
            stage.update(f"{sum(source.nbytes for pair in sources.values() for source in pair) / (1 << 30):.1f} GiB on disk")
        return cls(geometry, tuple(token_map), sources, inputs)

    def buffers(self):
        return {name: tensor for layer, inputs in self.inputs.items() for name, tensor in inputs.buffers(layer).items()}

    def events(self):
        return {f"engram_{layer}_ready": inputs.ready.cuda_event for layer, inputs in self.inputs.items()}

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
                hash_buffer=f"engram_{layer}_hashes", ready_event=f"engram_{layer}_ready")
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
