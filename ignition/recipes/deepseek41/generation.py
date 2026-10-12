"""Collective text execution and request completion."""

from copy import copy

from ignition.components.collectives import PreparedAllReduce
from ignition.components.sampling import PreparedCompletion, PreparedSampling
from ignition.components.state import save_hidden


class PreparedGeneration:
    def __init__(self, decoder, batch, records, frame, sequence, *, runtime, coordinator, session, name):
        self.decoder, self.batch = decoder, batch
        self.validation = PreparedAllReduce(batch.validation_input, batch.validation_sum,
            runtime=runtime, coordinator=coordinator, session=session, name=name + ".validation")
        b, rows = batch.inputs, batch.rows
        args = (batch.requests, b.row_slot_ids[:rows], b.row_generations[:rows], b.live_rows,
                records, frame, sequence)
        self.completion = (PreparedSampling(decoder.scratch.logits, *args, sample_enabled=b.sample_enabled[:rows])
            if batch.shard.rank == 0 else PreparedCompletion(*args))
        self.completion.prepare()
        self.peer_completion = PreparedCompletion(*args)
        self.peer_completion.prepare()
        self.compilation_variant = "sample" if batch.shard.rank == 0 else "consume"

    def compilation_variants(self):
        peer = copy(self)
        peer.batch = copy(self.batch)
        peer.batch.accept_unsampled = True
        peer.completion = self.peer_completion
        return {"sample": self, "consume": peer}

    def __call__(self):
        self.batch.begin(self.validation(self.batch.validate()))
        self.decoder()
        save_hidden(self.decoder.scratch.hidden, self.decoder.pool.last_hidden,
                    self.batch.metadata.state_rows, self.batch.token_state.active)
        self.batch.finish()
        self.completion()
