"""Compose a GLM rank's text forward, request consumption, and completion."""

from copy import copy

from ignition.components.collectives import PreparedAllReduce
from ignition.components.sampling import PreparedCompletion, PreparedSampling
from ignition.components.state import save_hidden

from .batch import PreparedBatch
from .stack import PreparedDecoder


class PreparedGeneration:
    """One captured rank step; rank zero samples and peers record consumption.

    All ranks receive identical token uploads before replay. Native execution
    must wait for every rank before retiring a step or reusing its cache pages.
    """

    def __init__(self, decoder: PreparedDecoder, batch: PreparedBatch, completion_records,
                 frame_index, submission_sequence, *, pool, runtime, coordinator, session, name, replicated_sampling=False):
        if decoder.metadata is not batch.metadata or decoder.weights.shard != batch.shard:
            raise ValueError("GLM generation requires the decoder's metadata and rank")
        if decoder.state is not pool.decoder:
            raise ValueError("GLM generation requires the decoder's continuation pool")
        self.decoder, self.batch, self.pool = decoder, batch, pool
        self.validation = PreparedAllReduce(batch.validation_input, batch.validation_sum,
            runtime=runtime, coordinator=coordinator, session=session, name=f"{name}.validation")
        b, rows = batch.inputs, batch.rows
        arguments = (batch.requests, b.row_slot_ids[:rows], b.row_generations[:rows], b.live_rows,
                     completion_records, frame_index, submission_sequence)
        self.completion = PreparedCompletion(*arguments) if decoder.weights.shard.rank != 0 and not replicated_sampling else PreparedSampling(
            decoder.scratch.logits, *arguments, sample_enabled=b.sample_enabled[:rows])
        self.completion.prepare()
        self.compilation_variant = "sample" if decoder.weights.shard.rank == 0 or replicated_sampling else "consume"
        self.replicated_sampling = replicated_sampling
        self.peer_completion = PreparedCompletion(*arguments)
        self.peer_completion.prepare()

    def compilation_variants(self):
        if self.replicated_sampling:
            return {"sample": self}
        peer = copy(self)
        peer.batch = copy(self.batch)
        peer.batch.accept_unsampled = True
        peer.completion = self.peer_completion
        return {"sample": self, "consume": peer}

    def __call__(self):
        self.batch.begin(self.validation(self.batch.validate()))
        self.decoder()
        save_hidden(self.decoder.scratch.hidden, self.pool.last_hidden,
                    self.batch.metadata.recurrent.convolution_slots, self.batch.token_state.active)
        self.batch.finish()
        self.completion()
