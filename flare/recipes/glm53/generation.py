"""Compose a GLM rank's text forward, request consumption, and completion."""

from flare.components.collectives import PreparedAllReduce
from flare.components.sampling import PreparedCompletion, PreparedSampling
from flare.components.state import save_hidden

from .batch import PreparedBatch
from .stack import PreparedDecoder


class PreparedGeneration:
    """One captured rank step; rank zero samples and peers record consumption.

    All ranks receive identical token uploads before replay. Native execution
    must wait for every rank before retiring a step or reusing its cache pages.
    """

    def __init__(self, decoder: PreparedDecoder, batch: PreparedBatch, completion_records,
                 frame_index, submission_sequence, *, pool, runtime, coordinator, session, name):
        if decoder.metadata is not batch.metadata or decoder.weights.shard != batch.shard:
            raise ValueError("GLM generation requires the decoder's metadata and rank")
        if decoder.state is not pool.decoder:
            raise ValueError("GLM generation requires the decoder's continuation pool")
        counts = pool.consumed_tokens[1:5, 0]
        if batch.requests.consumed_tokens.data_ptr() != counts.data_ptr() or batch.requests.consumed_tokens.stride() != counts.stride():
            raise ValueError("GLM request counts must alias the continuation pool live rows")
        self.decoder, self.batch, self.pool = decoder, batch, pool
        self.validation = PreparedAllReduce(batch.validation_input, batch.validation_sum,
            runtime=runtime, coordinator=coordinator, session=session, name=f"{name}.validation")
        b, rows = batch.inputs, batch.rows
        arguments = (batch.requests, b.row_slot_ids[:rows], b.row_generations[:rows], b.live_rows,
                     completion_records, frame_index, submission_sequence)
        self.completion = PreparedCompletion(*arguments) if decoder.scratch.logits is None else PreparedSampling(
            decoder.scratch.logits, *arguments, sample_enabled=b.sample_enabled[:rows])
        self.completion.prepare()

    def __call__(self):
        self.batch.begin(self.validation(self.batch.validate()))
        self.decoder()
        save_hidden(self.decoder.scratch.hidden, self.pool.last_hidden[1:5],
                    self.batch.inputs.row_slot_ids[:self.batch.rows], self.batch.token_state.active)
        self.batch.finish()
        self.completion()
