"""Compose a GLM rank's text forward, request consumption, and completion."""

from flare.components.collectives import PreparedAllReduce
from flare.components.sampling import PreparedCompletion, PreparedSampling

from .batch import PreparedBatch
from .stack import PreparedDecoder


class PreparedGeneration:
    """One captured rank step; rank zero samples and peers record consumption.

    All ranks receive identical token uploads before replay. Native execution
    must wait for every rank before retiring a step or reusing its cache pages.
    """

    def __init__(self, decoder: PreparedDecoder, batch: PreparedBatch, completion_records,
                 frame_index, submission_sequence, *, runtime, coordinator, session, name):
        if decoder.metadata is not batch.metadata or decoder.weights.shard != batch.shard:
            raise ValueError("GLM generation requires the decoder's metadata and rank")
        self.decoder, self.batch = decoder, batch
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
        self.batch.finish()
        self.completion()
