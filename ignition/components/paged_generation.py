"""Collective text execution and request completion."""

from ignition.components.collectives import PreparedAllReduce
from ignition.components.sampling import PreparedCompletion, PreparedSampling
from ignition.components.state import save_hidden


class PreparedGeneration:
    def __init__(
        self,
        decoder,
        batch,
        records,
        frame,
        sequence,
        *,
        runtime,
        coordinator,
        session,
        name,
        workspace,
        context_factory=None,
    ):
        self.decoder, self.batch = decoder, batch
        self.validation = PreparedAllReduce(
            batch.validation_input,
            batch.validation_sum,
            runtime=runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".validation",
        )
        b, rows = batch.inputs, batch.rows
        args = (
            batch.requests,
            b.row_slot_ids[:rows],
            b.row_generations[:rows],
            b.live_rows,
            records,
            frame,
            sequence,
        )
        self.replicated_sampling = decoder.weights.draft is not None
        self.completion = (
            PreparedSampling(decoder.scratch.logits, *args, sample_enabled=b.sample_enabled[:rows])
            if batch.shard.rank == 0 or self.replicated_sampling
            else PreparedCompletion(*args)
        )
        self.completion.prepare()
        self.compilation_variant = (
            "sample" if batch.shard.rank == 0 or self.replicated_sampling else "consume"
        )
        self.context = None
        if self.replicated_sampling:
            if context_factory is None:
                raise ValueError("speculative generation requires a draft context factory")
            self.context = context_factory(
                decoder.weights,
                decoder.auxiliary,
                batch.metadata,
                decoder.pool,
                session=session,
                name=name + ".context",
                workspace=workspace,
            )

    def __call__(self):
        self.batch.begin(self.validation(self.batch.validate()))
        self.decoder()
        if self.context is not None:
            self.context(self.decoder.auxiliary, self.batch.metadata.lengths)
        save_hidden(
            self.decoder.scratch.hidden,
            self.decoder.pool.last_hidden,
            self.batch.metadata.state_rows,
            self.batch.token_state.active,
        )
        self.batch.finish()
        self.completion()
