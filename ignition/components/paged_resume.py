"""Sample from a cached normalized hidden state without consuming a token."""

import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.resume import PreparedResumeState
from ignition.components.sampling import PreparedCompletion, PreparedSampling
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.preparation.scratch import bind_scratch


class PreparedResume:
    def __init__(
        self,
        weights,
        requests,
        pool,
        inputs,
        records,
        frame,
        sequence,
        *,
        rows,
        runtime,
        vocabulary_runtime,
        coordinator,
        session,
        name,
        activation_mode="a16",
        buffers=None,
    ):
        self.state = PreparedResumeState(
            requests,
            pool.last_hidden,
            inputs.row_slot_ids[:rows],
            inputs.row_generations[:rows],
            inputs.live_rows,
            world_size=weights.shard.world_size,
            state_rows=inputs.request_state_rows,
            buffers=buffers,
        )
        self.validation = PreparedAllReduce(
            self.state.validation_input,
            self.state.validation_sum,
            runtime=runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".validation",
        )
        def empty(shape):
            return (torch.empty(shape, dtype=torch.bfloat16, device=pool.backing.device) if buffers is None
                    else buffers.zeros(shape, torch.bfloat16))
        self.local_logits = empty((rows, weights.head.shape[0] if isinstance(weights.head, torch.Tensor)
                                   else weights.head.out_features))
        self.logits = empty((rows, requests.vocab_size))
        self.projection = PreparedVocabularyProjection(
            self.state.hidden,
            weights.head,
            self.local_logits,
            self.logits,
            runtime=vocabulary_runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".head",
            activation_mode=activation_mode,
            buffers=buffers,
        )
        args = (
            requests,
            inputs.row_slot_ids[:rows],
            inputs.row_generations[:rows],
            inputs.live_rows,
            records,
            frame,
            sequence,
        )
        self.replicated_sampling = weights.draft is not None
        # Shared forward artifacts require identical pool layouts on sampling and consuming ranks.
        sampling_workspace = None if buffers is None else bind_scratch(
            PreparedSampling.scratch_layout(rows, requests.vocab_size),
            device=self.logits.device, workspace=buffers)
        self.completion = (
            PreparedSampling(self.logits, *args, workspace=sampling_workspace)
            if weights.shard.rank == 0 or self.replicated_sampling
            else PreparedCompletion(*args)
        )
        self.completion.prepare()
        self.compilation_variant = (
            "sample" if weights.shard.rank == 0 or self.replicated_sampling else "consume"
        )

    def __call__(self):
        self.validation(self.state.validate())
        self.projection(self.state.gather())
        self.completion()
