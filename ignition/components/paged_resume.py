"""Sample from a cached normalized hidden state without consuming a token."""

from copy import copy

import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.resume import PreparedResumeState
from ignition.components.sampling import PreparedCompletion, PreparedSampling
from ignition.components.vocabulary import PreparedVocabularyProjection


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
    ):
        self.state = PreparedResumeState(
            requests,
            pool.last_hidden,
            inputs.row_slot_ids[:rows],
            inputs.row_generations[:rows],
            inputs.live_rows,
            world_size=weights.shard.world_size,
            state_rows=inputs.request_state_rows,
        )
        self.validation = PreparedAllReduce(
            self.state.validation_input,
            self.state.validation_sum,
            runtime=runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".validation",
        )
        self.local_logits = torch.empty(
            (rows, requests.vocab_size // weights.shard.world_size),
            dtype=torch.bfloat16,
            device=pool.backing.device,
        )
        self.logits = torch.empty(
            (rows, requests.vocab_size), dtype=torch.bfloat16, device=pool.backing.device
        )
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
        self.completion = (
            PreparedSampling(self.logits, *args)
            if weights.shard.rank == 0 or self.replicated_sampling
            else PreparedCompletion(*args)
        )
        self.completion.prepare()
        self.peer_completion = PreparedCompletion(*args)
        self.peer_completion.prepare()
        self.compilation_variant = (
            "sample" if weights.shard.rank == 0 or self.replicated_sampling else "consume"
        )

    def compilation_variants(self):
        if self.replicated_sampling:
            return {"sample": self}
        peer = copy(self)
        peer.completion = self.peer_completion
        return {"sample": self, "consume": peer}

    def __call__(self):
        self.validation(self.state.validate())
        self.projection(self.state.gather())
        self.completion()
