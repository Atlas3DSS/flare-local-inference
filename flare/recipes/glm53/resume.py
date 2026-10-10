"""Sample from GLM's cached final normalized hidden output without token consumption."""

import torch

from flare.components.collectives import PreparedAllReduce
from flare.components.resume import PreparedResumeState
from flare.components.sampling import PreparedCompletion, PreparedSampling
from flare.components.vocabulary import PreparedVocabularyProjection


class PreparedResume:
    def __init__(self, weights, requests, pool, inputs, records, frame, sequence, *, rows,
                 runtime, vocabulary_runtime, coordinator, session, name):
        counts = pool.consumed_tokens[1:1 + requests.capacity, 0]
        if requests.consumed_tokens.data_ptr() != counts.data_ptr() or requests.consumed_tokens.stride() != counts.stride():
            raise ValueError("GLM resume counts must alias continuation storage live rows")
        self.state = PreparedResumeState(requests, pool.last_hidden[1:1 + requests.capacity], inputs.row_slot_ids[:rows],
            inputs.row_generations[:rows], inputs.live_rows, world_size=weights.shard.world_size)
        self.validation = PreparedAllReduce(self.state.validation_input, self.state.validation_sum,
            runtime=runtime, coordinator=coordinator, session=session, name=f"{name}.validation")
        self.local_logits = torch.empty((rows, 154880 // weights.shard.world_size),
            dtype=torch.bfloat16, device=self.state.hidden.device)
        self.logits = torch.empty((rows, 154880), dtype=torch.bfloat16, device=self.state.hidden.device)
        self.projection = PreparedVocabularyProjection(self.state.hidden, weights.tree.lm_head.weight,
            self.local_logits, self.logits, runtime=vocabulary_runtime, coordinator=coordinator,
            session=session, name=f"{name}.lm_head")
        arguments = (requests, inputs.row_slot_ids[:rows], inputs.row_generations[:rows], inputs.live_rows,
                     records, frame, sequence)
        self.completion = PreparedCompletion(*arguments) if weights.shard.rank != 0 else PreparedSampling(self.logits, *arguments)
        self.completion.prepare()

    def __call__(self):
        self.validation(self.state.validate())
        hidden = self.state.gather()
        self.projection(hidden)
        self.completion()
