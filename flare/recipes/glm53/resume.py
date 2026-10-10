"""Sample from GLM's cached final normalized hidden output without token consumption."""

import torch

from flare.components.collectives import PreparedAllReduce
from flare.components.linear import UnquantizedProjection
from flare.components.resume import PreparedResumeState
from flare.components.sampling import PreparedCompletion, PreparedSampling


class PreparedResume:
    def __init__(self, weights, requests, pool, inputs, records, frame, sequence, *, rows,
                 runtime, coordinator, session, name):
        counts = pool.consumed_tokens[1:5, 0]
        if requests.consumed_tokens.data_ptr() != counts.data_ptr() or requests.consumed_tokens.stride() != counts.stride():
            raise ValueError("GLM resume counts must alias continuation storage live rows")
        self.state = PreparedResumeState(requests, pool.last_hidden[1:5], inputs.row_slot_ids[:rows],
            inputs.row_generations[:rows], inputs.live_rows, world_size=weights.shard.world_size)
        self.validation = PreparedAllReduce(self.state.validation_input, self.state.validation_sum,
            runtime=runtime, coordinator=coordinator, session=session, name=f"{name}.validation")
        self.logits, self.projection = None, None
        if weights.shard.rank == 0:
            self.logits = torch.empty((rows, 154880), dtype=torch.bfloat16, device=self.state.hidden.device)
            self.projection = UnquantizedProjection(self.state.hidden, weights.tree.lm_head.weight, self.logits,
                                                    session=session, name=f"{name}.lm_head")
        arguments = (requests, inputs.row_slot_ids[:rows], inputs.row_generations[:rows], inputs.live_rows,
                     records, frame, sequence)
        self.completion = PreparedCompletion(*arguments) if self.logits is None else PreparedSampling(self.logits, *arguments)
        self.completion.prepare()

    def __call__(self):
        self.validation(self.state.validate())
        hidden = self.state.gather()
        if self.projection is not None:
            self.projection(hidden)
        self.completion()
