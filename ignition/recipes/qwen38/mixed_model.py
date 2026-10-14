"""Mixed prompt and decode execution through one packed Qwen target forward."""

from copy import copy
from types import SimpleNamespace

import torch

from ignition.components.sampling import PreparedSampling

from .decoder import PreparedDecoder
from .decoder_state import PreparedDecoderState
from .mixed_state import MixedPacking, PreparedMixedState, role_inputs
from .speculative_model import PreparedPrefillAbsorption, PreparedSpeculativeRound
from .vision_state import PreparedEmbeddingInjection


class MixedEmbeddingInjection:
    def __init__(self, packing, features, state, embedding, scratch):
        self.packing, self.embedding, self.scratch = packing, embedding, scratch
        i = state.inputs
        self.prompt = PreparedEmbeddingInjection(features, state.requests, i.vision_token_indices,
            i.row_slot_ids, i.row_generations, i.query_lengths, i.live_rows, scratch)
        self.prompt.prepare()

    def validate(self):
        self.prompt.validate()

    def inject(self):
        self.scratch.copy_(self.embedding[self.packing.prompt_indices])
        self.prompt.inject()
        self.embedding.copy_(torch.where(self.packing.is_prompt[:, None],
            self.scratch[self.packing.prompt_source], self.embedding))


class PreparedMixedRound:
    """Keep role-specific sampling and commits around a shared target program."""

    def __init__(self, owner, batch, width, prefill, name, *, scratch, features, enabled):
        self.owner, self.enabled = owner, enabled
        self.prompt_inputs = role_inputs(owner.inputs, prefill)
        self.decode_inputs = role_inputs(owner.inputs, batch * width)
        self.packing = MixedPacking(owner.inputs, self.prompt_inputs, self.decode_inputs, batch, width, prefill)
        self.prompt_state = PreparedDecoderState(self.prompt_inputs, owner.requests, owner.pool,
            tokens=prefill, rows=1, mode="prefill", maintain_draft_state=enabled)
        self.prompt_completion = torch.zeros_like(owner.completion)
        self.decode_completion = torch.zeros_like(owner.completion)
        self.frames = torch.arange(2, device=owner.completion.device)
        self.slots = torch.arange(owner.requests.capacity, device=owner.completion.device)
        self.prompt_decoder = SimpleNamespace(state=self.prompt_state)
        self.absorb = None
        if enabled and owner.proposal_source == "mtp":
            prompt_owner = copy(owner)
            prompt_owner.inputs = self.prompt_inputs
            self.absorb = PreparedPrefillAbsorption(prompt_owner, self.prompt_decoder, f"{name}.prompt",
                                                    residual_output=scratch.prompt_residual[:prefill])
            owner.owners.append(self.absorb)
        else:
            self.prompt_decoder.residual_output = scratch.prompt_residual[:prefill]

        def decoder_factory(weights, decode_scratch, decode_state, cache, rotary, table, workspaces,
                            completion, frame, sequence, **kwargs):
            decode_residual = kwargs.pop("residual_output")
            kwargs.pop("verification", None)
            state = PreparedMixedState(self.packing, self.prompt_state, decode_state, scratch,
                prompt_residual=self.prompt_decoder.residual_output, decode_residual=decode_residual)
            sample_rows = batch * width + 1
            full_scratch = owner.scratch.rows(self.packing.tokens, sample_rows)
            injection = MixedEmbeddingInjection(self.packing, features, self.prompt_state,
                full_scratch.embedding, scratch.prompt_gdn.output[:prefill])
            self.decoder = PreparedDecoder(weights, full_scratch, state, cache, rotary, table, workspaces,
                completion, frame, sequence, verification=True, embedding_injection=injection, **kwargs)
            self.prompt_sampler = PreparedSampling(full_scratch.logits[-1:], owner.requests,
                self.prompt_inputs.row_slot_ids[:1], self.prompt_inputs.row_generations[:1],
                self.prompt_inputs.live_rows, self.prompt_completion, frame, sequence,
                sample_enabled=self.prompt_inputs.sample_enabled[:1], workspace=owner.sampling_workspace)
            self.prompt_sampler.prepare()
            return SimpleNamespace(scratch=decode_scratch, forward=self.decoder)

        if enabled:
            decode_owner = copy(owner)
            decode_owner.inputs, decode_owner.completion = self.decode_inputs, self.decode_completion
            # The speculative round requires a callable exposing only its verification logits.
            def speculative_decoder(*args, **kwargs):
                view = decoder_factory(*args, **kwargs)
                return DecodeTarget(view.scratch, view.forward)
            self.round = PreparedSpeculativeRound(decode_owner, batch, width, name,
                decoder_factory=speculative_decoder, residual_output=scratch.decode_residual[:batch * width])
            self.round.prepare()
            owner.owners.append(self.round)
        else:
            state = PreparedDecoderState(self.decode_inputs, owner.requests, owner.pool,
                tokens=batch, rows=batch, mode="decode", maintain_draft_state=False)
            residual = scratch.decode_residual[:batch]
            decoder_factory(owner.weights, owner.scratch.rows(batch, batch), state, owner.cache,
                owner.rotary, owner.table, owner.workspaces, self.decode_completion, owner.frame, owner.sequence,
                session=owner.session, name=name, max_cache_pages=owner.max_cache_pages,
                residual_output=residual, **owner.collectives)
            self.decode_sampler = PreparedSampling(self.decoder.scratch.logits[:batch], owner.requests,
                self.decode_inputs.row_slot_ids[:batch], self.decode_inputs.row_generations[:batch],
                self.decode_inputs.live_rows, self.decode_completion, owner.frame, owner.sequence,
                sample_enabled=self.decode_inputs.sample_enabled[:batch], workspace=owner.sampling_workspace)
            self.decode_sampler.prepare()

    def __call__(self):
        o = self.owner
        self.packing.upload_roles()
        if self.absorb is not None:
            self.absorb.begin()
        if self.enabled:
            self.round()
        else:
            self.decoder()
            self.decode_sampler()
        self.prompt_sampler()
        if self.absorb is not None:
            self.absorb.finish()
        prompt = self.slots == self.prompt_inputs.row_slot_ids[0]
        records = torch.where(prompt[None, :, None], self.prompt_completion, self.decode_completion)
        active_frame = self.frames == o.frame[0]
        o.completion.copy_(torch.where(active_frame[:, None, None], records, o.completion))


class DecodeTarget:
    """Expose the verification rows of a mixed target to shared acceptance machinery."""

    def __init__(self, scratch, forward):
        self.scratch, self.forward = scratch, forward

    def __call__(self):
        self.forward()
