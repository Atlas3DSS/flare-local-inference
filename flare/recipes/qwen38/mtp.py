"""Prepared single-layer MTP feedback, sparse attention and W4A16 experts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from b12x.preparation import PreparedCall
from b12x.sequence import mtp_feedback

from flare.components.embedding import PreparedEmbedding
from flare.components.linear import UnquantizedProjection
from flare.preparation.fixtures import mtp_feedback as feedback_fixtures
from flare.preparation.scratch import ScratchArena, planned_scratch, prime_owners
from flare.preparation.tuning import register_finalizer

from .attention import PreparedAttention, QSAScratch
from .hyperconnection import HyperconnectionScratch, PreparedHyperconnection
from .moe import PreparedQwenMoe, QwenMoeScratch
from .mtp_weights import MTPWeights


@dataclass(frozen=True)
class MTPScratch:
    """Persistent intermediate/output storage, separate from volatile kernel scratch."""

    embedding_ids: torch.Tensor
    embedding: torch.Tensor
    residual: torch.Tensor
    hidden: torch.Tensor
    selected_hidden: torch.Tensor
    logits: torch.Tensor
    hyperconnection: HyperconnectionScratch
    qsa: QSAScratch
    moe: QwenMoeScratch

    @classmethod
    def allocate(cls, tokens: int, output_rows: int, device):
        if (type(tokens) is not int or type(output_rows) is not int
                or not 0 < output_rows <= tokens):
            raise ValueError("MTP requires positive output rows no greater than token capacity")

        def zeros(rows, width):
            return torch.zeros((rows, width), dtype=torch.bfloat16, device=device)

        return cls(torch.zeros(tokens, dtype=torch.int64, device=device),
            zeros(tokens, 2560), zeros(tokens, 10240), zeros(tokens, 2560),
            zeros(output_rows, 2560), zeros(output_rows, 248320),
            HyperconnectionScratch.allocate(tokens, device), QSAScratch.allocate(tokens, device),
            QwenMoeScratch.allocate(tokens, device))

    def rows(self, tokens: int, output_rows: int):
        if (type(tokens) is not int or type(output_rows) is not int
                or not 0 < output_rows <= min(tokens, self.logits.shape[0])
                or not tokens <= self.residual.shape[0]):
            raise ValueError("MTP view exceeds persistent token/output capacity")
        qsa = QSAScratch(*(getattr(self.qsa, field)[:tokens] for field in self.qsa.__dataclass_fields__))
        return type(self)(self.embedding_ids[:tokens], self.embedding[:tokens], self.residual[:tokens],
            self.hidden[:tokens], self.selected_hidden[:output_rows], self.logits[:output_rows],
            self.hyperconnection.rows(tokens), qsa, self.moe.rows(tokens))


@dataclass(frozen=True)
class MTPOutput:
    residual: torch.Tensor
    hidden: torch.Tensor
    logits: torch.Tensor


class PreparedMTPFeedback:
    """Qwen's flattened-state normalization and shared-stream BF16 projection."""

    def __init__(self, module, token_embedding, target_hidden, output, workspace, *, session, name):
        self.eps = module.config.rms_norm_eps
        self.kwargs = {
            "token_embedding": token_embedding,
            "multi_state": target_hidden.view(-1, 4, 2560),
            "token_norm_weight": module.pre_fc_norm_embedding.weight,
            "state_norm_weight": module.pre_fc_norm_hidden.weight,
            "embedding_fc_weight": module.fc_embedding.weight,
            "hidden_fc_weight": module.fc_hidden.weight,
            "output": output.view(-1, 4, 2560),
        }
        self.plan = mtp_feedback.plan(
            mtp_feedback.Caps(device=target_hidden.device, max_tokens=target_hidden.shape[0]),
            invocation=mtp_feedback.invocation_from_tensors(**self.kwargs),
        )

        def prepare(state):
            scratch = planned_scratch(state.layout.scratch_specs(), workspace)
            binding = state.bind(scratch=scratch, **self.kwargs)
            return PreparedCall(run=lambda: state.run(binding, eps=self.eps),
                                owners=prime_owners(workspace, scratch, binding))

        def benchmark(state):
            return feedback_fixtures.benchmark_call(
                state, arguments=self.kwargs, session=session, name=name,
                workspace=workspace, eps=self.eps,
            )

        session.prepare((self.plan.request(name=name, prepare_call=prepare, benchmark_call=benchmark),))

        def finalize():
            self.scratch = planned_scratch(self.plan.scratch_specs(), workspace)
            self.binding = mtp_feedback.bind(self.plan, scratch=self.scratch, **self.kwargs)

        finalize()
        register_finalizer(session, finalize, workspace=workspace)

    def __call__(self):
        return mtp_feedback.run(self.binding, eps=self.eps)


class PreparedMTP:
    """Fixed storage for absorption and recursive proposal graphs.

    Each input row supplies target/recursive residual ``h[i]``, token ``x[i+1]``
    and unshifted position ``i`` through QSA metadata. The caller owns the
    accepted-history alignment and all cache commit/rollback transitions.
    ``output_indices`` selects vocabulary rows without allocating prefill-sized
    logits. Returned residuals precede the final mixer and retain all streams.
    """

    def __init__(self, weights: MTPWeights, scratch: MTPScratch, token_ids, target_hidden,
                 qsa_state, metadata, rotary, workspace: ScratchArena, *, output_indices,
                 session, name: str, draft_tokens: int = 7, max_cache_pages=None):
        tokens, device = scratch.residual.shape[0], scratch.residual.device
        if (target_hidden.shape != (tokens, 10240) or target_hidden.dtype != torch.bfloat16
                or target_hidden.device != device or not target_hidden.is_contiguous()):
            raise ValueError("MTP feedback requires contiguous BF16 [tokens,10240] residuals")
        if (token_ids.shape != (tokens,) or token_ids.dtype != torch.int64
                or token_ids.device != device or not token_ids.is_contiguous()):
            raise ValueError("MTP token IDs require contiguous int64 [tokens] on the model device")
        if (output_indices.shape != (scratch.logits.shape[0],) or output_indices.dtype != torch.int64
                or output_indices.device != device or not output_indices.is_contiguous()):
            raise ValueError("MTP output indices require contiguous int64 [output_rows]")
        if metadata.request_ids.shape != (tokens,) or metadata.request_ids.device != device:
            raise ValueError("MTP QSA row metadata must match feedback capacity and device")
        if torch._C._overlaps(target_hidden, scratch.residual):
            raise ValueError("MTP feedback input must not alias its residual output")
        self.weights, self.scratch, self.token_ids = weights, scratch, token_ids
        self.target_hidden, self.metadata, self.output_indices = target_hidden, metadata, output_indices
        self.workspace = workspace
        self.output = MTPOutput(scratch.residual, scratch.hidden, scratch.logits)
        self.embedding = PreparedEmbedding(weights.embedding, scratch.embedding_ids, scratch.embedding,
            session=session, name=f"{name}.embedding")
        self.feedback = PreparedMTPFeedback(weights.module, scratch.embedding, target_hidden,
            scratch.residual, workspace, session=session, name=f"{name}.feedback")
        layer = weights.module.layers[0]
        source = scratch.hyperconnection.block_input
        self.attn_hc = PreparedHyperconnection(layer.attn_hyper_connection, scratch.hyperconnection,
            priming_state=scratch.residual, session=session, name=f"{name}.attn_hc")
        self.attention = PreparedAttention(layer.self_attn, weights.attention, source, scratch.qsa,
            qsa_state, metadata, rotary, session=session, name=f"{name}.qsa", draft_tokens=draft_tokens,
            workspace=workspace, projection_workspace=workspace, max_cache_pages=max_cache_pages)
        self.mlp_hc = PreparedHyperconnection(layer.mlp_hyper_connection, scratch.hyperconnection,
            priming_state=scratch.residual, session=session, name=f"{name}.mlp_hc")
        self.moe = PreparedQwenMoe(source, weights.moe, session=session, name=f"{name}.moe",
            scratch=scratch.moe, workspace=workspace, projection_workspace=workspace)
        self.final_mixer = PreparedHyperconnection(weights.module.hyper_connection_mixer, scratch.hyperconnection,
            priming_state=scratch.residual, session=session, name=f"{name}.final_mixer")
        self.attn_hc.prepare_combine_norm(self.mlp_hc, priming_state=scratch.residual,
            session=session, name=f"{name}.attn_combine_norm")
        self.lm_head = UnquantizedProjection(scratch.selected_hidden, weights.lm_head, scratch.logits,
                                     session=session, name=f"{name}.lm_head")

    def rebind_pages_for_preparation(self, state):
        self.attention.rebind_pages_for_preparation(state)

    def __call__(self):
        s = self.scratch
        active = self.metadata.request_ids >= 0
        s.embedding_ids.copy_(self.token_ids)
        s.embedding_ids.masked_fill_(~active, 0)
        self.embedding()
        self.feedback()
        s.residual.masked_fill_(~active[:, None], 0)
        mixed = self.attn_hc.mix(s.residual)
        block = self.attention(mixed)
        self.attn_hc.combine_norm(s.residual, block)
        mixed = self.mlp_hc.mix(s.residual, pre_normalized=True)
        block = self.moe(mixed)
        s.residual.copy_(self.mlp_hc.combine(s.residual, block))
        s.residual.masked_fill_(~active[:, None], 0)
        s.hidden.copy_(self.final_mixer.mix(s.residual))
        torch.index_select(s.hidden, 0, self.output_indices, out=s.selected_hidden)
        self.lm_head(s.selected_hidden)
        return self.output
