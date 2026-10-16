"""Kimi K3 KDA with fused input projections and a full-rank output gate."""

import torch
from b12x.sequence import gdn_decode, kda_prefill, short_conv

from ignition.components.gated_norm import PreparedGatedRMSNorm
from ignition.components.linear import UnquantizedProjection
from ignition.components.recurrent import prepare_sequence_binding


class PreparedKDA:
    def __init__(self, weights, source, state, metadata, *, mode, eps, session, name, workspace, buffers):
        tokens, hidden = source.shape
        heads, width = state.recurrent.shape[1], state.recurrent.shape[1] * 128
        device = source.device

        def buffer(key, channels):
            if key not in buffers:
                buffers[key] = torch.zeros((tokens, channels), dtype=torch.bfloat16, device=device)
            return buffers[key][:tokens]

        self.input = buffer("input", 3 * width + heads + 128)
        qkv, beta, forget = self.input.split((3 * width, heads, 128), dim=-1)
        self.forget = buffer("forget_input", 128)
        raw_g, gate = buffer("raw_g", width), buffer("gate", width)
        mixed = buffer("mixed", 3 * width)
        recurrence, gated = buffer("recurrence", width), buffer("gated", width)
        self.projections = {key: UnquantizedProjection(x, weights[key], y, session=session, name=name + "." + key)
            for key, x, y in (("input", source, self.input), ("forget", self.forget, raw_g),
                             ("gate", source, gate), ("output", gated, buffer("output", hidden)))}
        sequences = metadata.query_start_loc.numel() - 1
        conv_args = dict(x=qkv, conv_weight=weights["convolution"], query_start_loc=metadata.query_start_loc,
            state_slot_ids=metadata.convolution_slots, state_is_fresh=metadata.fresh,
            num_accepted_tokens=metadata.accepted_tokens, num_seqs=metadata.num_seqs, num_tokens=metadata.num_tokens,
            conv_state=state.convolution, out=mixed)
        caps = short_conv.Caps(device=device, mode=mode, channels=3 * width, kernel_size=4,
            max_tokens=tokens, max_seqs=sequences, max_state_slots=state.convolution.shape[0], max_speculative_tokens=0)
        plan = short_conv.plan(caps, invocation=short_conv.invocation_from_tensors(**conv_args))
        prepare_sequence_binding(plan, conv_args, session=session, name=name + ".conv", mode=mode,
            public_bind=short_conv.bind, bind_plan=True, workspace=workspace,
            set_binding=lambda binding: setattr(self, "convolution", binding))
        common = dict(raw_g=raw_g.view(tokens, heads, 128), raw_beta=beta, A_log=weights["A_log"],
            dt_bias=weights["dt_bias"].view(heads, 128), recurrent_state=state.recurrent,
            num_seqs=metadata.num_seqs, num_tokens=metadata.num_tokens,
            output=(gated if mode == "decode" else recurrence).view(tokens, heads, 128))
        self.options, self.mode = {"lower_bound": -5.0}, mode
        if mode == "decode":
            self.options["eps"] = eps
            caps = gdn_decode.Caps(device=device, max_tokens=tokens, max_seqs=sequences,
                max_state_slots=state.recurrent.shape[0], key_heads=heads, value_heads=heads,
                state_index_columns=1, gate_activation="sigmoid", null_state_index=0)
            args = dict(common, mixed_qkv=mixed, z=gate.view(tokens, heads, 128), norm_weight=weights["norm"],
                query_start_loc=metadata.query_start_loc, num_accepted_tokens=metadata.accepted_tokens,
                state_indices=metadata.state_indices)
            plan = gdn_decode.plan(caps, invocation=gdn_decode.invocation_from_tensors(caps, **args))
            bind, method = gdn_decode.bind_kda, "bind_kda"
        else:
            caps = kda_prefill.Caps(device=device, max_tokens=tokens, max_seqs=sequences,
                max_state_slots=state.recurrent.shape[0], heads=heads, null_state_index=0)
            q, k, v = (x.view(tokens, heads, 128) for x in mixed.split(width, dim=-1))
            args = dict(common, q=q, k=k, v=v, cu_seqlens=metadata.query_start_loc,
                initial_state_indices=metadata.initial_state_indices, final_state_indices=metadata.final_state_indices,
                checkpoint_state_indices=metadata.checkpoint_state_indices, checkpoint_offsets=metadata.checkpoint_offsets)
            plan = kda_prefill.plan(caps, invocation=kda_prefill.invocation_from_tensors(**args))
            bind, method = kda_prefill.bind, "bind"
            self.norm = PreparedGatedRMSNorm(recurrence.view(tokens, heads, 128), gate.view(tokens, heads, 128),
                weights["norm"], gated.view(tokens, heads, 128), metadata.num_tokens, eps=eps)
        prepare_sequence_binding(plan, args, session=session, name=name + ".recurrence", mode=mode,
            public_bind=bind, bind_method=method, workspace=workspace, run_options=self.options,
            set_binding=lambda binding: setattr(self, "recurrence", binding))
        self.gated = gated

    def __call__(self, source):
        p = self.projections
        p["input"](source)
        self.forget.copy_(self.input[:, -128:])
        p["forget"](self.forget)
        p["gate"](source)
        short_conv.run(self.convolution)
        if self.mode == "decode":
            gdn_decode.run_kda(self.recurrence, **self.options)
        else:
            kda_prefill.run(self.recurrence, **self.options)
            self.norm()
        return p["output"](self.gated)
