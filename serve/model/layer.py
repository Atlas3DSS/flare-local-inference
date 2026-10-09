"""Universal transformer layer: composable attention + FFN blocks.

Any attention type (paged, GDN, MLA, DSA) plugs in as `attn_block`.
Any FFN type (MoE, dense) plugs in as `ffn_block`.
The layer handles the residual + norm skeleton.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerLayer(nn.Module):
    """One transformer layer: norm → attn → residual → norm → FFN → residual."""

    def __init__(self, *, attn, ffn, norm1, norm2):
        super().__init__()
        self.attn = attn    # Any attention block with forward_from_state(hidden, state).
        self.ffn = ffn      # Any FFN block with forward(hidden, state).
        self.norm1 = norm1  # Pre-attention norm.
        self.norm2 = norm2  # Pre-FFN norm.

    def forward(self, hidden_states: torch.Tensor, state) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn.forward_from_state(hidden_states, state)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn(hidden_states, state)
        hidden_states = residual + hidden_states
        return hidden_states

    # -- delegation for workspace/buffer injection (CUDA graphs) ---------------

    def bind_cache(self, **kwargs):
        """Bind cache refs to the attention block."""
        self.attn.bind_cache(**kwargs)

    def set_moe_workspace(self, workspace):
        self.ffn.set_moe_workspace(workspace)

    def set_moe_output_buffer(self, buf):
        self.ffn.set_moe_output_buffer(buf)
