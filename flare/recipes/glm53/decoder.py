"""GLM decoder-layer composition with b12x operators and tensor-parallel reductions."""

from dataclasses import dataclass

import torch

from flare.components.collectives import PreparedAllReduce
from flare.components.mhc import MHCScratch, MHCWeights, PreparedMHC, PreparedMHCPost
from flare.components.mlp import PreparedMLP

from .kda import KDAScratch, PreparedGlmKDA


@dataclass(frozen=True)
class DenseLayerScratch:
    attention_hc: MHCScratch
    mlp_hc: MHCScratch
    kda: KDAScratch
    attention_output: torch.Tensor
    gate_up: torch.Tensor
    intermediate: torch.Tensor
    mlp_partial: torch.Tensor
    mlp_output: torch.Tensor

    @classmethod
    def allocate(cls, tokens, world_size, device):
        if world_size not in (2, 4):
            raise ValueError("GLM layers require TP2 or TP4")
        intermediate = 12288 // world_size
        def buffer(width):
            return torch.zeros((tokens, width), dtype=torch.bfloat16, device=device)
        return cls(MHCScratch.allocate(tokens, 4096, device), MHCScratch.allocate(tokens, 4096, device),
                   KDAScratch.allocate(tokens, 64 // world_size, device), buffer(4096),
                   buffer(2 * intermediate), buffer(intermediate), buffer(4096), buffer(4096))


def _install_mhc(module, norm, weights, prefix, norm_name):
    for attribute in ("fn", "scale", "base"):
        setattr(module, attribute, torch.nn.Parameter(weights[f"{prefix}_{attribute}"].float(), requires_grad=False))
    norm.weight = torch.nn.Parameter(weights[f"{norm_name}.weight"], requires_grad=False)
    return MHCWeights(module.fn, module.scale, module.base, norm.weight)


class PreparedGlmDenseLayer:
    """KDA and dense-MLP layer with replicated mHC and two output all-reduces.

    Each rank owns a head/intermediate partition and identical four-stream
    residuals. Post-attention mixing, FFN mixing prediction, and FFN RMSNorm
    share one fused b12x operation. Scratch may be reused by serialized layers;
    recurrent state and escaping outputs must remain separately owned.
    """

    def __init__(self, module, attention_weights, mlp_weights, residual_weights,
                 source, output, scratch: DenseLayerScratch, state, metadata, *,
                 mode, runtime, coordinator, session, name, workspace):
        if module.block_type != "linear_attention" or not hasattr(module.mlp, "gate_proj"):
            raise ValueError("dense GLM layer requires canonical KDA and dense MLP modules")
        if (source.shape != (scratch.kda.output.shape[0], 4, 4096)
                or output.shape != source.shape or source.dtype != torch.bfloat16
                or output.dtype != source.dtype or output.device != source.device):
            raise ValueError("GLM layer residuals require BF16 [tokens, 4, 4096] on one device")
        self.module, self.scratch, self.output = module, scratch, output
        attention_mhc = _install_mhc(module.attn_hc, module.input_layernorm, residual_weights,
                                    "hc_attn", "input_layernorm")
        mlp_mhc = _install_mhc(module.ffn_hc, module.post_attention_layernorm, residual_weights,
                              "hc_ffn", "post_attention_layernorm")
        options = {"rms_eps": module.attn_hc.input_norm.eps,
                   "hc_eps": module.attn_hc.hc_eps, "sinkhorn_iters": module.attn_hc.hc_sinkhorn_iters,
                   "norm_eps": module.input_layernorm.variance_epsilon}
        self.attention_hc = PreparedMHC((source,), attention_mhc, scratch.attention_hc,
            session=session, name=f"{name}.attn_hc", workspace=workspace, **options)
        self.attention = PreparedGlmKDA(module.self_attn, attention_weights, scratch.attention_hc.normalized,
            scratch.kda, state, metadata, mode=mode, session=session, name=f"{name}.kda", workspace=workspace,
            checkpoint_export=mode == "prefill")
        self.attention_reduce = PreparedAllReduce(scratch.kda.output, scratch.attention_output, runtime=runtime,
            session=session, coordinator=coordinator, name=f"{name}.attention_reduce")
        self.mlp_hc = PreparedMHC((scratch.attention_output, scratch.attention_hc.residual,
                                  scratch.attention_hc.post, scratch.attention_hc.comb),
            mlp_mhc, scratch.mlp_hc, session=session, name=f"{name}.ffn_hc", workspace=workspace, **options)
        self.mlp = PreparedMLP(scratch.mlp_hc.normalized, mlp_weights, scratch.gate_up, scratch.intermediate,
            scratch.mlp_partial, session=session, name=f"{name}.mlp", workspace=workspace,
            swiglu_limit=module.mlp.swiglu_limit)
        self.mlp_reduce = PreparedAllReduce(scratch.mlp_partial, scratch.mlp_output, runtime=runtime,
            session=session, coordinator=coordinator, name=f"{name}.mlp_reduce")
        self.post = PreparedMHCPost(scratch.mlp_output, scratch.mlp_hc.residual, scratch.mlp_hc.post,
                                    scratch.mlp_hc.comb, output, session=session, name=f"{name}.post")

    def __call__(self, source):
        s = self.scratch
        self.attention_reduce(self.attention(self.attention_hc(source)))
        self.mlp_hc(s.attention_output, s.attention_hc.residual, s.attention_hc.post, s.attention_hc.comb)
        self.mlp_reduce(self.mlp(s.mlp_hc.normalized))
        return self.post(s.mlp_output, s.mlp_hc.residual, s.mlp_hc.post, s.mlp_hc.comb)
