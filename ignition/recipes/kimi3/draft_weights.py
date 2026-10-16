"""BF16 DFlash2 projections with sharded MLA heads and SwiGLU intermediates."""

from dataclasses import dataclass

import torch

from ignition.components.weights import DirectWeightLoader, MLPWeights
from ignition.preparation.checkpoint import inspect_indexed_checkpoint


@dataclass(frozen=True)
class DraftWeights:
    projection: torch.Tensor
    hidden_norm: torch.Tensor
    norm: torch.Tensor
    layers: tuple[dict, ...]
    selector: dict

    @classmethod
    def load(cls, snapshot, config, device, shard, *, progress=None):
        c = config.text
        heads = shard.partition(c["num_attention_heads"])
        intermediate = shard.partition(c["intermediate_size"], alignment=128)
        manifest = inspect_indexed_checkpoint(snapshot)
        with DirectWeightLoader(manifest, device, prefixes=("fc.", "hidden_norm.", "norm.", "layers.", "candidate_selector."),
                progress=progress, progress_label="Load draft weights") as loader:
            def tensor(name, extent=None):
                value = loader.load_tensor(name, extent=extent)
                if value.dtype != torch.bfloat16:
                    raise ValueError(f"DFlash2 requires BF16 weights: {name}")
                return value

            def matrix(name, rows=None, columns=None):
                return tensor(name + ".weight", (slice(*(rows or (None, None))), slice(*(columns or (None, None)))))

            projection = matrix("fc", rows=shard.partition(c["hidden_size"], alignment=64))
            hidden_norm, norm = tensor("hidden_norm.weight"), tensor("norm.weight")
            layers = []
            for i in range(config.layers):
                p = f"layers.{i}"
                a = p + ".self_attn"
                kv_a = matrix(a + ".kv_a_proj_with_mqa")
                kv_b = matrix(a + ".kv_b_proj", rows=(heads[0] * 256, heads[1] * 256)).view(-1, 256, 512)
                attention = dict(input=torch.cat((matrix(a + ".q_a_proj"), kv_a)), context=kv_a,
                    q_b=matrix(a + ".q_b_proj", rows=(heads[0] * 192, heads[1] * 192)),
                    key=kv_b[:, :128].contiguous(), value=kv_b[:, 128:].transpose(1, 2).contiguous(),
                    query_norm=tensor(a + ".q_a_layernorm.weight"), kv_norm=tensor(a + ".kv_a_layernorm.weight"),
                    output=matrix(a + ".o_proj", columns=(heads[0] * 128, heads[1] * 128)))
                convolutions = {kind: dict(base=tensor(p + f".{kind}.base_kernel"),
                    projection=matrix(p + f".{kind}.kernel_projection")) for kind in ("attention_conv", "mlp_conv")}
                mlp = MLPWeights(torch.cat([matrix(p + ".mlp." + k, rows=intermediate) for k in ("gate_proj", "up_proj")]),
                    matrix(p + ".mlp.down_proj", columns=intermediate))
                layers.append(dict(attention=attention, mlp=mlp, input_norm=tensor(p + ".input_layernorm.weight"),
                    post_norm=tensor(p + ".post_attention_layernorm.weight"), **convolutions))
            selector = dict(predecessor=tensor("candidate_selector.predecessor_codebook"),
                successor=tensor("candidate_selector.successor_codebook"),
                projection=matrix("candidate_selector.hidden_projection"))
        return cls(projection, hidden_norm, norm, tuple(layers), selector)
