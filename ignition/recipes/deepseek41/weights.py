"""Native FP8/MXFP4 checkpoint transfers and tensor-parallel partitions."""

from dataclasses import dataclass

import torch

from ignition.components.weights import DirectWeightLoader, MLPWeights, _aligned_range

from .config import ModelConfig


@dataclass(frozen=True)
class ExpertWeights:
    router: torch.Tensor
    correction: torch.Tensor
    experts: object
    shared: MLPWeights


@dataclass(frozen=True)
class LayerWeights:
    attention: dict
    experts: ExpertWeights
    residual: dict
    engram: dict | None


@dataclass(frozen=True)
class ModelWeights:
    config: ModelConfig
    shard: object
    embedding: torch.Tensor
    norm: torch.Tensor
    head: torch.Tensor
    layers: tuple[LayerWeights, ...]
    allocated_bytes: int

    @classmethod
    def load(cls, manifest, device, *, shard, progress=None):
        config = ModelConfig.load(manifest.snapshot)
        before = torch.cuda.memory_allocated(device)
        with DeepSeekWeightLoader(manifest, device, shard=shard, progress=progress) as loader:
            embedding, norm = loader.load_tensors(("embed.weight", "norm.weight"))
            head = loader.load_tensor("head.weight", extent=slice(*shard.partition(129280, alignment=64)))
            layers = tuple(loader.load_layer(i) for i in range(config.layers))
        return cls(config, shard, embedding, norm, head, layers,
                   torch.cuda.memory_allocated(device) - before)


class DeepSeekWeightLoader(DirectWeightLoader):
    def __init__(self, manifest, device, *, shard, progress=None, prefixes=None):
        super().__init__(manifest, device, progress=progress, prefixes=prefixes)
        self.config = ModelConfig.load(manifest.snapshot)
        self.shard = shard

    def _selected_names(self):
        return {name for name in super()._selected_names()
                if (name.startswith("layers.") or name in ("embed.weight", "norm.weight", "head.weight"))
                and ".engram.embed." not in name}

    def load_linear_group(self, prefixes, *, row_ranges=None, column_range=None):
        """Expand block-32 row scales without changing FP8 values or their scale bytes."""
        from b12x.gemm.blockscaled import pack_weight

        self._require_open()
        prefixes = tuple(prefixes)
        ranges = (None,) * len(prefixes) if row_ranges is None else row_ranges
        parts = [self._dense_blocks(prefix, rows=selected, columns=column_range)
                 for prefix, selected in zip(prefixes, ranges, strict=True)]
        self._session.flush()
        values = torch.cat([p[0] for p in parts]) if len(parts) > 1 else parts[0][0]
        scales = torch.cat([p[1].repeat_interleave(32, dim=0) for p in parts])
        return pack_weight(values, scales, recipe="mxfp8")

    def _dense_blocks(self, prefix, *, rows=None, columns=None):
        self._require_open()
        weight, scale = self._sources[f"{prefix}.weight"], self._sources[f"{prefix}.scale"]
        if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
            raise ValueError(f"{prefix}: expected native FP8 dense weights")
        n, k = weight.shape
        if (scale.dtype != torch.float8_e8m0fnu or scale.shape != ((n + 31) // 32, (k + 31) // 32)):
            raise ValueError(f"{prefix}: expected UE8M0 scales for 32-by-32 blocks")
        r0, r1 = _aligned_range(rows, n, 32)
        c0, c1 = _aligned_range(columns, k, 32)
        values = torch.empty((r1 - r0, c1 - c0), dtype=weight.dtype, device=self.device)
        scales = torch.empty(((r1 - r0) // 32, (c1 - c0) // 32), dtype=torch.uint8, device=self.device)
        self._copy(values, weight[r0:r1, c0:c1])
        self._copy(scales, scale[r0 // 32:r1 // 32, c0 // 32:c1 // 32].view(torch.uint8))
        return values, scales

    def load_attention_output(self, prefix, groups):
        from b12x.gemm import wo_projection

        a, sa = self._dense_blocks(prefix + ".wo_a", rows=groups)
        b, sb = self._dense_blocks(prefix + ".wo_b", columns=groups)
        self._session.flush()
        return wo_projection.pack_weights(a, sa.view(torch.float8_e8m0fnu), b, sb.view(torch.float8_e8m0fnu),
            groups=8 // self.shard.world_size, group_width=4096, rank=1024, hidden=5120, block_size=(32, 32))

    def load_layer(self, layer):
        prefix = f"layers.{layer}"
        text, shard = self.config.text, self.shard
        attn = prefix + ".attn"
        heads = shard.partition(64 * 512, alignment=512)
        groups = shard.partition(8 * 1024, alignment=1024)
        result = {
            "qkv_a": self.load_linear_group((attn + ".wq_a", attn + ".wkv")),
            "q_b": self.load_linear(attn + ".wq_b", rows=heads),
            "output": self.load_attention_output(attn, groups),
            "q_norm": self.load_tensor(attn + ".q_norm.weight"),
            "kv_norm": self.load_tensor(attn + ".kv_norm.weight"),
            "sink": self.load_tensor(attn + ".attn_sink", extent=slice(*shard.partition(64))),
        }
        if layer in text["kv_source_layer_ids"]:
            ratio = text["compress_ratios"][layer]
            result["compressor"] = {
                "values": self.load_tensor(attn + ".compressor.wkv.weight"),
                "norm": self.load_tensor(attn + ".compressor.norm.weight", dtype=torch.float32),
            }
            if ratio == 2:
                result["compressor"]["gates"] = self.load_tensor(attn + ".compressor.wgate.weight")
            result["index_k"] = self.load_tensor(attn + ".indexer.wk.weight")
            result["index_norm"] = self.load_tensor(attn + ".indexer.k_norm.weight")
        if layer in text["index_source_layer_ids"]:
            # Replicated index heads avoid a context-sized TP score reduction.
            result["index_q"] = self.load_linear(attn + ".indexer.wq_b")
            result["index_weights"] = self.load_tensor(attn + ".indexer.weights_proj.weight")
        ffn = prefix + ".ffn"
        interval = shard.partition(text["moe_intermediate_size"], alignment=32)
        experts = self.load_fp4_experts(
            [f"{ffn}.experts.{i}" for i in range(text["n_routed_experts"])],
            hidden_size=5120, intermediate_size=2304, format="mxfp4", activation_mode="a8",
            intermediate_range=interval, swiglu_limit=text["swiglu_limit"],
            projection_names=("w3", "w1", "w2"), scale_name="scale")
        shared = MLPWeights(self.load_linear_group((ffn + ".shared_experts.w1", ffn + ".shared_experts.w3"),
            row_ranges=(interval, interval)), self.load_linear(ffn + ".shared_experts.w2", columns=interval))
        routed = ExpertWeights(self.load_tensor(ffn + ".gate.weight"),
            self.load_tensor(ffn + ".gate.bias"), experts, shared)
        residual = {name: self.load_tensor(prefix + "." + name) for name in (
            "hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale",
            "attn_norm.weight", "ffn_norm.weight")}
        engram = None
        if layer in text["engram_layer_ids"]:
            engram = {"projection": self.load_linear(prefix + ".engram.wkv"),
                "weight": self.load_tensor(prefix + ".engram.q_weight").float()
                          * self.load_tensor(prefix + ".engram.k_weight").float()}
        return LayerWeights(result, routed, residual, engram)
