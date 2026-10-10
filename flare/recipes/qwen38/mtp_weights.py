"""Stored BF16 and NVFP4 W4A16 weights for the checkpoint's single MTP layer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json

import torch

from flare.preparation.checkpoint import CheckpointError, CheckpointManifest
from flare.preparation.qwen38 import CONFIG_SHA256, TRANSFORMERS_REVISION

from .moe import QwenMoeWeights, SharedExpertWeights
from .hyperconnection import packed_down_projection
from .weights import QwenWeightLoader, _expert_ids


def _global_scales(sources, expert_ids):
    w13, w2 = [], []
    for expert in expert_ids:
        prefix = f"mtp.layers.0.mlp.experts.{expert}"
        values = {}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = f"{prefix}.{projection}.weight_scale_2"
            value = sources[name]
            if value.device.type != "cpu" or value.dtype != torch.float32 or value.shape != ():
                raise ValueError(f"{name} requires a CPU float32 scalar")
            if not bool(torch.isfinite(value) & (value > 0)):
                raise ValueError(f"{name} must be finite and positive")
            values[projection] = value.item()
        if values["gate_proj"] != values["up_proj"]:
            raise ValueError(f"{prefix}: gate/up global scales differ")
        w13.append(values["up_proj"])
        w2.append(values["down_proj"])
    return torch.tensor(w13, dtype=torch.float32), torch.tensor(w2, dtype=torch.float32)


class MTPWeightLoader(QwenWeightLoader):
    """Reuse b12x direct transfers while selecting only stored ``mtp.*`` tensors."""

    def _selected_names(self):
        return {name for name, tensor in self.manifest.tensors.items()
                if tensor.component == "mtp"
                and (self.prefixes is None or name.startswith(self.prefixes))}

    def load_experts(self, *, expert_ids=None):
        """Prepare plain E4M3 scale planes and A16 activations, without CSF or QDQ.

        Explicit expert subsets are component-test inputs. The executable MTP
        layer requires all 512 experts in checkpoint order.
        """
        from b12x.moe import fused_moe

        self._require_open()
        ids = _expert_ids(0, expert_ids)
        scales = _global_scales(self._sources, ids)
        count = len(ids)
        w13 = torch.empty((count, 1280, 1280), dtype=torch.uint8, device=self.device)
        w2 = torch.empty((count, 2560, 320), dtype=torch.uint8, device=self.device)
        s13 = torch.empty((count, 1280, 160), dtype=torch.float8_e4m3fn, device=self.device)
        s2 = torch.empty((count, 2560, 40), dtype=torch.float8_e4m3fn, device=self.device)
        for local, expert in enumerate(ids):
            prefix = f"mtp.layers.0.mlp.experts.{expert}"
            for projection, values, blocks in (
                ("up_proj", w13[local, :640], s13[local, :640]),
                ("gate_proj", w13[local, 640:], s13[local, 640:]),
                ("down_proj", w2[local], s2[local]),
            ):
                self._copy(values, self._sources[f"{prefix}.{projection}.weight"])
                self._copy(blocks, self._sources[f"{prefix}.{projection}.weight_scale"])
        device_scales = []
        for value in scales:
            destination = torch.empty_like(value, device=self.device)
            self._copy(destination, value)
            device_scales.append(destination)
        self._session.flush()
        plan = fused_moe.plan_weights(
            source=fused_moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
            activation=fused_moe.ActivationSpec(mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16),
            geometry=fused_moe.MoEGeometry(num_experts=count, hidden_size=2560, intermediate_size=640),
        )
        return fused_moe.prepare_weights(plan=plan, weights=fused_moe.PackedWeights(
            w13=w13, w2=w2, w13_block_scales=s13, w2_block_scales=s2,
            w13_global_scales=device_scales[0], w2_global_scales=device_scales[1],
        ))

    def load_shared(self):
        prefix = "mtp.layers.0.mlp"
        gate_up = torch.empty((1280, 2560), dtype=torch.bfloat16, device=self.device)
        for projection, output in (("gate_proj", gate_up[:640]), ("up_proj", gate_up[640:])):
            self._copy(output, self._sources[f"{prefix}.shared_expert.{projection}.weight"])
        self._session.flush()
        return SharedExpertWeights(gate_up,
            self.load_tensor(f"{prefix}.shared_expert.down_proj.weight"),
            self.load_tensor(f"{prefix}.shared_expert_gate.weight"))


def create_mtp_modules(manifest: CheckpointManifest):
    """Create canonical HC/QSA parameter owners without a duplicate target model."""
    distribution = importlib.metadata.distribution("transformers")
    provenance = json.loads(distribution.read_text("direct_url.json") or "{}")
    if provenance.get("vcs_info", {}).get("commit_id") != TRANSFORMERS_REVISION:
        raise RuntimeError(f"MTP preparation requires Transformers {TRANSFORMERS_REVISION}")
    raw = (manifest.snapshot / "config.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA256:
        raise CheckpointError("config.json changed after checkpoint inspection")
    from transformers import Qwen4ExpConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextAttention, Qwen4ExpTextGatedResidual, Qwen4ExpTextRMSNorm,
    )

    config = Qwen4ExpConfig.from_dict(json.loads(raw)).text_config
    if config.mtp_num_hidden_layers != 1 or config.hc_count != 4 or config.hidden_size != 2560:
        raise ValueError("MTP requires one layer and four 2560-wide streams")
    with torch.device("meta"):
        module = torch.nn.Module()
        module.config = config
        module.fc_embedding = torch.nn.Linear(2560, 2560, bias=False)
        module.fc_hidden = torch.nn.Linear(2560, 2560, bias=False)
        module.pre_fc_norm_embedding = Qwen4ExpTextRMSNorm(2560, eps=config.rms_norm_eps)
        module.pre_fc_norm_hidden = Qwen4ExpTextRMSNorm(10240, eps=config.rms_norm_eps)
        module.hyper_connection_mixer = Qwen4ExpTextGatedResidual(config, use_combine=False)
        layer = torch.nn.Module()
        layer.attn_hyper_connection = Qwen4ExpTextGatedResidual(config)
        layer.mlp_hyper_connection = Qwen4ExpTextGatedResidual(config)
        layer.self_attn = Qwen4ExpTextAttention(config, layer_idx=48)
        module.layers = torch.nn.ModuleList([layer])
    module.eval().requires_grad_(False)
    return module


@dataclass(frozen=True)
class MTPWeights:
    module: torch.nn.Module
    attention: dict[str, torch.Tensor]
    moe: QwenMoeWeights
    embedding: torch.Tensor
    lm_head: torch.Tensor
    stored_bytes: int
    allocated_bytes: int

    @classmethod
    def load(cls, manifest: CheckpointManifest, device, *, embedding_weight, lm_head_weight):
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("MTP weights require a CUDA device")
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        for label, tensor in (("embedding", embedding_weight), ("lm_head", lm_head_weight)):
            if (tensor.shape != (248320, 2560) or tensor.dtype != torch.bfloat16
                    or tensor.device != device or not tensor.is_contiguous()):
                raise ValueError(f"shared MTP {label} must be contiguous BF16 [248320,2560] on {device}")
        stored = sum(t.nbytes for t in manifest.tensors.values() if t.component == "mtp")
        module = create_mtp_modules(manifest)
        before = torch.cuda.memory_allocated(device)
        with MTPWeightLoader(manifest, device) as loader:
            for name, _ in tuple(module.named_parameters()):
                value = loader.load_tensor(f"mtp.{name}")
                parent, _, attribute = name.rpartition(".")
                setattr(module.get_submodule(parent), attribute, torch.nn.Parameter(value, requires_grad=False))
            for mixer in (module.layers[0].attn_hyper_connection, module.layers[0].mlp_hyper_connection):
                packed_down_projection(mixer)
            shared = loader.load_shared()
            moe = QwenMoeWeights(loader.load_tensor("mtp.layers.0.mlp.gate.weight"),
                                loader.load_experts(), shared)
        attention = {name: module.layers[0].self_attn.get_submodule(name).weight
                     for name in ("q_proj", "k_proj", "v_proj", "o_proj", "indexer.index_qk_proj")}
        torch.cuda.synchronize(device)
        return cls(module, attention, moe, embedding_weight, lm_head_weight,
                   stored, torch.cuda.memory_allocated(device) - before)
