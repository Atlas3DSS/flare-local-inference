"""Prepared ViT and spatial aligner for DeepSeek V4 Flash Vision."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ignition.components.linear import UnquantizedProjection
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.rms_norm import PreparedRMSNorm
from ignition.components.varlen_attention import PreparedVarlenAttention
from ignition.components.weights import MLPWeights

PATCH_CAPACITIES = (576, 1152, 2304, 3456)
FEATURES = 1536
HIDDEN = 4096


@dataclass(frozen=True)
class VisionInputs:
    vision_pixel_values: torch.Tensor
    vision_position_ids: torch.Tensor
    vision_merge_indices: torch.Tensor
    vision_cu_seqlens: torch.Tensor
    vision_num_patches: torch.Tensor
    vision_num_features: torch.Tensor
    vision_slot: torch.Tensor
    vision_generation: torch.Tensor
    vision_feature_offset: torch.Tensor

    @classmethod
    def allocate(cls, device):
        def zeros(shape, dtype=torch.int32):
            return torch.zeros(shape, dtype=dtype, device=device)

        return cls(
            zeros((3456, 588), torch.float32),
            zeros((3456, 2), torch.int64),
            zeros((384, 9)),
            zeros(2),
            zeros(1),
            zeros(1),
            zeros(1) - 1,
            zeros(1, torch.int64),
            zeros(1),
        )

    def buffers(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class VisionScratch:
    patches: torch.Tensor
    residual: torch.Tensor
    normalized: torch.Tensor
    qkv: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attended: torch.Tensor
    projected: torch.Tensor
    mlp: MLPScratch
    merged: torch.Tensor
    aligned: torch.Tensor
    activated: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, patches, device, *, buffers=None):
        def zeros(*shape):
            return (torch.zeros(shape, dtype=torch.bfloat16, device=device) if buffers is None
                    else buffers.zeros(shape, torch.bfloat16))

        return cls(
            zeros(patches, 588),
            zeros(patches, 1024),
            zeros(patches, 1024),
            zeros(patches, 3072),
            zeros(patches, 16, 64),
            zeros(patches, 16, 64),
            zeros(patches, 16, 64),
            zeros(patches, 1024),
            zeros(patches, 1024),
            MLPScratch.allocate(patches, 1024, 2816, device, buffers=buffers),
            zeros(patches // 9, 9216),
            zeros(patches // 9, 4096),
            zeros(patches // 9, 4096),
            zeros(patches // 9, 4096),
        )

    def rows(self, patches):
        return type(self)(
            *(
                self.mlp.rows(patches)
                if name == "mlp"
                else getattr(self, name)[
                    : patches // 9 if name in ("merged", "aligned", "activated", "output") else patches
                ]
                for name in self.__dataclass_fields__
            )
        )


class PreparedVision:
    def __init__(self, weights, scratch, inputs, *, session, workspace, name, buffers=None):
        self.scratch, self.inputs = scratch, inputs
        tokens, device = scratch.patches.shape[0], scratch.patches.device
        self.frequencies = 10000 ** (-torch.arange(0, 32, 2, dtype=torch.float32, device=device) / 32)
        def zeros(shape, dtype=torch.int32):
            return (torch.zeros(shape, dtype=dtype, device=device) if buffers is None
                    else buffers.zeros(shape, dtype))
        self.positions = zeros((tokens, 2), torch.int64)
        self.cu = zeros((2,))
        self.indices = zeros((tokens // 9, 9)).fill_(-1)
        self.num_patches, self.num_features = zeros((1,)), zeros((1,))

        def project(key, source, out):
            return UnquantizedProjection(
                source,
                weights[key + ".weight"],
                out,
                bias=weights.get(key + ".bias"),
                session=session,
                name=name + "." + key,
            )

        self.patch = project("vision.patch_embed.proj", scratch.patches, scratch.residual)
        self.blocks = []
        for index in range(32):
            prefix = f"vision.blocks.{index}"
            norm1 = PreparedRMSNorm(
                scratch.residual,
                weights[prefix + ".norm1.weight"],
                scratch.normalized,
                eps=1e-6,
                session=session,
                name=f"{name}.{index}.norm1",
            )
            qkv = project(prefix + ".attn.wqkv", scratch.normalized, scratch.qkv)
            output = project(prefix + ".attn.wo", scratch.attended, scratch.projected)
            norm2 = PreparedRMSNorm(
                scratch.residual,
                weights[prefix + ".norm2.weight"],
                scratch.normalized,
                eps=1e-6,
                session=session,
                name=f"{name}.{index}.norm2",
            )
            mlp = PreparedMLP(
                scratch.normalized,
                MLPWeights(weights[prefix + ".mlp.w1.weight"], weights[prefix + ".mlp.w2.weight"]),
                scratch.mlp.gate_up,
                scratch.mlp.intermediate,
                scratch.mlp.output,
                session=session,
                name=f"{name}.{index}.mlp",
                workspace=workspace,
                round_silu=True,
            )
            self.blocks.append((norm1, qkv, output, norm2, mlp))
        self.attention = PreparedVarlenAttention(
            scratch.query,
            scratch.key,
            scratch.value,
            self.cu,
            scratch.attended.view(tokens, 16, 64),
            session=session,
            workspace=workspace,
            name=name + ".attention",
        )
        self.norm = PreparedRMSNorm(
            scratch.residual,
            weights["vision.norm.weight"],
            scratch.normalized,
            eps=1e-6,
            session=session,
            name=name + ".norm",
        )
        self.align1 = project("aligner.w1", scratch.merged, scratch.aligned)
        self.align2 = project("aligner.w2", scratch.activated, scratch.output)

    def __call__(self):
        s = self.scratch
        self.patch(s.patches)
        angles = (self.positions.float()[:, :, None] * self.frequencies).flatten(1)[:, None, :]
        cos, sin = angles.cos(), angles.sin()

        def rotate(source, destination):
            a, b = source.float().chunk(2, dim=-1)
            destination.copy_(torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1))

        for norm1, qkv, output, norm2, mlp in self.blocks:
            q, k, v = qkv(norm1(s.residual)).view(-1, 3, 16, 64).unbind(1)
            rotate(q, s.query)
            rotate(k, s.key)
            s.value.copy_(v)
            self.attention()
            s.residual.add_(output(s.attended))
            s.residual.add_(mlp(norm2(s.residual)))
        self.norm(s.residual)
        indices = self.indices.long()
        gathered = s.normalized[indices.clamp(0, s.normalized.shape[0] - 1)]
        gathered = torch.where(indices[:, :, None] >= 0, gathered, 0)
        s.merged.copy_(gathered.transpose(1, 2).reshape(s.merged.shape))
        s.activated.copy_(F.gelu(self.align1(s.merged)))
        return self.align2(s.activated)
