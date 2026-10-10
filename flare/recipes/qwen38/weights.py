"""Prepare the pinned Qwen checkpoint for b12x without dequantizing weights."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from flare.components.weights import DirectWeightLoader, join_nvfp4_csf
from flare.preparation.checkpoint import CheckpointManifest
from flare.preparation.qwen38 import TEXT_PREFIX, expected_quantization

if TYPE_CHECKING:
    import torch
    from b12x.gemm.blockscaled import Weight
    from b12x.moe.fused_moe import PreparedExperts


def _expert_ids(layer: int, expert_ids: Sequence[int] | None) -> tuple[int, ...]:
    if type(layer) is not int or not 0 <= layer < 48:
        raise ValueError("decoder layer must be an integer in [0, 48)")
    ids = tuple(range(512)) if expert_ids is None else tuple(expert_ids)
    if not ids or any(type(i) is not int or not 0 <= i < 512 for i in ids):
        raise ValueError("expert IDs must be a nonempty sequence of integers in [0, 512)")
    if len(set(ids)) != len(ids):
        raise ValueError("expert IDs must be unique")
    return ids


@dataclass(frozen=True, slots=True)
class ExpertMemoryPlan:
    """Exact source sizes and shared scratch; excludes b12x derived metadata."""

    expert_ids: tuple[int, ...]
    packed_bytes: int
    csf_bytes: int
    scalar_bytes: int
    scale_scratch_bytes: int


def expert_memory_plan(
    manifest: CheckpointManifest, layer: int, expert_ids: Sequence[int] | None = None
) -> ExpertMemoryPlan:
    ids = _expert_ids(layer, expert_ids)
    sizes = {"packed": 0, "csf": 0, "scalar": 0}
    for expert in ids:
        for projection in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"{TEXT_PREFIX}.layers.{layer}.mlp.experts.{expert}.{projection}"
            for suffix, kind in (
                ("weight", "packed"), ("weight_scale_2", "scalar"),
                ("input_scale", "scalar"),
                ("weight_scale.nvfp4_csf_fixed", "csf"),
                ("weight_scale.nvfp4_csf_exceptions", "csf"),
            ):
                sizes[kind] += manifest.tensors[f"{prefix}.{suffix}"].nbytes
    return ExpertMemoryPlan(ids, sizes["packed"], sizes["csf"], sizes["scalar"], len(ids) * 307200)


def allocate_expert_scale_scratch(
    device: torch.device | str, expert_count: int = 512
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate scale expansion storage shared by serial decoder layer calls."""
    import torch

    if type(expert_count) is not int or not 1 <= expert_count <= 512:
        raise ValueError("expert_count must be an integer in [1, 512]")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("expert scale scratch requires a CUDA device")
    return (
        torch.empty((expert_count, 1280, 160), dtype=torch.float8_e4m3fn, device=device),
        torch.empty((expert_count, 2560, 40), dtype=torch.float8_e4m3fn, device=device),
    )


def _expert_scale_vectors(sources: Mapping[str, torch.Tensor], layer: int, ids: tuple[int, ...]):
    import torch

    w13, w2, a1, a2 = [], [], [], []
    for expert in ids:
        prefix = f"{TEXT_PREFIX}.layers.{layer}.mlp.experts.{expert}"
        values = {}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight_scale_2", "input_scale"):
                name = f"{prefix}.{projection}.{suffix}"
                value = sources[name]
                if value.device.type != "cpu" or value.dtype != torch.float32 or value.numel() != 1:
                    raise ValueError(f"{name} requires a CPU float32 scalar")
                if not bool(torch.isfinite(value).all() & (value > 0).all()):
                    raise ValueError(f"{name} must be finite and positive")
                values[projection, suffix] = value.item()
        for suffix in ("weight_scale_2", "input_scale"):
            if values["gate_proj", suffix] != values["up_proj", suffix]:
                raise ValueError(f"{prefix}: gate/up {suffix} differ; fused NVFP4 requires equal scales")
        w13.append(values["up_proj", "weight_scale_2"])
        w2.append(values["down_proj", "weight_scale_2"])
        a1.append(values["up_proj", "input_scale"])
        a2.append(values["down_proj", "input_scale"])
    vectors = (
        torch.tensor(w13, dtype=torch.float32), torch.tensor(w2, dtype=torch.float32),
        torch.tensor(a1, dtype=torch.float32).reciprocal(),
        torch.tensor(a2, dtype=torch.float32).reciprocal(),
    )
    if any(not bool(torch.isfinite(vector).all()) for vector in vectors):
        raise ValueError("expert activation reciprocal scales must remain finite in float32")
    return vectors


class QwenWeightLoader(DirectWeightLoader):
    """Direct checkpoint transfers and explicit, bounded b12x weight preparation.

    ``prefixes`` bounds component qualification; omit it for a complete recipe.
    MTP and PLE table payloads are excluded. Returned tensors own GPU storage
    beyond this context's lifetime. Compressed expert scales use at most one
    layer of CPU staging; no full model or expert weight is decoded on the CPU.
    """

    def _selected_names(self):
        return {
            name for name, tensor in self.manifest.tensors.items()
            if tensor.component not in ("mtp", "ple_tables")
            and "nvfp4_csf_" not in name
            and (self.prefixes is None or name.startswith(self.prefixes))
        }

    def load_linear(self, prefix: str) -> Weight:
        from b12x.gemm.blockscaled import pack_weight

        self._require_open()
        recipe = expected_quantization().get(prefix, {}).get("quant_algo")
        if recipe not in ("MXFP8", "W4A16_NVFP4") or prefix.startswith("mtp."):
            raise ValueError(f"{prefix} is not a supported dense quantized projection")
        if recipe == "MXFP8":
            return pack_weight(
                self.load_tensor(f"{prefix}.weight"),
                self.load_tensor(f"{prefix}.weight_scale"), recipe="mxfp8",
            )
        import torch
        from b12x._lib.intrinsics import swizzle_block_scale

        scalar = self._sources[f"{prefix}.weight_scale_2"]
        if not bool(torch.isfinite(scalar).all() & (scalar > 0).all()):
            raise ValueError(f"{prefix}.weight_scale_2 must be finite and positive")
        source = self._sources[f"{prefix}.weight"]
        rows, packed_columns = source.shape
        logical_columns = packed_columns * 2
        stored_columns = ((logical_columns + 127) // 128) * 128
        values = torch.zeros((rows, stored_columns // 2), dtype=source.dtype, device=self.device)
        self._copy(values[:, :packed_columns], source)
        scales = self.load_tensor(f"{prefix}.weight_scale")
        global_scale = self.load_tensor(f"{prefix}.weight_scale_2")
        packed = pack_weight(
            values, swizzle_block_scale(scales), recipe="nvfp4",
            global_scale=global_scale, global_scale_kind="multiplier",
        )
        return replace(packed, in_features=logical_columns)

    def load_mxfp8_group(self, prefixes: Sequence[str]) -> Weight:
        """Pack serialized projections along output rows before scale swizzling."""
        import torch
        from b12x.gemm.blockscaled import pack_weight

        self._require_open()
        recipes = expected_quantization()
        if not prefixes or any(
            recipes.get(prefix, {}).get("quant_algo") != "MXFP8" for prefix in prefixes
        ):
            raise ValueError("projection packing requires declared MXFP8 weights")
        sources = [self._sources[f"{prefix}.weight"] for prefix in prefixes]
        if any(source.ndim != 2 or source.dtype != torch.float8_e4m3fn for source in sources):
            raise ValueError("packed MXFP8 projections must be FP8 matrices")
        columns = sources[0].shape[1]
        if any(source.shape[1] != columns for source in sources):
            raise ValueError("packed MXFP8 projections must share input width")
        rows = sum(source.shape[0] for source in sources)
        values = torch.empty((rows, columns), dtype=torch.float8_e4m3fn, device=self.device)
        scales = torch.empty((rows, columns // 32), dtype=torch.uint8, device=self.device)
        offset = 0
        for prefix, source in zip(prefixes, sources, strict=True):
            end = offset + source.shape[0]
            self._copy(values[offset:end], source)
            scale_rows = self._sources[f"{prefix}.weight_scale"][:source.shape[0], :columns // 32]
            self._copy(scales[offset:end], scale_rows.view(torch.uint8))
            offset = end
        self._session.flush()
        return pack_weight(values, scales, recipe="mxfp8")

    def load_gdn_projections(self, prefix: str) -> dict[str, Weight]:
        return {
            "in_proj_qkvz": self.load_mxfp8_group(tuple(f"{prefix}.{name}" for name in ("in_proj_qkv", "in_proj_z"))),
            "in_proj_ba": self.load_mxfp8_group(tuple(f"{prefix}.{name}" for name in ("in_proj_b", "in_proj_a"))),
            "out_proj": self.load_linear(f"{prefix}.out_proj"),
        }

    def load_qsa_projections(self, prefix: str) -> dict[str, Weight]:
        return {
            "qkv_proj": self.load_mxfp8_group(tuple(f"{prefix}.{name}" for name in ("q_proj", "k_proj", "v_proj"))),
            "o_proj": self.load_linear(f"{prefix}.o_proj"),
            "indexer.index_qk_proj": self.load_linear(f"{prefix}.indexer.index_qk_proj"),
        }

    def load_experts(
        self, layer: int, *, expert_ids: Sequence[int] | None = None,
        scale_scratch: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> PreparedExperts:
        """Prepare A4 routed experts; subset IDs define local ordering in tests.

        Complete model layers require all 512 experts. Reusing ``scale_scratch``
        across layers requires serialized execution on one CUDA stream.
        """
        import torch
        from b12x.moe import fused_moe as moe

        self._require_open()
        memory = expert_memory_plan(self.manifest, layer, expert_ids)
        if memory.csf_bytes > self.csf_byte_budget:
            raise ValueError(f"layer CSF planes require {memory.csf_bytes} bytes, exceeding the CPU staging budget")
        ids = memory.expert_ids
        scales = _expert_scale_vectors(self._sources, layer, ids)
        count = len(ids)
        if scale_scratch is None:
            scale_scratch = allocate_expert_scale_scratch(self.device, count)
        for scratch, shape in zip(scale_scratch, ((count, 1280, 160), (count, 2560, 40)), strict=True):
            if scratch.device != self.device or scratch.dtype != torch.float8_e4m3fn or tuple(scratch.shape) != shape or not scratch.is_contiguous():
                raise ValueError("expert scale scratch shape, dtype, device, or layout mismatch")
        w13 = torch.empty((count, 1280, 1280), dtype=torch.uint8, device=self.device)
        w2 = torch.empty((count, 2560, 320), dtype=torch.uint8, device=self.device)
        fixed13, exceptions13, fixed2, exceptions2 = [], [], [], []
        for local, expert in enumerate(ids):
            prefix = f"{TEXT_PREFIX}.layers.{layer}.mlp.experts.{expert}"
            planes = {}
            for projection, destination in (
                ("up_proj", w13[local, :640]),
                ("gate_proj", w13[local, 640:]),
                ("down_proj", w2[local]),
            ):
                self._copy(destination, self._sources[f"{prefix}.{projection}.weight"])
                planes[projection] = tuple(self._cpu_csf(
                    f"{prefix}.{projection}.weight_scale.nvfp4_csf_{suffix}"
                ) for suffix in ("fixed", "exceptions"))
            fixed, exceptions = join_nvfp4_csf(*planes["up_proj"], *planes["gate_proj"], rows=640, columns=160)
            fixed13.append(fixed)
            exceptions13.append(exceptions)
            fixed2.append(planes["down_proj"][0])
            exceptions2.append(planes["down_proj"][1])
        device_scales = []
        for scale in scales:
            target = torch.empty_like(scale, device=self.device)
            self._copy(target, scale)
            device_scales.append(target)
        self._session.flush()
        plan = moe.plan_weights(
            source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
            activation=moe.ActivationSpec(mode="a4", nonlinearity="silu", io_dtype=torch.bfloat16),
            geometry=moe.MoEGeometry(num_experts=count, hidden_size=2560, intermediate_size=640),
        )
        packed = moe.PackedWeights(
            w13=w13, w2=w2,
            w13_block_scales=scale_scratch[0], w2_block_scales=scale_scratch[1],
            w13_global_scales=device_scales[0], w2_global_scales=device_scales[1],
            input_scale=device_scales[2], intermediate_scale=device_scales[3],
            immutable_input_scales=True,
        )
        return moe.prepare_weights(plan=plan, weights=moe.Nvfp4CsfWeights(
            packed=packed,
            w13_scales=moe.CsfScalePlanes(tuple(fixed13), tuple(exceptions13)),
            w2_scales=moe.CsfScalePlanes(tuple(fixed2), tuple(exceptions2)),
        ))
