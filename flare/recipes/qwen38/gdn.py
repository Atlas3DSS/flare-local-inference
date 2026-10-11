"""Prepared Qwen GDN projections, convolution, recurrence, and output gating."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

import torch
from b12x.gemm.blockscaled import Weight
from b12x.preparation import PreparationSession
from b12x.sequence import gdn_decode, gdn_prefill, short_conv

from flare.components.linear import QuantizedProjection
from flare.components.recurrent import (
    DeltaMetadata,
    DeltaState,
    prepare_sequence_binding,
)
from flare.preparation.scratch import ScratchArena


@dataclass(frozen=True)
class GDNState(DeltaState):
    """Persistent chronological convolution history and transposed FP32 recurrence.

    Recurrence slots have shape ``[slot, local_value_heads, 128, 128]``.
    The canonical Transformers cache transposes the final two dimensions.
    Slot zero is reserved as the recurrence null index. Fresh decode slots must
    be zeroed by cache maintenance before replay; prefill selects null initial
    state through metadata. Convolution fresh flags discard prior history.
    """

    @classmethod
    def allocate(
        cls, slots: int, device: torch.device, *, speculative_tokens: int = 0,
        world_size: int = 1,
    ) -> GDNState:
        return super().allocate(slots, device, heads=48 // world_size, channels=10240 // world_size,
                                speculative_tokens=speculative_tokens)


@dataclass(frozen=True)
class GDNScratch:
    """Caller-owned graph operands reusable across serialized decoder layers."""

    projected_qkv: torch.Tensor
    convolved_qkv: torch.Tensor
    z: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    recurrence: torch.Tensor
    gated: torch.Tensor
    output: torch.Tensor
    projected_qkvz: torch.Tensor
    projected_ba: torch.Tensor

    @classmethod
    def allocate(cls, tokens: int, device: torch.device, *, world_size=1) -> GDNScratch:
        if type(tokens) is not int or tokens <= 0:
            raise ValueError("GDN token capacity must be positive")
        qkvz = torch.zeros((tokens, 16384 // world_size), device=device, dtype=torch.bfloat16)
        ba = torch.zeros((tokens, 96 // world_size), device=device, dtype=torch.bfloat16)
        convolved, recurrence, gated, output = (
            torch.zeros((tokens, width), device=device, dtype=torch.bfloat16)
            for width in (10240 // world_size, 6144 // world_size, 6144 // world_size, 2560))
        return cls(qkvz[:, :10240 // world_size], convolved, qkvz[:, 10240 // world_size:], ba[:, 48 // world_size:], ba[:, :48 // world_size],
                   recurrence, gated, output, qkvz, ba)

    def rows(self, tokens: int) -> GDNScratch:
        if type(tokens) is not int or not 0 < tokens <= self.projected_qkv.shape[0]:
            raise ValueError("GDN rows exceed scratch capacity")
        return GDNScratch(
            *(getattr(self, field.name)[:tokens] for field in fields(self))
        )


def _validate_geometry(module: torch.nn.Module, world_size=1) -> None:
    required = {
        "hidden_size": 2560,
        "num_k_heads": 16 // world_size,
        "num_v_heads": 48 // world_size,
        "head_k_dim": 128,
        "head_v_dim": 128,
        "conv_dim": 10240 // world_size,
        "conv_kernel_size": 4,
    }
    if any(getattr(module, name, None) != value for name, value in required.items()):
        raise ValueError("GDN module geometry differs from Qwen3.8 Flash Next")
    if module.activation != "silu" or module.norm.activation not in ("silu", "sigmoid"):
        raise ValueError(
            "Qwen GDN requires SiLU convolution and SiLU/sigmoid output gating"
        )
    if module.conv1d.bias is not None:
        raise ValueError("Qwen GDN convolution must not have bias")


class PreparedGDN:
    """One fixed-capacity decode or prefill pipeline over a canonical GDN module.

    The module supplies geometry and exact unquantized parameters. The projection
    mapping supplies packed MXFP8 QKV/Z, B/A, and output weights. Preparation
    requires zero live counts and must finish before graph capture. All plans,
    buffers, weights, and canonical norm parameters remain strongly owned.

    ``workspace`` and ``projection_workspace`` may be shared by serialized
    layers and graph buckets. Their storage must cover every selected plan and
    remain live until all referencing graphs drain. Concurrent graph execution
    over either shared workspace is invalid.

    Prefill uses canonical PyTorch gated RMSNorm inside the captured graph.
    Its temporary tensors remain private to that graph; gated and final outputs
    are copied/written into caller-owned scratch. No Python callback is needed
    after the graph is handed to the native replay worker.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        projections: Mapping[str, Weight],
        source: torch.Tensor,
        scratch: GDNScratch,
        state: GDNState,
        metadata: DeltaMetadata,
        *,
        mode: str,
        session: PreparationSession,
        name: str,
        checkpoint_export: bool = False,
        workspace: torch.Tensor | None = None,
        projection_workspace: torch.Tensor | None = None,
    ) -> None:
        self.world_size = 48 // module.num_v_heads
        self.key_heads, self.value_heads = module.num_k_heads, module.num_v_heads
        world_size = self.world_size
        _validate_geometry(module, world_size)
        if mode not in ("decode", "prefill"):
            raise ValueError("GDN graph mode must be decode or prefill")
        self.module, self.scratch, self.state, self.metadata = (
            module,
            scratch,
            state,
            metadata,
        )
        self.mode = mode
        self.tokens, self.device = source.shape[0], source.device
        if (
            source.ndim != 2
            or source.shape[1] != 2560
            or source.dtype != torch.bfloat16
        ):
            raise ValueError("GDN source must be BF16 [token_capacity, 2560]")
        if self.device.type != "cuda":
            raise ValueError("prepared GDN requires CUDA")
        widths = tuple(width if index == 7 else width // world_size for index, width in
                       enumerate((10240, 10240, 6144, 48, 48, 6144, 6144, 2560, 16384, 96)))
        strides = tuple(stride if index == 7 else stride // world_size for index, stride in
                        enumerate((16384, 10240, 16384, 96, 96, 6144, 6144, 2560, 16384, 96)))
        for field, width, stride in zip(fields(scratch), widths, strides, strict=True):
            tensor = getattr(scratch, field.name)
            if (
                tensor.device != self.device
                or tensor.dtype != torch.bfloat16
                or tuple(tensor.shape) != (self.tokens, width)
                or tensor.stride() != (stride, 1)
            ):
                raise ValueError(
                    f"GDN scratch {field.name} has incompatible shape, dtype, or layout"
                )
        for tensor in (
            module.conv1d.weight,
            module.A_log,
            module.dt_bias,
            module.norm.weight,
        ):
            if tensor.device != self.device:
                raise ValueError(
                    "canonical GDN parameters and graph buffers must share a device"
                )
        if state.convolution.ndim != 3 or state.convolution.shape[1] != module.conv_dim:
            raise ValueError(
                f"GDN convolution history must have shape [slots, {module.conv_dim}, 3 + drafts]"
            )
        self.speculative_tokens = state.convolution.shape[2] - 3
        if not 0 <= self.speculative_tokens <= 7:
            raise ValueError(
                "GDN history supports zero through seven speculative tokens"
            )
        if state.recurrent.ndim != 4 or tuple(state.recurrent.shape[1:]) != (
            self.value_heads,
            128,
            128,
        ):
            raise ValueError(f"GDN recurrence must have shape [slots, {self.value_heads}, 128, 128]")
        self.sequences = metadata.query_start_loc.numel() - 1
        if (
            self.sequences <= 0
            or metadata.state_indices.ndim != 2
            or metadata.state_indices.shape[0] != self.sequences
        ):
            raise ValueError("GDN metadata capacities disagree")
        columns = metadata.state_indices.shape[1]
        if mode == "decode" and (
            self.tokens > self.sequences * columns
            or columns > self.speculative_tokens + 1
        ):
            raise ValueError(
                "decode token/verification capacity exceeds its convolution history"
            )
        if metadata.num_tokens.item() or metadata.num_seqs.item():
            raise ValueError(
                "GDN preparation requires zero live token and sequence counts"
            )
        expected = {"in_proj_qkvz", "in_proj_ba", "out_proj"}
        if set(projections) != expected:
            raise ValueError("GDN requires packed QKV/Z, B/A, and output projection weights")
        self.projections = {
            projection: QuantizedProjection(
                source,
                projections[projection],
                output,
                session=session,
                name=f"{name}.{projection}",
                activation_mode="auto",
                workspace=projection_workspace,
            )
            for projection, output in (
                ("in_proj_qkvz", scratch.projected_qkvz),
                ("in_proj_ba", scratch.projected_ba),
            )
        }
        self.projections["out_proj"] = QuantizedProjection(
            scratch.gated,
            projections["out_proj"],
            scratch.output,
            session=session,
            name=f"{name}.out_proj",
            activation_mode="auto",
            workspace=projection_workspace,
        )
        self.parallel_projections = None
        if mode == "decode" and self.tokens <= 16 and isinstance(projection_workspace, ScratchArena):
            from flare.components.linear import ParallelProjections

            self.parallel_projections = ParallelProjections(
                self.projections["in_proj_qkvz"], self.projections["in_proj_ba"],
                session=session, workspace=projection_workspace)
        self.conv_weight = module.conv1d.weight.squeeze(1)
        conv_args = {
            "x": scratch.projected_qkv,
            "conv_weight": self.conv_weight,
            "query_start_loc": metadata.query_start_loc,
            "state_slot_ids": metadata.convolution_slots,
            "state_is_fresh": metadata.fresh,
            "num_accepted_tokens": metadata.accepted_tokens,
            "num_seqs": metadata.num_seqs,
            "num_tokens": metadata.num_tokens,
            "conv_state": state.convolution,
            "out": scratch.convolved_qkv,
        }
        conv_caps = short_conv.Caps(
            device=self.device,
            mode=mode,
            channels=module.conv_dim,
            kernel_size=4,
            max_tokens=self.tokens,
            max_seqs=self.sequences,
            max_state_slots=state.convolution.shape[0],
            max_speculative_tokens=self.speculative_tokens,
        )
        self.conv_plan = short_conv.plan(
            conv_caps,
            invocation=short_conv.invocation_from_tensors(**conv_args),
        )
        self.conv_binding = prepare_sequence_binding(
            self.conv_plan,
            conv_args,
            session=session,
            name=f"{name}.short_conv",
            public_bind=short_conv.bind,
            bind_plan=True,
            workspace=workspace,
            mode=mode,
            set_binding=lambda binding: setattr(self, "conv_binding", binding),
        )
        common = {
            "A_log": module.A_log,
            "dt_bias": module.dt_bias,
            "recurrent_state": state.recurrent,
            "num_seqs": metadata.num_seqs,
            "num_tokens": metadata.num_tokens,
            "output": (scratch.gated if mode == "decode" else scratch.recurrence).view(
                self.tokens, self.value_heads, 128),
            "a": scratch.a,
            "b": scratch.b,
        }
        if mode == "decode":
            caps = gdn_decode.Caps(
                device=self.device,
                max_tokens=self.tokens,
                max_seqs=self.sequences,
                max_state_slots=state.recurrent.shape[0],
                key_heads=self.key_heads,
                value_heads=self.value_heads,
                state_index_columns=columns,
                state_dtype=torch.float32,
                gate_activation=module.norm.activation,
                null_state_index=0,
            )
            args = dict(
                **common,
                mixed_qkv=scratch.convolved_qkv,
                z=scratch.z.view(self.tokens, self.value_heads, 128),
                norm_weight=module.norm.weight,
                query_start_loc=metadata.query_start_loc,
                num_accepted_tokens=metadata.accepted_tokens,
                state_indices=metadata.state_indices,
            )
            self.recurrent_plan = gdn_decode.plan(
                caps,
                invocation=gdn_decode.invocation_from_tensors(caps, **args),
            )
            operation = gdn_decode
        else:
            caps = gdn_prefill.Caps(
                device=self.device,
                max_tokens=self.tokens,
                max_seqs=self.sequences,
                max_state_slots=state.recurrent.shape[0],
                key_heads=self.key_heads,
                value_heads=self.value_heads,
                checkpoint_export=checkpoint_export,
                null_state_index=0,
                recurrence_policy="sequential_k1",
            )
            q, k, v = scratch.convolved_qkv.split((2048 // world_size, 2048 // world_size, 6144 // world_size), dim=-1)
            args = dict(
                **common,
                q=q.view(self.tokens, self.key_heads, 128),
                k=k.view(self.tokens, self.key_heads, 128),
                v=v.view(self.tokens, self.value_heads, 128),
                cu_seqlens=metadata.query_start_loc,
                initial_state_indices=metadata.initial_state_indices,
                final_state_indices=metadata.final_state_indices,
                checkpoint_state_indices=metadata.checkpoint_state_indices,
                checkpoint_offsets=metadata.checkpoint_offsets,
            )
            self.recurrent_plan = gdn_prefill.plan(
                caps,
                invocation=gdn_prefill.invocation_from_tensors(**args),
            )
            operation = gdn_prefill
        def set_recurrent_binding(binding):
            self.recurrent_binding = binding
            self.workspace = binding.scratch

        self.recurrent_binding = prepare_sequence_binding(
            self.recurrent_plan,
            args,
            session=session,
            name=f"{name}.{mode}",
            public_bind=operation.bind,
            workspace=workspace,
            mode=mode,
            set_binding=set_recurrent_binding,
            run_options={"eps": module.norm.variance_epsilon} if mode == "decode" else None,
        )
        self.row_ids = torch.arange(
            self.tokens, dtype=torch.int32, device=self.device
        ).view(-1, 1)
        self.inactive_rows = torch.empty(
            (self.tokens, 1), dtype=torch.bool, device=self.device
        )

    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        scratch = self.scratch
        if self.parallel_projections is None:
            self.projections["in_proj_qkvz"](source)
            self.projections["in_proj_ba"](source)
        else:
            self.parallel_projections(source)
        short_conv.run(self.conv_binding)
        if self.mode == "decode":
            gdn_decode.run(
                self.recurrent_binding, eps=self.module.norm.variance_epsilon
            )
        else:
            scratch.recurrence.zero_()
            gdn_prefill.run(self.recurrent_binding)
            normalized = self.module.norm(
                scratch.recurrence.view(self.tokens, self.value_heads, 128), scratch.z.view(self.tokens, self.value_heads, 128)
            )
            scratch.gated.copy_(normalized.view(self.tokens, 6144 // self.world_size))
            torch.ge(self.row_ids, self.metadata.num_tokens, out=self.inactive_rows)
            scratch.gated.masked_fill_(self.inactive_rows, 0)
        return self.projections["out_proj"](scratch.gated)


__all__ = [ "GDNScratch", "GDNState", "PreparedGDN"]
