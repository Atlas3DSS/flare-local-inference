"""Prepared Qwen GDN projections, convolution, recurrence, and output gating."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

import torch
from b12x.gemm.blockscaled import Weight
from b12x.preparation import PreparationSession, PreparedCall
from b12x.sequence import gdn_decode, gdn_prefill, short_conv

from flare.preparation.fixtures import gdn as gdn_fixtures
from flare.preparation.scratch import planned_scratch, prime_owners
from flare.preparation.tuning import register_finalizer

from .linear import QuantizedProjection


@dataclass(frozen=True)
class GDNState:
    """Persistent chronological convolution history and transposed FP32 recurrence.

    Recurrence slots have physical shape ``[slot, 48, value_dim=128, key_dim=128]``.
    The canonical Transformers cache transposes the final two dimensions.
    Slot zero is reserved as the recurrence null index. Fresh decode slots must
    be zeroed by cache maintenance before replay; prefill selects null initial
    state through metadata. Convolution fresh flags discard prior history.
    """

    convolution: torch.Tensor
    recurrent: torch.Tensor

    @classmethod
    def allocate(
        cls, slots: int, device: torch.device, *, speculative_tokens: int = 0
    ) -> GDNState:
        if type(slots) is not int or slots < 2:
            raise ValueError(
                "GDN requires a null slot and at least one live state slot"
            )
        if type(speculative_tokens) is not int or not 0 <= speculative_tokens <= 7:
            raise ValueError("GDN supports zero through seven speculative tokens")
        return cls(
            torch.zeros(
                (slots, 10240, 3 + speculative_tokens),
                device=device,
                dtype=torch.bfloat16,
            ),
            torch.zeros((slots, 48, 128, 128), device=device, dtype=torch.float32),
        )


@dataclass(frozen=True)
class GDNMetadata:
    """Fixed-address device metadata updated by the native scheduler.

    Decode consumes ``state_indices[:, accepted_tokens - 1]`` and writes each
    query token's checkpoint to its corresponding column. Prefill uses explicit
    initial/final indices and can export a checkpoint at a multiple of 16 tokens.
    Live nonnegative slots must be unique and in range. Counts and packed query
    boundaries must fit their buffer capacities. These are scheduler invariants.
    """

    query_start_loc: torch.Tensor
    convolution_slots: torch.Tensor
    fresh: torch.Tensor
    accepted_tokens: torch.Tensor
    state_indices: torch.Tensor
    initial_state_indices: torch.Tensor
    final_state_indices: torch.Tensor
    checkpoint_state_indices: torch.Tensor
    checkpoint_offsets: torch.Tensor
    num_seqs: torch.Tensor
    num_tokens: torch.Tensor

    @classmethod
    def allocate(
        cls, sequences: int, device: torch.device, *, verification_columns: int = 1
    ) -> GDNMetadata:
        if type(sequences) is not int or sequences <= 0:
            raise ValueError("GDN sequence capacity must be positive")
        if type(verification_columns) is not int or not 1 <= verification_columns <= 8:
            raise ValueError("GDN verification columns must be in [1, 8]")

        def zeros(shape, dtype=torch.int32):
            return torch.zeros(shape, dtype=dtype, device=device)

        return cls(
            zeros((sequences + 1,)),
            torch.full((sequences,), -1, dtype=torch.int64, device=device),
            zeros((sequences,), torch.bool),
            torch.ones(sequences, dtype=torch.int32, device=device),
            zeros((sequences, verification_columns), torch.int64),
            zeros((sequences,), torch.int64),
            zeros((sequences,), torch.int64),
            zeros((sequences,), torch.int64),
            zeros((sequences,)),
            zeros((1,)),
            zeros((1,)),
        )


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
    def allocate(cls, tokens: int, device: torch.device) -> GDNScratch:
        if type(tokens) is not int or tokens <= 0:
            raise ValueError("GDN token capacity must be positive")
        qkvz = torch.zeros((tokens, 16384), device=device, dtype=torch.bfloat16)
        ba = torch.zeros((tokens, 96), device=device, dtype=torch.bfloat16)
        convolved, recurrence, gated, output = (
            torch.zeros((tokens, width), device=device, dtype=torch.bfloat16)
            for width in (10240, 6144, 6144, 2560))
        return cls(qkvz[:, :10240], convolved, qkvz[:, 10240:], ba[:, 48:], ba[:, :48],
                   recurrence, gated, output, qkvz, ba)

    def rows(self, tokens: int) -> GDNScratch:
        if type(tokens) is not int or not 0 < tokens <= self.projected_qkv.shape[0]:
            raise ValueError("GDN rows exceed scratch capacity")
        return GDNScratch(
            *(getattr(self, field.name)[:tokens] for field in fields(self))
        )


def _validate_geometry(module: torch.nn.Module) -> None:
    required = {
        "hidden_size": 2560,
        "num_k_heads": 16,
        "num_v_heads": 48,
        "head_k_dim": 128,
        "head_v_dim": 128,
        "conv_dim": 10240,
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
        metadata: GDNMetadata,
        *,
        mode: str,
        session: PreparationSession,
        name: str,
        checkpoint_export: bool = False,
        workspace: torch.Tensor | None = None,
        projection_workspace: torch.Tensor | None = None,
    ) -> None:
        _validate_geometry(module)
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
        widths = (10240, 10240, 6144, 48, 48, 6144, 6144, 2560, 16384, 96)
        strides = (16384, 10240, 16384, 96, 96, 6144, 6144, 2560, 16384, 96)
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
        if state.convolution.ndim != 3 or state.convolution.shape[1] != 10240:
            raise ValueError(
                "GDN convolution history must have shape [slots, 10240, 3 + drafts]"
            )
        self.speculative_tokens = state.convolution.shape[2] - 3
        if not 0 <= self.speculative_tokens <= 7:
            raise ValueError(
                "GDN history supports zero through seven speculative tokens"
            )
        if state.recurrent.ndim != 4 or tuple(state.recurrent.shape[1:]) != (
            48,
            128,
            128,
        ):
            raise ValueError("GDN recurrence must have shape [slots, 48, 128, 128]")
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
            workspace=projection_workspace,
        )
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
            channels=10240,
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
        self.conv_binding = self._prepare_binding(
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
            "output": scratch.recurrence.view(self.tokens, 48, 128),
            "a": scratch.a,
            "b": scratch.b,
        }
        if mode == "decode":
            caps = gdn_decode.Caps(
                device=self.device,
                max_tokens=self.tokens,
                max_seqs=self.sequences,
                max_state_slots=state.recurrent.shape[0],
                key_heads=16,
                value_heads=48,
                state_index_columns=columns,
                state_dtype=torch.float32,
                gate_activation=module.norm.activation,
                null_state_index=0,
            )
            args = dict(
                **common,
                mixed_qkv=scratch.convolved_qkv,
                z=scratch.z.view(self.tokens, 48, 128),
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
                key_heads=16,
                value_heads=48,
                checkpoint_export=checkpoint_export,
                null_state_index=0,
                recurrence_policy="sequential_k1",
            )
            q, k, v = scratch.convolved_qkv.split((2048, 2048, 6144), dim=-1)
            args = dict(
                **common,
                q=q.view(self.tokens, 16, 128),
                k=k.view(self.tokens, 16, 128),
                v=v.view(self.tokens, 48, 128),
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

        self.recurrent_binding = self._prepare_binding(
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
        self.workspace = self.recurrent_binding.scratch
        self.row_ids = torch.arange(
            self.tokens, dtype=torch.int32, device=self.device
        ).view(-1, 1)
        self.inactive_rows = torch.empty(
            (self.tokens, 1), dtype=torch.bool, device=self.device
        )

    @staticmethod
    def _prepare_binding(
        plan, args, *, session, name, public_bind, mode, set_binding,
        bind_plan=False, workspace=None, run_options=None,
    ):
        run_options = {} if run_options is None else dict(run_options)

        def selected_scratch(specs):
            return planned_scratch(specs, workspace)

        def candidate_specs(prepared):
            return plan.scratch_specs() if bind_plan else prepared.layout.scratch_specs()

        def prepare_call(prepared):
            scratch = selected_scratch(candidate_specs(prepared))
            binding = prepared.bind(scratch=scratch, **args, **({"plan": plan} if bind_plan else {}))
            return PreparedCall(run=lambda: prepared.run(binding, **run_options), owners=prime_owners(workspace, scratch, binding))

        def benchmark_call(prepared):
            return gdn_fixtures.benchmark_call(
                prepared, arguments=args, session=session, name=name, mode=mode,
                scratch_specs=candidate_specs, workspace=workspace,
                run_options=run_options, binding_plan=plan if bind_plan else None,
            )

        session.prepare((plan.request(name=name, prepare_call=prepare_call,
            benchmark_call=benchmark_call),))
        selected = None

        def finalize():
            nonlocal selected
            scratch = selected_scratch(plan.scratch_specs())
            selected = public_bind(plan, scratch=scratch, **args)
            set_binding(selected)

        finalize()
        register_finalizer(session, finalize, workspace=workspace)
        return selected

    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        scratch = self.scratch
        self.projections["in_proj_qkvz"](source)
        self.projections["in_proj_ba"](source)
        short_conv.run(self.conv_binding)
        if self.mode == "decode":
            gdn_decode.run(
                self.recurrent_binding, eps=self.module.norm.variance_epsilon
            )
            scratch.gated.copy_(scratch.recurrence)
        else:
            scratch.recurrence.zero_()
            gdn_prefill.run(self.recurrent_binding)
            normalized = self.module.norm(
                scratch.recurrence.view(self.tokens, 48, 128), scratch.z.view(self.tokens, 48, 128)
            )
            scratch.gated.copy_(normalized.view(self.tokens, 6144))
            torch.ge(self.row_ids, self.metadata.num_tokens, out=self.inactive_rows)
            scratch.gated.masked_fill_(self.inactive_rows, 0)
        return self.projections["out_proj"](scratch.gated)


__all__ = ["GDNMetadata", "GDNScratch", "GDNState", "PreparedGDN"]
