"""Prepared Qwen GDN projections, convolution, recurrence, and output gating."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from functools import partial

import torch
from b12x.gemm.blockscaled import Weight
from b12x.preparation import PreparationSession, PreparedCall
from b12x.sequence import gdn_decode, gdn_prefill, short_conv

from sparkgap.preparation.scratch import planned_scratch, prime_owners
from sparkgap.preparation.tuning import benchmark_resource, register_finalizer

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

    @classmethod
    def allocate(cls, tokens: int, device: torch.device) -> GDNScratch:
        if type(tokens) is not int or tokens <= 0:
            raise ValueError("GDN token capacity must be positive")
        return cls(
            *(
                torch.zeros((tokens, width), device=device, dtype=torch.bfloat16)
                for width in (10240, 10240, 6144, 48, 48, 6144, 6144, 2560)
            )
        )

    def rows(self, tokens: int) -> GDNScratch:
        if type(tokens) is not int or not 0 < tokens <= self.projected_qkv.shape[0]:
            raise ValueError("GDN rows exceed scratch capacity")
        return GDNScratch(
            *(getattr(self, field.name)[:tokens] for field in fields(self))
        )


class _BenchmarkStateArena:
    """One full-stride semantic-state arena reused by serialized trials."""

    def __init__(self, template, *, storage=None):
        self.nbytes = template.shape[0] * template.stride(0) * template.element_size()
        self.borrowed_nbytes = 0 if storage is None else self.nbytes
        self.allocated_nbytes = self.nbytes - self.borrowed_nbytes
        if storage is None:
            storage = torch.empty(self.nbytes, dtype=torch.uint8, device=template.device)
        elif (not isinstance(storage, torch.Tensor) or storage.dtype != torch.uint8
                or not storage.is_contiguous() or storage.nbytes != self.nbytes
                or storage.device != template.device):
            raise ValueError("borrowed GDN arena requires contiguous uint8 storage with the exact extent and device")
        self.storage = storage.view(-1)

    def view(self, template):
        if (template.device != self.storage.device
                or template.shape[0] * template.stride(0) * template.element_size() != self.nbytes):
            raise ValueError("benchmark state shape/device differs from its strided arena")
        extent = sum((size - 1) * stride for size, stride in zip(template.shape, template.stride())) + 1
        if extent * template.element_size() > self.nbytes:
            raise ValueError("benchmark state view exceeds its strided arena")
        return self.storage.view(template.dtype).as_strided(template.shape, template.stride())


def _benchmark_state_key(template):
    return ("qwen_gdn_state", str(template.device), template.shape[0],
            template.stride(0) * template.element_size())


def register_gdn_benchmark_state(session, pool):
    """Borrow an unused hybrid pool during tuning and restore it before capture."""
    from sparkgap.preparation.tuning import RecordedPreparationSession

    if not isinstance(session, RecordedPreparationSession):
        raise TypeError("borrowing GDN state requires RecordedPreparationSession")
    template = next(iter(pool.gdn.values())).recurrent
    arena = _BenchmarkStateArena(template, storage=pool.backing)
    session.register_benchmark_resource(_benchmark_state_key(template), arena,
                                        restore=pool.reset_for_preparation)


def _strided_empty(template):
    return torch.empty_strided(template.shape, template.stride(), dtype=template.dtype, device=template.device)


def _packed_qkv(original):
    tensors = tuple(original[name] for name in ("q", "k", "v"))
    if len({tensor.untyped_storage().data_ptr() for tensor in tensors}) != 1:
        raise ValueError("GDN prefill benchmark requires packed QKV storage")
    first = min(tensor.storage_offset() for tensor in tensors)
    end = max(tensor.storage_offset() + 1 + sum(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
    ) for tensor in tensors)
    storage = torch.empty(end - first, dtype=tensors[0].dtype, device=tensors[0].device)
    views = {
        name: storage.as_strided(tensor.shape, tensor.stride(), tensor.storage_offset() - first)
        for name, tensor in zip(("q", "k", "v"), tensors)
    }
    return storage, views


class _GDNBenchmarkInputs:
    """Active recurrence/convolution operands with exact serving tensor layouts."""

    def __init__(self, original, *, mode, arena):
        self.mode, self.arena = mode, arena
        self.state_name = "conv_state" if "conv_state" in original else "recurrent_state"
        self.output_name = "out" if "out" in original else "output"
        immutable = {"A_log", "dt_bias", "norm_weight", "conv_weight"}
        self.packed_source, packed_views = _packed_qkv(original) if "q" in original else (None, {})
        self.args = {}
        for name, value in original.items():
            if name in immutable:
                self.args[name] = value
            elif name == self.state_name:
                self.args[name] = arena.view(value)
            elif name in packed_views:
                self.args[name] = packed_views[name]
            else:
                self.args[name] = _strided_empty(value)
        self.state = self.args[self.state_name]
        source_name = "x" if "x" in original else "mixed_qkv" if "mixed_qkv" in original else "q"
        self.tokens = original[source_name].shape[0]
        self.sequences = original.get("query_start_loc", original.get("cu_seqlens")).numel() - 1
        self.live_sequences = min(self.tokens, self.sequences)
        quotient, remainder = divmod(self.tokens, self.live_sequences)
        self.lengths = tuple(quotient + (index < remainder) for index in range(self.live_sequences))
        columns = original["state_indices"].shape[1] if "state_indices" in original else 1
        self.live_slots = tuple(1 + index * columns for index in range(self.live_sequences))
        self.state_rows = self.live_sequences * columns + 1
        if mode == "prefill" and "checkpoint_state_indices" in original:
            self.state_rows += self.live_sequences
        if self.state_rows > self.state.shape[0]:
            raise ValueError("benchmark state pool cannot hold live and checkpoint slots")
        generator = torch.Generator(device="cpu").manual_seed(104729)
        initial = torch.randn((self.state_rows, *self.state.shape[1:]), generator=generator) * .1
        initial[0].zero_()
        self.initial_state = initial.to(device=self.state.device, dtype=self.state.dtype)
        self.sources = {}
        self.packed_fixture = None
        if self.packed_source is not None:
            value = torch.randn(self.packed_source.shape, generator=generator) * .1
            self.packed_fixture = value.to(device=self.state.device, dtype=self.packed_source.dtype)
        for name, target in self.args.items():
            if name in immutable or name == self.state_name:
                continue
            if target.is_floating_point() and name != self.output_name and name not in packed_views:
                value = torch.randn(tuple(target.shape), generator=generator) * .1
                self.sources[name] = value.to(device=target.device, dtype=target.dtype)
            target.zero_()
        self._metadata(columns)
        self.producers = (partial(self.produce, False), partial(self.produce, True)) if mode == "prefill" else (partial(self.produce, False),)
        self._owned_tensors = (
            self.initial_state, self.packed_source, self.packed_fixture, *self.sources.values(),
            *(value for name, value in self.args.items() if name not in immutable and name != self.state_name),
        )

    @property
    def nbytes(self):
        storage = {(value.device, value.untyped_storage().data_ptr()): value.untyped_storage()
                   for value in self._owned_tensors if value is not None}
        return sum(value.nbytes() for value in storage.values())

    def _metadata(self, columns):
        a = self.args
        a["num_tokens"].fill_(self.tokens)
        a["num_seqs"].fill_(self.live_sequences)
        bounds = a.get("query_start_loc", a.get("cu_seqlens"))
        bounds.fill_(self.tokens)
        bounds[0] = 0
        offset = 0
        for row, length in enumerate(self.lengths):
            offset += length
            bounds[row + 1] = offset
            if "state_slot_ids" in a:
                a["state_slot_ids"][row] = self.live_slots[row]
            if "state_indices" in a:
                for column in range(columns):
                    a["state_indices"][row, column] = self.live_slots[row] + column
            if "final_state_indices" in a:
                a["final_state_indices"][row] = self.live_slots[row]
            if "checkpoint_offsets" in a:
                boundary = length // 64 * 64
                a["checkpoint_offsets"][row] = boundary
                a["checkpoint_state_indices"][row] = self.live_sequences + 1 + row if boundary else 0
        if "num_accepted_tokens" in a:
            a["num_accepted_tokens"].fill_(1)
        if "state_slot_ids" in a and self.live_sequences < self.sequences:
            a["state_slot_ids"][self.live_sequences:].fill_(-1)

    def reset(self):
        self.state[:self.state_rows].copy_(self.initial_state)
        self.args[self.output_name].zero_()

    def produce(self, fresh):
        if self.packed_source is not None:
            self.packed_source.copy_(self.packed_fixture)
        for name, value in self.sources.items():
            self.args[name].copy_(value)
        if "state_is_fresh" in self.args:
            self.args["state_is_fresh"].fill_(fresh)
        if "initial_state_indices" in self.args:
            self.args["initial_state_indices"].zero_()
            if not fresh:
                for row, slot in enumerate(self.live_slots):
                    self.args["initial_state_indices"][row] = slot


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
    mapping supplies the checkpoint's five packed MXFP8 weights. Preparation
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
        widths = (10240, 10240, 6144, 48, 48, 6144, 6144, 2560)
        for field, width in zip(fields(scratch), widths):
            tensor = getattr(scratch, field.name)
            if (
                tensor.device != self.device
                or tensor.dtype != torch.bfloat16
                or tuple(tensor.shape) != (self.tokens, width)
                or not tensor.is_contiguous()
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
        expected = {"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"}
        if set(projections) != expected:
            raise ValueError("GDN requires its five named packed projection weights")
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
                ("in_proj_qkv", scratch.projected_qkv),
                ("in_proj_z", scratch.z),
                ("in_proj_a", scratch.a),
                ("in_proj_b", scratch.b),
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
            template = args["conv_state"] if bind_plan else args["recurrent_state"]
            key = _benchmark_state_key(template)
            arena = benchmark_resource(session, key, lambda: _BenchmarkStateArena(template))
            operands = benchmark_resource(session, ("qwen_gdn_inputs", name),
                lambda: _GDNBenchmarkInputs(args, mode=mode, arena=arena))
            scratch = selected_scratch(candidate_specs(prepared))
            binding = prepared.bind(scratch=scratch, **operands.args, **({"plan": plan} if bind_plan else {}))
            return PreparedCall(run=lambda: prepared.run(binding, **run_options), output=operands.args[operands.output_name],
                produce=operands.producers[0], benchmark_producers=operands.producers,
                reset=operands.reset, restore=operands.reset, owners=(operands, scratch, binding))

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
        self.projections["in_proj_qkv"](source)
        self.projections["in_proj_z"](source)
        self.projections["in_proj_a"](source)
        self.projections["in_proj_b"](source)
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
                scratch.recurrence.view(-1, 128), scratch.z.view(-1, 128)
            )
            scratch.gated.copy_(normalized.view(self.tokens, 6144))
            torch.ge(self.row_ids, self.metadata.num_tokens, out=self.inactive_rows)
            scratch.gated.masked_fill_(self.inactive_rows, 0)
        return self.projections["out_proj"](scratch.gated)


__all__ = ["GDNMetadata", "GDNScratch", "GDNState", "PreparedGDN"]
