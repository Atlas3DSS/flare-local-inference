"""Direct checkpoint transfers and compressed-scale preparation."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from functools import cached_property
import json
import sys
from time import monotonic
from typing import TYPE_CHECKING

from ignition.preparation.checkpoint import IndexedCheckpoint
from ignition.progress import phase
from ignition.preparation.quantization import projection_group_format, quantized_layers

if TYPE_CHECKING:
    import torch
    from b12x.gemm.blockscaled import Weight


@dataclass(frozen=True)
class MLPWeights:
    gate_up: Weight | torch.Tensor
    down: Weight | torch.Tensor


def _release_unused_contexts(device):
    """Destroy primary contexts that checkpoint reads opened on other visible GPUs.

    GPUDirect Storage reads retain a CUDA context, about 0.5 GiB, on every visible
    device. A rank process allocates only on its own device, so those contexts only
    take memory from its peers' KV pages.
    """
    import ctypes

    import torch

    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return
    for ordinal in range(torch.cuda.device_count()):
        if ordinal == device or torch.cuda.memory_reserved(ordinal):
            continue
        handle, flags, active = ctypes.c_int(), ctypes.c_uint(), ctypes.c_int()
        if (driver.cuDeviceGet(ctypes.byref(handle), ordinal) == 0
                and driver.cuDevicePrimaryCtxGetState(handle, ctypes.byref(flags), ctypes.byref(active)) == 0
                and active.value):
            driver.cuDevicePrimaryCtxReset_v2(handle)


class DirectWeightLoader:
    """Own b12x tensor views until direct transfers finish.

    Recipe subclasses select tensors and validate model-specific quantization.
    Source views may partition weight rows or columns without CPU weight copies.
    Compressed scale payloads are gathered and indexed on CUDA one layer at a time.
    """

    def __init__(
        self, manifest: IndexedCheckpoint, device: torch.device | str,
        *, prefixes: Sequence[str] | None = None,
        progress=None,
        progress_label: str = "Load weights",
    ):
        self.manifest = manifest
        self.device = device
        self.prefixes = None if prefixes is None else tuple(prefixes)
        if self.prefixes is not None and not self.prefixes:
            raise ValueError("prefixes must be nonempty when supplied")
        self.progress, self.progress_label = progress, progress_label
        self._display = None
        self._load_stats = None
        self._session = None
        self._sources = {}
        self._stack = None

    @cached_property
    def quantization(self):
        return quantized_layers(json.loads((self.manifest.snapshot / "config.json").read_text()))

    def load_linear(self, prefix, *, rows=None, columns=None):
        """Load one projection using its checkpoint quantization declaration."""
        self._require_open()
        return self.load_linear_group((prefix,), row_ranges=(rows,), column_range=columns)

    def _load_nvfp4_linear(self, prefix, *, rows=None, columns=None):
        import torch
        from b12x._lib.intrinsics import swizzle_block_scale
        from b12x.gemm.blockscaled import pack_weight

        source = self._sources[f"{prefix}.weight"]
        scale = self._sources[f"{prefix}.weight_scale"]
        scalar = self._sources[f"{prefix}.weight_scale_2"]
        if source.ndim != 2 or source.dtype != torch.uint8:
            raise ValueError(f"{prefix}: NVFP4 weight must be a packed uint8 matrix")
        n, packed_k = source.shape
        k = packed_k * 2
        if (k % 16 or scale.ndim != 2 or scale.dtype != torch.float8_e4m3fn
                or scale.shape[0] < n or scale.shape[1] < k // 16):
            raise ValueError(f"{prefix}: NVFP4 block scale geometry or dtype mismatch")
        if (scalar.dtype != torch.float32 or scalar.numel() != 1
                or not bool((torch.isfinite(scalar) & (scalar > 0)).all())):
            raise ValueError(f"{prefix}.weight_scale_2 must be a positive finite FP32 scalar")
        r0, r1 = _aligned_range(rows, n, 1)
        c0, c1 = _aligned_range(columns, k, 16)
        logical_k = c1 - c0
        stored_k = (logical_k + 127) // 128 * 128
        values = torch.zeros((r1 - r0, stored_k // 2), dtype=torch.uint8, device=self.device)
        scales = torch.zeros((r1 - r0, stored_k // 16), dtype=torch.float8_e4m3fn, device=self.device)
        gain = torch.empty(scalar.shape, dtype=torch.float32, device=self.device)
        self._copy(values[:, :logical_k // 2], source[r0:r1, c0 // 2:c1 // 2])
        self._copy(scales[:, :logical_k // 16], scale[r0:r1, c0 // 16:c1 // 16])
        self._copy(gain, scalar)
        activation_scale = None
        if self.quantization[prefix]["quant_algo"] == "NVFP4":
            activation = self._sources[f"{prefix}.input_scale"]
            if (activation.dtype != torch.float32 or activation.numel() != 1
                    or not bool((torch.isfinite(activation) & (activation > 0)).all())):
                raise ValueError(f"{prefix}.input_scale must be a positive finite FP32 scalar")
            reciprocal = activation.reciprocal()
            if not bool(torch.isfinite(reciprocal).all()):
                raise ValueError(f"{prefix}.input_scale reciprocal must remain finite")
            activation_scale = torch.empty_like(activation, device=self.device)
            self._copy(activation_scale, reciprocal)
        self._session.flush()
        weight = pack_weight(values, swizzle_block_scale(scales), recipe="nvfp4",
                             global_scale=gain, global_scale_kind="multiplier",
                             activation_global_scale=activation_scale)
        return replace(weight, in_features=logical_k)

    def _selected_names(self):
        return {name for name in self.manifest.tensors
                if self.prefixes is None or name.startswith(self.prefixes)}

    @contextmanager
    def _loading_display(self):
        from b12x.loader import CheckpointDisplay

        with phase(self.progress, self.progress_label) as stage, CheckpointDisplay(
            console=stage.console, transient=True,
        ) as display:
            self._display = display
            started = monotonic()
            stats = {}
            yield stats
            display.complete(dict(ranks=1, payload_bytes=stats["payload_bytes"],
                                  physical_bytes=stats["physical_bytes"],
                                  load_seconds=monotonic() - started))

    def __enter__(self):
        import torch
        from b12x.loader import DirectWeightSession

        if self._stack is not None:
            raise RuntimeError("weight loader context is already open")
        device = torch.device(self.device)
        if device.type != "cuda":
            raise ValueError(f"{type(self).__name__} requires a CUDA device")
        self.device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        stack = ExitStack()
        try:
            if self.progress is not None:
                self._load_stats = stack.enter_context(self._loading_display())
            # Scalar calibration checks do not amortize intra-op thread dispatch.
            stack.callback(torch.set_num_threads, torch.get_num_threads())
            torch.set_num_threads(1)
            self._session = stack.enter_context(DirectWeightSession(device=self.device.index))
            selected = self._selected_names()
            shards = {self.manifest.tensors[name].shard for name in selected}
            if self._display is not None:
                self._display.source(len(shards))

            def paths():
                for shard in sorted(shards):
                    if self._display is not None:
                        self._display.file(shard)
                    yield self.manifest.snapshot / shard
                    if self._display is not None:
                        self._display.advance(self._session.payload_bytes)

            # b12x validates source view mutations through tensor version counters.
            with torch.inference_mode(False):
                self._sources = dict(self._session.weights(
                    paths(),
                    index_path=self.manifest.index_path,
                    skip=lambda name: name not in selected,
                    needs_values=lambda entry: entry.name.endswith((".weight_scale_2", ".input_scale")),
                ))
            if self._sources.keys() != selected:
                raise ValueError("direct loader tensor coverage differs from the inspected checkpoint")
            if self._display is not None:
                self._display.phase("execute")
            self._stack = stack
            return self
        except BaseException:
            stack.__exit__(*sys.exc_info())
            self._session = None
            self._display = self._load_stats = None
            self._sources.clear()
            raise

    def __exit__(self, *exc):
        try:
            if exc[0] is None and self._display is not None:
                try:
                    self._load_stats.update(self._session.stats())
                except BaseException:
                    self._stack.__exit__(*sys.exc_info())
                    raise
            return self._stack.__exit__(*exc)
        finally:
            self._stack = None
            self._session = None
            self._display = self._load_stats = None
            self._sources.clear()
            _release_unused_contexts(self.device.index)

    def _require_open(self):
        if self._session is None:
            raise RuntimeError(f"weight loading requires an open {type(self).__name__} context")

    def _copy(self, destination, source):
        from ignition.preparation.tensor_metadata import record_numeric_tensor

        if not self._session(destination, source):
            raise RuntimeError("b12x rejected the direct weight transfer")
        record_numeric_tensor(destination)
        if self._display is not None:
            self._display.update(self._session.payload_bytes)

    def load_tensor(self, name: str, *, dtype: torch.dtype | None = None, extent=None) -> torch.Tensor:
        """Transfer a selected resident weight or declared scalar to the GPU."""
        target = self._queue_tensor(name, dtype=dtype, extent=extent)
        self._session.flush()
        return target

    def load_tensors(self, names: Sequence[str], *, extents=None) -> tuple[torch.Tensor, ...]:
        """Transfer independent weights together, returning ready GPU tensors."""
        self._require_open()
        if extents is None:
            extents = (None,) * len(names)
        targets = tuple(self._queue_tensor(name, extent=extent)
                        for name, extent in zip(names, extents, strict=True))
        self._session.flush()
        return targets

    def _queue_tensor(self, name, *, dtype=None, extent=None):
        import torch

        self._require_open()
        source = self._sources[name]
        if extent is not None:
            source = source[extent]
        target = torch.empty(source.shape, dtype=source.dtype if dtype is None else dtype, device=self.device)
        self._copy(target, source)
        return target

    def load_linear_group(self, prefixes, *, row_ranges=None, column_range=None):
        """Pack compatible projection rows without changing their quantization."""
        self._require_open()
        prefixes = tuple(prefixes)
        fmt = projection_group_format(self.quantization, prefixes)
        import torch
        from b12x.gemm.blockscaled import pack_weight
        if row_ranges is None:
            row_ranges = (None,) * len(prefixes)
        if fmt == "nvfp4":
            (rows,) = tuple(row_ranges)
            return self._load_nvfp4_linear(prefixes[0], rows=rows, columns=column_range)
        fp4 = fmt == "mxfp4"
        pack = 2 if fp4 else 1
        dtype = torch.uint8 if fp4 else torch.float8_e4m3fn
        weights, block_scales = [], []
        for prefix, row_range in zip(prefixes, row_ranges, strict=True):
            weight = self._sources[f"{prefix}.weight"]
            scale = self._sources[f"{prefix}.weight_scale"]
            if weight.ndim != 2 or weight.dtype != dtype:
                raise ValueError(f"{prefix}: {fmt.upper()} weight must be a {dtype} matrix")
            rows, stored_columns = weight.shape
            columns = stored_columns * pack
            r0, r1 = _aligned_range(row_range, rows, 1)
            c0, c1 = _aligned_range(column_range, columns, 32)
            if (scale.ndim != 2 or scale.dtype not in (torch.uint8, torch.float8_e8m0fnu)
                    or scale.shape[0] < rows or scale.shape[1] < columns // 32):
                raise ValueError(f"{prefix}: {fmt.upper()} block scale geometry mismatch")
            weights.append(weight[r0:r1, c0 // pack:c1 // pack])
            block_scales.append(scale[r0:r1, c0 // 32:c1 // 32].view(torch.uint8))
        columns = weights[0].shape[1] * pack
        if any(weight.shape[1] * pack != columns for weight in weights):
            raise ValueError("packed projections must share input width")
        rows = sum(weight.shape[0] for weight in weights)
        stored_columns = (columns + 255) // 256 * 256 if fp4 else columns
        values = torch.zeros((rows, stored_columns // pack), dtype=dtype, device=self.device)
        scales = torch.full((rows, stored_columns // 32), 127, dtype=torch.uint8, device=self.device)
        offset = 0
        for weight, scale in zip(weights, block_scales, strict=True):
            end = offset + weight.shape[0]
            self._copy(values[offset:end, :columns // pack], weight)
            self._copy(scales[offset:end, :columns // 32], scale)
            offset = end
        self._session.flush()
        if fp4:
            from b12x._lib.intrinsics import swizzle_block_scale

            return replace(pack_weight(values, swizzle_block_scale(scales), recipe="mxfp4"),
                           in_features=columns)
        return pack_weight(values, scales, recipe="mxfp8")

    def load_fp4_experts(self, prefixes, *, hidden_size, intermediate_size, format, activation_mode,
                         intermediate_range=None, scale_vectors=None, swiglu_limit=None):
        """Prepare ordinary NVFP4 or MXFP4 expert storage through b12x."""
        import torch
        from b12x.moe import fused_moe as moe
        from b12x._lib.intrinsics import swizzle_block_scale

        self._require_open()
        prefixes = tuple(prefixes)
        if format not in ("nvfp4", "mxfp4"):
            raise ValueError("ordinary FP4 experts require NVFP4 or MXFP4")
        group = 16 if format == "nvfp4" else 32
        scale_dtype = torch.float8_e4m3fn if format == "nvfp4" else torch.uint8
        first, last = _aligned_range(intermediate_range, intermediate_size, group)
        local, count = last - first, len(prefixes)
        w13 = torch.empty((count, 2 * local, hidden_size // 2), dtype=torch.uint8, device=self.device)
        w2 = torch.empty((count, hidden_size, local // 2), dtype=torch.uint8, device=self.device)
        s13 = torch.empty((count, 2 * local, hidden_size // group), dtype=scale_dtype, device=self.device)
        s2 = torch.empty((count, hidden_size, local // group), dtype=scale_dtype, device=self.device)
        globals13, globals2 = [], []
        for index, prefix in enumerate(prefixes):
            scales = {}
            for projection, values, blocks in (
                ("up_proj", w13[index, :local], s13[index, :local]),
                ("gate_proj", w13[index, local:], s13[index, local:]),
                ("down_proj", w2[index], s2[index]),
            ):
                source = self._sources[f"{prefix}.{projection}.weight"]
                scale = self._sources[f"{prefix}.{projection}.weight_scale"]
                rows, columns = ((hidden_size, intermediate_size) if projection == "down_proj"
                                 else (intermediate_size, hidden_size))
                if source.dtype != torch.uint8 or source.shape != (rows, columns // 2):
                    raise ValueError(f"{prefix}.{projection}: packed FP4 weight geometry mismatch")
                if scale.shape != (rows, columns // group) or scale.dtype != scale_dtype:
                    raise ValueError(f"{prefix}.{projection}: {format.upper()} block scale geometry or dtype mismatch")
                self._copy(values, source[:, first // 2:last // 2] if projection == "down_proj" else source[first:last])
                self._copy(blocks, scale[:, first // group:last // group] if projection == "down_proj" else scale[first:last])
                if format == "mxfp4":
                    continue
                value = self._sources[f"{prefix}.{projection}.weight_scale_2"]
                if value.dtype != torch.float32 or value.numel() != 1 or not bool(torch.isfinite(value) & (value > 0)):
                    raise ValueError(f"{prefix}.{projection}: expected a positive FP32 global weight scale")
                scales[projection] = value.item()
            if format == "mxfp4":
                continue
            if scales["gate_proj"] != scales["up_proj"]:
                raise ValueError(f"{prefix}: gate/up global scales differ")
            globals13.append(scales["up_proj"])
            globals2.append(scales["down_proj"])
        if activation_mode == "a4":
            if scale_vectors is None or len(scale_vectors) != 4:
                raise ValueError("NVFP4 A4 experts require weight and activation scale vectors")
            vectors = scale_vectors
        elif format == "nvfp4":
            vectors = (torch.tensor(globals13, dtype=torch.float32), torch.tensor(globals2, dtype=torch.float32))
        else:
            vectors = (torch.ones(count, dtype=torch.float32), torch.ones(count, dtype=torch.float32))
        device_scales = []
        for values in vectors:
            if (values.shape != (count,) or values.dtype != torch.float32
                    or not bool((torch.isfinite(values) & (values > 0)).all())):
                raise ValueError("expert scales must be positive finite FP32 vectors")
            target = torch.empty(count, dtype=torch.float32, device=self.device)
            self._copy(target, values)
            device_scales.append(target)
        self._session.flush()
        plan = moe.plan_weights(
            source=moe.PackedSource(format="modelopt_nvfp4" if format == "nvfp4" else "fp4_e8m0_k32", w13_layout="w13"),
            activation=moe.ActivationSpec(mode=activation_mode, nonlinearity="silu", io_dtype=torch.bfloat16,
                                          swiglu_limit=swiglu_limit),
            geometry=moe.MoEGeometry(num_experts=count, hidden_size=hidden_size, intermediate_size=local))
        gains = {} if not device_scales else dict(w13_global_scales=device_scales[0], w2_global_scales=device_scales[1])
        if activation_mode == "a4":
            gains.update(input_scale=device_scales[2], intermediate_scale=device_scales[3], immutable_input_scales=True)
        return moe.prepare_weights(plan=plan, weights=moe.PackedWeights(
            w13=w13, w2=w2,
            w13_block_scales=swizzle_block_scale(s13) if format == "nvfp4" else s13,
            w2_block_scales=swizzle_block_scale(s2) if format == "nvfp4" else s2, **gains))

    def load_nvfp4_experts(
        self, prefixes, *, hidden_size, intermediate_size, scale_vectors,
        intermediate_range=None, scale_scratch=None, swiglu_limit=None, activation_mode="a4",
    ):
        """Prepare routed A4 experts from packed rank-local checkpoint views.

        ``scale_vectors`` supplies up/gate and down global multipliers followed
        by their activation reciprocal scales. Gate/up scalar compatibility is
        the recipe's responsibility. Scale scratch can be shared by serialized
        layers; both buffers remain caller-owned through replay.
        """
        import torch
        from b12x.moe import fused_moe as moe

        self._require_open()
        prefixes = tuple(prefixes)
        if not prefixes or hidden_size <= 0 or hidden_size % 128:
            raise ValueError("NVFP4 experts require nonempty prefixes and 128-aligned hidden width")
        first, last = _aligned_range(intermediate_range, intermediate_size, 32)
        local = last - first
        count = len(prefixes)
        shapes = ((count, (2 * local + 127) // 128 * 128, hidden_size // 16),
                  (count, hidden_size, (local // 16 + 3) // 4 * 4))
        if scale_scratch is None:
            scale_scratch = tuple(torch.empty(shape, dtype=torch.float8_e4m3fn, device=self.device)
                                  for shape in shapes)
        for scratch, shape in zip(scale_scratch, shapes, strict=True):
            if (scratch.device != self.device or scratch.dtype != torch.float8_e4m3fn
                    or tuple(scratch.shape) != shape or not scratch.is_contiguous()):
                raise ValueError("expert scale scratch shape, dtype, device, or layout mismatch")
        if len(scale_vectors) != 4 or any(
            scale.device.type != "cpu" or scale.dtype != torch.float32 or scale.shape != (count,)
            or not bool((torch.isfinite(scale) & (scale > 0)).all()) for scale in scale_vectors
        ):
            raise ValueError("expert scales require four positive finite CPU float32 vectors")

        w13 = torch.empty((count, 2 * local, hidden_size // 2), dtype=torch.uint8, device=self.device)
        w2 = torch.empty((count, hidden_size, local // 2), dtype=torch.uint8, device=self.device)
        for index, prefix in enumerate(prefixes):
            for projection, destination, rows, columns, rr, cr in (
                ("up_proj", w13[index, :local], intermediate_size, hidden_size, (first, last), None),
                ("gate_proj", w13[index, local:], intermediate_size, hidden_size, (first, last), None),
                ("down_proj", w2[index], hidden_size, intermediate_size, None, (first, last)),
            ):
                source = self._sources[f"{prefix}.{projection}.weight"]
                if source.dtype != torch.uint8 or source.shape != (rows, columns // 2):
                    raise ValueError(f"{prefix}.{projection}: packed NVFP4 weight geometry mismatch")
                r0, r1 = (0, rows) if rr is None else rr
                c0, c1 = (0, columns) if cr is None else cr
                self._copy(destination, source[r0:r1, c0 // 2:c1 // 2])
        device_scales = []
        for scale in scale_vectors:
            target = torch.empty_like(scale, device=self.device)
            self._copy(target, scale)
            device_scales.append(target)
        scales13 = self._queue_csf_sources(
            [f"{prefix}.{projection}.weight_scale" for prefix in prefixes
             for projection in ("up_proj", "gate_proj")],
            rows=intermediate_size, columns=hidden_size // 16, join=2,
            row_range=(first, last))
        scales2 = self._queue_csf_sources(
            [f"{prefix}.down_proj.weight_scale" for prefix in prefixes],
            rows=hidden_size, columns=intermediate_size // 16,
            column_range=(first // 16, last // 16))
        self._session.flush()
        if self._display is not None:
            self._display.update(self._session.payload_bytes)
        plan = moe.plan_weights(
            source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
            activation=moe.ActivationSpec(mode=activation_mode, nonlinearity="silu", io_dtype=torch.bfloat16,
                                          swiglu_limit=swiglu_limit),
            geometry=moe.MoEGeometry(num_experts=count, hidden_size=hidden_size, intermediate_size=local),
        )
        packed = moe.PackedWeights(
            w13=w13, w2=w2, w13_block_scales=scale_scratch[0], w2_block_scales=scale_scratch[1],
            w13_global_scales=device_scales[0], w2_global_scales=device_scales[1],
            input_scale=device_scales[2], intermediate_scale=device_scales[3],
            immutable_input_scales=True,
        )
        return moe.prepare_weights(plan=plan, weights=moe.Nvfp4CsfWeights(
            packed=packed, w13_scales=scales13, w2_scales=scales2))

    def _queue_csf_sources(self, prefixes, *, rows, columns, **selection):
        from b12x.loader._csf_source import stage_csf_sources
        from ignition.preparation.tensor_metadata import record_numeric_tensor

        encoding = "csf_32x2_v1" if f"{prefixes[0]}.csf32_bases" in self._sources else "csf"
        suffixes = (tuple(f"csf32_{field}" for field in
                    ("bases", "codes", "page_bases", "offsets", "exceptions"))
                    if encoding == "csf_32x2_v1" else ("nvfp4_csf_fixed", "nvfp4_csf_exceptions"))
        result = stage_csf_sources(
            [tuple(self._sources[f"{prefix}.{suffix}"] for suffix in suffixes) for prefix in prefixes],
            rows=rows, columns=columns, quantization="nvfp4", device=self.device,
            encoding=encoding, copy=self._session, **selection)
        for tensor in (*result.components, result.starts):
            record_numeric_tensor(tensor)
        return result


def _aligned_range(selected, extent, alignment):
    start, end = (0, extent) if selected is None else selected
    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= extent
            or start % alignment or end % alignment):
        raise ValueError("weight partition must be aligned, nonempty and in range")
    return start, end

