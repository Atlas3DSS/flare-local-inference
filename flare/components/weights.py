"""Direct checkpoint transfers and CUDA compressed-scale preparation."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import sys
from time import monotonic
from typing import TYPE_CHECKING

from flare.preparation.checkpoint import IndexedCheckpoint

if TYPE_CHECKING:
    import torch
    from b12x.gemm.blockscaled import Weight


@dataclass(frozen=True)
class MLPWeights:
    gate_up: Weight | torch.Tensor
    down: Weight | torch.Tensor


class DirectWeightLoader:
    """Own b12x tensor views until direct transfers finish.

    Recipe subclasses select tensors and validate model-specific quantization.
    Source views may partition weight rows or columns without CPU weight copies.
    Compressed scale payloads are converted on CUDA one layer at a time.
    """

    def __init__(
        self, manifest: IndexedCheckpoint, device: torch.device | str,
        *, prefixes: Sequence[str] | None = None,
        show_progress: bool = False,
    ):
        self.manifest = manifest
        self.device = device
        self.prefixes = None if prefixes is None else tuple(prefixes)
        if self.prefixes is not None and not self.prefixes:
            raise ValueError("prefixes must be nonempty when supplied")
        self.show_progress = show_progress
        self._display = None
        self._load_stats = None
        self._session = None
        self._sources = {}
        self._stack = None

    def _selected_names(self):
        return {name for name in self.manifest.tensors
                if self.prefixes is None or name.startswith(self.prefixes)}

    @contextmanager
    def _loading_display(self):
        from b12x.loader import CheckpointDisplay

        with CheckpointDisplay() as display:
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
            if self.show_progress:
                self._load_stats = stack.enter_context(self._loading_display())
            # Small CPU scale planes do not amortize intra-op thread dispatch.
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

    def _require_open(self):
        if self._session is None:
            raise RuntimeError(f"weight loading requires an open {type(self).__name__} context")

    def _copy(self, destination, source):
        if not self._session(destination, source):
            raise RuntimeError("b12x rejected the direct weight transfer")
        if self._display is not None:
            self._display.update(self._session.payload_bytes)

    def load_tensor(self, name: str, *, dtype: torch.dtype | None = None, extent=None) -> torch.Tensor:
        """Transfer a selected resident weight or declared scalar to the GPU."""
        target = self._queue_tensor(name, dtype=dtype, extent=extent)
        self._session.flush()
        return target

    def load_tensors(self, names: Sequence[str]) -> tuple[torch.Tensor, ...]:
        """Transfer independent weights together, returning ready GPU tensors."""
        self._require_open()
        targets = tuple(self._queue_tensor(name) for name in names)
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

    def load_mxfp8_group(self, prefixes, *, row_ranges=None, column_range=None):
        """Pack selected MXFP8 projection rows directly into one GPU matrix."""
        import torch
        from b12x.gemm.blockscaled import pack_weight

        self._require_open()
        prefixes = tuple(prefixes)
        if not prefixes:
            raise ValueError("MXFP8 projection group must be nonempty")
        if row_ranges is None:
            row_ranges = (None,) * len(prefixes)
        weights, block_scales = [], []
        for prefix, row_range in zip(prefixes, row_ranges, strict=True):
            weight = self._sources[f"{prefix}.weight"]
            scale = self._sources[f"{prefix}.weight_scale"]
            if weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn:
                raise ValueError(f"{prefix}: MXFP8 weight must be an FP8 matrix")
            rows, columns = weight.shape
            r0, r1 = _aligned_range(row_range, rows, 1)
            c0, c1 = _aligned_range(column_range, columns, 32)
            if (scale.ndim != 2 or scale.element_size() != 1
                    or scale.shape[0] < rows or scale.shape[1] < columns // 32):
                raise ValueError(f"{prefix}: MXFP8 block scale geometry mismatch")
            weights.append(weight[r0:r1, c0:c1])
            block_scales.append(scale[r0:r1, c0 // 32:c1 // 32].view(torch.uint8))
        columns = weights[0].shape[1]
        if any(weight.shape[1] != columns for weight in weights):
            raise ValueError("packed MXFP8 projections must share input width")
        rows = sum(weight.shape[0] for weight in weights)
        values = torch.empty((rows, columns), dtype=torch.float8_e4m3fn, device=self.device)
        scales = torch.empty((rows, columns // 32), dtype=torch.uint8, device=self.device)
        offset = 0
        for weight, scale in zip(weights, block_scales, strict=True):
            end = offset + weight.shape[0]
            self._copy(values[offset:end], weight)
            self._copy(scales[offset:end], scale)
            offset = end
        self._session.flush()
        return pack_weight(values, scales, recipe="mxfp8")

    def load_nvfp4_experts(
        self, prefixes, *, hidden_size, intermediate_size, scale_vectors,
        intermediate_range=None, scale_scratch=None, swiglu_limit=None,
    ):
        """Prepare routed A4 experts from packed rank-local checkpoint views.

        ``scale_vectors`` supplies up/gate and down global multipliers followed
        by their activation reciprocal scales. Gate/up scalar compatibility is
        the recipe's responsibility. Scale scratch can be shared by serialized
        layers; both buffers remain caller-owned through replay.
        """
        import torch
        from b12x.loader._nvfp4_csf import queue_nvfp4_csf
        from b12x.moe import fused_moe as moe

        self._require_open()
        prefixes = tuple(prefixes)
        if not prefixes or hidden_size <= 0 or hidden_size % 128:
            raise ValueError("NVFP4 experts require nonempty prefixes and 128-aligned hidden width")
        first, last = _aligned_range(intermediate_range, intermediate_size, 128)
        local = last - first
        count = len(prefixes)
        shapes = ((count, 2 * local, hidden_size // 16), (count, hidden_size, local // 16))
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
        transfers = []
        for projections, rows, columns, row_range, column_range in (
            (("up_proj", "gate_proj"), intermediate_size, hidden_size // 16, (first, last), None),
            (("down_proj",), hidden_size, intermediate_size // 16, None, (first // 16, last // 16)),
        ):
            names = [f"{prefix}.{projection}.weight_scale.nvfp4_csf_"
                     for prefix in prefixes for projection in projections]
            transfers.append(queue_nvfp4_csf(
                self._session, [self._sources[name + "fixed"] for name in names],
                [self._sources[name + "exceptions"] for name in names],
                rows=rows, columns=columns, join=len(projections),
                row_range=row_range, column_range=column_range,
            ))
        self._session.flush()
        planes = tuple(transfer.convert() for transfer in transfers)
        del transfers
        if self._display is not None:
            self._display.update(self._session.payload_bytes)
        plan = moe.plan_weights(
            source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
            activation=moe.ActivationSpec(mode="a4", nonlinearity="silu", io_dtype=torch.bfloat16,
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
            packed=packed, w13_scales=planes[0], w2_scales=planes[1]))


def _aligned_range(selected, extent, alignment):
    start, end = (0, extent) if selected is None else selected
    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= extent
            or start % alignment or end % alignment):
        raise ValueError("weight partition must be aligned, nonempty and in range")
    return start, end

