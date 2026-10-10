"""Direct checkpoint transfers into final GPU weights with bounded CPU scale staging."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack
from typing import TYPE_CHECKING

from flare.preparation.checkpoint import IndexedCheckpoint

if TYPE_CHECKING:
    import torch


def join_nvfp4_csf(up_fixed, up_exceptions, gate_fixed, gate_exceptions, *, rows, columns):
    import torch

    expected_shape = (rows // 16, 16 * (1 + columns // 2))
    for fixed, exceptions in ((up_fixed, up_exceptions), (gate_fixed, gate_exceptions)):
        if fixed.device.type != "cpu" or fixed.dtype != torch.uint8 or tuple(fixed.shape) != expected_shape:
            raise ValueError("CSF fixed plane has the wrong shape, dtype, or device")
        if exceptions.device.type != "cpu" or exceptions.dtype != torch.uint32 or exceptions.ndim != 1:
            raise ValueError("CSF exceptions must be a CPU uint32 vector")
        positions = exceptions.to(torch.int64) & 0xFFFFFF
        if positions.numel() and (positions[-1] >= rows * columns or (positions[1:] <= positions[:-1]).any()):
            raise ValueError("CSF exceptions must have unique, sorted, in-range positions")
    gate_words = gate_exceptions.to(torch.int64)
    gate_words = (gate_words & 0xFF000000) | ((gate_words & 0xFFFFFF) + rows * columns)
    return (
        torch.cat((up_fixed, gate_fixed)),
        torch.cat((up_exceptions.to(torch.int64), gate_words)).to(torch.uint32),
    )


class DirectWeightLoader:
    """Own b12x tensor views until direct transfers finish.

    Recipe subclasses select tensors and validate model-specific quantization.
    Source views may partition weight rows or columns without CPU weight copies.
    Compressed scale payloads alone use bounded CPU staging.
    """

    def __init__(
        self, manifest: IndexedCheckpoint, device: torch.device | str,
        *, prefixes: Sequence[str] | None = None, csf_byte_budget: int = 128 << 20,
    ):
        if csf_byte_budget <= 0:
            raise ValueError("csf_byte_budget must be positive")
        self.manifest = manifest
        self.device = device
        self.prefixes = None if prefixes is None else tuple(prefixes)
        if self.prefixes is not None and not self.prefixes:
            raise ValueError("prefixes must be nonempty when supplied")
        self.csf_byte_budget = csf_byte_budget
        self._session = None
        self._sources = {}
        self._stack = None
        self._csf_files = {}

    def _selected_names(self):
        return {name for name in self.manifest.tensors
                if "nvfp4_csf_" not in name
                and (self.prefixes is None or name.startswith(self.prefixes))}

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
            self._session = stack.enter_context(DirectWeightSession(device=self.device.index))
            selected = self._selected_names()
            shards = {self.manifest.tensors[name].shard for name in selected}
            # b12x validates source view mutations through tensor version counters.
            with torch.inference_mode(False):
                self._sources = dict(self._session.weights(
                    tuple(self.manifest.snapshot / shard for shard in sorted(shards)),
                    index_path=self.manifest.index_path,
                    skip=lambda name: name not in selected,
                    needs_values=lambda entry: entry.name.endswith((".weight_scale_2", ".input_scale")),
                ))
            if self._sources.keys() != selected:
                raise ValueError("direct loader tensor coverage differs from the inspected checkpoint")
            self._stack = stack
            return self
        except BaseException:
            stack.close()
            self._session = None
            self._sources.clear()
            raise

    def __exit__(self, *exc):
        try:
            return self._stack.__exit__(*exc)
        finally:
            self._stack = None
            self._session = None
            self._sources.clear()
            self._csf_files.clear()

    def _require_open(self):
        if self._session is None:
            raise RuntimeError(f"weight loading requires an open {type(self).__name__} context")

    def _copy(self, destination, source):
        if not self._session(destination, source):
            raise RuntimeError("b12x rejected the direct weight transfer")

    def load_tensor(self, name: str, *, dtype: torch.dtype | None = None, extent=None) -> torch.Tensor:
        """Transfer a selected resident weight or declared scalar to the GPU."""
        import torch

        self._require_open()
        source = self._sources[name]
        if extent is not None:
            source = source[extent]
        target = torch.empty(source.shape, dtype=source.dtype if dtype is None else dtype, device=self.device)
        self._copy(target, source)
        self._session.flush()
        return target

    def _cpu_csf(self, name):
        from safetensors import safe_open

        self._require_open()

        metadata = self.manifest.tensors[name]
        if "nvfp4_csf_" not in name:
            raise ValueError("CPU payload reads are restricted to compressed scale planes")
        if metadata.shard not in self._csf_files:
            self._csf_files[metadata.shard] = self._stack.enter_context(safe_open(
                str(self.manifest.snapshot / metadata.shard), framework="pt", device="cpu",
            ))
        return self._csf_files[metadata.shard].get_tensor(name).clone()

