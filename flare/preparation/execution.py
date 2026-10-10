"""Typed graph I/O metadata for the native two-frame execution ABI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from flare.components.capacity import BATCH_CAPACITIES, request_capacity

from .graphs import PreparedGraphBundle


@dataclass(frozen=True)
class GraphSignature:
    kind: str
    batch_capacity: int
    token_capacity: int
    inputs: tuple[str, ...] = ()
    sampling_mode: str = "general"

    def __post_init__(self):
        if self.kind not in ("init", "restore", "snapshot", "prefill", "decode", "resume", "vision", "speculative"):
            raise ValueError("unknown prepared graph operation")
        if self.sampling_mode != "general":
            raise ValueError("graphs use the general stochastic sampling path")
        if any(type(value) is not int or value <= 0 for value in (self.batch_capacity, self.token_capacity)):
            raise ValueError("graph capacities must be positive integers")
        if self.kind == "speculative" and (
                self.batch_capacity not in BATCH_CAPACITIES or not 1 <= self.token_capacity <= 8):
            raise ValueError("speculative graphs require power-of-two batch capacities through 64 and width 1 through 8")
        object.__setattr__(self, "inputs", tuple(self.inputs))
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("graph input names must be unique")
        if any(name in self.inputs for name in ("frame_index", "submission_sequence", "completion_records")):
            raise ValueError("reserved execution buffers are bound automatically")


class PreparedExecution(PreparedGraphBundle):
    """Retain immutable graph ownership and export the matching native ABI.

    Only graph I/O tensors belong in ``buffers``; packed weights, cache pools,
    graph scratch, and mapped-host storage belong in ``resources``. Native
    validation rejects overlapping I/O spans, incompatible graph capacities,
    and malformed reserved buffers before its worker becomes ready.
    """

    def __init__(self, *, signatures: Mapping[str, GraphSignature], max_active: int = 4, **kwargs):
        super().__init__(**kwargs)
        request_capacity(max_active)
        if signatures.keys() != self.graphs.keys():
            raise ValueError("every prepared graph requires exactly one signature")
        if any(not isinstance(value, GraphSignature) for value in signatures.values()):
            raise TypeError("prepared graph signatures must be GraphSignature objects")
        required = {"frame_index", "submission_sequence", "completion_records"}
        if not required <= self.buffers.keys():
            raise ValueError("execution requires frame index, submission sequence, and completion buffers")
        for signature in signatures.values():
            if not set(signature.inputs) <= self.buffers.keys():
                raise ValueError("graph signature references an unregistered input buffer")
        self.signatures = MappingProxyType(dict(signatures))
        self.max_active = max_active

    def manifest(self) -> dict[str, Any]:
        context, handles, spans = self._validate_resources()
        descriptors = []
        offsets = {}
        cursor = 0
        input_names = {"frame_index", "submission_sequence"}
        for signature in self.signatures.values():
            input_names.update(signature.inputs)
        for (name, tensor), (address, nbytes) in zip(self.buffers.items(), spans, strict=True):
            descriptors.append({"name": name, "address": address, "nbytes": nbytes,
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape), "strides": list(tensor.stride())})
            if name in input_names:
                cursor = (cursor + 7) // 8 * 8
                offsets[name] = (cursor, nbytes)
                cursor += nbytes
        graphs = []
        for (name, _), handle in zip(self.graphs.items(), handles, strict=True):
            signature = self.signatures[name]
            inputs = ("frame_index", "submission_sequence", *signature.inputs)
            graphs.append({"name": name, "kind": signature.kind, "handle": handle,
                "sampling_mode": signature.sampling_mode,
                "batch_capacity": signature.batch_capacity, "token_capacity": signature.token_capacity,
                "input_bindings": [{"buffer_name": field, "host_offset": offsets[field][0],
                                    "nbytes": offsets[field][1]} for field in inputs]})
        abi_version = 3 if self.buffers["completion_records"].shape[-1] == 24 else 2
        return {"abi_version": abi_version, "context": context, "device": self.device,
                "max_active": self.max_active, "input_frame_bytes": (cursor + 7) // 8 * 8,
                "buffers": descriptors, "graphs": graphs}

    def start_executor(self, *, peers=None):
        """Synchronize preparation and consume the bundle into the native executor."""
        import flare_helios
        import torch

        with self._handoff_lock:
            if self._consumed:
                raise RuntimeError("graph bundle has already been consumed by a native worker")
            manifest = self.manifest()
            torch.cuda.synchronize(self.device)
            arguments = {} if peers is None else {"peer_fds": [peer.fileno() for peer in peers]}
            executor = flare_helios.GraphExecutor(self, manifest, **arguments)
            self._consumed = True
            return executor

    def serve_rank(self, socket, *, rank, world):
        """Hand a peer to Rust until every rank drains; Python does no step work."""
        import flare_helios
        import torch

        with self._handoff_lock:
            if self._consumed:
                raise RuntimeError("graph bundle has already been consumed by a native worker")
            manifest = self.manifest()
            torch.cuda.synchronize(self.device)
            self._consumed = True
        flare_helios.serve_tensor_parallel_rank(self, manifest, socket.fileno(), rank, world)
