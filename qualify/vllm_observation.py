"""Read-only worker RPC metadata for the pinned vLLM serving oracle.

No model execution, plan resolution, tensor readback, or CUDA synchronization
occurs here. Installed b12x prepared payloads are inspected directly so the
receipt distinguishes declared precision from selected execution precision.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
import hashlib
import math
import os
from pathlib import Path
import sys


def _typename(value):
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _tensor_metadata(value):
    result = {
        "type": _typename(value), "dtype": str(value.dtype),
        "device": str(value.device), "shape": list(value.shape),
        "layout": str(value.layout), "requires_grad": value.requires_grad,
    }
    if str(value.layout) == "torch.strided":
        result["stride"] = list(value.stride())
        result["storage_offset"] = value.storage_offset()
    return result


def _metadata(value, *, _seen=None, _depth=0):
    """Serialize configuration without copying tensors or walking opaque owners."""
    import torch

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite_float": str(value)}
    if isinstance(value, Enum):
        return {"type": _typename(value), "name": value.name,
                "value": _metadata(value.value)}
    if isinstance(value, (torch.dtype, torch.device, Path)):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _tensor_metadata(value)
    if _depth > 24:
        raise ValueError(f"observation metadata nesting exceeded: {_typename(value)}")
    seen = set() if _seen is None else _seen
    if id(value) in seen:
        return {"type": _typename(value), "recursive_reference": True}
    seen.add(id(value))
    try:
        visit = lambda child: _metadata(child, _seen=seen, _depth=_depth + 1)
        if isinstance(value, Mapping):
            result = {}
            for key, child in value.items():
                name = str(key)
                if name in result:
                    raise ValueError(f"metadata key collision: {name}")
                result[name] = visit(child)
            return result
        if isinstance(value, (tuple, list, set, frozenset)):
            result = [visit(child) for child in value]
            if isinstance(value, (set, frozenset)):
                import json
                result.sort(key=lambda child: json.dumps(child, sort_keys=True))
            return result
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {field.name: visit(getattr(value, field.name))
                    for field in dataclasses.fields(value)}
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return {"type": _typename(value), "fields": visit(to_dict())}
        if callable(value):
            return {"callable": getattr(value, "__module__", "") + "."
                    + getattr(value, "__qualname__", type(value).__qualname__)}
        return {"type": _typename(value)}
    finally:
        seen.remove(id(value))


class _Plans:
    def __init__(self):
        self.records = {}
        self._objects = {}

    @staticmethod
    def is_plan(value):
        return (hasattr(value, "_prepared") and hasattr(value, "_handle")
                and hasattr(value, "component_id"))

    def add(self, plan):
        key = str(plan.handle)
        if key in self.records:
            if self._objects[key] is not plan:
                raise ValueError(f"distinct plans share handle {key}")
            return key
        self._objects[key] = plan
        record = {"type": _typename(plan), "component_id": plan.component_id}
        self.records[key] = record
        for name in ("query", "invocation", "override", "dependencies", "shared",
                     "capacity_metadata", "composite_semantic_version"):
            if hasattr(plan, name):
                record[name] = _metadata(getattr(plan, name))
        if hasattr(plan, "variants"):
            record["variants"] = {str(rows): self.add(child)
                                  for rows, child in plan.variants.items()}
        payload = plan._prepared
        record["prepared"] = payload is not None and not payload.closed
        if payload is not None:
            record["payload_closed"] = payload.closed
            record["selection"] = _metadata(payload.selection)
            record["scratch_specs"] = _metadata(payload.scratch)
            record["device"] = _metadata(payload.device)
            record["materialized_state"] = self.state(payload.state)
            if payload.variants is not None:
                record["prepared_variants"] = {
                    str(rows): self.add(child) for rows, child in payload.variants.items()
                }
        return key

    def state(self, state):
        if state is None:
            return None
        result = {"type": _typename(state)}
        # These are stored fields, not dispatch or materialization methods.
        fields = vars(state) if hasattr(state, "__dict__") else {}
        for name in ("query", "config", "layout", "mode", "activation_mode",
                     "dtype", "state_dtype", "intermediate_dtype", "required_workspace"):
            if name in fields:
                result[name] = _metadata(fields[name])
        # Blockscaled regime states retain exact-M and fallback execution states.
        for name in ("capacity", "a16_capacity"):
            if name in fields:
                result[name] = self.state(fields[name])
        if "exact" in fields:
            result["exact"] = {str(rows): self.state(child)
                               for rows, child in fields["exact"].items()}
        return result

    def references(self, value):
        if self.is_plan(value):
            return self.add(value)
        if isinstance(value, Mapping):
            refs = {str(key): self.references(child) for key, child in value.items()}
            return {key: ref for key, ref in refs.items() if ref is not None} or None
        if isinstance(value, (tuple, list)):
            refs = {str(index): self.references(child) for index, child in enumerate(value)}
            return {key: ref for key, ref in refs.items() if ref is not None} or None
        return None


def _fingerprint(path):
    with open(path, "rb") as source:
        before = os.fstat(source.fileno())
        digest = hashlib.file_digest(source, "sha256").hexdigest()
        after = os.fstat(source.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in fields):
        raise RuntimeError(f"file changed during observation: {path}")
    return {"sha256": digest, "size_bytes": before.st_size, "inode": before.st_ino,
            "device": f"{os.major(before.st_dev):x}:{os.minor(before.st_dev):x}"}


def _libraries():
    origins = {}
    for name, module in tuple(sys.modules.items()):
        if name.split(".", 1)[0] not in {"vllm", "flashinfer", "b12x", "torch", "triton", "cutlass", "cuda"}:
            continue
        path = getattr(module, "__file__", None)
        if path:
            origins.setdefault(str(Path(path).resolve()), []).append(name)
    source_files = []
    for path, modules in sorted(origins.items()):
        entry = {"path": path, "modules": sorted(modules)}
        try:
            entry.update(_fingerprint(path))
        except OSError as error:
            entry["error"] = str(error)
        source_files.append(entry)

    mappings = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(None, 5)
        if len(parts) != 6 or not parts[5].startswith("/"):
            continue
        address, permissions, offset, device, inode, path = parts
        if ".so" not in Path(path).name:
            continue
        key = (path, device, int(inode))
        entry = mappings.setdefault(key, {"path": path, "mapped_device": device,
                                         "mapped_inode": int(inode), "segments": []})
        entry["segments"].append({"address": address, "permissions": permissions,
                                  "offset": offset})
    for entry in mappings.values():
        try:
            entry["file"] = _fingerprint(entry["path"])
            stat = entry["file"]
            entry["file_matches_mapping"] = (
                stat["inode"] == entry["mapped_inode"]
                and tuple(int(part, 16) for part in stat["device"].split(":"))
                == tuple(int(part, 16) for part in entry["mapped_device"].split(":"))
            )
        except OSError as error:
            entry["file_matches_mapping"] = False
            entry["error"] = str(error)
    binaries = sorted(mappings.values(), key=lambda entry: entry["path"])
    return {"loaded_module_files": source_files, "mapped_native_binaries": binaries,
            "sampling_binary_paths": [entry["path"] for entry in binaries
                                      if Path(entry["path"]).name == "sampling.so"]}


def observe_worker(worker):
    """Return JSON metadata through ``LLM.collective_rpc(observe_worker)``."""
    model = worker.get_model()
    runner = worker.model_runner
    session = getattr(worker, "_b12x_session", None)
    if session is None:
        raise RuntimeError("worker has no b12x preparation session")
    plans = _Plans()
    session_plans = [plans.add(plan) for plan in session._plans]
    modules = {}
    for name, module in model.named_modules():
        entry = {"type": _typename(module)}
        references = {}
        for attr, value in vars(module).items():
            if "plan" in attr:
                refs = plans.references(value)
                if refs is not None:
                    references[attr] = refs
        if references:
            entry["plans"] = references
        linear = getattr(module, "b12x_linear", None)
        if linear is not None:
            entry["b12x_linear"] = {
                "type": _typename(linear),
                **{field: _metadata(getattr(linear, field)) for field in
                   ("recipe", "activation_mode", "a16_max_tokens", "a4_prefill_enabled", "_plan_key")
                   if hasattr(linear, field)},
                "plans": {field: plans.references(getattr(linear, field, None))
                          for field in ("plan", "a4_plan")},
            }
        for field in ("b12x_activation_mode", "gdn_prefill_backend", "gdn_decode_kernel",
                      "_use_hc_down_silu", "use_combine"):
            if hasattr(module, field):
                entry[field] = _metadata(getattr(module, field))
        entry["parameters"] = {key: _tensor_metadata(value)
                               for key, value in module.named_parameters(recurse=False)}
        entry["buffers"] = {key: _tensor_metadata(value)
                            for key, value in module.named_buffers(recurse=False)}
        modules[name] = entry
    return {
        "schema_version": 1, "pid": os.getpid(),
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "worker_class": _typename(worker), "runner_class": _typename(runner),
        "model_class": _typename(model),
        "model_state_class": _typename(runner.model_state) if hasattr(runner, "model_state") else None,
        "sampler_class": _typename(runner.sampler) if getattr(runner, "sampler", None) is not None else None,
        "config": _metadata(worker.vllm_config),
        "session": {"class": _typename(session), "state": session.state,
                    "namespace": _metadata(session.namespace), "autotune": session.autotune,
                    "plans": session_plans},
        "plans": plans.records, "modules": modules, "libraries": _libraries(),
    }
