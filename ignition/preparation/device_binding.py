"""Bind single-device compiled forwards to a tensor-parallel rank's CUDA ordinal."""

import ast
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace

import torch

_DEVICE = ContextVar("flare_compiled_device", default=None)


@contextmanager
def bind_device(device):
    token = _DEVICE.set(torch.device(device))
    try:
        yield
    finally:
        _DEVICE.reset(token)


def restore_device(kind, index):
    target = _DEVICE.get()
    return target if kind == "cuda" and target is not None else torch.device(kind, index)


def restore_prepared_device(kind, state):
    value = object.__new__(kind)
    target = _DEVICE.get()
    if target is not None:
        state["ordinal"] = target.index
        if isinstance(state.get("uuid"), str):
            state["uuid"] = str(torch.cuda.get_device_properties(target).uuid).strip()
    value.__dict__.update(state)
    return value


def _device_source(source, original, target, kernel_keys=None):
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    offsets, offset = [], 0
    for line in lines:
        offsets.append(offset)
        offset += len(line.encode())
    replacements = []

    def replace(node, value):
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        replacements.append((start, end, value.encode()))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        if name in ("torch.cuda.set_device", "torch.cuda._DeviceGuard", "get_raw_stream"):
            if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant) or node.args[0].value != original:
                raise ValueError(f"unsupported compiled CUDA device binding: {ast.unparse(node)}")
            replace(node.args[0], str(target))
        elif name == "DeviceProperties":
            for keyword in node.keywords:
                if keyword.arg == "index":
                    if not isinstance(keyword.value, ast.Constant) or keyword.value.value != original:
                        raise ValueError("compiled kernel spans multiple CUDA devices")
                    replace(keyword.value, str(target))
        elif name == "async_compile.triton":
            from torch._inductor.async_compile import CompiledTritonKernels

            kernel = node.args[1]
            rebound = _device_source(kernel.value, original, target)
            if kernel_keys is not None:
                kernel_keys[CompiledTritonKernels.key(kernel.value)] = CompiledTritonKernels.key(rebound)
            replace(kernel, repr(rebound))
    data = source.encode()
    for start, end, value in sorted(replacements, reverse=True):
        data = data[:start] + value + data[end:]
    return data.decode()


def relocate_inductor(entry, original):
    target = _DEVICE.get()
    if target is None or target == original:
        return
    from torch._inductor.codecache import code_hash
    from torch._inductor.output_code import CompiledFxGraph
    from torch.utils._ordered_set import OrderedSet

    graph = entry.compiled_fw.result
    if entry.compiled_bw is not None or not isinstance(graph, CompiledFxGraph):
        raise ValueError("device binding requires a compiled inference forward")
    if graph.device_idxs != OrderedSet((original.index,)):
        raise ValueError("compiled forward must use one CUDA device")
    kernel_keys = {}
    graph.source_code = _device_source(graph.source_code, original.index, target.index, kernel_keys)
    graph.cache_key = code_hash(graph.source_code)
    graph.device_idxs = OrderedSet((target.index,))
    if bundle := graph._triton_bundle:
        autotuners = []
        for entry_kernel in bundle.static_autotuners:
            kernel = entry_kernel.kernel
            kernel.device_props = kernel.device_props._replace(index=target.index)
            kernel.triton_meta["device"] = target.index
            for result in kernel.compile_results:
                result.compile_meta["device"] = target.index
            autotuners.append(replace(entry_kernel, cache_key=kernel_keys[entry_kernel.cache_key]))
        graph._triton_bundle = replace(bundle,
            kernel_artifacts=[replace(item, device=target.index) for item in bundle.kernel_artifacts],
            static_autotuners=autotuners)
