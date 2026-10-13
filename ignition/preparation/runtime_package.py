"""Export CUDA graph resources with explicit allocation-relative bindings."""

from __future__ import annotations

import ctypes as C
from dataclasses import asdict, dataclass, replace
import os
from pathlib import Path
import shutil
import tempfile


@dataclass(frozen=True)
class Reference:
    allocation: str
    offset: int = 0


@dataclass(frozen=True)
class Relocation:
    offset: int
    target: Reference
    kind: str = "pointer"


@dataclass(frozen=True)
class Argument:
    """Parameter bytes and explicit bindings needed when allocations relocate."""

    size: int
    relocations: tuple[Relocation, ...] = ()
    numeric: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class FileSource:
    """Immutable tensor bytes appended directly from a checkpoint file."""

    path: Path
    offset: int
    nbytes: int

    def __post_init__(self):
        if self.offset < 0 or self.nbytes <= 0 or self.offset + self.nbytes > self.path.stat().st_size:
            raise ValueError("tensor source must name a nonempty range within its checkpoint file")


@dataclass(frozen=True)
class Allocation:
    name: str
    address: int
    nbytes: int
    alignment: int = 256
    initialize: str = "snapshot"
    relocations: tuple[Relocation, ...] = ()
    preserve_address: bool = False
    memory: str = "device"
    snapshot_spans: tuple[tuple[int, int], ...] | None = None
    sizing: dict | None = None


@dataclass(frozen=True)
class Kernel:
    function: int
    module: str
    arguments: tuple[Argument, ...]


class _KernelParams(C.Structure):
    _fields_ = [("func", C.c_void_p), ("grid", C.c_uint * 3), ("block", C.c_uint * 3),
                ("shared", C.c_uint), ("params", C.POINTER(C.c_void_p)),
                ("extra", C.POINTER(C.c_void_p)), ("kern", C.c_void_p), ("ctx", C.c_void_p)]


class _MemsetParams(C.Structure):
    _fields_ = [("dst", C.c_uint64), ("pitch", C.c_size_t), ("value", C.c_uint),
                ("element_size", C.c_uint), ("width", C.c_size_t), ("height", C.c_size_t)]


class _CopyEndpoint(C.Structure):
    _fields_ = [("x", C.c_size_t), ("y", C.c_size_t), ("z", C.c_size_t), ("lod", C.c_size_t),
                ("memory_type", C.c_int), ("host", C.c_void_p), ("device", C.c_uint64),
                ("array", C.c_void_p), ("reserved", C.c_void_p), ("pitch", C.c_size_t),
                ("height", C.c_size_t)]


class _CopyParams(C.Structure):
    _fields_ = [("source", _CopyEndpoint), ("destination", _CopyEndpoint),
                ("width", C.c_size_t), ("height", C.c_size_t), ("depth", C.c_size_t)]


def _kernel_argument_data(params, layout):
    if params.params:
        if params.extra:
            raise ValueError("kernel specifies both parameter-array and packed arguments")
        if any(not params.params[index] for index in range(len(layout))):
            raise ValueError("kernel parameter array contains a null argument")
        return [C.string_at(params.params[index], size) for index, (_, size) in enumerate(layout)]
    if not params.extra:
        if layout:
            raise ValueError("kernel has parameters but no launch arguments")
        return []
    options = {}
    for index in range(0, 5, 2):
        tag = params.extra[index]
        if tag is None:
            break
        if tag not in (1, 2) or tag in options:
            raise ValueError("invalid packed kernel argument options")
        options[tag] = params.extra[index + 1]
    else:
        raise ValueError("packed kernel argument options lack a terminator")
    if not options.get(2):
        raise ValueError("packed kernel arguments lack a buffer size")
    nbytes = C.c_size_t.from_address(options[2]).value
    if nbytes and not options.get(1):
        raise ValueError("packed kernel arguments lack a buffer")
    if any(offset + size > nbytes for offset, size in layout):
        raise ValueError("kernel parameter extends beyond its packed argument buffer")
    return [C.string_at(options[1] + offset, size) for offset, size in layout]


class _Cuda:
    def __init__(self):
        self.lib = C.CDLL("libcuda.so.1")

    def call(self, name, *args):
        function = getattr(self.lib, name)
        function.restype = C.c_int
        status = function(*args)
        self.check(name, status)

    def check(self, name, status):
        if status:
            message = C.c_char_p()
            self.lib.cuGetErrorString(C.c_int(status), C.byref(message))
            raise RuntimeError(f"{name}: CUDA {status}: {message.value!r}")

    def download(self, address, nbytes):
        data = (C.c_ubyte * nbytes)()
        self.call("cuMemcpyDtoH_v2", data, C.c_uint64(address), C.c_size_t(nbytes))
        return bytes(data)


def _merge_ranges(ranges):
    merged = []
    for address, nbytes in sorted(ranges):
        if address <= 0 or nbytes <= 0:
            raise ValueError("storage ranges require a nonnull address and positive size")
        end = address + nbytes
        if merged and address <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((address, end))
    return tuple((start, end - start) for start, end in merged)


def _without_zero_ranges(allocation, zero_ranges):
    """Exclude disposable bytes without losing neighboring data in a shared segment."""
    if allocation.initialize == "zero":
        return allocation
    kept = []
    for offset, size in (allocation.snapshot_spans if allocation.snapshot_spans is not None
                         else ((0, allocation.nbytes),)):
        cursor, end = allocation.address + offset, allocation.address + offset + size
        for address, nbytes in zero_ranges:
            if address >= end:
                break
            if address + nbytes <= cursor:
                continue
            if cursor < address:
                kept.append((cursor - allocation.address, address - cursor))
            cursor = max(cursor, min(end, address + nbytes))
        if cursor < end:
            kept.append((cursor - allocation.address, end - cursor))
    return replace(allocation, initialize="snapshot" if kept else "zero",
                   snapshot_spans=tuple(kept) if kept else None)


# Payload bytes written between write-backs. A package holds the model's full weights; on a GPU that
# allocates from system memory, page cache filling with them would take the memory the model occupies.
_WRITE_BACK_BYTES = 256 << 20


def _write_back(output):
    """Write buffered payload to disk and drop it from the page cache."""
    output.flush()
    os.fdatasync(output.fileno())
    os.posix_fadvise(output.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def _write_allocation(cuda, allocation, path, *, visit=None):
    """Pack nonzero ranges into a file; the loader zeroes omitted bytes."""
    import numpy as np

    if allocation.initialize == "zero":
        return None, []
    stored = []
    block = 64 << 10
    with path.open("wb") as output:
        for start, nbytes in (allocation.snapshot_spans if allocation.snapshot_spans is not None
                              else ((0, allocation.nbytes),)):
            for offset in range(start, start + nbytes, 8 << 20):
                if offset > start and (offset - start) % _WRITE_BACK_BYTES == 0:
                    _write_back(output)
                size = min(8 << 20, start + nbytes - offset)
                data = (C.string_at(allocation.address + offset, size) if allocation.memory == "mapped_host"
                        else cuda.download(allocation.address + offset, size))
                if visit is not None:
                    visit(allocation, offset, data)
                full = size // block * block
                occupied = np.frombuffer(data, dtype=np.uint8, count=full).reshape(-1, block).any(axis=1)
                indices = np.flatnonzero(occupied).tolist()
                if full < size and any(data[full:]):
                    indices.append(full // block)
                for index in indices:
                    begin, length = offset + index * block, min(block, size - index * block)
                    output.write(memoryview(data)[index * block:index * block + length])
                    if stored and stored[-1][0] + stored[-1][1] == begin:
                        stored[-1][1] += length
                    else:
                        stored.append([begin, length])
        _write_back(output)
    if not stored:
        path.unlink()
        return None, []
    return path.name, stored


class RuntimePackage:
    """Preparation-time registry for a trusted native CUDA execution package.

    Relocatable resources require every pointer-bearing field to be declared.
    Recorded-address resources retain opaque argument and storage bytes. Unknown
    kernels and node types fail export. Resource owners remain alive until export
    completes; graph templates must have been captured with keep_graph=True.
    """

    def __init__(self, *, allocations, modules, kernels, owners=(), reservations=(), reachability=None, events=None,
                 transports=()):
        self.allocations = tuple(allocations)
        self.transports = tuple(transports)
        self.modules = dict(modules)
        kernels = tuple(kernels)
        self.kernels = {kernel.function: kernel for kernel in kernels}
        if len(self.kernels) != len(kernels):
            raise ValueError("kernel function registrations must be unique")
        self._function_indices = {function: index for index, function in enumerate(self.kernels)}
        self.owners = tuple(owners)
        self.reservations = tuple(reservations)
        self.reachability = reachability
        self.events = dict(events or {})
        if any(not name or handle <= 0 for name, handle in self.events.items()) or len(set(self.events.values())) != len(self.events):
            raise ValueError("external events require unique names and live handles")
        self._by_name = {allocation.name: allocation for allocation in self.allocations}
        if len(self._by_name) != len(self.allocations):
            raise ValueError("allocation names must be unique")
        spans = sorted((a.address, a.address + a.nbytes) for a in self.allocations)
        if any(end > following for (_, end), (following, _) in zip(spans, spans[1:])):
            raise ValueError("register shared backing storage once, with offset references for views")
        for allocation in self.allocations:
            if not allocation.name or allocation.address <= 0 or allocation.nbytes <= 0:
                raise ValueError("allocations require a name, nonnull address, and positive size")
            alignment = allocation.alignment
            if alignment <= 0 or alignment & (alignment - 1) or allocation.address % alignment:
                raise ValueError("allocation alignment must be a satisfied power of two")
            if allocation.initialize not in ("snapshot", "zero"):
                raise ValueError("allocation initializer must be snapshot or zero")
            if allocation.memory not in ("device", "mapped_host", "peer"):
                raise ValueError("unsupported allocation memory type")
            if allocation.memory == "peer" and (not allocation.preserve_address or allocation.initialize != "zero"
                                                or allocation.relocations):
                raise ValueError("peer allocations name another rank's storage at its recorded address")
            if allocation.memory == "mapped_host" and not allocation.preserve_address:
                raise ValueError("mapped host allocations require a recorded address")
            if allocation.snapshot_spans is not None:
                end = 0
                for offset, nbytes in allocation.snapshot_spans:
                    if offset < end or nbytes <= 0 or offset + nbytes > allocation.nbytes:
                        raise ValueError("snapshot spans must be ordered, disjoint, and within the allocation")
                    end = offset + nbytes
                if allocation.initialize != "snapshot":
                    raise ValueError("snapshot spans require a snapshot initializer")
            self._validate_relocations(allocation.nbytes, allocation.relocations)
        for kernel in self.kernels.values():
            if kernel.function <= 0 or kernel.module not in self.modules:
                raise ValueError("kernel requires a live function and registered module image")
            for argument in kernel.arguments:
                if argument.size <= 0:
                    raise ValueError("kernel arguments must have positive sizes")
                self._validate_relocations(argument.size, argument.relocations)
                end = 0
                for start, stop in argument.numeric:
                    if start < end or stop <= start or stop > argument.size:
                        raise ValueError("numeric argument fields must be ordered, disjoint, and within the parameter")
                    if any(start < relocation.offset + (8 if relocation.kind == "pointer" else 128)
                           and relocation.offset < stop for relocation in argument.relocations):
                        raise ValueError("numeric argument fields overlap a pointer binding")
                    end = stop
        if not self.allocations or not self.kernels:
            raise ValueError("runtime package requires allocations and kernels")
        if any(not isinstance(image, bytes) or not image.startswith(b"\x7fELF")
               for image in self.modules.values()):
            raise ValueError("module images must be compiled cubins")

    def _address(self, reference):
        allocation = self._by_name.get(reference.allocation)
        if allocation is None or not 0 <= reference.offset < allocation.nbytes:
            raise ValueError(f"invalid allocation reference {reference}")
        return allocation.address + reference.offset

    def _reference(self, address, nbytes=1):
        matches = [Reference(a.name, address - a.address) for a in self.allocations
                   if a.address <= address and address + nbytes <= a.address + a.nbytes]
        if len(matches) != 1:
            raise ValueError(f"unregistered CUDA span {address:#x}+{nbytes}")
        return asdict(matches[0])

    def _validate_relocations(self, size, relocations):
        end = 0
        for relocation in sorted(relocations, key=lambda value: value.offset):
            width = {"pointer": 8, "tensor_map": 128}.get(relocation.kind)
            if width is None or relocation.offset < end or relocation.offset + width > size:
                raise ValueError("invalid or overlapping relocation fields")
            self._address(relocation.target)
            end = relocation.offset + width

    def _check_bindings(self, cuda, data, relocations):
        for relocation in relocations:
            address = self._address(relocation.target)
            offset = relocation.offset
            if relocation.kind == "pointer":
                if int.from_bytes(data[offset:offset + 8], "little") != address:
                    raise ValueError(f"pointer binding does not match {relocation.target}")
            else:
                original = data[offset:offset + 128]
                storage = C.create_string_buffer(128 + 63)
                aligned = (C.addressof(storage) + 63) & ~63
                C.memmove(aligned, original, 128)
                cuda.call("cuTensorMapReplaceAddress", C.c_void_p(aligned), C.c_void_p(address))
                if C.string_at(aligned, 128) != original:
                    raise ValueError(f"tensor-map binding does not match {relocation.target}")

    def write(self, path, graphs, *, execution=None, server=None, progress=None, package_data=None):
        """Synchronize and publish an indexed model file without overwriting."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        if not graphs or any(not name for name in graphs):
            raise ValueError("named graph templates are required")
        cuda = _Cuda()
        cuda.call("cuCtxSynchronize")
        device = C.c_int()
        cuda.call("cuCtxGetDevice", C.byref(device))
        architecture = []
        for attribute in (75, 76):
            value = C.c_int()
            cuda.call("cuDeviceGetAttribute", C.byref(value), C.c_int(attribute), device)
            architecture.append(value.value)
        flags, cache = C.c_uint(), C.c_int()
        cuda.call("cuCtxGetFlags", C.byref(flags))
        cuda.call("cuCtxGetCacheConfig", C.byref(cache))
        limits = []
        for limit in range(7):
            value = C.c_size_t()
            status = cuda.lib.cuCtxGetLimit(C.byref(value), C.c_int(limit))
            if status == 215:
                continue
            if status:
                raise RuntimeError(f"cuCtxGetLimit({limit}) failed with CUDA status {status}")
            limits.append([limit, value.value])
        functions = []
        for kernel in self.kernels.values():
            name = C.c_char_p()
            cuda.call("cuFuncGetName", C.byref(name), C.c_void_p(kernel.function))
            attributes = []
            for attribute in (8, 9, 14, 15):
                value = C.c_int()
                cuda.call("cuFuncGetAttribute", C.byref(value), C.c_int(attribute), C.c_void_p(kernel.function))
                attributes.append([attribute, value.value])
            functions.append({"module": kernel.module, "symbol": name.value.decode(),
                              "parameter_sizes": [arg.size for arg in kernel.arguments],
                              "attributes": attributes})
        path.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
        try:
            from .package_proto import bundle, write_graph
            from .package_file import align, publish
            from .runtime_package_pb2 import ModelIndex

            serialized_graphs = []
            if self.reachability is not None and execution is not None:
                for buffer in execution["buffers"]:
                    self.reachability.reference(buffer["reference"])
            for index, (name, graph) in enumerate(graphs.items()):
                description = self._graph(cuda, name, int(graph.raw_cuda_graph()))
                if self.reachability is not None:
                    self.reachability.graph(description, tuple(self.kernels.values()))
                graph_flags = C.c_uint64()
                cuda.call("cuGraphExecGetFlags", C.c_void_p(graph.raw_cuda_graph_exec()), C.byref(graph_flags))
                description["flags"] = graph_flags.value
                serialized_graphs.append(write_graph(description, stage / f"graph-{index}.bin"))
                del description
            modules = []
            for index, (name, image) in enumerate(self.modules.items()):
                from .module_images import global_symbols, strip_debug, strip_global_source

                file = f"module-{index}.cubin"
                (stage / file).write_bytes(strip_debug(image))
                kernel = next(kernel for kernel in self.kernels.values() if kernel.module == name)
                module = C.c_void_p()
                cuda.call("cuFuncGetModule", C.byref(module), C.c_void_p(kernel.function))
                globals = []
                for symbol in global_symbols(image):
                    address, size = C.c_uint64(), C.c_size_t()
                    cuda.call("cuModuleGetGlobal_v2", C.byref(address), C.byref(size), module, C.c_char_p(symbol.encode()))
                    data_file = f"module-{index}-global-{len(globals)}.bin"
                    data = strip_global_source(symbol, cuda.download(address.value, size.value))
                    if self.reachability is not None:
                        self.reachability.blob(data)
                    (stage / data_file).write_bytes(data)
                    globals.append({"symbol": symbol, "file": data_file, "nbytes": size.value, "relocations": []})
                modules.append({"name": name, "file": file, "globals": globals})
            allocations = []
            selected = (self.allocations if self.reachability is None
                        else iter(self.reachability.take, None))
            for allocation in selected:
                index = len(allocations)
                if progress is not None and allocation.snapshot_spans is not None:
                    progress(f"Copy {allocation.name}: {allocation.address:#x}+{allocation.nbytes}; "
                             f"live spans {allocation.snapshot_spans}")
                file, data_spans = _write_allocation(cuda, allocation, stage / f"storage-{index}.bin",
                    visit=None if self.reachability is None else self.reachability.contents)
                for relocation in allocation.relocations:
                    width = 8 if relocation.kind == "pointer" else 128
                    data = cuda.download(allocation.address + relocation.offset, width)
                    self._check_bindings(cuda, data, [Relocation(0, relocation.target, relocation.kind)])
                if file is None and any(r.kind == "tensor_map" for r in allocation.relocations):
                    raise ValueError("tensor-map storage requires a snapshot initializer")
                allocations.append({"name": allocation.name, "nbytes": 0 if allocation.sizing else allocation.nbytes,
                    "alignment": allocation.alignment, "file": file,
                    "data_spans": data_spans,
                    "address": allocation.address if allocation.preserve_address else None,
                    "memory": allocation.memory,
                    "relocations": [asdict(r) for r in allocation.relocations], "sizing": allocation.sizing})
            if self.reachability is not None:
                omitted = [a for a in self.allocations if a.name not in self.reachability.reached]
                if progress is not None:
                    progress(f"Omitted {len(omitted)} unreachable allocations "
                             f"({sum(a.nbytes for a in omitted) / (1 << 20):.1f} MiB)")
                self.allocations = tuple(a for a in self.allocations if a.name in self.reachability.reached)
                self._by_name = {a.name: a for a in self.allocations}
                self.reservations = tuple(r for r in self.reservations if any(
                    a.memory == "device" and r["address"] <= a.address < r["address"] + r["nbytes"]
                    for a in self.allocations))
            document = {"architecture": architecture, "allocations": allocations,
                        "modules": modules, "functions": functions, "graphs": serialized_graphs,
                        "reservations": self.reservations, "execution": execution,
                        "context": {"flags": flags.value, "cache": cache.value, "limits": limits},
                        "events": list(self.events)}
            index = ModelIndex(version=1, ranks=[bundle(document)])
            for transport in self.transports:
                _roce_transport(index.ranks[0].roce, transport)
            asset_files = {}
            if server is not None:
                from google.protobuf.json_format import ParseDict

                ParseDict({k: v for k, v in server.items() if k != "assets"}, index.server)
                for name, asset in server["assets"].items():
                    source = Path(asset["path"])
                    destination = f"asset-{name}{source.suffix}"
                    shutil.copyfile(source, stage / destination)
                    asset_files[destination] = (name, asset["sha256"])
            payload, cursor = [], 0
            sources = sorted(stage.iterdir())
            storage_files = {allocation["file"] for allocation in allocations if allocation["file"]}

            def append_file(source):
                nonlocal cursor
                cursor = align(cursor)
                size = source.stat().st_size
                if source.name in asset_files:
                    name, digest = asset_files[source.name]
                    index.assets.add(name=name, sha256=digest, payload={"offset": cursor, "nbytes": size})
                else:
                    span = index.ranks[0].resources[source.name]
                    span.offset, span.nbytes = cursor, size
                payload.append((cursor, source, 0, size))
                cursor += size

            for source in sources:
                if source.name in storage_files:
                    append_file(source)
            if package_data is not None:
                cursor = package_data.append(index, payload, cursor)
            for source in sources:
                if source.name not in storage_files:
                    append_file(source)
            publish(path, index, payload)
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def _graph(self, cuda, name, graph):
        count = C.c_size_t()
        cuda.call("cuGraphGetNodes", C.c_void_p(graph), None, C.byref(count))
        handles = (C.c_void_p * count.value)()
        cuda.call("cuGraphGetNodes", C.c_void_p(graph), handles, C.byref(count))
        cupti = C.CDLL("libcupti.so")
        node_ids = {}
        for handle in handles:
            node_id = C.c_uint64()
            status = cupti.cuptiGetGraphNodeId(C.c_void_p(handle), C.byref(node_id))
            if status:
                raise RuntimeError(f"cuptiGetGraphNodeId: CUPTI {status}")
            node_ids[handle] = node_id.value
        # Preserve construction order, including independent branches.
        ordered = sorted(handles, key=node_ids.__getitem__)
        indices = {handle: index for index, handle in enumerate(ordered)}
        nodes = []
        for handle in ordered:
            kind = C.c_int()
            cuda.call("cuGraphNodeGetType", C.c_void_p(handle), C.byref(kind))
            if kind.value == 0:
                node = self._kernel(cuda, handle)
            elif kind.value == 1:
                params = _CopyParams()
                cuda.call("cuGraphMemcpyNodeGetParams", C.c_void_p(handle), C.byref(params))
                endpoints = (params.source, params.destination)
                # Unified endpoints (cudaMemcpyDefault) carry their address in the device field.
                if params.height != 1 or params.depth != 1 or any(
                        p.memory_type not in (2, 4) or p.y or p.z or p.lod
                        or p.memory_type == 4 and not self._is_device(cuda, p.device) for p in endpoints):
                    raise ValueError("only contiguous device-to-device graph copies are supported; got memory types "
                                     f"{[p.memory_type for p in endpoints]}")
                node = {"kind": "copy", "nbytes": params.width,
                        "source": self._reference(params.source.device + params.source.x, params.width),
                        "destination": self._reference(params.destination.device + params.destination.x, params.width)}
            elif kind.value == 2:
                params = _MemsetParams()
                cuda.call("cuGraphMemsetNodeGetParams", C.c_void_p(handle), C.byref(params))
                extent = (params.height - 1) * params.pitch + params.width * params.element_size
                node = {"kind": "memset", "destination": self._reference(params.dst, extent),
                        **{field: getattr(params, field) for field in
                           ("pitch", "value", "element_size", "width", "height")}}
            elif kind.value == 5:
                node = {"kind": "empty"}
            elif kind.value == 6:
                event = C.c_void_p()
                cuda.call("cuGraphEventWaitNodeGetEvent", C.c_void_p(handle), C.byref(event))
                event_name = next((name for name, value in self.events.items() if value == event.value), None)
                if event_name is None:
                    raise ValueError(f"graph {name}: undeclared external readiness event")
                node = {"kind": "event_wait", "event": event_name}
            else:
                raise ValueError(f"graph {name}: unsupported CUDA graph node type {kind.value}")
            nodes.append(node)
        count = C.c_size_t()
        cuda.call("cuGraphGetEdges_v2", C.c_void_p(graph), None, None, None, C.byref(count))
        if count.value == 0:
            return {"name": name, "nodes": nodes, "edges": []}
        sources, destinations = (C.c_void_p * count.value)(), (C.c_void_p * count.value)()
        data = ((C.c_ubyte * 8) * count.value)()
        cuda.call("cuGraphGetEdges_v2", C.c_void_p(graph), sources, destinations, data, C.byref(count))
        edges = [{"source": indices[a], "destination": indices[b], "data": list(d)}
                 for a, b, d in zip(sources, destinations, data, strict=True)]
        return {"name": name, "nodes": nodes, "edges": edges}

    @staticmethod
    def _is_device(cuda, address):
        memory = C.c_int()
        cuda.call("cuPointerGetAttribute", C.byref(memory), C.c_int(2), C.c_uint64(address))
        return memory.value == 2

    def _kernel(self, cuda, handle):
        params = _KernelParams()
        cuda.call("cuGraphKernelNodeGetParams_v2", C.c_void_p(handle), C.byref(params))
        function = C.c_void_p(params.func)
        if not function.value:
            cuda.call("cuKernelGetFunction", C.byref(function), C.c_void_p(params.kern))
        kernel = self.kernels.get(function.value)
        if kernel is None:
            name = C.c_char_p()
            cuda.call("cuFuncGetName", C.byref(name), function)
            raise ValueError(f"no explicit argument bindings for kernel {name.value!r}")
        layout = []
        for index, argument in enumerate(kernel.arguments):
            offset, size = C.c_size_t(), C.c_size_t()
            cuda.call("cuFuncGetParamInfo", function, C.c_size_t(index), C.byref(offset), C.byref(size))
            if size.value != argument.size:
                raise ValueError(f"kernel parameter {index} size is {size.value}, expected {argument.size}")
            layout.append((offset.value, size.value))
        arguments = []
        for argument, data in zip(kernel.arguments, _kernel_argument_data(params, layout), strict=True):
            self._check_bindings(cuda, data, argument.relocations)
            arguments.append({"data": list(data), "relocations": [asdict(r) for r in argument.relocations]})
        offset, size = C.c_size_t(), C.c_size_t()
        status = cuda.lib.cuFuncGetParamInfo(function, C.c_size_t(len(arguments)), C.byref(offset), C.byref(size))
        if status != 1:
            raise ValueError("argument layout does not cover exactly the kernel's parameters")
        attributes = []
        for attribute in (1, 2, 4, 5, 8, 9, 10, 11, 13, 14):
            value = (C.c_uint64 * 8)()
            status = cuda.lib.cuGraphKernelNodeGetAttribute(C.c_void_p(handle), C.c_int(attribute), C.byref(value))
            # Drivers without preferred-cluster support reject even its default-value query.
            if attribute == 11 and status == 1:
                continue
            cuda.check(f"cuGraphKernelNodeGetAttribute({attribute})", status)
            data = bytes(value)
            if attribute == 13:
                if any(data):
                    raise ValueError("device-updatable kernel nodes require explicit export support")
                continue
            # An unset preferred shared-memory carveout reads as zero; restoring it explicitly would
            # request the minimum shared memory and cut occupancy of every kernel using shared memory.
            if attribute == 14 and not any(data):
                continue
            relocations = []
            if attribute == 1 and value[1]:
                relocations = [{"offset": 0, "kind": "pointer", "target": self._reference(value[0], value[1])}]
            attributes.append({"id": attribute, "value": {"data": list(data), "relocations": relocations}})
        return {"kind": "kernel", "function": self._function_indices[function.value],
                "grid": list(params.grid), "block": list(params.block), "shared": params.shared,
                "arguments": arguments, "attributes": attributes}


def _external_allocations(spans, granularity):
    pages = []
    for address, nbytes in sorted(spans):
        start = address & -granularity
        end = (address + nbytes + granularity - 1) & -granularity
        if pages and start <= pages[-1][1]:
            first, last, contents = pages[-1]
            pages[-1] = (first, max(last, end), contents + [(address, nbytes)])
        else:
            pages.append((start, end, [(address, nbytes)]))
    return [Allocation(f"external-{index}", start, end - start, alignment=granularity,
                       preserve_address=True,
                       snapshot_spans=tuple((address - start, nbytes) for address, nbytes in contents))
            for index, (start, end, contents) in enumerate(pages)]


def _roce_transport(message, transport):
    """Fill a bundle's RoceTransport from b12x's `AllReduce.native_description()`."""
    message.proxy_abi = transport["proxy_abi"]
    message.world, message.rank = transport["world_size"], transport["rank"]
    message.region_address, message.region_bytes = transport["region"]["address"], transport["region"]["nbytes"]
    message.counters_address = transport["counters"]["address"]
    message.counters_bytes = transport["counters"]["nbytes"]
    message.slot_bytes, message.pipe_bytes = transport["slot_bytes"], transport["pipe_bytes"]
    message.ctrl_offset = transport["ctrl_offset"]
    message.hcas.extend(transport["hca_names"])
    message.gid_index, message.traffic_class = transport["gid_index"], transport["traffic_class"]
    message.rails = transport["rails"]
    for links in transport["routes"]:
        route = message.routes.add()
        for local, remote in links:
            route.links.add(local=local, remote=remote)


def export_model(prepared, images, storage, path, *, options, progress=None, bundle_only=False):
    """Export one rank's graphs and storage with recorded virtual addresses.

    With `bundle_only`, a multi-host peer's export omits frontend assets and package-level
    tables: rank zero's package already holds them once for every host.
    """
    cuda = _Cuda()
    cuda.call("cuCtxSynchronize")
    if storage.error is not None:
        raise RuntimeError("native package allocator failed") from storage.error
    if prepared._consumed:
        raise RuntimeError("runtime package export requires an unused prepared model")
    inactive = storage.unused_ranges()
    dynamic = {region.address: region for region in storage.dynamic_regions}
    zero_ranges = [(address, size) for address, size in prepared.export_zero_ranges
                   if not any(region.address <= address < region.address + region.extent for region in dynamic.values())]
    zero_ranges.extend((segment["address"], segment["total_size"])
                       for segment in storage.graph_pool.snapshot(include_traces=False))
    zero_ranges.extend((start, end - start) for start, end in inactive)
    # A native rank rebuilds each RoCE transport over its recorded region and counters, which
    # restore zeroed so every rank starts again at sequence 1. The proxy stops first: its doorbell
    # lives in the region.
    transports = [runtime.native_description() for runtime in storage.transports]
    if len(transports) > 1:
        raise ValueError("a rank exports at most one RoCE transport")
    for runtime in storage.transports:
        runtime.close()
    for transport in transports:
        zero_ranges.extend((transport[part]["address"], transport[part]["nbytes"])
                           for part in ("region", "counters"))
    zero_ranges = _merge_ranges(zero_ranges)
    for address, nbytes in zero_ranges:
        cuda.call("cuMemsetD8_v2", C.c_uint64(address), C.c_ubyte(0), C.c_size_t(nbytes))
    cuda.call("cuCtxSynchronize")
    if progress is not None:
        progress(f"Cleared {sum(size for _, size in zero_ranges) / (1 << 30):.2f} GiB of disposable storage")
    mappings = storage.mappings
    imports = {} if storage.peer_storage is None else storage.peer_storage.imported
    imported_mappings = (() if storage.peer_storage is None else
                         (part for parts in storage.peer_storage.imported_mappings.values() for part in parts))
    for address, size in imported_mappings:
        if images.allocations.get(address) != (size, "device"):
            raise ValueError(f"peer mapping {address:#x}+{size} differs from the recorded allocation")
    outside = [(address, size) for address, (size, kind) in images.allocations.items()
               if kind == "device" and address not in mappings
               and not any(base <= address < base + size for base, size in imports.items())
               and not any(region.address <= address < region.address + region.extent for region in dynamic.values())]
    current = C.c_void_p()
    cuda.call("cuCtxGetCurrent", C.byref(current))
    for address, nbytes in outside:
        memory, context = C.c_int(), C.c_void_p()
        cuda.call("cuPointerGetAttribute", C.byref(memory), C.c_int(2), C.c_uint64(address))
        cuda.call("cuPointerGetAttribute", C.byref(context), C.c_int(1), C.c_uint64(address))
        if progress is not None:
            progress(f"External storage {address:#x}+{nbytes}: memory type {memory.value}, "
                     f"context {context.value}, recorded context {images.allocation_contexts.get(address)}, "
                     f"serving context {current.value}")
        if memory.value != 2 or context.value != current.value:
            raise ValueError(f"external allocation {address:#x}+{nbytes} is not device memory in the serving context")
    allocations = [Allocation(f"device-{index}", mapping.address, mapping.nbytes,
                             alignment=storage.granularity, preserve_address=True)
                   for index, mapping in enumerate(sorted(mappings.values(), key=lambda m: m.address))]
    allocations.extend(_external_allocations(outside, storage.granularity))
    # Graph copies may address peer mappings; the model index declares their owners and access.
    allocations.extend(Allocation(f"peer-{index}", address, size, alignment=storage.granularity,
                                  initialize="zero", preserve_address=True, memory="peer")
                       for index, (address, size) in enumerate(sorted(imports.items())))
    allocations.extend(Allocation(f"host-{index}", region.address, region.extent, alignment=4096,
                                  preserve_address=True, memory="mapped_host")
                       for index, region in enumerate(storage.host_regions) if not region.closed)
    allocations = [_without_zero_ranges(allocation, zero_ranges) for allocation in allocations]
    allocations.extend(Allocation(region.name, region.address, region.extent, alignment=storage.granularity,
        initialize="zero", preserve_address=True, sizing={"capacity": region.group, "unit_bytes": region.unit_bytes,
        "maximum": region.maximum}) for region in dynamic.values())
    from .tensor_metadata import live_metadata
    from .reachability import Reachability

    numeric, parameter_layouts = live_metadata(allocations, storage.numeric_storage, prepared.buffers)
    if progress is not None:
        progress(f"Classified {len(numeric)} numeric tensor spans and "
                 f"{len(parameter_layouts)} typed kernel signatures")
    modules, module_names, kernels = {}, {}, {}
    for graph in prepared.graphs.values():
        count = C.c_size_t()
        handle = C.c_void_p(graph.raw_cuda_graph())
        cuda.call("cuGraphGetNodes", handle, None, C.byref(count))
        nodes = (C.c_void_p * count.value)()
        cuda.call("cuGraphGetNodes", handle, nodes, C.byref(count))
        for node in nodes:
            kind = C.c_int()
            cuda.call("cuGraphNodeGetType", C.c_void_p(node), C.byref(kind))
            if kind.value != 0:
                continue
            params = _KernelParams()
            cuda.call("cuGraphKernelNodeGetParams_v2", C.c_void_p(node), C.byref(params))
            function = C.c_void_p(params.func)
            if not function.value:
                cuda.call("cuKernelGetFunction", C.byref(function), C.c_void_p(params.kern))
            if function.value in kernels:
                continue
            module = C.c_void_p()
            cuda.call("cuFuncGetModule", C.byref(module), function)
            name = module_names.setdefault(module.value, f"module-{len(module_names)}")
            if name not in modules:
                modules[name] = images.image_for(cuda, function.value)
            arguments = []
            for index in range(4096):
                offset, size = C.c_size_t(), C.c_size_t()
                status = cuda.lib.cuFuncGetParamInfo(function, C.c_size_t(index), C.byref(offset), C.byref(size))
                if status == 1:
                    break
                if status:
                    raise RuntimeError(f"cuFuncGetParamInfo failed with CUDA status {status}")
                arguments.append(Argument(size.value))
            else:
                raise ValueError("kernel exceeds supported parameter count")
            layout = parameter_layouts.get(function.value)
            if layout is not None:
                if tuple(size for size, _ in layout) != tuple(argument.size for argument in arguments):
                    raise ValueError("compiler parameter metadata differs from the CUDA function layout")
                arguments = [replace(argument, numeric=((0, argument.size),) if numeric else ())
                             for argument, (_, numeric) in zip(arguments, layout, strict=True)]
            kernels[function.value] = Kernel(function.value, name, tuple(arguments))
    ranges = [(storage.DEVICE_BASE, storage.DEVICE_BASE + storage.RESERVATION_BYTES)]
    granule = storage.granularity
    ranges.extend((address & -granule, (address + size + granule - 1) & -granule)
                  for address, size in outside)
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    peer_addresses = set() if storage.peer_storage is None else storage.peer_storage.owned.keys()
    # RoCE kernels take their region and counters as integer arguments, which reachability cannot follow.
    pinned = [transport[part]["address"] for transport in transports for part in ("region", "counters")]
    reachability = Reachability(allocations, numeric=numeric, inactive=inactive,
                               roots=[a.name for a in allocations if a.address in peer_addresses
                                      or any(a.address <= p < a.address + a.nbytes for p in pinned)])
    package = RuntimePackage(allocations=allocations, modules=modules, kernels=kernels.values(), owners=(prepared, storage),
        reservations=[{"address": start, "nbytes": end - start} for start, end in merged], reachability=reachability,
        events=prepared.export_events, transports=transports)
    execution = prepared.manifest(all_variants=True)
    execution.pop("context")
    execution.pop("device")
    for buffer in execution["buffers"]:
        buffer["reference"] = package._reference(buffer.pop("address"), buffer["nbytes"])
    for graph in execution["graphs"]:
        graph.pop("handle")
        if graph["kind"] == "speculative":
            graph["draft_tokens"] = graph["token_capacity"] - 1
            graph["verify_tokens"] = graph["token_capacity"]
            graph["draft_source"] = "COPY_DRAFT" if prepared.server_config.get("speculative_proposal_source") == "copy" else "MTP"
    server = {**prepared.server_config, **options}
    if bundle_only:
        server["assets"] = {}
    package.write(path, prepared.graphs, execution=execution, server=server, progress=progress,
                  package_data=None if bundle_only else prepared.package_data)
    return package
