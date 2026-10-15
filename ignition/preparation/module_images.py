"""Collect loaded cubins through preparation-scoped CUPTI resource callbacks."""

import ctypes as C
from dataclasses import dataclass
from pathlib import Path
import struct
import threading


_MATERIALIZERS = {
    "cuModuleLoad", "cuModuleLoadData", "cuModuleLoadDataEx", "cuModuleLoadFatBinary",
    "cuModuleGetFunction", "cuLibraryLoadData", "cuLibraryLoadFromFile", "cuLibraryGetModule",
    "cuLibraryGetKernel", "cuKernelGetFunction", "cuKernelSetAttribute", "cuFuncSetAttribute",
    "cuFuncLoad", "cuLaunchKernel", "cuLaunchKernel_ptsz", "cuLaunchKernelEx", "cuLaunchKernelEx_ptsz",
    "cuLaunchCooperativeKernel", "cuLaunchCooperativeKernel_ptsz",
}


class _Resource(C.Structure):
    _fields_ = [("context", C.c_void_p), ("handle", C.c_void_p), ("descriptor", C.c_void_p)]


class _Module(C.Structure):
    _fields_ = [("id", C.c_uint32), ("size", C.c_size_t), ("image", C.c_void_p)]


class _Call(C.Structure):
    _fields_ = [("site", C.c_int), ("name", C.c_char_p), ("params", C.c_void_p),
                ("result", C.POINTER(C.c_int)), ("symbol", C.c_char_p), ("context", C.c_void_p),
                ("context_id", C.c_uint32), ("correlation", C.c_void_p), ("correlation_id", C.c_uint32)]


def _elf_image(address):
    header = C.string_at(address, 64)
    if not header.startswith(b"\x7fELF\x02\x01"):
        return None
    offset = struct.unpack_from("<Q", header, 40)[0]
    entry_size, count = struct.unpack_from("<HH", header, 58)
    if entry_size != 64 or not count:
        raise ValueError("unsupported cubin ELF section table")
    sections = C.string_at(address + offset, count * entry_size)
    programs = struct.unpack_from("<Q", header, 32)[0]
    program_size, program_count = struct.unpack_from("<HH", header, 54)
    end = max(offset + len(sections), programs + program_size * program_count)
    for index in range(count):
        section = sections[index * entry_size:(index + 1) * entry_size]
        kind = struct.unpack_from("<I", section, 4)[0]
        start, size = struct.unpack_from("<QQ", section, 24)
        if kind != 8:
            end = max(end, start + size)
    return C.string_at(address, end)


def strip_debug(image):
    """Erase source-bearing cubin sections without renumbering CUDA metadata.

    CUDA-specific sections contain symbol and section indices that generic ELF
    stripping tools do not update. Empty debug sections retain those indices.
    """
    if not image.startswith(b"\x7fELF\x02\x01"):
        raise ValueError("expected an ELF64 little-endian cubin")
    offset = struct.unpack_from("<Q", image, 40)[0]
    entry_size, count, names = struct.unpack_from("<HHH", image, 58)
    if entry_size != 64 or not 0 < names < count or offset + count * entry_size > len(image):
        raise ValueError("invalid cubin ELF section table")
    sections = [struct.unpack_from("<IIQQQQIIQQ", image, offset + i * entry_size) for i in range(count)]
    name_table = sections[names]
    strings = image[name_table[4]:name_table[4] + name_table[5]]
    debug = set()
    for i, section in enumerate(sections):
        name = strings[section[0]:strings.index(b"\0", section[0])]
        if b".debug" in name or b".nv_debug" in name:
            debug.add(i)
    result = bytearray(image)
    for i in debug:
        section = sections[i]
        start, size = section[4:6]
        if section[2] & 2 or start + size > len(image):
            raise ValueError("invalid or device-allocated cubin debug section")
        for other_index, other in enumerate(sections):
            if other_index not in debug and other[1] != 8 and other[5]:
                if start < other[4] + other[5] and other[4] < start + size:
                    raise ValueError("cubin debug section overlaps executable metadata")
        result[start:start + size] = bytes(size)
        struct.pack_into("<Q", result, offset + i * entry_size + 32, 0)
    for table in sections:
        if table[1] not in (2, 0x70000085):
            continue
        symbols = sections[table[6]]
        strings = image[symbols[4]:symbols[4] + symbols[5]]
        for position in range(table[4], table[4] + table[5], table[9]):
            name, info, _, index, value, size = struct.unpack_from("<IBBHQQ", image, position)
            symbol = strings[name:strings.index(b"\0", name)].decode()
            if info & 15 != 1 or not symbol.startswith("assertFile_") or not 0 < index < count:
                continue
            section = sections[index]
            if section[1] == 8 or value + size > section[5]:
                raise ValueError("invalid CUDA assertion filename symbol")
            start = section[4] + value
            result[start:start + size] = strip_global_source(symbol, image[start:start + size])
    return bytes(result)


def strip_global_source(symbol, data):
    """Keep CUDA assertion diagnostics while excluding preparation source paths."""
    if symbol.startswith("assertFile_"):
        if not data or data[-1] != 0:
            raise ValueError("CUDA assertion filename must be a terminated string")
        return b"kernel"[:len(data) - 1].ljust(len(data), b"\0")
    return data


def global_symbols(image):
    """Enumerate defined CUDA global and constant objects in an ELF64 cubin."""
    offset = struct.unpack_from("<Q", image, 40)[0]
    entry_size, count, names = struct.unpack_from("<HHH", image, 58)
    if entry_size != 64:
        raise ValueError("unsupported cubin ELF section table")
    sections = [struct.unpack_from("<IIQQQQIIQQ", image, offset + index * entry_size)
                for index in range(count)]
    def contents(section):
        return image[section[4]:section[4] + section[5]]
    def string(table, start):
        return table[start:table.index(b"\0", start)].decode()
    section_names = contents(sections[names])
    symbols = set()
    for table in sections:
        if table[1] != 2:
            continue
        strings = contents(sections[table[6]])
        for position in range(table[4], table[4] + table[5], table[9]):
            name, info, _, section, _, size = struct.unpack_from("<IBBHQQ", image, position)
            if info & 15 != 1 or not size or not 0 < section < len(sections):
                continue
            section_name = string(section_names, sections[section][0])
            if section_name.startswith((".nv.global", ".nv.constant")):
                symbols.add(string(strings, name))
    return sorted(symbols)


@dataclass(frozen=True)
class ModuleImage:
    context: int
    module_id: int
    cubin: bytes


def cupti_library(library=None):
    if library is None:
        from cuda.pathfinder import load_nvidia_dynamic_lib

        library = load_nvidia_dynamic_lib("cupti").abs_path
    return C.CDLL(library)


class ModuleImages:
    """Copy module images while their callback-owned storage is valid.

    Enter before loading or warming kernels. The callback performs no CUDA calls.
    Driver load callbacks associate cubins with module and library handles.
    Resource callbacks also collect modules loaded internally by CUDA libraries.
    """

    # Recorders active in this process; tensor-parallel peers record exactly when their root does.
    active = 0

    def __init__(self, library=None):
        self._library = cupti_library(library)
        self._subscriber = C.c_void_p()
        self._callback = C.CFUNCTYPE(None, C.c_void_p, C.c_int, C.c_uint, C.c_void_p)(self._record)
        self.images = []
        self.modules = {}
        self.libraries = {}
        self.allocations = {}
        self.allocation_contexts = {}
        self._pending = {}
        self._functions = {}
        self._kernels = {}
        self._error = None

    def _check(self, name, *args):
        status = getattr(self._library, name)(*args)
        if status:
            message = C.c_char_p()
            self._library.cuptiGetResultString(C.c_int(status), C.byref(message))
            raise RuntimeError(f"{name}: CUPTI {status}: {message.value!r}")

    def _record(self, _user, domain, callback, data):
        try:
            if domain == 1:
                self._driver(C.cast(data, C.POINTER(_Call)).contents)
                return
            if domain != 3 or callback not in (2, 6, 7):
                return
            resource = C.cast(data, C.POINTER(_Resource)).contents
            if callback == 2:
                for address, context in list(self.allocation_contexts.items()):
                    if context == resource.context:
                        self.allocations.pop(address, None)
                        self.allocation_contexts.pop(address)
                return
            module = C.cast(resource.descriptor, C.POINTER(_Module)).contents
            if callback == 7:
                for bindings in (self._functions, self._kernels):
                    for handle, image in list(bindings.items()):
                        if (image.context, image.module_id) == (resource.context, module.id):
                            bindings.pop(handle)
                return
            if not module.image or not module.size:
                raise RuntimeError("CUPTI returned a module without its cubin")
            image = ModuleImage(resource.context, module.id, C.string_at(module.image, module.size))
            self.images.append(image)
            pending = self._pending.get(threading.get_ident())
            if pending:
                pending[-1][1].append(image)
        except BaseException as error:
            self._error = error

    def _driver(self, call):
        name = call.name.decode()
        images = []
        if name in _MATERIALIZERS:
            pending = self._pending.setdefault(threading.get_ident(), [])
            if call.site == 0:
                pending.append((name, []))
                return
            entered, images = pending.pop()
            if entered != name:
                raise RuntimeError("unbalanced CUDA driver callbacks")
        if call.site != 1 or call.result[0] != 0:
            return
        values = C.cast(call.params, C.POINTER(C.c_uint64))
        if len(images) == 1:
            image, = images
            if name.startswith(("cuLaunchKernelEx",)):
                self._functions[values[1]] = image
            elif name.startswith(("cuLaunchKernel", "cuLaunchCooperativeKernel")) or name in ("cuFuncLoad", "cuFuncSetAttribute"):
                self._functions[values[0]] = image
            elif name == "cuKernelSetAttribute":
                self._kernels[values[1]] = image
            elif name == "cuKernelGetFunction":
                self._functions[C.cast(values[0], C.POINTER(C.c_void_p))[0]] = image
            elif name == "cuModuleGetFunction":
                self.modules[values[1]] = image.cubin
            elif name == "cuLibraryGetKernel":
                self.libraries[values[1]] = image.cubin
            elif name == "cuLibraryGetModule":
                self.modules[C.cast(values[0], C.POINTER(C.c_void_p))[0]] = image.cubin
        if name in ("cuModuleLoadData", "cuModuleLoadDataEx", "cuLibraryLoadData", "cuModuleLoad", "cuLibraryLoadFromFile"):
            handle = C.cast(values[0], C.POINTER(C.c_void_p))[0]
            image = (Path(C.string_at(values[1]).decode()).read_bytes()
                     if name in ("cuModuleLoad", "cuLibraryLoadFromFile") else _elf_image(values[1]))
            if image is not None and image.startswith(b"\x7fELF"):
                target = self.libraries if name.startswith("cuLibrary") else self.modules
                target[handle] = image
            elif len(images) == 1:
                target = self.libraries if name.startswith("cuLibrary") else self.modules
                target[handle] = images[0].cubin
        elif name == "cuLibraryGetModule" and values[1] in self.libraries:
            self.modules[C.cast(values[0], C.POINTER(C.c_void_p))[0]] = self.libraries[values[1]]
        elif name == "cuModuleUnload":
            self.modules.pop(values[0], None)
        elif name == "cuLibraryUnload":
            self.libraries.pop(values[0], None)
        elif name in ("cuMemAlloc_v2", "cuMemAllocAsync", "cuMemAllocAsync_ptsz", "cuMemAllocFromPoolAsync", "cuMemAllocFromPoolAsync_ptsz"):
            address = C.cast(values[0], C.POINTER(C.c_uint64))[0]
            self.allocations[address] = (values[1], "device")
            self.allocation_contexts[address] = call.context
        elif name in ("cuMemFree_v2", "cuMemFreeAsync", "cuMemFreeAsync_ptsz", "cuMemUnmap", "cuMemFreeHost", "cuMemHostUnregister"):
            self.allocations.pop(values[0], None)
            self.allocation_contexts.pop(values[0], None)
        elif name == "cuMemMap":
            self.allocations[values[0]] = (values[1], "device")
            self.allocation_contexts[values[0]] = call.context
        elif name in ("cuMemHostAlloc", "cuMemHostAlloc_v2", "cuMemAllocHost_v2"):
            address = C.cast(values[0], C.POINTER(C.c_void_p))[0]
            self.allocations[address] = (values[1], "mapped_host")
        elif name in ("cuMemHostRegister", "cuMemHostRegister_v2"):
            self.allocations[values[0]] = (values[1], "mapped_host")

    def image_for(self, cuda, function):
        module = C.c_void_p()
        cuda.call("cuFuncGetModule", C.byref(module), C.c_void_p(function))
        if module.value not in self.modules:
            for kernel, image in self._kernels.items():
                handle = C.c_void_p()
                cuda.call("cuKernelGetFunction", C.byref(handle), C.c_void_p(kernel))
                self._functions[handle.value] = image
            for handle, image in self._functions.items():
                parent = C.c_void_p()
                status = cuda.lib.cuFuncGetModule(C.byref(parent), C.c_void_p(handle))
                if status == 400:
                    # Launch APIs also accept context-independent CUkernel handles.
                    resolved = C.c_void_p()
                    cuda.call("cuKernelGetFunction", C.byref(resolved), C.c_void_p(handle))
                    cuda.call("cuFuncGetModule", C.byref(parent), resolved)
                elif status:
                    raise RuntimeError(f"cuFuncGetModule failed with CUDA status {status}")
                self.modules[parent.value] = image.cubin
            for library, image in self.libraries.items():
                handle = C.c_void_p()
                cuda.call("cuLibraryGetModule", C.byref(handle), C.c_void_p(library))
                self.modules[handle.value] = image
        if module.value in self.modules:
            return self.modules[module.value]
        name = C.c_char_p()
        cuda.call("cuFuncGetName", C.byref(name), C.c_void_p(function))
        matches = {image.cubin for image in self.images if name.value + b"\0" in image.cubin}
        if len(matches) != 1:
            raise ValueError(f"cannot uniquely associate captured kernel {name.value!r} with its loaded cubin")
        image, = matches
        self.modules[module.value] = image
        return image

    def __enter__(self):
        if self._subscriber.value:
            raise RuntimeError("module image recorder is already active")
        self._check("cuptiSubscribe", C.byref(self._subscriber), self._callback, None)
        try:
            for callback in (2, 6, 7):
                self._check("cuptiEnableCallback", C.c_uint(1), self._subscriber, C.c_int(3), C.c_uint(callback))
            for callback in (18, 19, 20, 22, 23, 678, 679, 680, 681, 682, 683, 687, 742,
                             307, 442, 477, 478, 481, 652, 653, 243, 245, 38, 39, 271, 294,
                             301, 302, 379, 551, 552, 598, 599, 600, 601, 611, 612):
                self._check("cuptiEnableCallback", C.c_uint(1), self._subscriber, C.c_int(1), C.c_uint(callback))
        except BaseException:
            self._check("cuptiUnsubscribe", self._subscriber)
            self._subscriber = C.c_void_p()
            raise
        ModuleImages.active += 1
        return self

    def __exit__(self, kind, value, traceback):
        ModuleImages.active -= 1
        self._check("cuptiUnsubscribe", self._subscriber)
        self._subscriber = C.c_void_p()
        if kind is None and self._error is not None:
            raise RuntimeError("module image callback failed") from self._error
