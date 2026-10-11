"""Preparation storage whose virtual addresses can be reserved by native restore."""

import ctypes as C
from contextlib import contextmanager
from dataclasses import dataclass
import math
import json

from .runtime_package import _Cuda


@dataclass
class Mapping:
    address: int
    nbytes: int
    kind: str


class PackageStorage:
    """Back a PyTorch caching pool with CUDA VMM and own mapped host regions.

    The pool preserves PyTorch's capture and allocation-lifetime machinery.
    Only backing segments are allocated here. Callbacks and reservations remain
    alive until all tensors and graphs using the pool have been released.
    """

    DEVICE_BASE = 0x200000000000
    HOST_BASE = 0x100000000000
    RESERVATION_BYTES = 1 << 40

    def __init__(self, *, rank=0, world=1, device=0):
        import torch

        self.rank, self.world, self.device = rank, world, device
        self.DEVICE_BASE += rank * self.RESERVATION_BYTES
        self.HOST_BASE += rank * self.RESERVATION_BYTES
        self.cuda = _Cuda()
        from flare_heliopause import flare_heliopause as native

        self.host_regions = []
        self._host_cursor = self.HOST_BASE
        self.library = C.CDLL(native.__file__)
        self.library.flare_package_error.restype = C.c_char_p
        self.library.flare_package_snapshot.restype = C.c_void_p
        if self.library.flare_package_initialize(rank, world, device):
            raise self.error or RuntimeError("cannot initialize native package allocator")
        self.granularity = self._snapshot()["granularity"]
        allocator = torch.cuda.memory.CUDAPluggableAllocator(
            native.__file__, "flare_package_allocate", "flare_package_free")
        self._allocator = allocator
        self.pool = torch.cuda.MemPool(allocator.allocator())
        self.graph_pool = torch.cuda.MemPool(allocator.allocator())
        self._scope = torch.cuda.use_mem_pool(self.pool, device=device)
        self.peer_storage = None
        if world > 1:
            from .peer_storage import PeerStorage

            self.peer_storage = PeerStorage(self)

    def _snapshot(self):
        pointer = self.library.flare_package_snapshot()
        if not pointer:
            raise RuntimeError("native package allocator is not initialized")
        try:
            return json.loads(C.string_at(pointer))
        finally:
            self.library.flare_package_string_free(C.c_void_p(pointer))

    @property
    def mappings(self):
        return {int(address): Mapping(int(address), size, "device")
                for address, size in self._snapshot()["mappings"].items()}

    @property
    def error(self):
        message = self.library.flare_package_error()
        return None if message is None else RuntimeError(message.decode())

    @contextmanager
    def transient(self):
        """Route allocations to a package-backed pool whose free segments return to the device on exit.

        A live pool never releases cached segments, so work that frees most of
        what it allocates, such as kernel tuning, would otherwise strand that
        memory for the rest of preparation. Allocations still live on exit
        stay valid, package-backed, and exported.
        """
        import torch

        pool = torch.cuda.MemPool(self._allocator.allocator())
        try:
            with torch.cuda.use_mem_pool(pool, device=self.device):
                yield
        finally:
            torch.cuda.synchronize(self.device)
            del pool
            torch.cuda.empty_cache()

    def host_allocate(self, name, shape, dtype):
        size = math.prod(shape) * dtype.itemsize
        extent = (size + 4095) // 4096 * 4096
        if self._host_cursor + extent > self.HOST_BASE + self.RESERVATION_BYTES:
            raise MemoryError("native package host-address range exhausted")
        region = HostRegion(self.cuda, self._host_cursor, extent, shape, dtype, self.device)
        self._host_cursor += extent
        self.host_regions.append(region)
        return region

    def __enter__(self):
        self._scope.__enter__()
        return self

    def __exit__(self, kind, value, traceback):
        self._scope.__exit__(kind, value, traceback)
        if kind is None and self.error is not None:
            raise RuntimeError("native package allocator failed") from self.error

class HostRegion:
    def __init__(self, cuda, address, extent, shape, dtype, device):
        import torch
        from b12x.sequence._shared.disk_table import _tensor_from_pointer

        self.cuda, self.address, self.extent = cuda, address, extent
        self.nbytes = math.prod(shape) * dtype.itemsize
        self.closed = True
        self.libc = C.CDLL(None, use_errno=True)
        self.libc.mmap.restype = C.c_void_p
        pointer = self.libc.mmap(C.c_void_p(address), C.c_size_t(extent), C.c_int(3),
                                 C.c_int(0x100022), C.c_int(-1), C.c_long(0))
        if pointer != address:
            raise OSError(C.get_errno(), f"cannot reserve native package host address {address:#x}")
        try:
            cuda.call("cuMemHostRegister_v2", C.c_void_p(address), C.c_size_t(extent), C.c_uint(2))
        except BaseException:
            self.libc.munmap(C.c_void_p(address), C.c_size_t(extent))
            raise
        self.closed = False
        try:
            gpu = C.c_uint64()
            cuda.call("cuMemHostGetDevicePointer_v2", C.byref(gpu), C.c_void_p(address), C.c_uint(0))
            if gpu.value != address:
                raise RuntimeError("mapped host storage requires identical CPU and CUDA addresses")
            views = [_tensor_from_pointer(address, shape=shape, dtype=dtype, device=device, nbytes=self.nbytes)
                     for device in (torch.device("cpu"), torch.device("cuda", device))]
            self.host_view, self.device_view = views
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self.closed:
            self.cuda.call("cuCtxSynchronize")
            self.cuda.call("cuMemHostUnregister", C.c_void_p(self.address))
            if self.libc.munmap(C.c_void_p(self.address), C.c_size_t(self.extent)):
                raise OSError(C.get_errno(), "cannot unmap native package host storage")
            self.closed = True
