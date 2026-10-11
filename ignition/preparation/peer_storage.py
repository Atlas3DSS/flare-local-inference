"""Symmetric VMM mappings for b12x collectives during rank preparation."""

import ctypes as C
from multiprocessing.reduction import DupFd
import os
import pickle

from b12x.comm.pcie._cuda_ipc import CudaRTLibrary


class _Location(C.Structure):
    _fields_ = [("kind", C.c_int), ("device", C.c_int)]


class _Access(C.Structure):
    _fields_ = [("location", _Location), ("flags", C.c_uint64)]


class PeerStorage(CudaRTLibrary):
    """Use rank arenas and descriptor passing for the b12x shared-buffer API."""

    def __init__(self, storage):
        super().__init__()
        self.storage = storage
        self.owned = {}
        self.imported = {}
        storage.library.flare_package_allocate.restype = C.c_void_p

    def cudaMalloc(self, size):
        pointer = self.storage.library.flare_package_allocate(
            C.c_size_t(size), C.c_int(self.storage.device), None)
        if pointer is None:
            raise self.storage.error or MemoryError("collective allocation failed")
        self.owned[pointer] = self.storage.mappings[pointer].nbytes
        return pointer

    def cudaFree(self, pointer):
        self.storage.library.flare_package_free(C.c_void_p(pointer), C.c_size_t(0),
                                               C.c_int(self.storage.device), None)
        if self.storage.error:
            raise self.storage.error
        del self.owned[pointer]

    def cudaIpcGetMemHandleBytes(self, pointer):
        cuda = self.storage.cuda
        physical, descriptor = C.c_uint64(), C.c_int()
        cuda.call("cuMemRetainAllocationHandle", C.byref(physical), C.c_void_p(pointer))
        try:
            cuda.call("cuMemExportToShareableHandle", C.byref(descriptor), physical,
                      C.c_int(1), C.c_uint64(0))
        finally:
            cuda.call("cuMemRelease", physical)
        try:
            descriptors = [None if rank == self.storage.rank else DupFd(descriptor.value)
                           for rank in range(self.storage.world)]
            return pickle.dumps((pointer, self.owned[pointer], descriptors))
        finally:
            os.close(descriptor.value)

    def cudaIpcOpenMemHandleBytes(self, data):
        address, size, descriptors = pickle.loads(data)
        descriptor = descriptors[self.storage.rank].detach()
        physical = C.c_uint64()
        cuda = self.storage.cuda
        try:
            cuda.call("cuMemImportFromShareableHandle", C.byref(physical),
                      C.c_void_p(descriptor), C.c_int(1))
        finally:
            os.close(descriptor)
        try:
            cuda.call("cuMemMap", C.c_uint64(address), C.c_size_t(size), C.c_size_t(0),
                      physical, C.c_uint64(0))
        finally:
            cuda.call("cuMemRelease", physical)
        try:
            access = _Access(_Location(1, self.storage.device), 3)
            cuda.call("cuMemSetAccess", C.c_uint64(address), C.c_size_t(size), C.byref(access), C.c_size_t(1))
        except BaseException:
            cuda.call("cuMemUnmap", C.c_uint64(address), C.c_size_t(size))
            raise
        self.imported[address] = size
        return address

    def cudaIpcCloseMemHandle(self, pointer):
        self.storage.cuda.call("cuMemUnmap", C.c_uint64(pointer), C.c_size_t(self.imported[pointer]))
        del self.imported[pointer]

    def reset(self):
        """Reset all local signals and generation-tagged payloads after group drain."""
        for address, size in self.owned.items():
            self.cudaMemset(address, 0, size)
