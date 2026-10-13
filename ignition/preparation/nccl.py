"""An NCCL communicator for prefill-sized reductions between hosts, captured in CUDA graphs.

RoCEnante's one-shot all-reduce sends every rank's whole payload to every peer,
which costs more than NCCL's ring once activations reach megabytes on four
hosts. Prefill graphs therefore reduce through NCCL, which records its network
transfers as graph work served by its own proxy thread. A captured NCCL graph
costs about a millisecond of host launch time on GB10, which a prefill chunk
amortizes and a decode step would not, so decode keeps RoCEnante.

The library is PyTorch's own NCCL, called directly on the caller's stream: the
reduction is ordinary stream work under capture, with no process-group watchdog.
"""

import ctypes
from pathlib import Path

import torch
import torch.distributed as dist

_BFLOAT16 = 9
_SUM = 0
_LIBRARY = None


class _UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


def _library():
    global _LIBRARY
    if _LIBRARY is None:
        import nvidia.nccl

        library = ctypes.CDLL(str(Path(nvidia.nccl.__path__[0]) / "lib" / "libnccl.so.2"))
        library.ncclGetErrorString.restype = ctypes.c_char_p
        library.ncclGetUniqueId.argtypes = [ctypes.POINTER(_UniqueId)]
        library.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _UniqueId, ctypes.c_int]
        library.ncclAllReduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                                          ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        library.ncclCommDestroy.argtypes = [ctypes.c_void_p]
        library.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        _LIBRARY = library
    return _LIBRARY


def _check(result):
    if result != 0:
        raise RuntimeError(f"NCCL failed: {_library().ncclGetErrorString(result).decode()}")


class NcclCommunicator:
    """One communicator over a Gloo group's ranks; every rank constructs it collectively."""

    def __init__(self, group, device):
        library = _library()
        self.group, self.device = group, torch.device(device)
        self.rank, self.world_size = dist.get_rank(group), dist.get_world_size(group)
        identity = _UniqueId()
        if self.rank == 0:
            _check(library.ncclGetUniqueId(ctypes.byref(identity)))
        payload = [bytes(identity.internal) if self.rank == 0 else None]
        dist.broadcast_object_list(payload, src=dist.get_global_rank(group, 0), group=group)
        ctypes.memmove(identity.internal, payload[0], 128)
        self.comm = ctypes.c_void_p()
        with torch.cuda.device(self.device):
            _check(library.ncclCommInitRank(ctypes.byref(self.comm), self.world_size, identity, self.rank))
        version = ctypes.c_int()
        _check(library.ncclGetVersion(ctypes.byref(version)))
        self.version = version.value
        self._closed = False

    def all_reduce(self, source, output):
        """Sum BF16 `source` across ranks into `output` on the current stream."""
        if self._closed:
            raise RuntimeError("NCCL communicator is closed")
        _check(_library().ncclAllReduce(source.data_ptr(), output.data_ptr(), source.numel(), _BFLOAT16, _SUM,
                                        self.comm, torch.cuda.current_stream(self.device).cuda_stream))

    def close(self):
        if self._closed:
            return
        self._closed = True
        torch.cuda.synchronize(self.device)
        _check(_library().ncclCommDestroy(self.comm))
