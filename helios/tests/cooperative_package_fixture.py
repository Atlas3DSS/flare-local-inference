"""Export cooperative worker grids that exceed the assigned GPU's residency limit."""

import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from ignition.preparation.runtime_package import (
    Allocation, Argument, Kernel, Reference, Relocation, RuntimePackage, _Cuda, _KernelParams,
)
from signing import publisher


def prepare(root):
    cuda = _Cuda()
    cuda.call("cuInit", C.c_uint(0))
    device, context = C.c_int(), C.c_void_p()
    cuda.call("cuDeviceGet", C.byref(device), C.c_int(0))
    cuda.call("cuCtxCreate_v2", C.byref(context), C.c_uint(0), device)
    attributes = []
    for attribute in (75, 76, 16):
        value = C.c_int()
        cuda.call("cuDeviceGetAttribute", C.byref(value), C.c_int(attribute), device)
        attributes.append(value.value)
    major, minor, sms = attributes
    source = root / "workers.cu"
    source.write_text('''
#include <cooperative_groups.h>
template<bool Cooperative> __device__ void work(int *out, int n) {
    int block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    int blocks = gridDim.x * gridDim.y * gridDim.z;
    int thread = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    int threads = blockDim.x * blockDim.y * blockDim.z;
    int start = block * threads + thread;
    for (int i = start; i < n; i += blocks * threads) out[i] = i * 3 + 7;
    if constexpr (Cooperative) {
        cooperative_groups::this_grid().sync();
        for (int i = start; i < n; i += blocks * threads) out[n + i] = out[n - 1 - i] + 11;
    }
    if (start == 0) {
        out[2 * n] = gridDim.x;
        out[2 * n + 1] = gridDim.y;
        out[2 * n + 2] = gridDim.z;
    }
}
extern "C" __global__ void cooperative(int *out, int n) { work<true>(out, n); }
extern "C" __global__ void ordinary(int *out, int n) { work<false>(out, n); }
''')
    cubin = root / "workers.cubin"
    subprocess.run(["nvcc", "--cubin", "-std=c++17", f"-arch=sm_{major}{minor}",
                    str(source), "-o", str(cubin)], check=True)
    module = C.c_void_p()
    cuda.call("cuModuleLoad", C.byref(module), C.c_char_p(str(cubin).encode()))
    functions = {}
    for name in ("cooperative", "ordinary"):
        function = C.c_void_p()
        cuda.call("cuModuleGetFunction", C.byref(function), module, C.c_char_p(name.encode()))
        cuda.call("cuFuncSetAttribute", function, C.c_int(8), C.c_int(65536))
        functions[name] = function
    length = 500003
    allocations, graphs, expected, exported_grids = [], {}, {}, {}
    for name, block, shared, yz, size in (
        ("smaller", (32, 4, 1), 0, (1, 1), "small"),
        ("exact", (32, 4, 1), 0, (1, 1), "exact"),
        ("oversized", (32, 4, 1), 0, (1, 1), "large"),
        ("thread_limited", (32, 8, 4), 0, (1, 1), "large"),
        ("shared_limited", (32, 4, 1), 65536, (1, 1), "large"),
        ("partitioned", (32, 4, 1), 0, (2, 3), "large"),
        ("ordinary", (32, 4, 1), 0, (1, 1), "large"),
    ):
        function = functions["ordinary" if name == "ordinary" else "cooperative"]
        occupancy = C.c_int()
        cuda.call("cuOccupancyMaxActiveBlocksPerMultiprocessor", C.byref(occupancy), function,
                  C.c_int(block[0] * block[1] * block[2]), C.c_size_t(shared))
        capacity = occupancy.value * sms
        max_x = capacity // (yz[0] * yz[1])
        captured_x = 3 if size == "small" else max_x if size == "exact" else max_x + 7
        expected_grid = (captured_x if name == "ordinary" else min(captured_x, max_x), *yz)
        output = C.c_uint64()
        nbytes = (length * 2 + 3) * 4
        cuda.call("cuMemAlloc_v2", C.byref(output), C.c_size_t(nbytes))
        allocations.append(Allocation(name, output.value, nbytes, initialize="zero"))
        count = C.c_int(length)
        arguments = (C.c_void_p * 2)(C.addressof(output), C.addressof(count))
        graph, node, executable = C.c_void_p(), C.c_void_p(), C.c_void_p()
        cuda.call("cuGraphCreate", C.byref(graph), C.c_uint(0))
        params = _KernelParams(func=function.value, grid=(C.c_uint * 3)(captured_x, *yz),
                               block=(C.c_uint * 3)(*block), shared=shared, params=arguments)
        cuda.call("cuGraphAddKernelNode_v2", C.byref(node), graph, None, C.c_size_t(0), C.byref(params))
        if name != "ordinary":
            cooperative = (C.c_uint64 * 8)(1)
            status = cuda.lib.cuGraphKernelNodeSetAttribute(node, C.c_int(2), C.byref(cooperative))
            if size == "large":
                assert status == 720, f"oversized {name} grid returned CUDA status {status}"
                params.grid[0] = expected_grid[0]
                cuda.call("cuGraphKernelNodeSetParams_v2", node, C.byref(params))
                cuda.call("cuGraphKernelNodeSetAttribute", node, C.c_int(2), C.byref(cooperative))
            else:
                cuda.check("cuGraphKernelNodeSetAttribute", status)
        cuda.call("cuGraphInstantiateWithFlags", C.byref(executable), graph, C.c_uint64(0))
        graphs[name] = SimpleNamespace(raw_cuda_graph=lambda g=graph: g.value,
                                      raw_cuda_graph_exec=lambda e=executable: e.value)
        exported_grids[node.value] = (captured_x, *yz)
        expected[name] = expected_grid

    class WorkerPackage(RuntimePackage):
        def _kernel(self, cuda, handle):
            params = _KernelParams()
            cuda.call("cuGraphKernelNodeGetParams_v2", C.c_void_p(handle), C.byref(params))
            address = C.c_uint64.from_address(params.params[0]).value
            allocation = next(a for a in allocations if a.address == address)
            self.kernels[params.func] = Kernel(params.func, "workers", (
                Argument(8, (Relocation(0, Reference(allocation.name)),)), Argument(4)))
            node = super()._kernel(cuda, handle)
            # Encode the larger preparation grid without requiring a larger physical GPU.
            node["grid"] = exported_grids[handle]
            return node

    package = WorkerPackage(allocations=allocations, modules={"workers": cubin.read_bytes()},
        kernels=[Kernel(f.value, "workers", (Argument(8), Argument(4))) for f in functions.values()])
    package.write(root / "package", graphs, publisher=publisher())
    (root / "expected.json").write_text(json.dumps({"length": length, "grids": expected}))
    cuda.call("cuCtxDestroy_v2", context)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    prepare(root)
