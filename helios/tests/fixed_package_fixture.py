"""Export graph bindings using the production VMM allocator and CUPTI collector."""

import ctypes as C
import json
import os
from pathlib import Path
import sys
import subprocess

import torch
from b12x.sequence._shared.disk_table import _tensor_from_pointer

from ignition.preparation.execution import GraphSignature, PreparedExecution
from ignition.preparation.module_images import ModuleImages
from ignition.preparation.package_storage import PackageStorage
from ignition.preparation.runtime_package import export_model


def global_kernel(root, storage, source):
    code = root / "globals.cu"
    code.write_text('''
extern "C" {
__device__ float bias;
__constant__ float scale;
__device__ float *base;
__device__ char assertFile_0[] = "/preparation/internal/kernel.py";
__global__ void transform(float *output) {
    int i = threadIdx.x;
    output[i] = output[i] * scale + bias + base[i];
}
}
''')
    cubin = root / "globals.cubin"
    subprocess.run(["/opt/cuda/bin/nvcc", "--cubin", "-arch=sm_120", str(code), "-o", str(cubin)], check=True)
    cuda = storage.cuda
    module, function = C.c_void_p(), C.c_void_p()
    cuda.call("cuModuleLoad", C.byref(module), C.c_char_p(str(cubin).encode()))
    for name, value in (("bias", C.c_float(2)), ("scale", C.c_float(3)), ("base", C.c_uint64(source.data_ptr()))):
        address, size = C.c_uint64(), C.c_size_t()
        cuda.call("cuModuleGetGlobal_v2", C.byref(address), C.byref(size), module, C.c_char_p(name.encode()))
        assert size.value == C.sizeof(value)
        cuda.call("cuMemcpyHtoD_v2", address, C.byref(value), size)
    cuda.call("cuModuleGetFunction", C.byref(function), module, C.c_char_p(b"transform"))
    return module, function


def prepare(root):
    with ModuleImages() as images:
        torch.cuda.set_device(0)
        with PackageStorage() as storage:
            external = C.c_uint64()
            storage.cuda.call("cuMemAlloc_v2", C.byref(external), C.c_size_t(65536))
            source = _tensor_from_pointer(external.value, shape=(128,), dtype=torch.float32,
                                          device=torch.device("cuda", 0), nbytes=65536)
            source.copy_(torch.arange(128, device="cuda", dtype=torch.float32))
            output = torch.zeros_like(source)
            region = storage.host_allocate("table", (128,), torch.float32)
            region.host_view.fill_(3)
            module, function = global_kernel(root, storage, source)
            buffers = {"frame_index": torch.zeros(1, device="cuda", dtype=torch.int32),
                       "submission_sequence": torch.zeros(1, device="cuda", dtype=torch.int64),
                       "completion_records": torch.zeros((2, 1, 8), device="cuda", dtype=torch.int64),
                       "source": source, "output": output}
            def run():
                torch.add(source, region.device_view, out=output)
                pointer = C.c_uint64(output.data_ptr())
                params = (C.c_void_p * 1)(C.addressof(pointer))
                storage.cuda.call("cuLaunchKernel", function, C.c_uint(1), C.c_uint(1), C.c_uint(1),
                                  C.c_uint(128), C.c_uint(1), C.c_uint(1), C.c_uint(0),
                                  C.c_void_p(torch.cuda.current_stream().cuda_stream), params, None)
            run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, pool=storage.graph_pool.id):
                run()
            graph.instantiate()
            graph.replay()
            torch.cuda.synchronize()
            expected = output.view(torch.uint8).cpu().numpy().tobytes()
            output.zero_()
            prepared = PreparedExecution(device=0, graphs={"decode": graph}, buffers=buffers,
                signatures={"decode": GraphSignature("decode", 1, 1, ("source",))}, max_active=1,
                resources=(storage, region))
            prepared.server_config = {"assets": {}}
    package = export_model(prepared, images, storage, root / "package", options={})
    assert any(a.name.startswith("external-") and a.address <= external.value < a.address + a.nbytes
               for a in package.allocations)
    (root / "expected.json").write_text(json.dumps({
        "reference": package._reference(output.data_ptr(), output.nbytes), "bytes": list(expected),
        "addresses": {allocation.name: allocation.address for allocation in package.allocations},
    }))
    print(f"exported fixed-address graph with {len(package.allocations)} allocations and {len(package.modules)} cubins", flush=True)
    storage.cuda.call("cuMemFree_v2", external)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    prepare(root)
    if len(sys.argv) == 3:
        os.execv(sys.argv[2], [sys.argv[2], str(root)])
