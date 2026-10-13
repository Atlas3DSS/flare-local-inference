"""Bound preparation's transient memory on a GPU that allocates from system memory.

An integrated GPU such as GB10 (DGX Spark) allocates device memory from the
operating system's memory, so CUDA's free memory is what the whole host has
left. Autotuning trials and compiler processes draw on it alongside the
operating system and processes that start later. Preparation therefore keeps
the plan's headroom and reserve free while it works, as it does once KV pages
are allocated, instead of sizing its transient work from all free memory.
"""

import torch

# One spawned b12x compiler: Python, PyTorch and the CuTe DSL take 0.34 GiB on GB10 before compiling,
# and the process grows by about 2.4 MB per CuTe program it compiles.
B12X_COMPILER_BYTES = 1 << 30
# The preparing process's own growth while tuning: kernel modules it loads to race candidates, Triton
# kernels it compiles itself, and allocator segments; about 0.3-0.5 GiB per Qwen tuning pass on GB10.
PROCESS_GROWTH_BYTES = 1 << 30
# Host memory the preparing process holds for each compiled forward it restores to warm and capture:
# its generated module and kernel launchers; 94 MB on average for Qwen 3.8 forwards on GB10. Serving
# replays graphs without them, but they share memory with the KV pages until export completes.
COMPILED_FORWARD_BYTES = 128 << 20
# Package export in the preparing process: graph descriptions and payload downloads, about 1.5 GiB
# for Qwen 3.8 on GB10, while the KV pages, first written when export clears them, hold their memory.
EXPORT_BYTES = 2 << 30


def spare_bytes(device, protected_bytes: int) -> int:
    """Free memory above the protected amount, which transient preparation work may use."""
    return torch.cuda.mem_get_info(device)[0] - protected_bytes


def shortfall(work: str, device, protected_bytes: int) -> MemoryError:
    gib = 1 << 30
    free = torch.cuda.mem_get_info(device)[0]
    return MemoryError(
        f"{work} needs memory beyond the {protected_bytes / gib:.1f} GiB of headroom and [memory] reserve that a "
        f"unified-memory GPU keeps for the system; {free / gib:.1f} GiB is free. "
        "A lower --max-concurrency or a smaller [memory] reserve leaves more.")
