"""Process environment that preparation libraries read, set from Flare's configuration.

PyTorch, b12x, Triton, and the CuTe DSL read these variables when they are
imported or first used, so `apply` runs before any of them is imported. Rank
and compiler processes inherit the result.
"""

import os
from importlib import util
from pathlib import Path

_DLL_DIRECTORIES = []


def toolkit(configured: Path | None) -> Path | None:
    """Locate a CUDA toolkit in installed packages, else use the configured directory."""
    try:
        package = util.find_spec("nvidia")
    except (ImportError, ValueError):
        package = None
    for root in ([] if package is None else package.submodule_search_locations or []):
        for candidate in sorted(Path(root).glob("cu*")):
            if (candidate / "bin" / "nvcc").is_file():
                return candidate
    return configured


def apply(configuration, gpus) -> None:
    """Select the plan's GPUs as process-local devices 0 through N-1 and place every cache."""
    kernel_cache = configuration.directory("kernel_cache")
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": ",".join(gpu.uuid for gpu in gpus),
        "HF_HUB_CACHE": str(configuration.directory("hub_cache")),
        "B12X_COMPILE_CACHE_DIR": str(kernel_cache / "b12x"),
        "TRITON_CACHE_DIR": str(kernel_cache / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(kernel_cache / "inductor"),
    })
    cuda = toolkit(configuration.directory("cuda"))
    if cuda is not None:
        os.environ["CUDA_HOME"] = str(cuda)
        if os.name == "nt":
            os.environ["CUDA_PATH"] = str(cuda)
            os.environ["TRITON_PTXAS.EXE_PATH"] = str(cuda / "bin/ptxas.exe")
            for path in (cuda / "bin", cuda / "bin/x64", cuda / "lib/x64"):
                if path.is_dir():
                    _DLL_DIRECTORIES.append(os.add_dll_directory(str(path)))
