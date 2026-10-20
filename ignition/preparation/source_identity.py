"""Source compatibility identity for preparation and package rebuild metadata."""

import hashlib
from pathlib import Path


def source_identity():
    """Digest of the Python and native sources every rank must run identically."""
    from importlib import util

    # Locate FlashInfer without importing it: CUDA device selection must precede PyTorch's import.
    flashinfer = util.find_spec("flashinfer")
    if flashinfer is None:
        raise RuntimeError("the preparation environment requires FlashInfer")
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for source in (root / "ignition", root / "helios" / "src", root / "heliopause" / "src",
                   Path(flashinfer.origin).resolve().parent):
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".rs", ".c", ".cu", ".cuh", ".h"):
                digest.update(str(path.relative_to(source)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()
