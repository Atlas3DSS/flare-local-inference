"""Locate native executables and cache packages by their preparation inputs."""

from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
from importlib import metadata, util
import json
from pathlib import Path
import shutil

from ignition.errors import UsageError


def binary():
    installed = shutil.which("flare")
    if installed:
        return Path(installed)
    root = Path(__file__).resolve().parents[2]
    executable = root / "flare/target/release/flare"
    if not executable.is_file():
        raise UsageError("flare executable is missing; build flare/Cargo.toml or install flare on PATH")
    return executable


def package_key(model, plan, no_compile):
    inputs = {"version": 3, "checkpoint": model.checkpoint.revision,
              "repository": model.checkpoint.repository, "compile": not no_compile,
              "concurrency": plan.concurrency, "prefill_chunk": plan.prefill_chunk,
              "context_parallel": plan.context_parallel,
              "cache_states": plan.cache_states, "reserve_bytes": plan.reserve_bytes,
              "gpus": [(gpu.name, gpu.total_bytes) for gpu in plan.gpus],
              "speculation": None if plan.speculation is None else asdict(plan.speculation)}
    inputs["libraries"] = {name: metadata.version(name) for name in
                           ("torch", "transformers", "flashinfer-python", "cuda-python")}
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode())
    root = Path(__file__).resolve().parents[2]
    source_roots = [root / "ignition", root / "heliopause/src"]
    flashinfer = util.find_spec("flashinfer")
    if flashinfer is None:
        raise UsageError("the preparation environment requires FlashInfer")
    source_roots.append(Path(flashinfer.origin).parent)
    for source_root in source_roots:
        for source in sorted(source_root.rglob("*")):
            if source.is_file() and source.suffix in (".py", ".rs", ".cu", ".cuh", ".cpp", ".h"):
                key.update(str(source.relative_to(source_root)).encode())
                key.update(source.read_bytes())
    if model.directory is not None:
        for source in sorted(model.directory.rglob("*")):
            if source.is_file():
                stat = source.stat()
                key.update(str((source.relative_to(model.directory), stat.st_size, stat.st_mtime_ns)).encode())
    return key.hexdigest()


@contextmanager
def cached_package(configuration, model, plan, no_compile, publisher):
    root = configuration.directory("package_cache")
    root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256((package_key(model, plan, no_compile) + publisher.fingerprint.hex()
                          + publisher.namespace + publisher.official_source).encode()).hexdigest()
    with (root / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield root / f"{key}.safetensors"
