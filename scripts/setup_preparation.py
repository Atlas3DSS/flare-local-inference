"""Install locked dependencies with a clean, pinned editable FlashInfer checkout."""

from __future__ import annotations

import os
import json
from pathlib import Path
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def verify_flashinfer() -> dict:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    revisions = project["tool"]["flare"]["source-revisions"]
    revision = revisions["flashinfer"]
    source = project["tool"]["uv"]["sources"]["flashinfer-python"]
    if source.get("editable") is not True or not isinstance(source.get("path"), str):
        raise RuntimeError("FlashInfer must use an editable local source")
    checkout = (ROOT / source["path"]).resolve()
    if not checkout.is_dir():
        raise RuntimeError(f"FlashInfer checkout does not exist: {checkout}")
    if Path(git(checkout, "rev-parse", "--show-toplevel")).resolve() != checkout:
        raise RuntimeError(f"FlashInfer source must be a Git checkout root: {checkout}")
    actual = git(checkout, "rev-parse", "HEAD")
    if actual != revision:
        raise RuntimeError(f"{checkout}: expected HEAD {revision}, found {actual}")
    status = git(checkout, "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none")
    if status:
        raise RuntimeError(f"{checkout}: reproducible setup requires a clean FlashInfer worktree:\n{status}")
    return {"checkout": str(checkout), "head": actual, "expected_revision": revision,
            "editable": True, "clean": True, "status": status}


def main() -> int:
    try:
        source = verify_flashinfer()
        print(json.dumps({"flashinfer_source": source}, sort_keys=True), flush=True)
        environment = dict(os.environ, BUILD_NVEP="0", FLASHINFER_BUILD_NO_PIP="1")
        subprocess.run(
            ["uv", "sync", "--locked", "--extra", "prepare", "--extra", "test",
             "--reinstall-package", "flashinfer-python", "--reinstall-package", "flare-helios"],
            cwd=ROOT, env=environment, check=True,
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"preparation environment setup failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
