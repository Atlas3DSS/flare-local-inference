"""Packaging and install-smoke checks for the serving stack."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tomllib


ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_pyproject_declares_serve_runtime_deps_and_packages():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    deps = set(pyproject["project"]["dependencies"])
    includes = set(pyproject["tool"]["setuptools"]["packages"]["find"]["include"])

    assert "serve*" in includes
    assert any(dep.startswith("fastapi") for dep in deps)
    assert any(dep.startswith("uvicorn") for dep in deps)
    assert any(dep.startswith("transformers") for dep in deps)
    assert any(dep.startswith("safetensors") for dep in deps)
    assert pyproject["project"]["scripts"]["b12x-serve"] == "serve.cli:main"


def test_cli_help_smoke():
    result = subprocess.run(
        [sys.executable, "-m", "serve.cli", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "b12x serve CLI" in result.stdout
    assert "--web-ui" in result.stdout
