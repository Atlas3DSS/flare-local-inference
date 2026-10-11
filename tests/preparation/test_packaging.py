"""CPU preparation package and command-line smoke checks."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tomllib


ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_pyproject_declares_preparation_entrypoint_and_packages():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    includes = set(pyproject["tool"]["setuptools"]["packages"]["find"]["include"])

    assert "flare*" in includes
    assert not any(name.startswith("serve") for name in includes)
    assert pyproject["project"]["dependencies"] == []
    assert pyproject["project"]["scripts"]["flare"] == "flare.__main__:main"
    assert pyproject["project"]["scripts"]["flare-inspect"] == "flare.preparation.__main__:main"


def test_cli_help_smoke():
    result = subprocess.run(
        [sys.executable, "-m", "flare.preparation", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "snapshot" in result.stdout
    assert "without loading tensor payloads" in result.stdout


def test_package_import_has_no_runtime_or_cuda_side_effects():
    result = subprocess.run(
        [sys.executable, "-c", "import sys, flare; "
         "assert flare.__all__ == ['__version__']; "
         "assert 'torch' not in sys.modules; "
         "assert 'flare_helios' not in sys.modules; "
         "assert not any(n == 'serve' or n.startswith('serve.') for n in sys.modules)"],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not (ROOT / "serve").exists()


def test_server_cli_help_does_not_prepare_model():
    result = subprocess.run(
        [sys.executable, "-m", "flare", "--help"],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    for command in ("run", "models", "pull", "status", "bench", "config", "skill"):
        assert f"\n    {command} " in result.stdout
    for command in ("prepare", "plan"):
        assert f"\n    {command} " not in result.stdout
