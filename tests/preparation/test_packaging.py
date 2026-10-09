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

    assert "sparkgap*" in includes
    assert not any(name.startswith("serve") for name in includes)
    assert pyproject["project"]["dependencies"] == []
    assert pyproject["project"]["scripts"]["sparkgap"] == "sparkgap.__main__:main"
    assert pyproject["project"]["scripts"]["sparkgap-inspect"] == "sparkgap.preparation.__main__:main"


def test_cli_help_smoke():
    result = subprocess.run(
        [sys.executable, "-m", "sparkgap.preparation", "--help"],
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
        [sys.executable, "-c", "import sys, sparkgap; "
         "assert sparkgap.__all__ == ['__version__']; "
         "assert 'torch' not in sys.modules; "
         "assert 'sparkgap_native' not in sys.modules; "
         "assert not any(n == 'serve' or n.startswith('serve.') for n in sys.modules)"],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not (ROOT / "serve").exists()


def test_server_cli_help_does_not_prepare_model():
    result = subprocess.run(
        [sys.executable, "-m", "sparkgap", "--help"],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--prefill-chunk-size" in result.stdout
    assert "--bind" in result.stdout
