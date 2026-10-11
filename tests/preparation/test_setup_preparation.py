"""Local editable source verification never mutates FlashInfer or repairs its tree."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("setup_preparation", ROOT / "scripts/setup_preparation.py")
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


def git(checkout, *args):
    return subprocess.check_output(["git", "-C", str(checkout), *args], text=True).strip()


@pytest.fixture
def source(tmp_path, monkeypatch):
    project, checkout = tmp_path / "ignition", tmp_path / "flashinfer"
    project.mkdir()
    checkout.mkdir()
    git(checkout, "init", "--quiet")
    (checkout / "source.py").write_text("VERSION = 1\n")
    git(checkout, "add", "source.py")
    git(checkout, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--quiet", "-m", "Test source")
    revision = git(checkout, "rev-parse", "HEAD")
    (project / "pyproject.toml").write_text(
        '[tool.uv.sources]\nflashinfer-python = { path = "../flashinfer", editable = true }\n'
        f'[tool.ignition.source-revisions]\nflashinfer = "{revision}"\n')
    monkeypatch.setattr(setup, "ROOT", project)
    return project, checkout, revision


def test_clean_pinned_checkout_is_verified_without_mutation(source):
    project, checkout, revision = source
    assert setup.verify_flashinfer() == {
        "checkout": str(checkout), "head": revision, "expected_revision": revision,
        "editable": True, "clean": True, "status": "",
    }
    assert not (project / ".build-sources").exists()
    assert git(checkout, "status", "--porcelain=v1") == ""


@pytest.mark.parametrize("kind", ["tracked", "untracked", "staged"])
def test_dirty_checkout_is_rejected_and_preserved(source, kind):
    _, checkout, revision = source
    path = checkout / ("untracked.py" if kind == "untracked" else "source.py")
    path.write_text("VERSION = 2\n")
    if kind == "staged":
        git(checkout, "add", "source.py")
    before = git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
    with pytest.raises(RuntimeError, match="clean FlashInfer worktree"):
        setup.verify_flashinfer()
    assert git(checkout, "status", "--porcelain=v1", "--untracked-files=all") == before
    assert git(checkout, "rev-parse", "HEAD") == revision
    assert path.read_text() == "VERSION = 2\n"


def test_wrong_head_is_rejected_without_checkout(source):
    project, checkout, revision = source
    configuration = project / "pyproject.toml"
    configuration.write_text(configuration.read_text().replace(revision, "0" * 40))
    with pytest.raises(RuntimeError, match="expected HEAD"):
        setup.verify_flashinfer()
    assert git(checkout, "rev-parse", "HEAD") == revision


def test_missing_checkout_is_rejected_without_clone(source):
    project, checkout, _ = source
    checkout.rename(checkout.with_name("saved-flashinfer"))
    with pytest.raises(RuntimeError, match="does not exist"):
        setup.verify_flashinfer()
    assert not checkout.exists()
    assert not (project / ".build-sources").exists()


def test_noneditable_source_is_rejected(source):
    project, _, _ = source
    configuration = project / "pyproject.toml"
    configuration.write_text(configuration.read_text().replace("editable = true", "editable = false"))
    with pytest.raises(RuntimeError, match="editable local source"):
        setup.verify_flashinfer()


def test_setup_reports_verified_source_before_locked_sync(monkeypatch, capsys):
    record = {"checkout": "/example/flashinfer", "head": "a" * 40, "clean": True}
    monkeypatch.setattr(setup, "verify_flashinfer", lambda: record)
    calls = []
    monkeypatch.setattr(setup.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert setup.main() == 0
    assert json.loads(capsys.readouterr().out) == {"flashinfer_source": record}
    assert len(calls) == 1
    command = calls[0][0][0]
    assert command[:3] == ["uv", "sync", "--locked"]
    assert calls[0][1]["env"]["BUILD_NVEP"] == "0"


def test_failed_source_verification_never_runs_sync(monkeypatch, capsys):
    def fail():
        raise RuntimeError("dirty source")

    monkeypatch.setattr(setup, "verify_flashinfer", fail)
    monkeypatch.setattr(setup.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("unexpected sync"))
    assert setup.main() == 1
    assert "dirty source" in capsys.readouterr().err


def test_project_and_lock_use_only_main_editable_flashinfer():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    assert project["tool"]["uv"]["sources"]["flashinfer-python"] == {"path": "../flashinfer", "editable": True}
    assert set(project["tool"]["ignition"]["source-revisions"]) == {"flashinfer"}
    dependency = next(package for package in lock["package"] if package["name"] == "flashinfer-python")
    assert dependency["source"] == {"editable": "../flashinfer"}
    assert project["tool"]["setuptools"]["packages"]["find"]["include"] == ["ignition*"]
