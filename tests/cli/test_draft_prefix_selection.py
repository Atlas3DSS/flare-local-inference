"""GPU prefix allocation is an explicit preparation choice, not a packed-mode default."""

import pytest

from ignition.cli import _parser, preparation, display
from ignition.errors import UsageError
from ignition.recipes import resolve
from ignition.recipes.declaration import Speculation


def settings(*flags):
    args = _parser().parse_args(["run", "deepseek-v4.1-flash", *flags])
    return preparation.speculation(args, resolve("deepseek-v4.1-flash"))


def test_existing_packed_command_keeps_cpu_prefix_selection():
    assert settings().draft_prefix_selection == "cpu"
    assert settings("--packed-verification") == Speculation(5, packed_verification=True)
    assert settings("--packed-verification", "--draft-prefix-selection", "cpu") == settings("--packed-verification")


def test_gpu_prefix_selection_is_retained_and_printed():
    selected = settings("--packed-verification", "--draft-prefix-selection", "gpu")
    assert selected == Speculation(5, packed_verification=True, draft_prefix_selection="gpu")
    assert "GPU prefix selection" in display.speculation(selected)
    assert "GPU prefix selection" in display.speculation(selected, brief=True)


@pytest.mark.parametrize("flags", [
    (), ("--no-mtp",), ("--packed-verification", "--no-mtp-adaptive-verification"),
    ("--packed-verification", "--mtp-draft-tokens", "0"),
    ("--packed-verification", "--mtp-proposal-source", "copy"),
    ("--packed-verification", "--history-drafts"),
])
def test_incompatible_gpu_selection_fails_before_preparation(flags):
    with pytest.raises(UsageError):
        settings(*flags, "--draft-prefix-selection", "gpu")


def test_plan_records_selection_and_recipe_receives_it():
    from tests.cli.test_planning import derive
    from ignition.recipes.deepseek41.bootstrap import PreparationConfig

    selected = Speculation(5, packed_verification=True, draft_prefix_selection="gpu")
    assert derive(speculation=selected).to_dict()["speculation"]["draft_prefix_selection"] == "gpu"
    config = PreparationConfig(split_speculation=True, packed_verification=True, draft_prefix_selection="gpu")
    assert config.draft_prefix_selection == "gpu"
    assert PreparationConfig(split_speculation=True, packed_verification=True).draft_prefix_selection == "cpu"
    with pytest.raises(ValueError, match="requires packed"):
        PreparationConfig(draft_prefix_selection="gpu")
    with pytest.raises(ValueError, match="cpu or gpu"):
        PreparationConfig(draft_prefix_selection="automatic")


def test_rebuild_metadata_records_prefix_selection(monkeypatch):
    from types import SimpleNamespace
    from tests.cli.test_planning import derive
    from ignition.preparation import rebuild_manifest, source_identity, runtime_package_pb2 as p

    monkeypatch.setattr(source_identity, "source_identity", lambda: "test-source")
    selected = Speculation(5, packed_verification=True, draft_prefix_selection="gpu")
    encoded = rebuild_manifest.from_plan(SimpleNamespace(directory=None), derive(speculation=selected),
        compiled_forwards=True)
    assert p.RebuildManifest.FromString(encoded).speculation.draft_prefix_selection == "gpu"
    legacy = p.RebuildSpeculation.FromString(b"\x08\x05\x10\x01\x1a\x03mtp\x30\x01")
    assert legacy.draft_prefix_selection == ""
    assert legacy.draft_tokens == 5 and legacy.packed_verification


def test_gpu_prefix_catalog_selects_a_distinct_package_cache_entry(monkeypatch, tmp_path):
    from dataclasses import replace
    from pathlib import Path
    from types import SimpleNamespace
    from ignition.cli import runtime
    from tests.cli.test_planning import derive, SOLO

    cpu = derive(speculation=Speculation(5, packed_verification=True))
    gpu = replace(cpu, speculation=replace(cpu.speculation, draft_prefix_selection="gpu"))
    model = SimpleNamespace(checkpoint=SOLO.checkpoints[0], directory=None)
    monkeypatch.setattr(runtime.metadata, "version", lambda _: "test-version")
    monkeypatch.setattr(runtime.util, "find_spec", lambda _: SimpleNamespace(origin=str(tmp_path / "__init__.py")))
    monkeypatch.setattr(Path, "rglob", lambda *args: ())
    assert runtime.package_key(model, cpu, False) != runtime.package_key(model, gpu, False)
