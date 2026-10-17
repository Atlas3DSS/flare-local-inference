"""External draft models share the configured target-checkpoint hub cache."""

from importlib import import_module
import sys
from types import SimpleNamespace

import pytest

from ignition import planning
from ignition.cli.config import Configuration, Setting
from ignition.cli.preparation import Host, derive
from ignition.recipes import ResolvedModel, recipes


@pytest.mark.parametrize("name", ["nemotron_h", "kimi3"])
@pytest.mark.parametrize("custom_cache", [False, True])
@pytest.mark.parametrize("drafting", [False, True])
def test_draft_preparation_reuses_the_hub_snapshot(monkeypatch, tmp_path, name, custom_cache, drafting):
    import huggingface_hub

    monkeypatch.setenv("HOME", str(tmp_path))
    values = {"paths": {"hub_cache": str(tmp_path / "custom-hub")}} if custom_cache else {}
    configuration = Configuration(tmp_path / "config.toml", values, exists=True)
    hub_cache = configuration.directory("hub_cache")
    recipe = next(recipe for recipe in recipes() if recipe.name == name)
    model = ResolvedModel(recipe, recipe.checkpoints[0])
    world = recipe.tensor_parallel[0]
    gpus = tuple(planning.Gpu(i, "Blackwell", f"GPU-{i}", 256 << 30, 250 << 30, (12, 0))
                 for i in range(world))
    plan = derive(configuration, model, Host(gpus, None), gpus=f"0-{world - 1}",
                  concurrency=Setting(4, "test"), speculation=recipe.speculation if drafting else None)
    assert plan.hub_cache == hub_cache

    draft = import_module(f"ignition.recipes.{name}.draft_config")
    snapshot = hub_cache / f"models--{draft.REPOSITORY.replace('/', '--')}" / "snapshots" / draft.REVISION
    if drafting:
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
        (snapshot / "model.safetensors").touch()

    # Use the real hub reader offline: it must find this exact cached revision,
    # even when the library was imported before Flare's cache configuration.
    download = huggingface_hub.snapshot_download
    downloads = []

    def cached_download(repository, **options):
        downloads.append(repository)
        assert options["allow_patterns"] == ("config.json", "model.safetensors")
        return download(repository, **options, local_files_only=True)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", cached_download)

    def prepare_model(snapshot, config, **options):
        return config

    def preparation_config(**options):
        return SimpleNamespace(**(dict(draft_checkpoint=None) | options))

    monkeypatch.setitem(sys.modules, f"ignition.recipes.{name}.bootstrap", SimpleNamespace(
        PreparationConfig=preparation_config, prepare=prepare_model, prepare_rank=prepare_model))
    monkeypatch.setitem(sys.modules, "ignition.preparation.rank_group", SimpleNamespace(
        prepare_group=lambda snapshot, config, prepare, **options: config))
    config = recipe.prepare(tmp_path / "target", plan, compile_forwards=True, kernel_cache=None, progress=None)
    assert config.draft_checkpoint == (str(snapshot) if drafting else None)
    assert downloads == ([draft.REPOSITORY] if drafting else [])
