"""GLM graph catalog capacities and rank geometry before loading weights."""

import pytest

from flare.recipes.glm53.bootstrap import PreparationConfig


def test_default_catalog_covers_4096_prefill_and_four_requests():
    config = PreparationConfig()
    assert config.prefill_capacities == (128, 256, 512, 1024, 2048, 4096)
    assert config.batch_capacities == (4, 2, 1)
    assert config.page_capacity == 4 * (262144 // 256)
    assert PreparationConfig(prefill_chunk_size=3072).prefill_capacities[-2:] == (2048, 3072)
    assert PreparationConfig(max_active=1, pages=64).batch_capacities == (1,)
    assert PreparationConfig(max_active=3).batch_capacities == (4, 2, 1)


@pytest.mark.parametrize("values", [
    {"devices": (0,)}, {"devices": (0, 0)}, {"devices": (0, 1, 2)}, {"devices": (-1, 0)},
    {"context": 513}, {"context": 2097152}, {"prefill_chunk_size": 4097},
    {"context": 256, "prefill_chunk_size": 512}, {"max_active": 65}, {"pages": 0},
    {"snapshot_budget_bytes": 0}, {"headroom_bytes": -1},
])
def test_invalid_prepared_geometry_rejects_before_gpu_use(values):
    with pytest.raises((ValueError, TypeError)):
        PreparationConfig(**values)


@pytest.mark.parametrize("concurrency", [1, 5, 64])
@pytest.mark.parametrize("no_compile", [False, True])
def test_cli_passes_compile_policy_to_rank_group(monkeypatch, tmp_path, no_compile, concurrency):
    from flare.__main__ import main

    (tmp_path / "config.json").write_text('{"model_type":"glm5_next"}')
    configs = []

    class Model:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def manifest(self):
            return {}

    def prepare_group(snapshot, config, prepare, *, progress):
        configs.append(config)
        return Model()

    monkeypatch.setattr("flare.preparation.rank_group.prepare_group", prepare_group)
    assert main(["--snapshot", str(tmp_path), "--tp", "2", "--prepare-only", "--concurrency", str(concurrency),
                 *(["--no-compile"] if no_compile else [])]) == 0
    assert configs[0].compile_decoder is not no_compile
    assert configs[0].max_active == concurrency
