"""GLM graph catalog capacities and rank geometry before loading weights."""

import pytest

from flare.recipes.glm53.bootstrap import PreparationConfig


def test_default_catalog_covers_4096_prefill_and_four_requests():
    config = PreparationConfig()
    assert config.prefill_capacities == (1, 16, 64, 128, 256, 512, 1024, 2048, 4096)
    assert config.batch_capacities == (4, 2, 1)
    assert config.page_capacity == 4 * (262144 // 256)
    assert PreparationConfig(prefill_chunk_size=3072).prefill_capacities[-2:] == (2048, 3072)
    assert PreparationConfig(max_active=1, pages=64).batch_capacities == (1,)
    assert PreparationConfig(max_active=3).batch_capacities == (4, 2, 1)


@pytest.mark.parametrize("values", [
    {"devices": (0,)}, {"devices": (0, 0)}, {"devices": (0, 1, 2)}, {"devices": (-1, 0)},
    {"context": 513}, {"context": 2097152}, {"prefill_chunk_size": 4097},
    {"context": 256, "prefill_chunk_size": 512}, {"max_active": 5}, {"pages": 0},
    {"snapshot_budget_bytes": 0}, {"headroom_bytes": -1},
])
def test_invalid_prepared_geometry_rejects_before_gpu_use(values):
    with pytest.raises((ValueError, TypeError)):
        PreparationConfig(**values)
