from types import SimpleNamespace

import pytest

from ignition.direct.executor import StepRunner


def test_eager_execution_requires_the_recipe_to_retain_its_forwards():
    with pytest.raises(RuntimeError, match="retains no prepared forwards"):
        StepRunner(SimpleNamespace(forwards=None), eager=True)
