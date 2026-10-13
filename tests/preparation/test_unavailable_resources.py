"""Placeholders for process-local resources survive the pickling of compiled-forward guards."""

import pickle

import pytest

from ignition.preparation.forward_metadata import _Unavailable


def test_placeholder_round_trips_through_pickle():
    restored = pickle.loads(pickle.dumps(_Unavailable("RoCE proxy")))
    assert restored.description == "RoCE proxy"
    with pytest.raises(RuntimeError, match="RoCE proxy.post"):
        restored.post
