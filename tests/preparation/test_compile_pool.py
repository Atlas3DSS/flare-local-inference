"""Compiler workers preserve symbolic metadata without serializing tracing state."""

import torch
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from flare.preparation.compile_pool import _GraphPickler


def test_symbolic_metadata_uses_worker_shape_environment():
    source = ShapeEnv()
    dimension = source.create_unbacked_symint()
    values = (dimension, dimension > 0, dimension / 2)
    mode = FakeTensorMode(shape_env=ShapeEnv())
    restored = _GraphPickler.loads(_GraphPickler.dumps(values), mode)
    assert tuple(type(value) for value in restored) == (torch.SymInt, torch.SymBool, torch.SymFloat)
    assert all(value.node.shape_env is mode.shape_env for value in restored)
    assert [value.node._expr for value in restored] == [value.node._expr for value in values]
    assert [value.node._hint for value in restored] == [value.node._hint for value in values]


def test_fake_tensor_views_preserve_aliases_in_worker_mode():
    source = FakeTensorMode(shape_env=ShapeEnv())
    with source:
        base = torch.empty((16, 32))
        view = base[2:6, ::2]
    mode = FakeTensorMode(shape_env=ShapeEnv())
    restored_base, restored_view = _GraphPickler.loads(_GraphPickler.dumps((base, view)), mode)
    assert restored_base.fake_mode is mode and restored_view.fake_mode is mode
    assert restored_view._base is restored_base
    assert restored_view.untyped_storage()._cdata == restored_base.untyped_storage()._cdata
    assert restored_view.shape == view.shape
    assert restored_view.stride() == view.stride()
    assert restored_view.storage_offset() == view.storage_offset()
