"""Compatibility shim for the old MambaPool import path."""

from serve.cache.tensor_arena import TensorArena


MambaPool = TensorArena
