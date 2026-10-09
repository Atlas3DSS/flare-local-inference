"""Tests for standalone kernel imports — verify no sglang runtime dependency."""


def test_causal_conv1d_imports():
    from serve.kernels.mamba.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    assert callable(causal_conv1d_fn)
    assert callable(causal_conv1d_update)


def test_fused_gdn_gating_imports():
    from serve.kernels.fla.fused_gdn_gating import fused_gdn_gating
    assert callable(fused_gdn_gating)


def test_no_sglang_imports_in_model():
    """serve/model/ should not import from sglang at module level."""
    import importlib
    import sys

    # Track sglang imports.
    original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    sglang_imports = []

    def tracking_import(name, *args, **kwargs):
        if name.startswith("sglang"):
            sglang_imports.append(name)
        return original_import(name, *args, **kwargs)

    # Check that gdn module source doesn't have sglang imports.
    import inspect
    from serve.model import gdn
    source = inspect.getsource(gdn)
    # No "from sglang" in the module source.
    assert "from sglang" not in source, f"Found sglang import in gdn.py: {source}"


def test_paged_attention_module_imports():
    from serve.model.attention import B12xPagedAttention

    assert B12xPagedAttention is not None
