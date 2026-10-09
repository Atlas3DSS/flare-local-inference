"""Test that MambaPool's per-layer SSM state views have the same
memory layout and strides as SSMStatePool's independent tensors.

The FLA Triton kernels use raw pointer arithmetic (initial_state + index * stride_h),
so contiguity and stride correctness are critical.
"""

import torch
from serve.cache.mamba_pool import MambaPool
from serve.cache.ssm_pool import SSMStatePool


POOL_KWARGS = dict(
    num_linear_layers=3,
    num_heads=4,
    head_v_dim=16,
    head_k_dim=16,
    conv_dim=128,
    conv_kernel=4,
)


def test_ssm_state_contiguous():
    """Per-layer SSM state view must be contiguous."""
    pool = MambaPool(num_slots=8, device="cpu", **POOL_KWARGS)
    for i in range(3):
        layer_state = pool.ssm_state_for_layer(i)
        assert layer_state.is_contiguous(), f"Layer {i} SSM state is not contiguous"


def test_conv_state_contiguous():
    """Per-layer conv state must be contiguous."""
    pool = MambaPool(num_slots=8, device="cpu", **POOL_KWARGS)
    for i in range(3):
        layer_conv = pool.conv_state_for_layer(i)
        assert layer_conv.is_contiguous(), f"Layer {i} conv state is not contiguous"


def test_ssm_strides_match_old_pool():
    """Strides of MambaPool per-layer view must match SSMStatePool tensors."""
    old = SSMStatePool(num_slots=8, device="cpu", **POOL_KWARGS)
    new = MambaPool(num_slots=8, device="cpu", **POOL_KWARGS)

    for i in range(3):
        old_strides = old.ssm_state[i].stride()
        new_strides = new.ssm_state_for_layer(i).stride()
        # Strides should match (the only difference is dim0 size: 8 vs 9).
        assert old_strides == new_strides, (
            f"Layer {i} stride mismatch: old={old_strides}, new={new_strides}"
        )


def test_conv_strides_match_old_pool():
    """Strides of MambaPool per-layer conv view must match SSMStatePool tensors."""
    old = SSMStatePool(num_slots=8, device="cpu", **POOL_KWARGS)
    new = MambaPool(num_slots=8, device="cpu", **POOL_KWARGS)

    for i in range(3):
        old_strides = old.conv_state[i].stride()
        new_strides = new.conv_state_for_layer(i).stride()
        assert old_strides == new_strides, (
            f"Layer {i} stride mismatch: old={old_strides}, new={new_strides}"
        )


def test_ssm_pointer_arithmetic():
    """Verify that index * stride gives the correct element, matching
    what the Triton kernel does: initial_state + index * stride_h."""
    pool = MambaPool(num_slots=8, device="cpu", **POOL_KWARGS)
    layer_state = pool.ssm_state_for_layer(0)  # [9, 4, 16, 16]

    # Write known values to slot 5.
    layer_state[5].fill_(42.0)

    # The kernel computes: base_ptr + index * (H * V * K)
    # In our case: H=4, V=16, K=16, so stride_h = 4 * 16 * 16 = 1024
    stride_h = layer_state.stride(0)  # Should be 4*16*16 = 1024
    assert stride_h == 4 * 16 * 16, f"Expected stride 1024, got {stride_h}"

    # Flat view — verify the data at offset index*stride_h is correct.
    flat = layer_state.view(-1)
    offset = 5 * stride_h
    assert flat[offset].item() == 42.0
    assert flat[offset + stride_h - 1].item() == 42.0


def test_ssm_state_not_a_copy():
    """ssm_state_for_layer should return a view, not a copy.
    Writes through the view must be visible in the underlying tensor."""
    pool = MambaPool(num_slots=8, device="cpu", **POOL_KWARGS)
    view = pool.ssm_state_for_layer(1)
    view[3].fill_(99.0)
    assert pool.state.ssm[1, 3].sum().item() == 99.0 * 4 * 16 * 16
