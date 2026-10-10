"""Prepared Triton launches that remain visible to whole-forward compilation."""

from functools import partial

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode


def prepare_triton(kernel, args, grid, **options):
    compiled = kernel.warmup(*args, grid=grid, **options)
    return partial(launch_triton, kernel, compiled, args, grid, **options)


def launch_triton(kernel, compiled, args, grid, **options):
    if torch.compiler.is_compiling() or get_proxy_mode() is not None:
        torch.library.wrap_triton(kernel)[grid](*args, **options)
    else:
        compiled[grid](*args)
