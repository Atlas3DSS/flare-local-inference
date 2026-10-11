"""Prepared Triton launches that remain visible to whole-forward compilation."""

from dataclasses import dataclass

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode


@dataclass
class PreparedTriton:
    kernel: object
    compiled: object
    args: tuple
    grid: tuple
    options: dict

    def __call__(self):
        if torch.compiler.is_compiling() or get_proxy_mode() is not None:
            torch.library.wrap_triton(self.kernel)[self.grid](*self.args, **self.options)
        else:
            self.compiled[self.grid](*self.args)


def prepare_triton(kernel, args, grid, **options):
    compiled = kernel.warmup(*args, grid=grid, **options)
    return PreparedTriton(kernel, compiled, args, grid, options)


def launch_triton(kernel, compiled, args, grid, **options):
    if torch.compiler.is_compiling() or get_proxy_mode() is not None:
        torch.library.wrap_triton(kernel)[grid](*args, **options)
    else:
        compiled[grid](*args)
