"""Identify explicitly declared disposable backing storage for package export."""

from dataclasses import fields, is_dataclass

import torch

from ignition.preparation.scratch import GraphBufferPool


def storage_ranges(values):
    """Deduplicate backing storage for explicitly declared scratch tensor trees."""
    storage = {}

    def visit(value):
        if value is None:
            return
        if isinstance(value, torch.Tensor):
            if value.is_meta:
                raise ValueError("export scratch must have materialized storage")
            backing = value.untyped_storage()
            if backing.nbytes():
                storage[backing.data_ptr()] = backing.nbytes()
        elif is_dataclass(value):
            for field in fields(value):
                visit(getattr(value, field.name))
        elif isinstance(value, GraphBufferPool):
            visit(value.chunks)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)
        else:
            raise TypeError(f"unsupported scratch declaration: {type(value).__name__}")

    visit(values)
    return tuple(sorted(storage.items()))
