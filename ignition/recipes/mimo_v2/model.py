"""Checkpoint-supplied residual layers with recipe-owned state and vision delivery."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DecoderEnvironment:
    canonical: tuple[type, type]
    vision: object

    @classmethod
    def load(cls, snapshot, vision):
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        classes = tuple(get_class_from_dynamic_module("modeling_mimo_v2." + name,
                        str(snapshot), local_files_only=True)
                        for name in ("MiMoV2DecoderLayer", "MiMoV2RMSNorm"))
        return cls(classes, vision)
