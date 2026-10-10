"""Canonical Transformers module structure for Qwen graph preparation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from typing import TYPE_CHECKING

from flare.preparation.checkpoint import CheckpointError, CheckpointManifest
from flare.preparation.qwen38 import CONFIG_SHA256, TRANSFORMERS_REVISION

if TYPE_CHECKING:
    from transformers import Qwen4ExpForConditionalGeneration


def create_model_tree(manifest: CheckpointManifest) -> Qwen4ExpForConditionalGeneration:
    """Construct the pinned canonical module tree on meta storage without loading weights.

    The returned tree supplies module names, geometry, and canonical component
    methods. Quantized weights and prepared forwards must be attached before
    execution; Transformers' generic ModelOpt checkpoint loader is unsupported.
    """
    distribution = importlib.metadata.distribution("transformers")
    provenance = json.loads(distribution.read_text("direct_url.json") or "{}")
    if provenance.get("vcs_info", {}).get("commit_id") != TRANSFORMERS_REVISION:
        raise RuntimeError(f"Qwen preparation requires Transformers {TRANSFORMERS_REVISION}")
    raw = (manifest.snapshot / "config.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA256:
        raise CheckpointError("config.json changed after checkpoint inspection")

    import torch
    from transformers import Qwen4ExpConfig, Qwen4ExpForConditionalGeneration

    config = Qwen4ExpConfig.from_dict(json.loads(raw))
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen4ExpForConditionalGeneration(config)
    model.eval().requires_grad_(False)
    return model
