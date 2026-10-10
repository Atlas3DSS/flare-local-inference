"""Shared scratch and persistent state ownership for complete GLM text stacks."""

import os
from pathlib import Path

import pytest
import torch

from flare.components.parallel import TensorParallelShard
from flare.recipes.glm53.model import create_model_tree
from flare.recipes.glm53.stack import DecoderScratch
from flare.recipes.glm53.state import DecoderMetadata, DecoderState


@pytest.mark.parametrize("world,rank", [(2, 0), (4, 3)])
def test_canonical_layers_share_intermediates_and_own_distinct_state(world, rank):
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT",
        "/data/checkpoints-masters/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"))
    if not snapshot.is_dir():
        pytest.skip("GLM checkpoint is not available")
    tree = create_model_tree(snapshot)
    layers = tree.model.language_model.layers
    initialized = torch.cuda.is_initialized()
    scratch = DecoderScratch.allocate(65, 1, "prefill", "cpu",
        shard=TensorParallelShard(rank, world), sample_rows=1)
    state = DecoderState.allocate(tree.config.text_config.layer_types, world, 3, 1, "cpu")
    metadata = DecoderMetadata.allocate(65, 1, 1, "cpu")
    buffers = list(state.named_tensors())
    assert len(buffers) == 34 * 2 + 11 * 3
    assert len({name for name, _ in buffers}) == len(buffers)
    assert len({value.untyped_storage().data_ptr() for _, value in buffers}) == len(buffers)
    for index, module in enumerate(layers):
        layer = scratch.layer(module.block_type, sparse=hasattr(module.mlp, "experts"))
        assert layer.attention_hc is scratch.attention_hc and layer.mlp_hc is scratch.mlp_hc
        assert layer.attention_output is scratch.attention_output and layer.mlp_output is scratch.mlp_output
        assert layer.attention is (scratch.mla if module.block_type == "indexed_attention" else scratch.kda)
        assert layer.mlp is (scratch.dense if index < 3 else scratch.moe)
    assert (scratch.logits is not None) == (rank == 0)
    assert not metadata.recurrent.num_tokens.count_nonzero()
    assert not metadata.recurrent.num_seqs.count_nonzero()
    assert (metadata.indexed.positions == -1).all()
    assert (metadata.indexed.token_slots == -1).all()
    assert (metadata.indexed.state_slots == -1).all()
    for _, buffer in buffers:
        buffer.fill_(1)
    state.clear_for_preparation()
    assert all(not value.count_nonzero() for _, value in buffers)
    assert torch.cuda.is_initialized() == initialized
