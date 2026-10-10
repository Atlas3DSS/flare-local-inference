"""Exact-weight MTP operator and graph checks on an explicitly assigned GPU."""

import os
from pathlib import Path

import pytest
import torch

from flare.preparation import inspect_qwen38_checkpoint
from flare.preparation.qwen38 import TEXT_PREFIX
from flare.preparation.scratch import ScratchArena
from flare.recipes.qwen38.attention import QSAMetadata, QSAState, RotaryTables
from flare.recipes.qwen38.mtp import MTPScratch, PreparedMTP, PreparedMTPFeedback
from flare.recipes.qwen38.mtp_weights import MTPWeights
from flare.recipes.qwen38.weights import QwenWeightLoader

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an assigned CUDA_VISIBLE_DEVICES and FLARE_RUN_CUDA_TESTS=1")

TRACE = Path("/data/sparkgap-artifacts/text-layer-trace/vllm-moe-inprocess-repeat/vllm-layers.pt")


@pytest.fixture(scope="module")
def weights():
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability(0) in ((12, 0), (12, 1))
    manifest = inspect_qwen38_checkpoint("/data/checkpoints-masters/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD")
    with QwenWeightLoader(manifest, "cuda:0", prefixes=(f"{TEXT_PREFIX}.embed_tokens.", "lm_head.")) as loader:
        embedding = loader.load_tensor(f"{TEXT_PREFIX}.embed_tokens.weight")
        head = loader.load_tensor("lm_head.weight")
        assert loader._session.torch_copy_bytes == 0
    value = MTPWeights.load(manifest, "cuda:0", embedding_weight=embedding, lm_head_weight=head)
    assert value.embedding is embedding and value.lm_head is head
    assert value.stored_bytes == 1596720640
    return value


def inputs():
    trace = torch.load(TRACE, map_location="cpu", weights_only=True)
    hidden = trace["layer_outputs"][0][-1].bfloat16()
    assert torch.equal(hidden.float(), trace["layer_outputs"][0][-1])
    tokens = torch.tensor(trace["prompt_tokens"][1:] + trace["teacher_tokens"][:1], dtype=torch.int64)
    assert hidden.shape == (len(tokens), 10240)
    return hidden, tokens


def test_exact_weight_feedback_matches_published_operator_reference(weights):
    from b12x.preparation import PreparationSession
    from b12x.sequence import mtp_feedback

    hidden, tokens = inputs()
    hidden = hidden.cuda()
    embedding = weights.embedding.index_select(0, tokens.cuda())
    output = torch.empty_like(hidden)
    arena = ScratchArena("cuda:0")
    with PreparationSession(device="cuda:0", autotune=False, compile_workers=0) as session:
        op = PreparedMTPFeedback(weights.module, embedding, hidden, output, arena, session=session, name="mtp.feedback")
        arena.finalize(); arena.freeze()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.stream(stream):
                op()
            stream.synchronize()
            with session.capture(), torch.cuda.graph(graph, stream=stream):
                op()
            session.freeze()
            expected = mtp_feedback.reference.feedback(embedding, hidden.view(-1, 4, 2560),
                weights.module.pre_fc_norm_embedding.weight, weights.module.pre_fc_norm_hidden.weight,
                weights.module.fc_embedding.weight, weights.module.fc_hidden.weight, eps=weights.module.config.rms_norm_eps)
            for poison in (0xA5, 0x5A):
                arena.tensor.fill_(poison)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay(); torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                torch.testing.assert_close(output.view_as(expected), expected, rtol=2e-2, atol=4e-2)
        finally:
            stream.synchronize(); graph.reset()


def test_exact_weight_mtp_capture_replay_preserves_inputs_and_persistent_outputs(weights):
    from b12x.preparation import PreparationSession

    host_hidden, host_tokens = inputs()
    live, capacity = len(host_tokens), 8
    scratch = MTPScratch.allocate(capacity, 1, "cuda:0")
    hidden = torch.zeros_like(scratch.residual)
    tokens = torch.full((capacity,), -1, dtype=torch.int64, device="cuda")
    selected = torch.tensor([live - 1], dtype=torch.int64, device="cuda")
    metadata = QSAMetadata.allocate(tokens=capacity, batch=1, context=128, device=torch.device("cuda:0"))
    state = QSAState.allocate(pages=2, slots=1, device=torch.device("cuda:0"), draft_tokens=7)
    rotary = RotaryTables.allocate(weights.module.config, 128, torch.device("cuda:0"))
    arena = ScratchArena("cuda:0")
    with PreparationSession(device="cuda:0", autotune=False, compile_workers=0) as session:
        op = PreparedMTP(weights, scratch, tokens, hidden, state, metadata, rotary, arena,
            output_indices=selected, session=session, name="mtp", max_cache_pages=2)
        arena.finalize(); arena.freeze()
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.stream(stream):
                op()
            stream.synchronize()
            with session.capture(), torch.cuda.graph(graph, stream=stream):
                op()
            session.freeze()
            hidden[:live].copy_(host_hidden); tokens[:live].copy_(host_tokens)
            metadata.main_block_table[0].copy_(torch.tensor([1, 0], device="cuda", dtype=torch.int32))
            metadata.compressed_block_table.copy_(metadata.main_block_table)
            metadata.raw_state_slot_ids.fill_(0)
            metadata.request_ids[:live].fill_(0)
            metadata.query_positions[:live].copy_(torch.arange(live, device="cuda"))
            metadata.rope_positions[:live].copy_(torch.arange(live, device="cuda")[:, None].expand(live, 3))
            metadata.sequence_lengths.fill_(live); metadata.query_start_loc[-1] = live
            metadata.num_accepted_tokens.fill_(1); metadata.is_prefilling.fill_(True)
            metadata.slot_mapping[:live].copy_(torch.arange(64, 64 + live, device="cuda"))
            observed = []
            for poison in (0xA5, 0x5A):
                for field in state.__dataclass_fields__:
                    value = getattr(state, field)
                    value.fill_(-1 if value.dtype == torch.int64 else 0)
                arena.tensor.fill_(poison)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay(); torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                result = tuple(x.cpu().clone() for x in (scratch.residual, scratch.hidden, scratch.logits))
                assert all(torch.isfinite(x).all() for x in result)
                assert torch.count_nonzero(result[0][live:]) == 0
                assert torch.count_nonzero(result[1][live:]) == 0
                assert torch.equal(hidden[:live].cpu(), host_hidden)
                assert torch.equal(tokens[:live].cpu(), host_tokens)
                observed.append(result)
            for first, second in zip(*observed):
                assert torch.equal(first.view(torch.uint8), second.view(torch.uint8))
        finally:
            stream.synchronize(); graph.reset()
