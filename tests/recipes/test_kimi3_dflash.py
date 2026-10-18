"""Checkpoint-backed DFlash2 arithmetic, attention visibility, and KDA rollback."""

from dataclasses import replace
import os
from pathlib import Path

import pytest
import torch

TARGET = Path(os.getenv("FLARE_KIMI3_TARGET", "/data/models/Kimi-K3-MXFP4-CSF"))
DRAFT = Path(os.getenv("FLARE_KIMI3_DFLASH2", "/data/models/Kimi-K3-DFlash2"))
cuda = pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned CUDA device")


def test_draft_configuration_and_state_geometry():
    from ignition.recipes.kimi3.config import ModelConfig
    from ignition.recipes.kimi3.draft_config import DraftConfig
    from ignition.recipes.kimi3.pool import continuation_layout, page_bytes
    from ignition.recipes.kimi3.recipe import RECIPE

    if not TARGET.is_dir() or not DRAFT.is_dir():
        pytest.skip("Kimi target and DFlash2 checkpoints are unavailable")
    target = ModelConfig.load(TARGET)
    config = replace(target, draft=DraftConfig.load(DRAFT, target))
    layout = continuation_layout(config, 16, context_ranks=16)
    assert layout.region("layer0.convolution").shape == (2304, 10)
    assert layout.region("draft0.ring").shape == (4096, 576) and "draft4.ring" not in {r.name for r in layout.regions}
    assert layout.row_bytes == RECIPE.ranks[16].state_row_bytes
    assert page_bytes(config, 16) == RECIPE.ranks[16].page_bytes
    assert page_bytes(config, 16) - page_bytes(replace(config, draft=None), 16) == RECIPE.ranks[16].draft_page_bytes
    assert RECIPE.max_draft_tokens == 7
    assert config.draft.target_layers == (19, 37, 66, 78, 90)


@pytest.mark.parametrize("kind", ["dflash2", "dspark"])
def test_draft_scratch_export_ranges(kind):
    import ctypes
    from dataclasses import fields, is_dataclass
    from types import SimpleNamespace
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.disposable import storage_ranges
    from ignition.recipes.kimi3.decoder import Scratch
    from ignition.recipes.kimi3.dflash import DraftScratch as DFlashScratch
    from ignition.recipes.kimi3.dspark import DraftScratch as DSparkScratch

    config = SimpleNamespace(hidden=128, vocabulary=512, draft=SimpleNamespace(
        heads=2, kv_heads=1, head_dim=64, window=128, text={"intermediate_size": 256}))
    if kind == "dflash2":
        activations = Scratch.allocate(32, 32, config, 1, "cpu")
        buffers = {"projection": torch.empty((32, 64)), "alias": activations.activations[:8]}
        scratch = DFlashScratch(activations, buffers, 32)
    else:
        scratch = DSparkScratch.allocate(config, TensorParallelShard(0, 1), 32, 8, "cpu")

    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif is_dataclass(value):
            for field in fields(value):
                yield from tensors(getattr(value, field.name))
        elif isinstance(value, dict):
            for child in value.values():
                yield from tensors(child)

    owned = tuple(tensors(scratch))
    for tensor in owned:
        tensor.fill_(1)
    ranges = storage_ranges(scratch.scratch_tensors())
    assert len(ranges) == len({tensor.untyped_storage().data_ptr() for tensor in owned})
    assert ranges == storage_ranges(scratch.rows(8, 8).scratch_tensors())
    for address, size in ranges:
        ctypes.memset(address, 0, size)
    assert all(not tensor.count_nonzero() for tensor in owned)
    if kind == "dflash2":
        assert scratch.token_capacity == 32
        with pytest.raises(ValueError, match="token capacity"):
            scratch.rows(40, 40)


@cuda
@torch.no_grad()
def test_dynamic_convolution_preserves_bf16_rounding_and_block_boundaries():
    from ignition.components.grouped_convolution import _convolve
    from ignition.preparation.compilation import compile_forward

    torch.cuda.set_device(0)
    torch.manual_seed(2605)
    rows, hidden, group, width = 24, 7168, 16, 8
    x = torch.randn((rows, hidden), dtype=torch.bfloat16, device="cuda")
    delta = torch.randn((rows, 2, 2, hidden // group), dtype=x.dtype, device=x.device)
    base = torch.randn((2, 2, hidden), dtype=x.dtype, device=x.device)
    output = torch.empty_like(x)
    for side in (0, 1):
        coeff = base[side].view(1, 2, hidden // group, group) + delta[:, side, :, :, None]
        shifted = torch.zeros_like(x).view(-1, width, hidden // group, group)
        shifted[:, 1:] = x.view_as(shifted)[:, :-1]
        expected = ((coeff[:, 0] * x.view(rows, hidden // group, group)) +
                    (coeff[:, 1] * shifted.view(rows, hidden // group, group))).flatten(1)
        def forward():
            torch.library.wrap_triton(_convolve)[((x.numel() + 255) // 256,)](
                x, delta, base, output, rows, hidden, group, width, 2, delta.stride(0), side, 256)
        compiled = compile_forward(forward)
        compiled()
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        output.zero_()
        graph.replay()
        torch.testing.assert_close(output, expected, atol=0, rtol=0)


@pytest.fixture(scope="module")
def draft_weights():
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.kimi3.config import ModelConfig
    from ignition.recipes.kimi3.draft_config import DraftConfig
    from ignition.recipes.kimi3.draft_weights import DraftWeights

    torch.cuda.set_device(0)
    target = ModelConfig.load(TARGET)
    config = DraftConfig.load(DRAFT, target)
    with torch.no_grad():
        weights = DraftWeights.load(DRAFT, config, "cuda:0", TensorParallelShard(0, 16))
    return config, weights


@cuda
@torch.no_grad()
@pytest.mark.parametrize("window", [0, 4096])
def test_checkpoint_draft_attention_context_and_noncausal_block(draft_weights, window):
    from torch.nn import functional as F
    from ignition.components.paged_batch import Metadata
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.dflash import DraftAttention
    from ignition.recipes.kimi3.pool import DRAFT_RING

    config, weights = draft_weights
    w = weights.layers[0]["attention"]
    device, fp8 = torch.device("cuda:0"), torch.float8_e4m3fn
    torch.manual_seed(171)
    source = torch.randn((8, 7168), dtype=torch.bfloat16, device=device) * .2
    # A sliding layer reads its continuation row's ring (row 1 of 3, rows strided like continuations); the
    # full layer reads pages of 128 positions.
    if window:
        rows = torch.randn((3, DRAFT_RING * 576 + 4096), device=device) * .2
        cache = rows.to(fp8)[:, :DRAFT_RING * 576].view(3, DRAFT_RING, 576)
    else:
        cache = (torch.randn((34, 128, 576), device=device) * .2).to(fp8)
    metadata = Metadata.allocate(8, 1, 34, device, samples=8)
    with RecordedPreparationSession(device=device, checkpoint="kimi-dflash2-attention",
            workload_schema="kimi-dflash2-attention", numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        attention = DraftAttention(w, source, metadata, cache, config, window, session=session,
            name="attention", workspace=arena, buffers={})
        session.tune(show_progress=False)
        compiled = compile_forward(lambda: attention(source))
        compiled()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        for length in (0, 5, 127, 129, 4100):
            metadata.set(torch.tensor([length], device=device), torch.tensor([8], device=device),
                torch.ones(1, dtype=torch.int64, device=device), torch.arange(34, device=device)[None],
                torch.zeros((1, 8), dtype=torch.int64, device=device), torch.ones(1, device=device))
            graph.replay()
            torch.cuda.synchronize()
            positions = torch.arange(length, device=device)
            context = cache[1, positions % DRAFT_RING] if window else cache.flatten(0, 1)[:length]
            # The context half attends E4M3 records with a saturated E4M3 query; the block half in BF16.
            absorbed = attention.absorbed.float()
            quantized = absorbed.clamp(-448., 448.).to(fp8).float()
            scores = torch.cat((torch.einsum("qhk,tk->hqt", quantized, context.float()),
                                torch.einsum("qhk,tk->hqt", absorbed, attention.latent.float())), -1)
            scores = scores * attention.rotary.scale
            latent = torch.cat((context.float(), attention.latent.float()))
            if window:
                allowed = torch.arange(latent.shape[0], device=device)[None] >= (length + torch.arange(8, device=device)[:, None] - window + 1)
                scores.masked_fill_(~allowed[None], -torch.inf)
            values = torch.einsum("tk,hkd->thd", latent[:, :512], w["value"].float())
            attended = torch.einsum("hqk,khd->qhd", scores.softmax(-1), values).bfloat16().flatten(1)
            expected = F.linear(attended, w["output"])
            error = (attention.output.float() - expected.float()).square().mean().sqrt() / expected.float().square().mean().sqrt()
            assert error < .025, (length, window, error.item())


@cuda
@torch.no_grad()
def test_checkpoint_kda_verification_commits_only_accepted_prefix():
    from ignition.components.kda_speculation import KDARecords, PreparedKDAReplay
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.recurrent import DeltaMetadata
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.config import ModelConfig
    from ignition.recipes.kimi3.draft_config import DraftConfig
    from ignition.recipes.kimi3.decoder import _commit_convolution
    from ignition.recipes.kimi3.kda import PreparedKDA
    from ignition.recipes.kimi3.pool import ContinuationPool
    from ignition.recipes.kimi3.weights import KimiWeightLoader

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    c = ModelConfig.load(TARGET)
    c = replace(c, draft=DraftConfig.load(DRAFT, c))
    pool = ContinuationPool.for_model(c, 16, 3, 1, device)
    with KimiWeightLoader(inspect_indexed_checkpoint(TARGET), device, shard=TensorParallelShard(0, 16),
            prefixes=("language_model.model.layers.0.self_attn.",)) as loader:
        w = loader.load_attention(0)
    state = pool.recurrent(0)
    records = KDARecords.allocate({0: w}, 3, 6, device)
    verify_meta, decode_meta = DeltaMetadata.allocate(1, device, verification_columns=8), DeltaMetadata.allocate(1, device)
    source = torch.randn((8, 7168), dtype=torch.bfloat16, device=device) * .2
    counts = torch.zeros(1, dtype=torch.int32, device=device)
    region = pool.layout.region("layer0.convolution")
    offsets = torch.tensor([region.offset // 2], dtype=torch.int64, device=device)
    with RecordedPreparationSession(device=device, checkpoint="kimi-dflash2-kda",
            workload_schema="kimi-dflash2-kda", numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        buffers = {}
        decode = PreparedKDA(w, source[:1], state, decode_meta, mode="decode", eps=1e-5,
            session=session, name="decode", workspace=arena, buffers=buffers)
        verify = PreparedKDA(w, source, state, verify_meta, mode="verify", eps=1e-5,
            session=session, name="verify", workspace=arena, buffers=buffers, records=records.layer(0))
        replay = PreparedKDAReplay(pool.backing, [state.recurrent], records, verify.plan,
            verify_meta.state_indices[:, 0], session=session, workspace=arena)
        session.tune(show_progress=False)
        for meta, tokens in ((verify_meta, 8), (decode_meta, 1)):
            meta.query_start_loc[1] = tokens
            meta.convolution_slots.fill_(1)
            meta.state_indices[:, 0].fill_(1)
            meta.num_seqs.fill_(1)
            meta.num_tokens.fill_(tokens)
        def verify_commit():
            verify(source)
            replay(counts)
            torch.library.wrap_triton(_commit_convolution)[(1, 1, 18)](
                pool.backing.view(torch.bfloat16), offsets, verify_meta.state_indices[:, 0], counts,
                pool.layout.row_bytes // 2, 2304, 10, 128)
        compiled = compile_forward(verify_commit)
        counts.fill_(1)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        pool.backing.zero_()
        for i in range(3):
            decode(source[i:i+1])
        initial = pool.backing.clone()
        initial_recurrent = state.recurrent.clone()
        expected, outputs = [], []
        for i in range(8):
            outputs.append(decode(source[i:i+1]).clone())
            expected.append((state.convolution[1, :, :3].clone(), state.recurrent[1].clone()))
        for count in range(1, 9):
            pool.backing.copy_(initial)
            actual = verify(source).clone()
            torch.testing.assert_close(state.recurrent, initial_recurrent, rtol=0, atol=0)
            counts.fill_(count)
            pool.backing.copy_(initial)
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            torch.testing.assert_close(state.convolution[1, :, :3], expected[count-1][0], rtol=.02, atol=.02)
            torch.testing.assert_close(state.recurrent[1], expected[count-1][1], rtol=.02, atol=.002)
        torch.testing.assert_close(actual, torch.cat(outputs), rtol=.04, atol=.02)


@cuda
@torch.no_grad()
@pytest.mark.parametrize("batch", [1, 4])
def test_complete_draft_and_context_replay_with_checkpoint_weights(batch):
    from types import SimpleNamespace
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.paged_batch import Metadata
    from ignition.components.weights import DirectWeightLoader
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena, scratch_views
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.config import ModelConfig
    from ignition.recipes.kimi3.decoder import Scratch
    from ignition.recipes.kimi3.dflash import DraftScratch, PreparedContext, PreparedDraft
    from ignition.recipes.kimi3.draft_config import DraftConfig
    from ignition.recipes.kimi3.draft_weights import DraftWeights

    torch.cuda.set_device(0)
    torch.manual_seed(531)
    device, shard = torch.device("cuda:0"), TensorParallelShard(0, 1)
    c = ModelConfig.load(TARGET)
    c = replace(c, draft=DraftConfig.load(DRAFT, c))
    draft = DraftWeights.load(DRAFT, c.draft, device, shard)
    names = ("language_model.model.embed_tokens.weight", "language_model.lm_head.weight")
    with DirectWeightLoader(inspect_indexed_checkpoint(TARGET), device, prefixes=names) as loader:
        embedding, head = loader.load_tensors(names)
    weights = SimpleNamespace(config=c, draft=draft, shard=shard, embedding=embedding, head=head)
    # Sliding layers write rings in continuation rows 0-1; the full layer pages of 128 positions.
    cache = [torch.zeros((2, 4096, 576), dtype=torch.float8_e4m3fn, device=device) for _ in range(4)]
    cache.append(torch.zeros((2, 128, 576), dtype=torch.float8_e4m3fn, device=device))
    pool = SimpleNamespace(draft_attention=lambda i: cache[i])
    work = torch.empty(129 * 7168 * 10, dtype=torch.bfloat16, device=device)
    activation, total, kda, mla, moe, auxiliary = scratch_views(work, ((129, 7168),) * 5 + ((129, 5 * 7168),))
    auxiliary.normal_()
    context_buffers = dict(hidden=activation, local=total,
        kv=kda.flatten()[:129 * 576].view(129, 576),
        latent=mla.flatten()[:129 * 576].view(129, 576))
    context_meta = Metadata.allocate(129, 1, 2, device)
    scratch = DraftScratch(Scratch.allocate(8 * batch, 8 * batch, c, 1, device,
                                          activations=activation[:8 * batch]), {}, 8 * batch)
    with RecordedPreparationSession(device=device, checkpoint="kimi-dflash2-forward",
            workload_schema="kimi-dflash2-forward", numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        context = PreparedContext(weights, auxiliary, context_meta, pool, session=session, name="context",
            workspace=arena, runtime=None, coordinator=None, buffers=context_buffers)
        forwards = []
        # Every draft batch shares storage sized for the maximum request batch.
        for draft_batch in (1,) if batch == 1 else (1, 2, 4):
            draft_meta = Metadata.allocate(8 * draft_batch, draft_batch, 2, device, samples=8 * draft_batch)
            forward = PreparedDraft(weights, scratch.rows(8 * draft_batch, 8 * draft_batch), draft_meta,
                pool, session=session, name=f"draft{draft_batch}", workspace=arena, runtime=None,
                vocabulary_runtime=None, embedding_runtime=None, router_runtime=None, vision_state=None,
                coordinator=None)
            forwards.append(forward)
            assert forward.layers[0][2].output.shape == (8 * draft_batch, c.hidden)
            assert forward.layers[0][5].output.shape == (8 * draft_batch, c.hidden)
        for prepared in forwards:
            assert prepared.layers[0][2].output.data_ptr() == forward.layers[0][2].output.data_ptr()
            assert prepared.layers[0][5].output.data_ptr() == forward.layers[0][5].output.data_ptr()
        session.tune(show_progress=False)
        context_meta.set(torch.zeros(1, device=device), torch.tensor([129], device=device),
            torch.ones(1, device=device), torch.tensor([[0, 1]], device=device),
            torch.zeros((1, 129), dtype=torch.int64, device=device), torch.ones(1, device=device))
        context(auxiliary, context_meta.lengths)
        expected_context = [x.clone() for x in cache]
        compiled_context = compile_forward(lambda: context(auxiliary, context_meta.lengths))
        compiled_context()
        for actual, reference in zip(cache, expected_context, strict=True):
            torch.testing.assert_close(actual.view(torch.uint8), reference.view(torch.uint8), atol=0, rtol=0)
        inputs = torch.full((batch, 8), c.draft.noise_token, dtype=torch.int64, device=device)
        inputs[:, 0] = 42
        lengths = torch.full((batch,), 8, device=device)
        if batch > 1:
            lengths[1] = 0
        draft_meta.set(torch.full((batch,), 129, device=device), lengths,
            torch.ones(batch, device=device), torch.tensor([[0, 1]], device=device).expand(batch, -1),
            inputs, torch.tensor([batch], device=device))
        forward()
        expected = scratch.logits.clone()
        if batch > 1:
            torch.testing.assert_close(expected[:8], expected[16:24], rtol=.02, atol=.1)
        expected_context = [x.clone() for x in cache]
        compiled = compile_forward(forward)
        compiled()
        delta = scratch.logits.float() - expected.float()
        relative = delta.norm() / expected.float().norm()
        # The E4M3 context attention also quantizes the block's queries, so compiled and eager rounding differ.
        assert relative < .02, relative.item()
        compiled_output = scratch.logits.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        torch.testing.assert_close(scratch.logits, compiled_output, rtol=0, atol=0)
        for actual, reference in zip(cache, expected_context, strict=True):
            torch.testing.assert_close(actual.view(torch.uint8), reference.view(torch.uint8), atol=0, rtol=0)
        previous = inputs[:, 0]
        for position in range(1, 8):
            logits = scratch.logits.view(batch, 8, -1)[:, position].float().clone()
            unary, ids = logits.topk(16, sorted=False)
            hidden = torch.nn.functional.linear(scratch.hidden.view(batch, 8, -1)[:, position], draft.selector["projection"])
            coded = draft.selector["predecessor"][previous] * hidden
            reference = unary + torch.einsum("br,bkr->bk", coded, draft.selector["successor"][ids]).float()
            forward.selector(logits, previous, position)
            torch.testing.assert_close(logits.gather(1, ids), reference, atol=.02, rtol=.01)
            assert logits.isfinite().sum() == 16 * batch
            previous = logits.argmax(-1)
