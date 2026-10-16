"""Prepared GQA retains strided cache layouts and live metadata under replay."""

import os

import pytest
import torch

from ignition.components.paged_attention import PreparedPagedAttention
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU")


@pytest.mark.parametrize("mode,tokens", [("decode", 4), ("prefill", 64), ("prefill", 4096)])
@pytest.mark.parametrize("dimension,value_dimension,heads", [(192, 128, 16), (192, 192, 8), (256, 256, 6)])
def test_gqa_strided_pages_and_sinks(mode, tokens, dimension, value_dimension, heads):
    device = torch.device("cuda", 0)
    rows = tokens if mode == "decode" else 1
    pages = max(2, (tokens + 127) // 128)
    key_size, value_size = 128 * dimension, 128 * value_dimension
    padding = 16384 if dimension == 256 else 12288 if value_dimension == 192 else 8192
    backing = torch.randn((pages, key_size + value_size + padding), device=device, dtype=torch.bfloat16) * .1
    key = backing[:, :key_size].view(pages, 128, 1, dimension)
    value = backing[:, key_size:key_size + value_size].view(pages, 128, 1, value_dimension)
    if mode == "prefill":
        key, value = key.contiguous(), value.contiguous()
    query = torch.randn((tokens, heads, dimension), device=device, dtype=torch.bfloat16) * .1
    output = torch.empty((tokens, heads, value_dimension), device=device, dtype=torch.bfloat16)
    table = torch.arange(pages, device=device, dtype=torch.int32)[None].repeat(rows, 1)
    lengths = torch.full((rows,), max(128, tokens), device=device, dtype=torch.int32)
    cu = torch.arange(rows + 1, device=device, dtype=torch.int32) * (tokens // rows)
    sink = torch.linspace(-1, 1, heads, device=device)
    with RecordedPreparationSession(device=device, checkpoint="paged-gqa",
            workload_schema="paged-gqa", numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        attention = PreparedPagedAttention(query, key, value, table, lengths, cu, output,
            sink=sink, mode=mode, window=127, session=session, workspace=arena, name="attention")
        session.tune()
        compiled = compile_forward(attention)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        cases = ([(length, tokens) for length in (128, 33, 1)] if mode == "decode"
                 else [(max(128, tokens), tokens), (tokens, tokens),
                       (max(128, tokens), tokens // 2), (17, 17)])
        for length, live_tokens in cases:
            lengths.fill_(length)
            if mode == "prefill":
                cu[1] = live_tokens
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            # Sample both ends and interior query positions without a context-sized reference matrix.
            for token in sorted({0, live_tokens // 2, live_tokens - 1}):
                position = length - 1 if mode == "decode" else length - live_tokens + token
                first, last = max(0, position - 127), position + 1
                k = key.flatten(0, 1)[first:last, 0].float()
                v = value.flatten(0, 1)[first:last, 0].float()
                logits = query[token].float() @ k.T / (dimension ** .5)
                probabilities = torch.cat((logits, sink[:, None]), 1).softmax(1)[:, :-1]
                expected = probabilities @ v
                torch.testing.assert_close(output[token].float(), expected, atol=.002, rtol=.02)
        graph.reset()


@pytest.mark.parametrize('dimension,window,causal', [(192, 128, True), (128, 1024, False)])
def test_parallel_verification_has_per_query_windows(dimension, window, causal):
    device = torch.device('cuda', 0)
    pages_per_row = (window + 8 + 127) // 128
    key = torch.randn((4 * pages_per_row, 128, 1, dimension), device=device).bfloat16() * .1
    value = torch.randn((4 * pages_per_row, 128, 1, 128), device=device).bfloat16() * .1
    query = torch.randn((32, 16, dimension), device=device).bfloat16() * .1
    output = torch.empty((32, 16, 128), dtype=torch.bfloat16, device=device)
    table = torch.arange(4 * pages_per_row, dtype=torch.int32, device=device).view(4, pages_per_row)
    lengths = torch.full((4,), window + 8, dtype=torch.int32, device=device)
    cu = torch.arange(5, dtype=torch.int32, device=device) * 8
    sink = torch.linspace(-1, 1, 16, device=device)
    with RecordedPreparationSession(device=device, checkpoint='parallel-gqa',
            workload_schema='parallel-gqa', numerical_policy='default') as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        attention = PreparedPagedAttention(query, key, value, table, lengths, cu, output,
            sink=sink, mode='verify', width=8, window=window - 1, causal=causal,
            session=session, workspace=arena, name='attention')
        session.tune()
        compiled = compile_forward(attention)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        for counts in ([8, 0, 3, 1], [1, 1, 1, 1], [8, 8, 0, 8]):
            lengths.copy_(torch.tensor([window + n if n else 0 for n in counts], device=device))
            cu.copy_(torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()], device=device))
            output.fill_(float('nan'))
            graph.replay()
            torch.cuda.synchronize()
            begin = 0
            for row, count in enumerate(counts):
                for local in range(count):
                    first = local + 1
                    last = window + local + 1 if causal else window + count
                    k = key.view(4, -1, 1, dimension)[row, first:last, 0].float()
                    v = value.view(4, -1, 1, 128)[row, first:last, 0].float()
                    logits = query[begin + local].float() @ k.T / dimension ** .5
                    probabilities = torch.cat((logits, sink[:, None]), 1).softmax(1)[:, :-1]
                    torch.testing.assert_close(output[begin + local].float(), probabilities @ v,
                                               atol=.002, rtol=.02)
                begin += count
        graph.reset()
