"""Compile prepared text forwards and capture one native executable per shape."""

import gc
from contextlib import contextmanager, nullcontext
from time import monotonic

import torch

from .compilation import ForwardTracer, compile_forward


@contextmanager
def _defer_full_collection():
    thresholds = gc.get_threshold()
    # Repeated full-heap scans dominate collection while FX graphs are constructed.
    gc.set_threshold(thresholds[0], thresholds[1], max(thresholds[2], 1000))
    try:
        yield
    finally:
        gc.set_threshold(*thresholds)


@_defer_full_collection()
def capture_forwards(functions, signatures, *, session, device, stream, pool,
                     collective=None, compile_decoder=True, progress=None):
    """Retain compiled callables and capture with an optional b12x IPC runtime.

    Persistent state and graph outputs must live in caller-owned storage.
    Only graph-local intermediates may use the shared graph memory pool;
    serving replays variants serially and may choose them in any order.
    """
    graphs, compiled = {}, {}
    tracer = ForwardTracer() if compile_decoder else None
    for name, function in functions.items():
        started = monotonic()
        if compile_decoder and signatures[name].kind in ("prefill", "decode", "resume", "speculative"):
            if progress is not None:
                progress(f"compiling {name}")
            function = compile_forward(function, tracer=tracer)
            compiled[name] = function
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            function()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        communication = nullcontext() if collective is None else collective.capture(stream=stream)
        with communication, session.capture(), torch.cuda.graph(graph, pool=pool, stream=stream):
            function()
        graphs[name] = graph
        if progress is not None:
            cache = " (compiled cache hit)" if name in compiled and compiled[name].cache_hit else ""
            progress(f"captured {name} in {monotonic() - started:.2f}s{cache}")
    torch.cuda.synchronize(device)
    return graphs, compiled
