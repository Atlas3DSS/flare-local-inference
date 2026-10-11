"""Compile prepared text forwards and capture one native executable per shape."""

from contextlib import nullcontext
from time import monotonic

import torch

from flare.progress import StartupProgress, phase

from .aot import ForwardCache
from .compilation import _defer_full_collection


@_defer_full_collection()
def capture_forwards(functions, signatures, *, session, device, stream, pool,
                     collective=None, coordinator=None, compile_decoder=True, progress=None, label="Prepare serving graphs"):
    """Retain compiled callables and capture with an optional b12x IPC runtime.

    Persistent state and graph outputs must live in caller-owned storage.
    Only graph-local intermediates may use the shared graph memory pool;
    serving replays variants serially and may choose them in any order.
    """
    graphs, compiled = {}, {}
    if compile_decoder:
        cache = ForwardCache(namespace=session.namespace.to_dict(), device=device)
        forwards = {name: function for name, function in functions.items()
                    if signatures[name].kind in ("prefill", "decode", "resume", "speculative")}
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            compiled = cache.prepare_many(forwards, signatures, progress=progress, coordinator=coordinator)
        stream.synchronize()
    with phase(progress, label, total=len(functions)) as stage:
        for name, function in functions.items():
            started = monotonic()
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                if name in compiled:
                    function = compiled[name]
                else:
                    function()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            communication = nullcontext() if collective is None else collective.capture(stream=stream)
            with communication, session.capture(), torch.cuda.graph(graph, pool=pool, stream=stream):
                function()
            graphs[name] = graph
            stage.update(advance=1)
            if progress is not None and not isinstance(progress, StartupProgress):
                cached = " (compiled cache hit)" if name in compiled and compiled[name].cache_hit else ""
                progress(f"captured {name} in {monotonic() - started:.2f}s{cached}")
    torch.cuda.synchronize(device)
    return graphs, compiled
