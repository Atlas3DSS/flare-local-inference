"""Compile lowered FX graphs in isolated processes without model storage."""

import importlib

import torch

from .compilation import _OPTIONS, _artifact_kernel_table, _defer_full_collection


@_defer_full_collection()
@torch.no_grad()
def _compile_graph(payload, registrations, device, options):
    from torch._inductor import standalone_compile
    from torch._subclasses import FakeTensorMode
    from torch.fx._graph_pickler import GraphPickler
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    for module in registrations:
        importlib.import_module(module)
    mode = FakeTensorMode(shape_env=ShapeEnv(), allow_non_fake_inputs=True)
    module, examples = GraphPickler.loads(payload, mode)
    module.recompile()
    artifact = standalone_compile(module, examples, dynamic_shapes="from_example_inputs",
        fake_mode=mode, options={"config_patches": options}, aot=True, donate_graph_module=True)
    with _artifact_kernel_table():
        return artifact.serialize()


class CompilePool:
    def __init__(self, device):
        self.device = torch.device(device)
        budget = max(1, _OPTIONS["compile_threads"])
        self.workers = min(4, budget)
        self.options = dict(_OPTIONS, compile_threads=max(1, budget // self.workers))
        self.pool = None

    def submit(self, module, examples):
        from torch._inductor.compile_worker.subproc_pool import SubprocKind, SubprocPool
        from torch._library.custom_ops import _maybe_get_opdef
        from torch.fx._graph_pickler import GraphPickler, Options

        registrations = set()
        for node in module.graph.nodes:
            if isinstance(node.target, torch._ops.OpOverload):
                definition = _maybe_get_opdef(node.target)
                if definition is not None:
                    registrations.add(definition._init_fn.__module__)
        payload = GraphPickler.dumps((module, examples), Options(ops_filter=None))
        if self.pool is None:
            self.pool = SubprocPool(self.workers, kind=SubprocKind.SPAWN)
        return self.pool.submit(_compile_graph, payload, tuple(sorted(registrations)), self.device, self.options)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if self.pool is not None:
            self.pool.shutdown()
