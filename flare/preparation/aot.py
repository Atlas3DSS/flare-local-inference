"""Serializable compiled forwards with caller-owned storage and runtime handles."""

import hashlib
import importlib
import inspect
import operator
import os
import pickle
import tempfile
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import torch
from torch._dynamo.aot_compile_types import SerializableCallable
from torch._dynamo.guards import GuardsStatePickler, _Missing
from torch.fx.experimental.proxy_tensor import make_fx
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
from torch.utils._pytree import tree_flatten, tree_unflatten, treespec_dumps, treespec_loads

from .compilation import _INVOCATIONS, _OPTIONS, _Invocation, _artifact_kernel_table, _byte_storage, _lower_storage

_GUARD_LOCK = RLock()
_GUARD_TENSOR_MODE = ContextVar("flare_guard_tensor_mode", default=None)
_COMPILE_POOL = ContextVar("flare_compile_pool", default=None)


class BindingMismatch(ValueError):
    """Prepared tensor storage differs from the compiled input contract."""


def _guard_object(cls):
    return object.__new__(cls)


class _GuardPickler(GuardsStatePickler):
    """Exclude unguarded kernel resources from prepared Python objects."""

    @classmethod
    def _unpickle_tensor(cls, meta_tensor, device, pytype, dispatch_keys_raw, grad):
        mode = _GUARD_TENSOR_MODE.get()
        if mode is None:
            return super()._unpickle_tensor(meta_tensor, device, pytype, dispatch_keys_raw, grad)
        value = mode.fake_tensor_converter.from_meta_and_device(
            mode, meta_tensor, device, pytype,
            torch._C.DispatchKeySet.from_raw_repr(dispatch_keys_raw))
        value.grad = grad
        return value

    def reducer_override(self, obj):
        from triton.runtime import JITFunction
        from torch._dynamo.aot_compile_types import _deserialize_triton_kernel, _serialize_triton_kernel

        if isinstance(obj, JITFunction):
            return _deserialize_triton_kernel, (_serialize_triton_kernel(obj),)
        if (hasattr(obj, '__dict__')
                and type(obj).__module__.startswith(('flare.', 'b12x.'))
                and not inspect.isclass(obj) and not isinstance(obj, (torch.Tensor, torch.nn.Module))):
            state = {key: value if id(value) in self.guard_tree_values
                     else _Missing('unguarded attribute')
                     for key, value in vars(obj).items()}
            return _guard_object, (type(obj),), state
        return super().reducer_override(obj)


@contextmanager
def _guard_serialization():
    from torch._dynamo import guards

    with _GUARD_LOCK:
        original = guards.GuardsStatePickler
        guards.GuardsStatePickler = _GuardPickler
        try:
            yield
        finally:
            guards.GuardsStatePickler = original


@contextmanager
def _guard_code_cache(root):
    from torch._dynamo import guards

    with _GUARD_LOCK:
        original = guards.build_guard_function

        def build(code_parts, closure_args):
            key = hashlib.sha256(pickle.dumps((tuple(code_parts), closure_args))).hexdigest()
            path = root / (key + '.bin')
            if path.is_file():
                return pickle.loads(path.read_bytes())
            result = original(code_parts, closure_args)
            root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=root, suffix='.tmp', delete=False) as file:
                temporary = Path(file.name)
                try:
                    pickle.dump(result, file)
                    file.close()
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            return result

        token = _GUARD_TENSOR_MODE.set(torch._subclasses.FakeTensorMode())
        guards.build_guard_function = build
        try:
            yield
        finally:
            guards.build_guard_function = original
            _GUARD_TENSOR_MODE.reset(token)


def _signature(inputs):
    aliases, result = {}, []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            alias = aliases.setdefault(storage._cdata, len(aliases))
            result.append((tuple(map(int, value.shape)), tuple(map(int, value.stride())),
                           int(value.storage_offset()), value.dtype, value.device, alias))
        elif isinstance(value, (int, torch.SymInt)):
            result.append(int)
        else:
            raise TypeError(f'unsupported compiled input: {type(value)}')
    return tuple(result)


@dataclass
class _Input:
    index: int
    view: tuple | None = None

    def bind(self, args):
        value = args[self.index]
        if self.view is not None:
            dtype, shape, stride, offset = self.view
            value = _byte_storage(value).view(dtype).as_strided(shape, stride, offset)
        return value


class StorageBackend(SerializableCallable):
    def __init__(self, module, examples):
        from torch._guards import detect_fake_mode
        from torch._inductor import standalone_compile

        module = make_fx(module, tracing_mode='fake', _allow_non_fake_inputs=True,
                         _disable_torch_fn_metadata_mode=True)(*examples)
        graph = module.graph
        arguments = [node for node in graph.nodes if node.op == 'placeholder']
        examples = [node.meta['val'] for node in arguments]
        self.signature = _signature(examples)
        storages, scalar_inputs = {}, {}
        for index, node in enumerate(arguments):
            value = node.meta['val']
            if not isinstance(value, torch.Tensor):
                scalar_inputs[node] = _Input(index)
                continue
            storages.setdefault(value.untyped_storage()._cdata, index)
            name = f'_input_{index}'
            module.register_buffer(name, value)
            with graph.inserting_before(node):
                replacement = graph.get_attr(name)
            replacement.meta = node.meta.copy()
            node.replace_all_uses_with(replacement)
            graph.erase_node(node)

        self.invocations = _lower_storage(module)
        bindings = dict(scalar_inputs)
        for node in list(graph.nodes):
            if node.op != 'get_attr':
                continue
            value = operator.attrgetter(node.target)(module)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f'compiled constant must be a tensor: {node.target}')
            index = storages[value.untyped_storage()._cdata]
            with graph.inserting_before(next(iter(graph.nodes))):
                argument = graph.placeholder(f'buffer_{len(bindings)}')
            argument.meta['val'] = value
            bindings[argument] = _Input(index, (value.dtype, tuple(map(int, value.shape)),
                                              tuple(map(int, value.stride())), int(value.storage_offset())))
            node.replace_all_uses_with(argument)
            graph.erase_node(node)
        graph.eliminate_dead_code()
        graph.lint()
        arguments = [node for node in graph.nodes if node.op == 'placeholder']
        self.inputs = [bindings[node] for node in arguments]
        examples = [node.meta['val'] for node in arguments]
        module = torch.fx.GraphModule({}, graph)
        self._pending = None
        pool = _COMPILE_POOL.get()
        if pool is None:
            self.artifact = standalone_compile(module, examples, dynamic_shapes='from_example_inputs',
                fake_mode=detect_fake_mode(examples), options={'config_patches': _OPTIONS},
                aot=True, donate_graph_module=True)
        else:
            self.artifact = None
            self._pending = pool.submit(module, examples)
        self._bound = None

    def _finish_compile(self):
        if self._pending is not None:
            from torch._inductor.standalone_compile import AOTCompiledArtifact

            data = self._pending.result()
            with _artifact_kernel_table():
                self.artifact = AOTCompiledArtifact.deserialize(data)
            self._pending = None

    def __call__(self, *args):
        self._finish_compile()
        identity = tuple((id(value), value.untyped_storage()._cdata, value.storage_offset())
                         if isinstance(value, torch.Tensor) else value for value in args)
        if self._bound is None or identity != self._bound[0]:
            if _signature(args) != self.signature:
                raise BindingMismatch('compiled forward input shapes, layouts, or storage aliases changed')
            self._bound = identity, [item.bind(args) for item in self.inputs]
        token = _INVOCATIONS.set(self.invocations)
        try:
            return self.artifact(*self._bound[1])
        finally:
            _INVOCATIONS.reset(token)

    @classmethod
    def serialize_compile_artifacts(cls, fn):
        from torch._library.custom_ops import _maybe_get_opdef
        from torch._dynamo.aot_compile_types import _serialize_triton_kernel

        fn._finish_compile()
        with _artifact_kernel_table():
            artifact = fn.artifact.serialize()
        calls = []
        for call in fn.invocations:
            target = call.target
            leaves = call.leaves
            if target is torch.ops.higher_order.triton_kernel_wrapper_mutation:
                args, kwargs = tree_unflatten(leaves, call.tree)
                kwargs = dict(kwargs)
                kernel = kernel_side_table.get_kernel(kwargs['kernel_idx'])
                constants = kernel_side_table.get_constant_args(kwargs['constant_args_idx'])
                name = ('triton', _serialize_triton_kernel(kernel), constants)
                kwargs['kernel_idx'] = kwargs['constant_args_idx'] = None
                leaves, _ = tree_flatten((args, kwargs))
            elif isinstance(target, torch._ops.OpOverload):
                definition = _maybe_get_opdef(target)
                registration = None if definition is None else definition._init_fn.__module__
                name = ('op', target.namespace, target._opname, target._overloadname, registration)
            else:
                raise TypeError(f'cannot serialize native operator: {target}')
            calls.append((name, leaves, treespec_dumps(call.tree), call.operands, call.integers))
        return pickle.dumps((artifact, fn.inputs, calls, fn.signature))

    @classmethod
    def deserialize_compile_artifacts(cls, data):
        from torch._inductor.standalone_compile import AOTCompiledArtifact
        from torch._dynamo.aot_compile_types import _deserialize_triton_kernel

        fn = cls.__new__(cls)
        artifact, fn.inputs, calls, fn.signature = pickle.loads(data)
        fn.invocations = []
        for name, leaves, tree, operands, integers in calls:
            tree = treespec_loads(tree)
            if name[0] == 'triton':
                _, kernel, constants = name
                target = torch.ops.higher_order.triton_kernel_wrapper_mutation
                args, kwargs = tree_unflatten(leaves, tree)
                kwargs = dict(kwargs)
                kwargs['kernel_idx'] = kernel_side_table.add_kernel(_deserialize_triton_kernel(kernel))
                kwargs['constant_args_idx'] = kernel_side_table.add_constant_args(constants)
                leaves, _ = tree_flatten((args, kwargs))
            else:
                _, namespace, opname, overload, registration = name
                if registration is not None:
                    importlib.import_module(registration)
                target = getattr(getattr(getattr(torch.ops, namespace), opname), overload)
            fn.invocations.append(_Invocation(target, leaves, tree, operands, integers))
        with _artifact_kernel_table():
            fn.artifact = AOTCompiledArtifact.deserialize(artifact)
        fn._bound = None
        fn._pending = None
        return fn


def _execute(forward):
    return forward()


def aot_forward(forward, *, compiler=None):
    sources = r'.*\.(?:_?handle|stream_handle|plans\[\d+\])$'
    token = _COMPILE_POOL.set(compiler)
    try:
        with _guard_serialization(), torch.compiler.config.patch(unbacked_sources=sources):
            return torch.compile(_execute, backend=StorageBackend, fullgraph=True,
                                 dynamic=False).aot_compile(((forward,), {}))
    finally:
        _COMPILE_POOL.reset(token)


@dataclass
class BoundForward:
    artifact: object
    owner: object
    cache_hit: bool

    def __call__(self):
        return self.artifact(self.owner)


class ForwardCache:
    """Load complete compiled forwards before tracing the serving catalog."""

    def __init__(self, *, namespace, device):
        import flashinfer
        from torch._inductor.codecache import torch_key

        self.device = torch.device(device)
        properties = torch.cuda.get_device_properties(device)
        options = {name: value for name, value in _OPTIONS.items() if name != 'compile_threads'}
        key = hashlib.sha256(pickle.dumps((namespace, options, torch_key(),
            properties.name, properties.major, properties.minor, properties.multi_processor_count)))
        for root in (Path(__file__).resolve().parents[1], Path(flashinfer.__file__).resolve().parent):
            for path in sorted(root.rglob('*.py')):
                if root.name == 'flare' and (path.name == '__main__.py' or path == root / 'progress.py'):
                    continue
                key.update(str(path.relative_to(root)).encode())
                key.update(path.read_bytes())
        root = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
        self.root = root / 'flare' / 'forwards' / key.hexdigest()

    def prepare(self, forward, *, name, signature, progress=None):
        """Bind and warm one callable on the caller's preparation stream."""
        with _guard_code_cache(self.root / 'guards'):
            bound, path = self._prepare(forward, name=name, signature=signature, progress=progress)
            return self._warm(bound, path, name=name, progress=progress)

    def prepare_many(self, functions, signatures, *, progress=None):
        """Pipeline tracing and compilation, then warm each shape on the owning rank."""
        from flare.progress import phase
        from .compile_pool import CompilePool

        compiled, pending = {}, deque()
        with phase(progress, "Compile model forwards", total=len(functions)) as stage, CompilePool(self.device) as compiler:
            def finish():
                name, bound, path = pending.popleft()
                with _guard_code_cache(self.root / 'guards'):
                    compiled[name] = self._warm(bound, path, name=name, compiler=compiler)
                stage.update(advance=1)

            for name, forward in functions.items():
                with _guard_code_cache(self.root / 'guards'):
                    bound, path = self._prepare(forward, name=name, signature=signatures[name], compiler=compiler)
                pending.append((name, bound, path))
                if len(pending) >= compiler.workers:
                    finish()
            while pending:
                finish()
        return compiled

    def _prepare(self, forward, *, name, signature, compiler=None, progress=None):
        def report(operation):
            if progress is not None:
                progress.update(f"{name} · {operation}")
        identity = (type(forward).__module__, type(forward).__qualname__, name, signature)
        path = self.root / (hashlib.sha256(pickle.dumps(identity)).hexdigest() + '.bin')
        if path.is_file():
            report("restoring compiled forward")
            with path.open('rb') as file:
                artifact = torch.compiler.load_compiled_function(file)
            report("checking compiled bindings")
            if artifact.guard_check(forward):
                return BoundForward(artifact, forward, True), path
        report("tracing and compiling forward")
        artifact = aot_forward(forward, compiler=compiler)
        return BoundForward(artifact, forward, False), path

    def _warm(self, bound, path, *, name, compiler=None, progress=None):
        def report(operation):
            if progress is not None:
                progress.update(f"{name} · {operation}")

        if bound.cache_hit:
            try:
                report("warming restored forward")
                bound()
            except BindingMismatch:
                report("tracing and compiling forward")
                bound = BoundForward(aot_forward(bound.owner, compiler=compiler), bound.owner, False)
            else:
                return bound
        report("warming compiled forward")
        bound()
        path.parent.mkdir(parents=True, exist_ok=True)
        report("saving compiled forward")
        bound.artifact.save_compiled_function(str(path))
        return bound
