"""Serializable compiled forwards with caller-owned storage and runtime handles."""

import hashlib
import fcntl
import importlib
import inspect
import operator
import os
import pickle
import tempfile
from concurrent.futures import FIRST_COMPLETED, wait
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

from .compilation import (DEFAULT_CACHE_ROOT, _INVOCATIONS, _OPTIONS, _Invocation, _artifact_kernel_table,
                          _byte_storage, _lower_storage, _compiler_recursion)
from .device_binding import bind_device, relocate_inductor, restore_device, restore_prepared_device

_GUARD_LOCK = RLock()
_GUARD_TENSOR_MODE = ContextVar("flare_guard_tensor_mode", default=None)
_DEFERRED_BINDINGS = ContextVar("flare_deferred_bindings", default=False)


class BindingMismatch(ValueError):
    """Prepared tensor storage differs from the compiled input contract."""


def _guard_object(cls):
    return object.__new__(cls)


class _GuardPickler(GuardsStatePickler):
    """Exclude unguarded kernel resources from prepared Python objects."""

    @classmethod
    def _unpickle_tensor(cls, meta_tensor, device, pytype, dispatch_keys_raw, grad, deferred=False):
        if deferred and _DEFERRED_BINDINGS.get():
            pytype = torch._subclasses.FakeTensor
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

        if isinstance(obj, torch.device):
            return restore_device, (obj.type, obj.index)
        if isinstance(obj, torch._subclasses.FakeTensor):
            with torch._C._DisableTorchDispatch():
                meta = torch.empty_strided(obj.shape, obj.stride(), dtype=obj.dtype,
                                          device="meta", requires_grad=obj.requires_grad)
            from .forward_metadata import _TENSOR_KINDS
            kinds = _TENSOR_KINDS.get()
            kind = torch._subclasses.FakeTensor if kinds is None else kinds.get(obj, torch._subclasses.FakeTensor)
            deferred = kind is torch._subclasses.FakeTensor
            return type(self)._unpickle_tensor, (meta, obj.device, torch.Tensor if deferred else kind,
                torch._C._dispatch_keys(obj).raw_repr(), obj.grad, deferred)
        if isinstance(obj, JITFunction):
            return _deserialize_triton_kernel, (_serialize_triton_kernel(obj),)
        if (hasattr(obj, '__dict__')
                and type(obj).__module__.startswith(('ignition.', 'b12x.'))
                and not inspect.isclass(obj) and not isinstance(obj, (torch.Tensor, torch.nn.Module))):
            state = {key: value if id(value) in self.guard_tree_values
                     else _Missing('unguarded attribute')
                     for key, value in vars(obj).items()}
            from b12x.preparation.device import DetectedDevice

            if isinstance(obj, DetectedDevice):
                return restore_prepared_device, (type(obj), state)
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
        with _compiler_recursion(module):
            self.artifact = standalone_compile(module, examples, dynamic_shapes='from_example_inputs',
                fake_mode=detect_fake_mode(examples), options={'config_patches': _OPTIONS},
                aot=True, donate_graph_module=True)
        self._bound = None

    def __call__(self, *args):
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
        devices = {item[4] for item in fn.signature if isinstance(item, tuple) and item[4].type == "cuda"}
        if len(devices) != 1:
            raise ValueError("compiled forwards require exactly one CUDA device")
        artifact = relocate_inductor(artifact, devices.pop())
        fn.signature = tuple((*item[:4], restore_device(item[4].type, item[4].index), item[5])
                             if isinstance(item, tuple) else item for item in fn.signature)
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
        return fn


def _execute(forward):
    return forward()


def aot_forward(forward):
    sources = r'.*\.(?:_?handle|stream_handle|plans\[\d+\])$'
    with _guard_serialization(), torch.compiler.config.patch(unbacked_sources=sources):
        return torch.compile(_execute, backend=StorageBackend, fullgraph=True,
                             dynamic=False).aot_compile(((forward,), {}))


@dataclass
class BoundForward:
    artifact: object
    owner: object
    cache_hit: bool
    pending: object = None

    def __call__(self):
        return self.artifact(self.owner)


def _completed(pending):
    while True:
        for index, item in enumerate(pending):
            future = item[-2].pending
            if future is None or future.done():
                return pending.pop(index)
        wait([item[-2].pending for item in pending], return_when=FIRST_COMPLETED)


# Command-line, planning, and display sources do not change compiled forwards.
_COMMAND_SOURCES = ('cli', 'errors.py', 'planning.py', 'progress.py')


class ForwardCache:
    """Load complete compiled forwards before tracing the serving catalog."""

    def __init__(self, *, namespace, device, root=None, headroom_bytes=0):
        import flashinfer
        from torch._inductor.codecache import torch_key

        self.device = torch.device(device)
        self.headroom_bytes = headroom_bytes
        properties = torch.cuda.get_device_properties(device)
        options = {name: value for name, value in _OPTIONS.items() if name != 'compile_threads'}
        key = hashlib.sha256(pickle.dumps((namespace, options, torch_key(),
            properties.name, properties.major, properties.minor, properties.multi_processor_count)))
        for source in (Path(__file__).resolve().parents[1], Path(flashinfer.__file__).resolve().parent):
            for path in sorted(source.rglob('*.py')):
                if source.name == 'ignition' and (path.name == '__main__.py'
                                               or path.relative_to(source).parts[0] in _COMMAND_SOURCES):
                    continue
                key.update(str(path.relative_to(source)).encode())
                key.update(path.read_bytes())
        root = DEFAULT_CACHE_ROOT if root is None else Path(root)
        self.root = root / 'forwards' / key.hexdigest()

    def prepare(self, forward, *, name, signature, progress=None):
        """Bind and warm one callable on the caller's preparation stream."""
        with self._publication_lock(), _guard_code_cache(self.root / 'guards'):
            bound, path = self._prepare(forward, name=name, signature=signature, progress=progress)
            return self._warm(bound, path, name=name, progress=progress)

    @contextmanager
    def _publication_lock(self, coordinator=None):
        if coordinator is not None:
            import torch.distributed as dist

            if dist.get_rank() != coordinator.ranks[0]:
                yield
                return
        self.root.mkdir(parents=True, exist_ok=True)
        # Cached variants share filenames. Hold publication through binding on
        # every rank so another launch cannot replace a catalog still in use.
        with (self.root / 'publication.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def prepare_many(self, functions, signatures, *, progress=None, coordinator=None, materialize=None):
        """Trace and compile in workers, then bind and warm on the owning rank."""
        # Compiler workers own parallel compilation; artifact binding needs no second worker pool.
        with self._publication_lock(coordinator), torch._inductor.config.patch(compile_threads=1):
            return self._prepare_many(functions, signatures, progress=progress,
                                      coordinator=coordinator, materialize=materialize)

    def _prepare_many(self, functions, signatures, *, progress, coordinator, materialize):
        from ignition.progress import phase
        from .compile_pool import CompilePool

        if coordinator is not None:
            return self._prepare_ranks(functions, signatures, coordinator, progress=progress, materialize=materialize)
        compiled, paths, pending = {}, {}, []
        with phase(progress, "Compile model forwards", total=len(functions)) as stage, CompilePool(self.device, headroom_bytes=self.headroom_bytes) as compiler:
            def finish():
                name, bound, path = _completed(pending)
                try:
                    with _guard_code_cache(self.root / 'guards'):
                        if bound.pending is not None:
                            bound.pending.result()
                        if materialize is None:
                            if not bound.cache_hit:
                                bound = self._load(bound.owner, path, required=True)
                                bound.cache_hit = False
                            bound()
                            compiled[name] = bound
                        else:
                            paths[name] = path
                except Exception as error:
                    raise RuntimeError(f"failed to prepare compiled forward {name}") from error
                stage.update(advance=1)

            for name, forward in functions.items():
                with _guard_code_cache(self.root / 'guards'):
                    bound, path = self._prepare(forward, name=name, signature=signatures[name], compiler=compiler,
                                                deferred=materialize is not None)
                pending.append((name, bound, path))
                if len(pending) >= compiler.workers:
                    finish()
            while pending:
                finish()
        if materialize is not None:
            materialize()
            with phase(progress, "Warm compiled forwards", total=len(functions)) as stage:
                for name, forward in functions.items():
                    with _guard_code_cache(self.root / 'guards'):
                        compiled[name] = self._load(forward, paths[name], required=True)
                    compiled[name]()
                    stage.update(advance=1)
        return compiled

    def _prepare_ranks(self, functions, signatures, coordinator, *, progress, materialize):
        import torch.distributed as dist
        from ignition.progress import phase
        from .compile_pool import CompilePool

        group, root = coordinator.group, coordinator.ranks[0]
        primary = dist.get_rank() == root
        selected, publication = {}, [None]
        if primary:
            try:
                variants = {name: function.compilation_variants() if hasattr(function, "compilation_variants")
                            else {"shared": function} for name, function in functions.items()}
                pending, paths = [], {}
                with phase(progress, "Compile model forwards", total=sum(map(len, variants.values()))) as stage, CompilePool(self.device, headroom_bytes=self.headroom_bytes) as compiler:
                    def finish():
                        name, variant, bound, path = _completed(pending)
                        if bound.pending is not None:
                            bound.pending.result()
                            if materialize is None:
                                with _guard_code_cache(self.root / 'guards'):
                                    bound = self._load(bound.owner, path, required=True)
                                bound.cache_hit = False
                        paths[name, variant] = str(path)
                        if variant == getattr(functions[name], "compilation_variant", "shared"):
                            selected[name] = bound
                        stage.update(advance=1)

                    for name, choices in variants.items():
                        for variant, forward in choices.items():
                            with _guard_code_cache(self.root / 'guards'):
                                bound, path = self._prepare(forward, name=f"{name}.{variant}",
                                    signature=signatures[name], compiler=compiler, deferred=materialize is not None)
                            pending.append((name, variant, bound, path))
                            if len(pending) >= compiler.workers:
                                finish()
                    while pending:
                        finish()
                publication[0] = paths, None
            except Exception as error:
                publication[0] = None, f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(publication, src=root, group=group)
        paths, failure = publication[0]
        if failure is not None:
            raise RuntimeError(f"rank-zero forward compilation failed: {failure}")
        error = None
        try:
            if materialize is not None:
                materialize()
            if not primary or materialize is not None:
                with phase(progress, "Load compiled forwards", total=len(functions)) as stage:
                    for name, forward in functions.items():
                        path = paths[name, getattr(forward, "compilation_variant", "shared")]
                        with _guard_code_cache(self.root / 'guards'):
                            selected[name] = self._load(forward, Path(path), required=True)
                        stage.update(advance=1)
        except Exception as failure:
            import traceback

            traceback.print_exc(limit=-40)
            error = f"{type(failure).__name__}: {failure}"
        errors = [None] * len(coordinator.ranks)
        dist.all_gather_object(errors, error, group=group)
        if any(errors):
            raise RuntimeError(f"compiled forward binding failed: {errors}")
        with phase(progress, "Warm compiled forwards", total=len(functions)) as stage:
            for name in functions:
                selected[name]()
                stage.update(advance=1)
        return selected

    def _load(self, forward, path, *, required=False):
        with bind_device(self.device), path.open('rb') as file:
            artifact = torch.compiler.load_compiled_function(file)
        if artifact.guard_check(forward):
            return BoundForward(artifact, forward, True)
        if required:
            reason = artifact._artifacts.guard_manager.check_verbose(artifact.prepare_f_locals(forward))
            raise BindingMismatch(f"rank-local forward does not match compiled guards: {reason}")
        return None

    def _prepare(self, forward, *, name, signature, compiler=None, progress=None, deferred=False):
        def report(operation):
            if progress is not None:
                progress.update(f"{name} · {operation}")
        identity = (type(forward).__module__, type(forward).__qualname__, name, signature)
        path = self.root / (hashlib.sha256(pickle.dumps(identity)).hexdigest() + '.bin')
        if path.is_file():
            report("restoring compiled forward")
            token = _DEFERRED_BINDINGS.set(deferred)
            try:
                bound = self._load(forward, path)
            finally:
                _DEFERRED_BINDINGS.reset(token)
            report("checking compiled bindings")
            if bound is not None:
                return bound, path
        report("tracing and compiling forward")
        if compiler is not None:
            return BoundForward(None, forward, False, compiler.submit(forward, path)), path
        return BoundForward(aot_forward(forward), forward, False), path

    def _warm(self, bound, path, *, name, progress=None):
        def report(operation):
            if progress is not None:
                progress.update(f"{name} · {operation}")

        if bound.cache_hit:
            try:
                report("warming restored forward")
                bound()
            except BindingMismatch:
                report("tracing and compiling forward")
                bound = BoundForward(aot_forward(bound.owner), bound.owner, False)
            else:
                return bound
        report("warming compiled forward")
        bound()
        path.parent.mkdir(parents=True, exist_ok=True)
        report("saving compiled forward")
        bound.artifact.save_compiled_function(str(path))
        return bound
