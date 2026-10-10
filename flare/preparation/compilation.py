"""Compile prepared forwards with explicit byte-storage mutation boundaries."""

import itertools
import math
import operator
import weakref
from dataclasses import dataclass

import torch
from torch._higher_order_ops.triton_kernel_wrap import get_mutated_tensors
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_flatten, tree_unflatten

_INVOCATIONS = weakref.WeakValueDictionary()
_HANDLES = itertools.count()


@dataclass
class _Invocation:
    target: object
    leaves: list
    tree: object
    operands: list

    def run(self, reads, writes):
        leaves = self.leaves.copy()
        for position, mutable, index, view in self.operands:
            tensor = writes[index] if mutable else reads[index]
            if view is not None:
                dtype, shape, stride, offset = view
                tensor = tensor.view(dtype).as_strided(shape, stride, offset)
            leaves[position] = tensor
        args, kwargs = tree_unflatten(leaves, self.tree)
        self.target(*args, **kwargs)


@torch.library.custom_op('flare::storage_call', mutates_args=('writes',))
def _storage_call(reads: list[torch.Tensor], writes: list[torch.Tensor], handle: int) -> None:
    _INVOCATIONS[handle].run(reads, writes)


@_storage_call.register_fake
def _fake_storage_call(reads, writes, handle):
    pass


def _byte_storage(tensor):
    storage = tensor.untyped_storage()
    return tensor.new_empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))


def compile_forward(fn):
    """Trace and compile one fixed-buffer forward before CUDA graph capture.

    Tracing executes the prepared forward once. Mutating native operators
    receive explicit storage operands, including aliases of different dtypes.
    Their typed views are reconstructed inside each opaque invocation so
    Inductor can preserve mutations without cloning strided state regions.
    Tensor operations remain visible to ordinary Inductor compilation.
    The compiled module retains the forward and its prepared kernel owners.
    """
    module = make_fx(fn, tracing_mode='real')()
    graph = module.graph
    for node in reversed(list(graph.nodes)):
        if (node.op == 'call_function' and isinstance(node.target, torch._ops.OpOverload)
                and node.target._schema.name in (
                    'profiler::_record_function_enter', 'profiler::_record_function_enter_new',
                    'profiler::_record_function_exit')):
            graph.erase_node(node)
    originals = list(graph.nodes)
    bases = {}
    keys = {}
    for node in originals:
        value = node.meta.get('val')
        if not isinstance(value, torch.Tensor):
            continue
        keys[node] = value.untyped_storage()._cdata
        if node.op == 'get_attr':
            actual = operator.attrgetter(node.target)(module)
            keys[node] = actual.untyped_storage()._cdata
            node.meta['val'] = actual
        elif node.op == 'call_function' and isinstance(node.target, torch._ops.OpOverload):
            schema = node.target._schema
            if len(schema.returns) == 1 and schema.returns[0].alias_info is not None:
                aliases = schema.returns[0].alias_info.before_set
                for index, argument in enumerate(schema.arguments):
                    if argument.alias_info is not None and aliases & argument.alias_info.before_set:
                        arg = node.args[index] if index < len(node.args) else node.kwargs.get(argument.name)
                        if isinstance(arg, torch.fx.Node) and arg in keys:
                            keys[node] = keys[arg]
                            if schema.is_mutable:
                                node.meta['val'] = arg.meta['val']
                            else:
                                node.meta['val'] = node.target(
                                    *torch.fx.map_arg(node.args, lambda arg: arg.meta['val']),
                                    **torch.fx.map_arg(node.kwargs, lambda arg: arg.meta['val']))
                            break
    dtypes = {}
    for node, key in keys.items():
        dtypes.setdefault(key, set()).add(node.meta['val'].dtype)

    for node in originals:
        if node.op != 'get_attr' or not isinstance(node.meta.get('val'), torch.Tensor):
            continue
        key = keys[node]
        if len(dtypes[key]) <= 1:
            continue
        if key not in bases:
            tensor = operator.attrgetter(node.target)(module)
            name = '_storage_' + str(len(bases))
            module.register_buffer(name, _byte_storage(tensor))
            with graph.inserting_before(originals[0]):
                bases[key] = graph.get_attr(name)

    def tensor_nodes(value):
        leaves, _ = tree_flatten(value)
        return [node for node in leaves if isinstance(node, torch.fx.Node)
                and isinstance(node.meta.get('val'), torch.Tensor)]

    def key_for(node):
        return keys[node]

    calls = []
    for node in originals:
        if node.op != 'call_function':
            continue
        target = node.target
        mutated = []
        if target is torch.ops.higher_order.triton_kernel_wrapper_mutation:
            kwargs = node.kwargs
            names = get_mutated_tensors(kwargs['kernel_idx'], kwargs['constant_args_idx'],
                torch.fx.map_arg(kwargs['kwargs'], lambda arg: arg.meta['val']),
                kwargs['tma_descriptor_metadata'])
            mutated = [kwargs['kwargs'][name] for name in names]
        elif isinstance(target, torch._ops.OpOverload):
            for index, argument in enumerate(target._schema.arguments):
                if argument.alias_info is not None and argument.alias_info.is_write:
                    value = node.args[index] if index < len(node.args) else node.kwargs.get(argument.name)
                    mutated.extend(tensor_nodes(value))
        if not mutated:
            continue
        # Keep pointwise tensor work visible unless its operands alias shared storage.
        is_external = target is torch.ops.higher_order.triton_kernel_wrapper_mutation or (
            isinstance(target, torch._ops.OpOverload) and target.namespace != 'aten')
        operand_nodes = tensor_nodes((node.args, node.kwargs))
        if not is_external and all(arg.meta['val'].is_contiguous() for arg in mutated) and not any(
            len(dtypes[key_for(arg)]) > 1
            for arg in operand_nodes):
            continue
        mutable_keys = {key_for(arg) for arg in mutated}
        reads, writes, read_indices, write_indices = [], [], {}, {}
        leaves, tree = tree_flatten((node.args, node.kwargs))
        operands = []
        for position, arg in enumerate(leaves):
            if not isinstance(arg, torch.fx.Node):
                continue
            value = arg.meta.get('val')
            if not isinstance(value, torch.Tensor):
                raise TypeError(f'non-tensor dynamic operand: {arg}')
            key = key_for(arg)
            mutable = key in mutable_keys
            array, indices = (writes, write_indices) if mutable else (reads, read_indices)
            base = bases.get(key)
            identity = key if base is not None else arg
            if identity not in indices:
                indices[identity] = len(array)
                array.append(base if base is not None else arg)
            descriptor = (value.dtype, tuple(value.shape), value.stride(), value.storage_offset()) if base is not None else None
            operands.append((position, mutable, indices[identity], descriptor))
            leaves[position] = None
        invocation = _Invocation(target, leaves, tree, operands)
        handle = next(_HANDLES)
        _INVOCATIONS[handle] = invocation
        calls.append(invocation)
        with graph.inserting_before(node):
            graph.call_function(torch.ops.flare.storage_call.default, (reads, writes, handle))
        if node.users:
            result = node.meta.get('val')
            matches = [arg for arg in operand_nodes if isinstance(result, torch.Tensor)
                and key_for(arg) == key_for(node)
                and arg.meta['val'].shape == result.shape
                and arg.meta['val'].stride() == result.stride()
                and arg.meta['val'].storage_offset() == result.storage_offset()
                and arg.meta['val'].dtype == result.dtype]
            if not matches:
                raise RuntimeError(f'unsupported returned value: {node}')
            node.replace_all_uses_with(matches[0])
        graph.erase_node(node)

    for node in originals:
        if node.op != 'get_attr' or not isinstance(node.meta.get('val'), torch.Tensor):
            continue
        if key_for(node) not in bases:
            continue
        value = node.meta['val']
        with graph.inserting_before(node):
            typed = graph.call_function(torch.ops.aten.view.dtype, (bases[key_for(node)], value.dtype))
            if value.is_contiguous():
                sliced = graph.call_function(torch.ops.aten.slice.Tensor,
                    (typed, 0, value.storage_offset(), value.storage_offset() + math.prod(value.shape)))
                view = graph.call_function(torch.ops.aten.view.default, (sliced, tuple(value.shape)))
            else:
                view = graph.call_function(torch.ops.aten.as_strided.default,
                    (typed, tuple(value.shape), value.stride(), value.storage_offset()))
        node.replace_all_uses_with(view)
        graph.erase_node(node)
    graph.eliminate_dead_code()
    graph.lint()
    module.recompile()
    module._storage_invocations = calls
    module._forward_owner = fn
    return torch.compile(module, fullgraph=True, dynamic=False,
        options={'max_autotune': True, 'emulate_precision_casts': True,
                 'triton.cudagraphs': False, 'compile_threads': 4})
