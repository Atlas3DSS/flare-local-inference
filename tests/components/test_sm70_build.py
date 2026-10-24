"""CPU checks for explicit native-build paths and serialization.

Load the build functions without GPU dependencies, and replace only nvcc/CDLL;
the production path resolution, header discovery and cache logic run unchanged.
"""

import ast
import ctypes
from dataclasses import dataclass
import fcntl
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import pickle
import re
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


class BuildPathsTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        self.cache = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.other = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.default = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.commands = []
        self.capturing = False
        self.torch = SimpleNamespace(cuda=SimpleNamespace(
            is_current_stream_capturing=lambda: self.capturing))
        self.modules = {
            'torch': self.torch,
            'torch.utils.cpp_extension': SimpleNamespace(CUDA_HOME='/configured/cuda'),
            'ignition.preparation.compilation': SimpleNamespace(DEFAULT_CACHE_ROOT=self.default),
        }
        patch.dict(sys.modules, self.modules).start()

    def build(self, command, *, check):
        self.assertTrue(check)
        self.commands.append(command)
        Path(command[command.index('-o') + 1]).touch()

    def load(self, name, *, classes=()):
        path = ROOT / 'ignition/components' / (name + '.py')
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in ('compile_library', 'library', '_cutlass_headers', '_sol20')
            or isinstance(node, ast.Assign) and any(getattr(target, 'id', None) == 'TILES' for target in node.targets)
            or isinstance(node, ast.ClassDef) and node.name in classes]
        module = ModuleType('_sm70_build_test_' + name)
        module.__package__ = 'ignition.components'
        module.__dict__.update(__file__=str(path), dataclass=dataclass, fcntl=fcntl, lru_cache=lru_cache,
            hashlib=hashlib, os=os, Path=Path, re=re, torch=self.torch, checked=lambda status: self.assertEqual(status, 0),
            subprocess=SimpleNamespace(run=self.build), C=SimpleNamespace(
                CDLL=lambda path: SimpleNamespace(**{symbol: (lambda *args: 0) for symbol in
                    ('sol9_dense', 'sol12_commit', 'atlas_prefill', 'atlas_prefill_init', 'atlas_produce',
                     'atlas_project', 'atlas_repack', 'atlas_position_exact', 'atlas_fp8_exact', 'atlas_init',
                     'atlas_tuple_status', 'atlas_pgemm_init', 'atlas_pgemm4', 'atlas_pgemm4_describe',
                     'sol9_dense_rows', 'atlas_produce_swiglu')}),
                c_void_p=ctypes.c_void_p, c_int=ctypes.c_int, c_float=ctypes.c_float))
        patch.dict(sys.modules, {module.__name__: module}).start()
        exec(compile(tree, str(path), 'exec'), module.__dict__)
        return module

    def headers(self):
        root = Path('/configured/cutlass/include')
        self.header_data = {
            root / 'cutlass/gemm/warp/mma_tensor_op_sm70.h': b'#include "cutlass/detail/shared.h"\n',
            root / 'cutlass/gemm/warp/mma_tensor_op_policy.h': b'#include <cutlass/detail/shared.h>\n',
            root / 'cutlass/detail/shared.h': b'#include <cuda_runtime.h>\n// transitive dependency\n',
        }
        original_read, original_is_file = Path.read_bytes, Path.is_file
        patch.object(Path, 'read_bytes', lambda path: self.header_data[path]
            if path in self.header_data else original_read(path)).start()
        patch.object(Path, 'is_file', lambda path: path in self.header_data or original_is_file(path)).start()
        return root

    def test_explicit_and_default_caches_are_used_by_every_library(self):
        for name in ('sm70_dense', 'sm70_projection_tc', 'sm70_commit', 'sm70_prefill'):
            with self.subTest(name=name):
                module = self.load(name)
                include = self.headers() if name == 'sm70_prefill' else None
                for cache in (self.cache, self.other, None):
                    args = (cache, include) if include is not None else (cache,)
                    start = len(self.commands)
                    module.library(*args)
                    command, = self.commands[start:]
                    self.assertEqual(Path(command[command.index('-o') + 1]).parent,
                                     self.default if cache is None else cache)
                    self.assertEqual(command[0], '/configured/cuda/bin/nvcc')
                    if include is not None:
                        self.assertIn('-I' + str(include), command)
                        self.assertIn('-DSOL_STAGE=1', command)
                    # A preloaded key can be used during capture. A new key
                    # fails before filesystem work or compiler invocation.
                    self.capturing = True
                    module.library(*args)
                    with self.assertRaisesRegex(RuntimeError, 'before capture'):
                        module.library('/not/preloaded', *(() if include is None else (include,)))
                    self.assertEqual(len(self.commands), start + 1)
                    self.capturing = False

    def test_cutlass_is_explicit_and_transitive_headers_change_the_cache_key(self):
        module = self.load('sm70_prefill')
        with self.assertRaisesRegex(ValueError, 'configured cutlass_include'):
            module.library(self.cache)
        with self.assertRaisesRegex(ValueError, 'header is missing'):
            module.library(self.cache, '/missing/cutlass/include')
        root = self.headers()
        found_root, headers = module._cutlass_headers(root)
        self.assertEqual(found_root, root)
        self.assertEqual(len(headers), 3)
        module.library(self.cache, root)
        first = self.commands[-1][-1]
        self.header_data[root / 'cutlass/detail/shared.h'] += b'// changed\n'
        module.library.cache_clear()
        module.library(self.cache, root)
        self.assertNotEqual(first, self.commands[-1][-1])

    def test_served_sol20_tiles_are_compiled_reviewed_instantiations(self):
        # astra-18: every served (capacity, shape) id is one of the pgemm4 instantiations compiled into the library
        # (sm70_pgemm_configs.inc), of the right format, inside the reviewed r4 template space, and the kernel header
        # is the reviewed sol-20/kernel/pgemm.cuh (r4, sha256 60a4ed9f...) plus astra-21's output row stride `ldy`.
        module = self.load('sm70_prefill')
        directory = ROOT / 'ignition/components'
        self.assertEqual(hashlib.sha256((directory / 'sm70_pgemm.cuh').read_bytes()).hexdigest(),
                         '7d979e47d804aedbc63c8ad343448f4ab0e3a85c49aa5af3cee8002159592735')
        configs = (directory / 'sm70_pgemm_configs.inc').read_text()
        self.assertRegex(configs, r'#define SOL20_CONFIGS\(X\)\n')
        self.assertRegex(configs, r'#define SOL20_CONFIGS2\(X\)\n')
        compiled = set()
        for fp4, mt, ng, w, bk, pf, minb in re.findall(
                r'X\((true|false),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+)\)', configs):
            mt, ng, w, bk, pf, minb = map(int, (mt, ng, w, bk, pf, minb))
            self.assertTrue(1 <= mt <= 8 and ng in (1, 2) and 1 <= w <= 8 and bk in (32, 64, 128)
                            and pf in (1, 2) and 1 <= minb <= 8)
            compiled.add((fp4 == 'true', module._sol20(mt, ng, w, bk, pf, minb)))
        shapes = {('tensor_fp8', 6144, 5120), ('tensor_fp8', 5120, 10240), ('tensor_fp8', 5120, 6144),
                  ('tensor_fp8', 5120, 12288), ('tensor_fp8', 5120, 1024), ('nvfp4', 5120, 17408),
                  ('nvfp4', 17408, 5120), ('nvfp4', 5120, 248320)}
        self.assertEqual(set(module.TILES), {128, 256})
        used = set()
        for capacity, choices in module.TILES.items():
            self.assertEqual(set(choices), shapes)
            for (kind, _, _), config in choices.items():
                self.assertIn((kind == 'nvfp4', config), compiled)
                used.add((kind == 'nvfp4', config))
        self.assertEqual(used, compiled)
        self.assertEqual(module._sol20(4, 1, 4, 64, 1, 4), 0x10412414)
        self.assertEqual(module._sol20(8, 2, 8, 64, 2, 1), 0x10122828)

    def test_prepared_projection_serializes_paths_and_routes_both_backends(self):
        module = self.load('sm70_projection', classes=('ProjectionQuery', 'PreparedProjection'))
        query = module.ProjectionQuery('dense', 1, 5120, 5120, 1024)
        calls = []
        def run(*args, **kwargs):
            calls.append(kwargs)
            return 'output'
        patch.dict(sys.modules, {
            'ignition.components.sm70_dense': SimpleNamespace(run=run),
            'ignition.components.sm70_projection_tc': SimpleNamespace(run=run),
        }).start()
        for tactic in (('row_dense', 54), ('row_tc', 64, 2, 2, 60), ('prefill_tc', 512)):
            prepared = module.PreparedProjection(query, tactic, kernel_cache=str(self.cache),
                                                  cutlass_include='/configured/cutlass/include')
            restored = pickle.loads(pickle.dumps(prepared))
            self.assertEqual(restored.run(None, None, None, None, None, None, None), 'output')
            self.assertEqual(calls[-1]['kernel_cache'], str(self.cache))
            if tactic[0] != 'row_dense':
                self.assertEqual(calls[-1]['cutlass_include'], '/configured/cutlass/include')
                self.assertIsNone(calls[-1]['live_count'])
                # The device live count reaches the tensor-core backend (which only the prefill tactic consumes).
                restored.run(None, None, None, None, None, None, None, None, 'count')
                self.assertEqual(calls[-1]['live_count'], 'count')


if __name__ == '__main__':
    unittest.main()
