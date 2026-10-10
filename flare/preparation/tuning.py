"""Default preparation followed by one measured batch before graph capture."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import gc
from threading import get_ident
from typing import Hashable

import torch
from b12x.preparation import MemoryRequirements, PreparationSession

from .scratch import ScratchArena


WORKLOAD_SCHEMA = "qwen38-active-operators-v1"


class RecordedPreparationSession(PreparationSession):
    """Retain declarations, tune them together, and refresh selected bindings.

    Default priming permits existing constructors to bind inactive buffers.
    Benchmark callbacks own trial state. Finalizers rebuild persistent bindings
    even when b12x shares a selected payload without invoking a priming callback.
    """

    def __init__(self, *, device, checkpoint: str, namespace=None, race_budget=None):
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("tuning requires an exact checkpoint revision")
        identity = dict(namespace or {})
        reserved = {"checkpoint", "workload_schema", "numerical_policy"}
        if reserved.intersection(identity):
            raise ValueError("tuning namespace cannot override its workload identity")
        identity.update(checkpoint=checkpoint, workload_schema=WORKLOAD_SCHEMA,
                        numerical_policy="qsa16-gdn-sequential-k1-dense-mxfp8-quantized-vision-nvfp4-a16-moe-a4-route-fp32")
        super().__init__(device=device, autotune=False, compile_workers=0,
                         namespace=identity, race_budget=race_budget)
        self._recorded_requests = {}
        self._finalizers = []
        self._scratch_arena = None
        self._benchmark_resources = {}
        self._benchmark_restorers = {}
        self._preparation_thread = get_ident()
        self._tune_status = "collecting"
        self.tuning_report = None

    def _require_collecting(self):
        self._check_thread()
        if self._tune_status != "collecting":
            raise RuntimeError("preparation declarations are closed after tuning begins")

    def prepare(self, requests, *, coordinator=None, progress=None, autotune=None):
        self._require_collecting()
        requests = tuple(requests)
        if (autotune is not None and autotune is not False) or coordinator is not None:
            raise ValueError("recorded preparation uses default local priming before tune()")
        names = [request.name for request in requests]
        if len(set(names)) != len(names) or any(name in self._recorded_requests for name in names):
            raise ValueError("preparation request names must be unique across the model")
        if any(request.retain_benchmark_call for request in requests):
            raise ValueError("serving preparation cannot retain benchmark calls after tuning")
        result = super().prepare(requests, progress=progress, autotune=False)
        self._recorded_requests.update(zip(names, requests))
        return result

    def register_finalizer(self, callback):
        self._require_collecting()
        if not callable(callback):
            raise TypeError("binding finalizer must be callable")
        self._finalizers.append(callback)

    def register_scratch(self, arena):
        self._require_collecting()
        if not isinstance(arena, ScratchArena):
            raise TypeError("planned operator scratch requires a ScratchArena")
        if self._scratch_arena is not None:
            raise ValueError("operator scratch arena is already registered")
        self._scratch_arena = arena

    def benchmark_resource(self, key: Hashable, factory):
        self._check_thread()
        if self._tune_status != "tuning":
            raise RuntimeError("shared benchmark resources exist only during candidate tuning")
        if key not in self._benchmark_resources:
            self._benchmark_resources[key] = factory()
        return self._benchmark_resources[key]

    def register_benchmark_resource(self, key: Hashable, resource, *, restore):
        """Borrow preparation-only storage until trial cleanup and restoration finish."""
        self._require_collecting()
        if key in self._benchmark_resources:
            raise ValueError("benchmark resource keys must be unique")
        if not callable(restore):
            raise TypeError("borrowed benchmark resources require a restore callback")
        self._benchmark_resources[key] = resource
        self._benchmark_restorers[key] = restore

    def _restore_benchmark_resources(self):
        if not self._benchmark_restorers:
            return
        self._synchronize()
        failures = []
        for restore in self._benchmark_restorers.values():
            try:
                restore()
            except BaseException as error:
                failures.append(error)
        try:
            self._synchronize()
        except BaseException as error:
            failures.append(error)
        if failures:
            # Keep callbacks and their state owners for a subsequent close attempt.
            if len(failures) == 1:
                raise failures[0]
            raise BaseExceptionGroup("benchmark state restoration failed", failures)
        self._benchmark_restorers.clear()

    def _reserve_scratch(self, requests):
        if self._scratch_arena is None:
            return {}
        arena = self._scratch_arena
        envelope = self.candidate_memory_envelope(requests)
        before = arena.nbytes
        arena.reserve(envelope.scratch_nbytes)
        return {"initial_bytes": before, "candidate_scratch_bytes": envelope.scratch_nbytes,
                "candidate_reserved_bytes": arena.nbytes,
                "pending_persistent_bytes": envelope.pending_persistent_nbytes}

    def tune(self, *, progress=None):
        self._require_collecting()
        if not self._recorded_requests:
            raise RuntimeError("tuning requires recorded preparation requests")
        self._tune_status = "tuning"
        result = None
        try:
            allocated_before = self._allocated()
            if self._stop.is_set():
                raise RuntimeError("full serving preparation requires enabled, uncancelled b12x tuning")
            requests = tuple(self._recorded_requests.values())
            scratch_arena = self._reserve_scratch(requests)
            self.configure_compile_workers(None)
            result = super().prepare(requests, progress=progress, autotune=True)
            if self._stop.is_set():
                raise RuntimeError("b12x tuning was cancelled before serving preparation completed")
            selections = []
            for request in requests:
                selection = request.plan.selection
                if selection is None or selection.source == "default":
                    raise RuntimeError(f"request {request.name!r} has no completed tuning selection")
                selections.append({"name": request.name, "component": request.plan.component_id,
                    "source": selection.source,
                    "config": request.plan.contract.config_payload(selection.config).to_dict(),
                    "coverage": dict(result.coverage.get(request.name, {}))})
            self._tune_status = "finalizing"
            if self._scratch_arena is not None:
                if self._scratch_arena.nbytes != scratch_arena["candidate_reserved_bytes"]:
                    raise RuntimeError("candidate binding exceeded the planned scratch envelope")
                self._synchronize()
                selected = MemoryRequirements.sequential(request.plan.memory_requirements() for request in requests)
                self._scratch_arena.trim(selected.scratch_nbytes)
                self._scratch_arena.freeze()
                scratch_arena.update(selected_scratch_bytes=selected.scratch_nbytes,
                                     reserved_bytes=self._scratch_arena.nbytes)
                self._scratch_arena.finalize()
            for callback in self._finalizers:
                callback()
            self._synchronize()
            self.tuning_report = {
                "workload_schema": WORKLOAD_SCHEMA, "namespace": self.namespace.to_dict(),
                "requests": len(requests), "compile_workers": self.compile_workers,
                "cache_hits": result.cache_hits, "benchmarked_candidates": result.benchmarked_candidates,
                "elapsed_seconds": result.elapsed_seconds,
                "parent_cute_compilations": result.parent_cute_compilations,
                "parent_triton_compilations": result.parent_triton_compilations,
                "selection_counts": dict(Counter(item["source"] for item in selections)),
                "selections": selections, "scratch_arena": scratch_arena,
                "shared_benchmark_bytes": sum(int(getattr(value, "nbytes", 0))
                                              for value in self._benchmark_resources.values()),
                "borrowed_benchmark_bytes": sum(int(getattr(value, "borrowed_nbytes", 0))
                                                for value in self._benchmark_resources.values()),
                "allocated_shared_benchmark_bytes": sum(int(getattr(value, "allocated_nbytes",
                    getattr(value, "nbytes", 0))) for value in self._benchmark_resources.values()),
                "search_limits": {
                    "bf16_projection": "all eligible b12x backends and rows_per_tile configurations",
                    "mxfp8_projection": "fixed quantized wrapper with default inner dense lowering",
                    "qsa": "fixed attention_splits=16 and singleton kernel configuration",
                    "gdn_prefill": "sequential k_split=1; v_split, stages, and window_tiles remain tunable",
                    "operator_scratch": "one arena sized to the complete candidate envelope",
                },
                "allocated_before_tuning": allocated_before,
                "allocated_before_trial_release": self._allocated(),
            }
            self._tune_status = "releasing"
            return self.tuning_report
        except BaseException:
            self._tune_status = "failed"
            raise
        finally:
            try:
                if result is not None:
                    result.close()
            except BaseException:
                self._tune_status = "failed"
                raise
            finally:
                result = None
                requests = ()
                request = None
                callback = None
                self._benchmark_resources.clear()
                self._finalizers.clear()
                self._recorded_requests.clear()
                try:
                    gc.collect()
                    self._restore_benchmark_resources()
                except BaseException:
                    self._tune_status = "failed"
                    raise
            if self._tune_status == "releasing":
                try:
                    self.tuning_report["allocated_after_trial_release"] = self._allocated()
                    self.tuning_report["released_preparation_bytes"] = (
                        self.tuning_report["allocated_before_trial_release"] - self._allocated())
                    with torch.cuda.device(self.device.ordinal):
                        torch.cuda.empty_cache()
                    self._tune_status = "ready"
                except BaseException:
                    self._tune_status = "failed"
                    raise

    @contextmanager
    def capture(self):
        if self._tune_status != "ready":
            raise RuntimeError("graph capture requires completed tuning and binding finalization")
        with super().capture():
            yield

    def freeze(self):
        if self._tune_status != "ready":
            raise RuntimeError("preparation freeze requires completed tuning and binding finalization")
        return super().freeze()

    def close(self):
        if get_ident() != self._preparation_thread:
            raise RuntimeError("preparation cleanup requires the owning thread")
        if self._tune_status == "closed":
            return
        try:
            try:
                super().close()
            finally:
                self._benchmark_resources.clear()
                self._finalizers.clear()
                self._recorded_requests.clear()
                gc.collect()
                self._restore_benchmark_resources()
                self._synchronize()
                if self._scratch_arena is not None:
                    self._scratch_arena._finalizers.clear()
                self._scratch_arena = None
        except BaseException:
            self._tune_status = "failed"
            raise
        else:
            self._tune_status = "closed"


def register_finalizer(session, callback, *, workspace=None):
    if isinstance(workspace, ScratchArena):
        workspace.register_finalizer(callback)
    elif isinstance(session, RecordedPreparationSession):
        session.register_finalizer(callback)


def benchmark_resource(session, key, factory):
    if isinstance(session, RecordedPreparationSession):
        return session.benchmark_resource(key, factory)
    return factory()
