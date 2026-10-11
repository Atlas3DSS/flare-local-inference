"""Declare operators, prepare one kernel batch, then bind serving storage."""

from __future__ import annotations

import gc
from collections import Counter
from collections.abc import Hashable
from contextlib import ExitStack, contextmanager
from threading import get_ident

import torch
from b12x.preparation import MemoryRequirements, PreparationDisplay, PreparationSession
from b12x.preparation.session import _declaration_key

from .scratch import ScratchArena


class RecordedPreparationSession(PreparationSession):
    """Retain declarations, tune them together, and refresh selected bindings.

    Construction records requests without compiling or launching their kernels.
    All kernels prepare in one batch using b12x's declaration coalescing and
    compiler pool. Finalizers bind the selected plans after the shared scratch
    arena has its serving capacity.
    """

    def __init__(self, *, device, checkpoint: str, workload_schema: str,
                 numerical_policy: str, search_limits=None, namespace=None, race_budget=None,
                 coordinator=None):
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("tuning requires an exact checkpoint revision")
        if any(not isinstance(value, str) or not value for value in (workload_schema, numerical_policy)):
            raise ValueError("tuning requires a workload schema and numerical policy")
        identity = dict(namespace or {})
        reserved = {"checkpoint", "workload_schema", "numerical_policy"}
        if reserved.intersection(identity):
            raise ValueError("tuning namespace cannot override its workload identity")
        identity.update(checkpoint=checkpoint, workload_schema=workload_schema,
                        numerical_policy=numerical_policy)
        super().__init__(device=device, autotune=True, compile_workers=0,
                         namespace=identity, race_budget=race_budget,
                         collective_barrier=None if coordinator is None else coordinator.barrier)
        self._coordinator = coordinator
        self._recorded_requests = {}
        self._finalizers = []
        self._scratch_arena = None
        self._benchmark_resources = {}
        self._benchmark_restorers = {}
        self._preparation_thread = get_ident()
        self._tune_status = "collecting"
        self._workload_schema = workload_schema
        self._search_limits = dict(search_limits or {})
        self.tuning_report = None

    def _require_collecting(self):
        self._check_thread()
        if self._tune_status != "collecting":
            raise RuntimeError("preparation declarations are closed after tuning begins")

    def declare(self, requests, *, coordinator=None):
        self._require_collecting()
        requests = tuple(requests)
        if coordinator is not None and coordinator is not self._coordinator:
            raise ValueError("collective declarations must use the session's bootstrap coordinator")
        names = [request.name for request in requests]
        if len(set(names)) != len(names) or any(name in self._recorded_requests for name in names):
            raise ValueError("preparation request names must be unique across the model")
        if any(request.retain_benchmark_call for request in requests):
            raise ValueError("serving preparation cannot retain benchmark calls after tuning")
        self._recorded_requests.update(zip(names, requests))

    def _prepare_job(self, requests, *, progress, autotune):
        if self._coordinator is None or not any(request.collective is not None for request in requests):
            return super().prepare(requests, progress=progress, autotune=autotune)
        return self._coordinator.prepare(self, requests, progress=progress, autotune=autotune)

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
        groups, keys, envelopes = {}, {}, {}
        for request in requests:
            key = _declaration_key(request.plan)
            keys[id(request.plan)] = key
            groups.setdefault(key, []).append(request)

        def scratch_bytes(plan):
            key = keys[id(plan)]
            if key not in envelopes:
                # Bound every binding in the group while enumerating configs once.
                envelopes[key] = self.candidate_memory_envelope(groups[key])
            return envelopes[key].scratch_nbytes

        parallel = arena.parallel_nbytes(scratch_bytes)
        remaining = [request for request in requests if keys[id(request.plan)] not in envelopes]
        requirements = list(envelopes.values())
        if remaining:
            requirements.append(self.candidate_memory_envelope(remaining))
        envelope = MemoryRequirements.sequential(requirements)
        before = arena.nbytes
        required = max(envelope.scratch_nbytes, parallel)
        arena.reserve(required)
        return {"initial_bytes": before, "candidate_scratch_bytes": required,
                "parallel_candidate_scratch_bytes": parallel,
                "candidate_reserved_bytes": arena.nbytes,
                "pending_persistent_bytes": envelope.pending_persistent_nbytes}

    def tune(self, *, progress=None, show_progress=False, console=None):
        self._require_collecting()
        if not self._recorded_requests:
            raise RuntimeError("tuning requires recorded preparation requests")
        self._tune_status = "tuning"
        results = []
        try:
            allocated_before = self._allocated()
            if self._stop.is_set():
                raise RuntimeError("full serving preparation requires enabled, uncancelled b12x tuning")
            requests = tuple(self._recorded_requests.values())
            scratch_arena = self._reserve_scratch(requests)
            self.configure_compile_workers(None)
            with ExitStack() as display_stack:
                update = progress
                if show_progress:
                    display = display_stack.enter_context(PreparationDisplay(
                        global_rank=0, console=console, transient=True))

                    def update(status):
                        display.update(status)
                        if progress is not None:
                            progress(status)

                results.append(self._prepare_job(requests, progress=update, autotune=True))
            if self._stop.is_set():
                raise RuntimeError("b12x tuning was cancelled before serving preparation completed")
            selections = []
            coverage = {name: value for result in results for name, value in result.coverage.items()}
            for request in requests:
                selection = request.plan.selection
                if selection is None or selection.source == "default":
                    raise RuntimeError(f"request {request.name!r} has no completed tuning selection")
                selections.append({"name": request.name, "component": request.plan.component_id,
                    "source": selection.source,
                    "config": request.plan.contract.config_payload(selection.config).to_dict(),
                    "coverage": dict(coverage.get(request.name, {}))})
            self._tune_status = "finalizing"
            if self._scratch_arena is not None:
                if self._scratch_arena.nbytes != scratch_arena["candidate_reserved_bytes"]:
                    raise RuntimeError("candidate binding exceeded the planned scratch envelope")
                self._synchronize()
                selected = MemoryRequirements.sequential(request.plan.memory_requirements() for request in requests)
                parallel = self._scratch_arena.parallel_nbytes(lambda plan:
                    plan.memory_requirements().scratch_nbytes)
                required = max(selected.scratch_nbytes, parallel)
                self._scratch_arena.trim(required)
                self._scratch_arena.freeze()
                scratch_arena.update(selected_scratch_bytes=required,
                                     parallel_selected_scratch_bytes=parallel,
                                     reserved_bytes=self._scratch_arena.nbytes)
                self._scratch_arena.finalize()
            for callback in self._finalizers:
                callback()
            self._synchronize()
            self.tuning_report = {
                "workload_schema": self._workload_schema, "namespace": self.namespace.to_dict(),
                "requests": len(requests), "compile_workers": self.compile_workers,
                "cache_hits": sum(result.cache_hits for result in results),
                "benchmarked_candidates": sum(result.benchmarked_candidates for result in results),
                "elapsed_seconds": sum(result.elapsed_seconds for result in results),
                "parent_cute_compilations": sum(result.parent_cute_compilations for result in results),
                "parent_triton_compilations": sum(result.parent_triton_compilations for result in results),
                "selection_counts": dict(Counter(item["source"] for item in selections)),
                "selections": selections, "scratch_arena": scratch_arena,
                "shared_benchmark_bytes": sum(int(getattr(value, "nbytes", 0))
                                              for value in self._benchmark_resources.values()),
                "borrowed_benchmark_bytes": sum(int(getattr(value, "borrowed_nbytes", 0))
                                                for value in self._benchmark_resources.values()),
                "allocated_shared_benchmark_bytes": sum(int(getattr(value, "allocated_nbytes",
                    getattr(value, "nbytes", 0))) for value in self._benchmark_resources.values()),
                "search_limits": dict(self._search_limits),
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
                failures = []
                for result in reversed(results):
                    try:
                        result.close()
                    except BaseException as error:
                        failures.append(error)
                if len(failures) == 1:
                    raise failures[0]
                if failures:
                    raise BaseExceptionGroup("preparation result cleanup failed", failures)
            except BaseException:
                self._tune_status = "failed"
                raise
            finally:
                result = None
                results.clear()
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


def declare(session, requests, *, coordinator=None):
    """Collect model requests, or prepare immediately in a standalone b12x session."""
    if isinstance(session, RecordedPreparationSession):
        session.declare(requests, coordinator=coordinator)
    else:
        options = {} if coordinator is None else {"coordinator": coordinator}
        session.prepare(requests, **options)


def register_finalizer(session, callback, *, workspace=None):
    """Bind after model tuning, or immediately after standalone preparation."""
    if not isinstance(session, RecordedPreparationSession):
        callback()
    if isinstance(workspace, ScratchArena):
        workspace.register_finalizer(callback)
    elif isinstance(session, RecordedPreparationSession):
        session.register_finalizer(callback)


def benchmark_resource(session, key, factory):
    if isinstance(session, RecordedPreparationSession):
        return session.benchmark_resource(key, factory)
    return factory()
