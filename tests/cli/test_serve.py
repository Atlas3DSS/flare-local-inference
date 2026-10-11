"""Preparation, package caching, and process replacement by the Rust Flare runtime."""

import json
import os
import socket
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

from ignition import planning
from ignition.cli import environment, main as cli_main, runtime, serve
from ignition.recipes import ResolvedModel, resolve
from ignition.recipes.declaration import RankUsage

GIB = 1 << 30
FREE = 95 * GIB


class Executed(BaseException):
    pass


def main(arguments):
    try:
        return cli_main(arguments)
    except Executed:
        return 0


class Prepared:
    def __init__(self, plan, *, extra=0, pages=None):
        self.plan, self.extra, self.pages = plan, extra, pages
        self.closed = False
        self.started = None

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.closed = True

    def usage(self):
        plan = self.plan
        state = (plan.live_rows + plan.cache_states) * plan.costs.state_row_bytes
        held = plan.weights_bytes + plan.graph_bytes + self.extra + state + plan.page_bytes
        return (RankUsage(self.pages or plan.pages, FREE - held),)

    def start_server(self, **kwargs):
        raise AssertionError("serving must start in the native executable")


@pytest.fixture
def host(monkeypatch, tmp_path):
    from ignition.preparation import module_images, package_storage, runtime_package

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    gpus = tuple(planning.Gpu(index, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{index:04d}", 96 * GIB, FREE, (12, 0))
                 for index in range(2))
    monkeypatch.setattr(planning, "discover_gpus", lambda: gpus)
    monkeypatch.setattr(environment, "apply", lambda configuration, selected: calls.update(gpus=selected))
    monkeypatch.setattr(ResolvedModel, "cached", lambda self, hub_cache: True)
    calls = {"behavior": {}, "exports": 0, "signals": types.SimpleNamespace(entered=0, exited=0)}

    class CliSignals:
        def __enter__(self):
            calls["signals"].entered += 1
        def __exit__(self, *exception):
            calls["signals"].exited += 1

    class Scope:
        def __enter__(self):
            return self
        def __exit__(self, *exception):
            pass

    class OutOfMemoryError(RuntimeError):
        pass

    monkeypatch.setitem(sys.modules, "flare_heliopause", types.SimpleNamespace(CliSignals=CliSignals))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        cuda=types.SimpleNamespace(set_device=lambda _: None), OutOfMemoryError=OutOfMemoryError))
    monkeypatch.setattr(module_images, "ModuleImages", Scope)
    monkeypatch.setattr(package_storage, "PackageStorage", Scope)

    def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs, package_storage):
        assert retain_graphs and isinstance(package_storage, Scope)
        calls.update(snapshot=snapshot, plan=plan, compile_forwards=compile_forwards, kernel_cache=kernel_cache)
        behavior = dict(calls["behavior"])
        if "raises" in behavior:
            raise behavior["raises"]
        calls["prepared"] = Prepared(plan, **behavior)
        return calls["prepared"]

    model = resolve("qwen3.8-flash")
    recipe = replace(model.recipe, prepare=prepare, usage=lambda prepared: prepared.usage())
    monkeypatch.setattr(serve, "resolve_model", lambda configuration, name: ResolvedModel(recipe, model.checkpoint))
    monkeypatch.setattr(runtime, "binary", lambda: Path("/native/flare"))
    monkeypatch.setattr(runtime, "package_key", lambda model, plan, no_compile:
        str((plan.concurrency, plan.prefill_chunk, plan.speculation, no_compile)))

    def export_model(prepared, images, storage, path, *, options, progress):
        assert prepared is calls["prepared"] and not prepared.closed
        assert isinstance(images, Scope) and isinstance(storage, Scope)
        assert "api_key" not in options
        calls["options"] = options
        calls["exports"] += 1
        path.mkdir()
        (path / "package.json").write_text("{}")
    monkeypatch.setattr(runtime_package, "export_model", export_model)

    def execute(path, arguments):
        assert path == "/native/flare" and arguments[1] == "run"
        assert calls["prepared"].closed
        assert "api_key" not in calls["options"]
        calls["exec"] = arguments
        if "--api-key-file" in arguments:
            key_file = arguments[arguments.index("--api-key-file") + 1]
            calls["api_key"] = Path(key_file).read_text()
        raise Executed
    monkeypatch.setattr(serve.os, "execv", execute)
    return calls


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def run(*arguments):
    return main(["run", *arguments, "--port", str(free_port())])


def test_run_exports_then_replaces_python_with_flare_and_reuses_the_package(host):
    assert run("qwen3.8-flash", "--no-compile") == 0
    assert host["compile_forwards"] is False and host["exports"] == 1
    assert [gpu.index for gpu in host["gpus"]] == [0]
    assert host["plan"].speculation == host["plan"].recipe.speculation
    assert host["exec"][3:5] == ["--device", "0"]
    assert host["prepared"].closed and host["prepared"].started is None
    assert run("qwen3.8-flash", "--no-compile") == 0
    assert host["exports"] == 1


def test_preparation_flags_select_distinct_packages(host):
    assert run("qwen3.8-flash", "--max-concurrency", "2", "--prefill-chunk-size", "1024",
               "--mtp-draft-tokens", "2", "--no-mtp-adaptive-verification", "--history-drafts") == 0
    plan = host["plan"]
    assert (plan.concurrency, plan.prefill_chunk) == (2, 1024)
    assert (plan.speculation.draft_tokens, plan.speculation.adaptive, plan.speculation.history_drafts) == (2, False, True)
    assert run("qwen3.8-flash", "--no-mtp") == 0
    assert host["exports"] == 2 and host["plan"].speculation is None


def test_export_does_not_launch_a_server_and_preserves_the_serving_plan(host, tmp_path):
    output = tmp_path / "package"
    assert main(["export", "qwen3.8-flash", "--output", str(output)]) == 0
    assert host["exports"] == 1 and "exec" not in host
    assert host["plan"].prefill_chunk == 4096 and host["compile_forwards"]
    assert host["options"]["queue_capacity"] == 16
    assert host["options"]["served_names"] == ["qwen3.8-flash"]
    assert host["prepared"].closed
    assert main(["export", "qwen3.8-flash", "--output", str(output)]) == 2
    assert host["exports"] == 1


def test_tensor_parallel_fails_before_preparation(host, tmp_path):
    assert run("qwen3.8-flash", "--gpus", "0,1") == 2
    assert "prepared" not in host
    assert main(["export", "qwen3.8-flash", "--gpus", "0,1", "--output", str(tmp_path / "package")]) == 2
    assert "prepared" not in host


def test_listener_and_authentication_are_supplied_after_export(host, tmp_path):
    port = free_port()
    config = tmp_path / "config.toml"
    config.write_text(f'[server]\nport = {port}\napi_key = "secret"\n')
    assert main(["run", "qwen3.8-flash", "--config", str(config)]) == 0
    assert host["api_key"] == "secret"
    assert host["exec"][5:7] == ["--bind", f"127.0.0.1:{port}"]
    assert "secret" not in host["exec"]


def test_an_address_in_use_fails_before_preparation(host, capsys):
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        assert main(["run", "qwen3.8-flash", "--port", str(occupied.getsockname()[1])]) == 2
    assert "cannot listen on" in capsys.readouterr().err and "prepared" not in host
    assert run("qwen3.8-flash", "--host", "localhost") == 2


@pytest.mark.parametrize("exception,code,message", [
    (RuntimeError("kernel tuning failed"), 5, "preparation failed: RuntimeError: kernel tuning failed"),
    (MemoryError("insufficient memory"), 4, "fewer than the minimum 512 pages"),
])
def test_export_failures_prevent_native_launch(host, capsys, exception, code, message):
    host["behavior"] = {"raises": exception}
    assert run("qwen3.8-flash") == code
    assert message in capsys.readouterr().err and "exec" not in host
    assert host["signals"].exited >= 1


def test_exhausted_gpu_memory_is_a_capacity_error(host, capsys):
    host["behavior"] = {"raises": sys.modules["torch"].OutOfMemoryError("CUDA out of memory.")}
    assert run("qwen3.8-flash") == 4
    assert "Fewer concurrent requests, --no-mtp, or more GPUs" in capsys.readouterr().err


def test_preparation_preserves_the_configured_reserve(host, tmp_path, capsys):
    config = tmp_path / "config.toml"
    config.write_text('[memory]\nreserve = "2G"\n')
    host["behavior"] = {"extra": 2 * GIB}
    assert run("qwen3.8-flash", "--config", str(config)) == 4
    assert "preparation left 1.0 GiB free" in capsys.readouterr().err
    assert host["prepared"].closed and "exec" not in host


def test_uncached_checkpoint_downloads_before_export(host, monkeypatch):
    fetched = []
    monkeypatch.setattr(ResolvedModel, "cached", lambda self, hub_cache: False)
    monkeypatch.setattr(serve, "fetch", lambda model, hub_cache, progress_bars: fetched.append(model.checkpoint.alias))
    assert run("qwen3.8-flash") == 0
    assert fetched == ["qwen3.8-flash"]


def test_environment_selects_gpus_by_uuid_and_places_caches(monkeypatch, tmp_path):
    from ignition.cli.config import Configuration

    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("CUDA_VISIBLE_DEVICES", "CUDA_HOME", "B12X_COMPILE_CACHE_DIR", "TRITON_CACHE_DIR",
                 "TORCHINDUCTOR_CACHE_DIR", "HF_HUB_CACHE"):
        monkeypatch.setenv(name, "previous")
    path = tmp_path / "config.toml"
    path.write_text(f'[paths]\ncuda = "{tmp_path}/cuda"\nkernel_cache = "{tmp_path}/kernels"\n')
    monkeypatch.setattr(environment, "toolkit", lambda configured: configured)
    gpus = (types.SimpleNamespace(uuid="GPU-b"), types.SimpleNamespace(uuid="GPU-a"))
    environment.apply(Configuration.load(path), gpus)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-b,GPU-a"
    assert os.environ["B12X_COMPILE_CACHE_DIR"] == f"{tmp_path}/kernels/b12x"
    assert os.environ["TRITON_CACHE_DIR"] == f"{tmp_path}/kernels/triton"
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == f"{tmp_path}/kernels/inductor"
    assert os.environ["HF_HUB_CACHE"] == f"{tmp_path}/.cache/huggingface/hub"
    assert os.environ["CUDA_HOME"] == f"{tmp_path}/cuda"
    assert environment.toolkit(None) is None or (environment.toolkit(None) / "bin/nvcc").is_file()
