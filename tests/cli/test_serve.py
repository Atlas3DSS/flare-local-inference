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
        state = plan.live_rows * plan.costs.state_row_bytes
        held = plan.weights_bytes + plan.graph_bytes + self.extra + state + plan.page_bytes
        return (RankUsage(self.pages or plan.pages, FREE - held),)

    def start_server(self, **kwargs):
        raise AssertionError("serving must start in the native executable")


@pytest.fixture
def host(monkeypatch, tmp_path):
    from ignition.preparation import module_images, package_storage, model_package

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    gpus = tuple(planning.Gpu(index, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{index:04d}", 96 * GIB, FREE, (12, 0))
                 for index in range(2))
    monkeypatch.setattr(planning, "discover_gpus", lambda: gpus)
    monkeypatch.setattr(environment, "apply", lambda configuration, selected, b12x=(): calls.update(gpus=selected, b12x=b12x))
    monkeypatch.setattr(ResolvedModel, "cached", lambda self, hub_cache: True)
    calls = {"behavior": {}, "exports": 0, "signals": types.SimpleNamespace(entered=0, exited=0)}

    class CliSignals:
        def __enter__(self):
            calls["signals"].entered += 1
        def __exit__(self, *exception):
            calls["signals"].exited += 1

    class Scope:
        def __init__(self, **kwargs):
            self.host_regions = []
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
        calls.update(snapshot=snapshot, plan=plan, compile_forwards=compile_forwards,
                     kernel_cache=kernel_cache, storage=package_storage)
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
        str((plan.concurrency, plan.prefill_chunk, plan.speculation, plan.stream_overlap,
             plan.mixed_prefill, plan.expert_scale_compression, no_compile)))

    def export_model(prepared, images, storage, path, *, options, publisher, progress, console, staging):
        assert prepared is calls["prepared"] and not prepared.closed
        assert isinstance(images, Scope) and isinstance(storage, Scope)
        assert "api_key" not in options
        calls["publisher"] = publisher
        calls["options"] = options
        calls["exports"] += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"runtime package")
    monkeypatch.setattr(model_package, "export_package", export_model)

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
    assert not host["plan"].mixed_prefill and host["plan"].mixed_shapes == ()
    assert host["exec"][3:5] == ["--gpus", "0"]
    assert host["prepared"].closed and host["prepared"].started is None
    assert run("qwen3.8-flash", "--no-compile") == 0
    assert host["exports"] == 1


def test_runtime_memory_flags_are_forwarded_without_preparing_another_package(host):
    assert run("qwen3.8-flash") == 0
    original = host["exec"][2]
    assert not any(flag.startswith(("--memory-limit", "--kv-cache-limit")) for flag in host["exec"])
    for limit, mode in [("30G", "concurrency"), ("-30G", "context"), ("0.8", "memory")]:
        assert run("qwen3.8-flash", "--memory-limit", limit, "--kv-cache-limit", mode) == 0
        assert host["exec"][2] == original and host["exports"] == 1
        assert f"--memory-limit={limit}" in host["exec"]
        if mode != "memory":
            assert host["exec"][host["exec"].index("--kv-cache-limit") + 1] == mode
        assert host["options"]["reserve_bytes"] == 0


def test_served_model_name_is_forwarded_without_changing_the_package_alias(host):
    assert run("qwen3.8-flash") == 0
    original = host["exec"][2]
    assert "--served-model-name" not in host["exec"]
    for alias in ("local-model", "organization/model", "local model"):
        assert run("qwen3.8-flash", "--served-model-name", alias) == 0
        arguments = host["exec"]
        assert arguments[arguments.index("--served-model-name") + 1] == alias
        assert arguments[2] == original and host["exports"] == 1
        assert host["options"]["served_names"] == ["qwen3.8-flash"]


def test_diagnostic_serving_retains_storage_until_native_shutdown(host, monkeypatch):
    closed = []

    def serve_prepared(prepared, bind, options):
        assert prepared is host["prepared"] and not prepared.closed
        assert host["exports"] == 0

        def close_region():
            assert prepared.closed
            closed.append(True)

        host["storage"].host_regions.append(types.SimpleNamespace(close=close_region))
        with prepared:
            return 0

    monkeypatch.setattr(serve, "_serve_prepared", serve_prepared)
    assert main(["serve", "qwen3.8-flash", "--gpus", "0",
                 "--port", str(free_port())]) == 0
    assert host["exports"] == 0 and closed == [True]


def test_diagnostic_startup_failure_retains_undrained_host_storage(host, monkeypatch):
    from ignition.preparation.model import PreparedModel

    original = RuntimeError("native startup retained resources")
    original.resources_retained = True
    released = []

    def start_server(self, **options):
        host["storage"].host_regions.append(types.SimpleNamespace(close=lambda: released.append(True)))
        raise original

    def close(self):
        raise RuntimeError("native drain was not confirmed")

    monkeypatch.setattr(Prepared, "memory_report", {}, raising=False)
    monkeypatch.setattr(Prepared, "start_server", start_server)
    monkeypatch.setattr(Prepared, "close", close, raising=False)
    monkeypatch.setattr(Prepared, "__exit__", PreparedModel.__exit__)
    with pytest.raises(RuntimeError, match="native startup retained resources") as caught:
        main(["serve", "qwen3.8-flash", "--gpus", "0", "--port", str(free_port())])
    assert caught.value is original and "native drain was not confirmed" in original.__notes__[0]
    assert released == [] and not host["prepared"].closed


def test_preparation_flags_select_distinct_packages(host):
    assert run("qwen3.8-flash", "--max-concurrency", "2", "--prefill-chunk-size", "1024",
               "--mtp-draft-tokens", "2", "--no-mtp-adaptive-verification", "--history-drafts") == 0
    plan = host["plan"]
    assert (plan.concurrency, plan.prefill_chunk) == (2, 1024)
    assert (plan.speculation.draft_tokens, plan.speculation.adaptive, plan.speculation.history_drafts) == (2, False, True)
    assert run("qwen3.8-flash", "--no-mtp") == 0
    assert host["exports"] == 2 and host["plan"].speculation is None


@pytest.mark.parametrize("overlap", [False, True])
def test_mixed_prefill_opt_in_prepares_and_reuses_a_separate_package(host, capsys, overlap):
    options = () if overlap else ("--no-stream-overlap",)
    assert run("qwen3.8-flash", *options) == 0
    ordinary_package = host["exec"][2]
    assert run("qwen3.8-flash", "--mixed-prefill", *options) == 0
    mixed_package = host["exec"][2]
    assert mixed_package != ordinary_package and host["exports"] == 2
    assert host["plan"].mixed_prefill and len(host["plan"].mixed_shapes) == 42
    assert host["plan"].stream_overlap == overlap
    assert "mixed prefill on" in capsys.readouterr().err
    assert run("qwen3.8-flash", "--mixed-prefill", *options) == 0
    assert host["exec"][2] == mixed_package and host["exports"] == 2
    assert run("qwen3.8-flash", "--mixed-prefill", "--no-mtp", *options) == 0
    assert host["exports"] == 3 and host["plan"].speculation is None
    assert host["plan"].stream_overlap == overlap
    assert len(host["plan"].mixed_shapes) == 20
    assert all(not enabled for enabled, _, _, _ in host["plan"].mixed_shapes)


@pytest.mark.parametrize("overlap", [False, True])
def test_package_cache_key_includes_mixed_capture(overlap):
    model = resolve("qwen3.8-flash")
    gpu = planning.Gpu(0, "NVIDIA RTX PRO 6000 Blackwell", "GPU-test", 96 * GIB, FREE, (12, 0))
    plan = planning.fit(model.recipe, model.checkpoint, (gpu,), concurrency=4,
                        reserve_bytes=0, speculation=model.recipe.speculation, stream_overlap=overlap)
    assert runtime.package_key(model, plan, False) != runtime.package_key(
        model, replace(plan, mixed_prefill=True), False)


@pytest.mark.parametrize("compression", [["--expert-scale-compression"], ["--expert-scale-compression", "csf"]])
def test_expert_scale_compression_reaches_preparation_and_package_cache(host, monkeypatch, capsys, compression):
    original = serve.resolve_model(None, None)
    checkpoint = replace(original.checkpoint, scale_compressed_device_bytes={"csf": original.checkpoint.device_bytes})
    model = ResolvedModel(original.recipe, checkpoint)
    monkeypatch.setattr(serve, "resolve_model", lambda *_: model)
    assert run("qwen3.8-flash") == 0
    native = host["exec"][2]
    assert run("qwen3.8-flash", *compression) == 0
    compressed = host["exec"][2]
    assert compressed != native and host["exports"] == 2
    assert host["plan"].expert_scale_compression == "csf"
    assert "expert scales csf" in capsys.readouterr().err
    assert run("qwen3.8-flash", "--expert-scale-compression", "csf") == 0
    assert host["exec"][2] == compressed and host["exports"] == 2


def test_package_cache_key_includes_expert_scale_compression():
    model = resolve("step5-preview-nvfp4")
    gpus = tuple(planning.Gpu(i, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{i}", 96 * GIB, FREE, (12, 0))
                 for i in range(4))
    plan = planning.fit(model.recipe, model.checkpoint, gpus, concurrency=4,
                        reserve_bytes=0, speculation=model.recipe.speculation)
    assert runtime.package_key(model, plan, False) != runtime.package_key(
        model, replace(plan, expert_scale_compression="csf"), False)


def test_serialized_branches_reach_the_plan_and_the_startup_line(host, capsys):
    assert run("qwen3.8-flash") == 0
    assert host["plan"].stream_overlap and "branches serialized" not in capsys.readouterr().err
    assert run("qwen3.8-flash", "--no-stream-overlap") == 0
    assert host["plan"].stream_overlap is False
    assert "branches serialized" in capsys.readouterr().err


def test_package_cache_keys_distinguish_serialized_branches():
    model = resolve("glm-5.3-flash")
    sparks = tuple(planning.Gpu(index, "NVIDIA GB10", f"GPU-{index}", 121 * GIB, 110 * GIB, (12, 1))
                   for index in range(2))
    settings = dict(concurrency=4, reserve_bytes=0, speculation=model.recipe.speculation)
    overlapped = planning.fit(model.recipe, model.checkpoint, sparks, **settings)
    serial = planning.fit(model.recipe, model.checkpoint, sparks, stream_overlap=False, **settings)
    assert overlapped.stream_overlap and not serial.stream_overlap
    assert runtime.package_key(model, overlapped, False) != runtime.package_key(model, serial, False)
    assert runtime.package_key(model, overlapped, False) == runtime.package_key(
        model, replace(serial, stream_overlap=True), False)


@pytest.mark.parametrize("reserve", [0, 2 * GIB])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_export_does_not_launch_a_server_and_preserves_the_serving_plan(host, tmp_path, reserve, mixed, overlap):
    output = tmp_path / "package"
    config = tmp_path / "config.toml"
    config.write_text(f'[memory]\nreserve = "{reserve}"\n')
    assert main(["export", "qwen3.8-flash", "--output", str(output), "--config", str(config),
                 *(["--mixed-prefill"] if mixed else []), *([] if overlap else ["--no-stream-overlap"])]) == 0
    assert host["exports"] == 1 and "exec" not in host
    assert host["plan"].prefill_chunk == 4096 and host["compile_forwards"]
    assert host["plan"].mixed_prefill == mixed
    assert host["plan"].stream_overlap == overlap
    assert host["options"]["queue_capacity"] == 16
    assert host["options"]["served_names"] == ["qwen3.8-flash"]
    assert host["options"]["reserve_bytes"] == reserve
    assert "headroom_bytes" not in host["options"]
    assert host["prepared"].closed
    assert output.with_suffix(".safetensors").is_file()
    assert main(["export", "qwen3.8-flash", "--output", str(output)]) == 2
    assert host["exports"] == 1


def test_listener_and_authentication_are_supplied_after_export(host, tmp_path):
    port = free_port()
    config = tmp_path / "config.toml"
    config.write_text(f'[server]\nport = {port}\napi_key = "secret"\n')
    assert main(["run", "qwen3.8-flash", "--config", str(config)]) == 0
    assert host["api_key"] == "secret"
    assert host["exec"][5:9] == ["--host", "127.0.0.1", "--port", str(port)]
    assert "secret" not in host["exec"]


def test_listener_flags_override_configuration(host, tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('[server]\nhost = "0.0.0.0"\nport = 8000\n')
    port = free_port()
    assert main(["run", "qwen3.8-flash", "--config", str(config),
                 "--host", "127.0.0.1", "--port", str(port)]) == 0
    assert host["exec"][5:9] == ["--host", "127.0.0.1", "--port", str(port)]


def test_prefix_cache_settings_are_forwarded_only_when_set(host, tmp_path):
    assert run("qwen3.8-flash") == 0
    assert not any(flag.startswith("--prefix-cache") for flag in host["exec"])
    config = tmp_path / "config.toml"
    config.write_text('[prefix_cache]\nmemory = "8G"\ndisk_limit = "100G"\n[paths]\nprefix_cache = "~/records"\n')
    assert main(["run", "qwen3.8-flash", "--config", str(config), "--port", str(free_port())]) == 0
    arguments = host["exec"]
    assert arguments[arguments.index("--prefix-cache-disk") + 1] == str(tmp_path / "records")
    assert arguments[arguments.index("--prefix-cache-memory") + 1] == "8G"
    assert arguments[arguments.index("--prefix-cache-disk-limit") + 1] == "100G"
    assert "--prefix-cache-disk-transport" not in arguments


def test_an_address_in_use_fails_before_preparation(host, capsys):
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        assert main(["run", "qwen3.8-flash", "--port", str(occupied.getsockname()[1])]) == 2
    assert "cannot listen on" in capsys.readouterr().err and "prepared" not in host
    assert run("qwen3.8-flash", "--host", "localhost") == 2


@pytest.mark.parametrize("exception,code,message", [
    (RuntimeError("kernel tuning failed"), 5, "preparation failed: RuntimeError: kernel tuning failed"),
    (MemoryError("insufficient memory"), 4, "insufficient memory during model preparation"),
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
                 "TORCHINDUCTOR_CACHE_DIR", "HF_HUB_CACHE", "FLASHINFER_WORKSPACE_BASE"):
        monkeypatch.setenv(name, "previous")
    path = tmp_path / "config.toml"
    path.write_text(f'[paths]\ncuda = "{tmp_path}/cuda"\nkernel_cache = "{tmp_path}/kernels"\n')
    monkeypatch.setattr(environment, "toolkit", lambda configured: configured)
    gpus = (types.SimpleNamespace(uuid="GPU-b"), types.SimpleNamespace(uuid="GPU-a"))
    environment.apply(Configuration.load(path), gpus)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-b,GPU-a"
    assert os.environ["B12X_COMPILE_CACHE_DIR"] == f"{tmp_path}/kernels/b12x"
    assert Path(os.environ["FLASHINFER_WORKSPACE_BASE"]).is_relative_to(tmp_path / "kernels" / "flashinfer")
    assert os.environ["TRITON_CACHE_DIR"] == f"{tmp_path}/kernels/triton"
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == f"{tmp_path}/kernels/inductor"
    assert os.environ["HF_HUB_CACHE"] == f"{tmp_path}/.cache/huggingface/hub"
    assert os.environ["CUDA_HOME"] == f"{tmp_path}/cuda"
    assert environment.toolkit(None) is None or (environment.toolkit(None) / "bin/nvcc").is_file()


def test_environment_sets_the_recipe_b12x_settings_over_inherited_values(monkeypatch, tmp_path):
    from ignition.cli.config import Configuration
    from ignition.recipes.kimi3.recipe import RECIPE

    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("CUDA_VISIBLE_DEVICES", "CUDA_HOME", "B12X_COMPILE_CACHE_DIR", "TRITON_CACHE_DIR",
                 "TORCHINDUCTOR_CACHE_DIR", "HF_HUB_CACHE", "FLASHINFER_WORKSPACE_BASE", *RECIPE.b12x_environment):
        monkeypatch.setenv(name, "inherited")
    monkeypatch.setenv("B12X_PCIE_ALLREDUCE_ALGORITHM", "hierarchical")
    path = tmp_path / "config.toml"
    path.write_text(f'[paths]\nkernel_cache = "{tmp_path}/kernels"\n')
    environment.apply(Configuration.load(path), (types.SimpleNamespace(uuid="GPU-a"),), RECIPE.b12x_environment)
    assert {name: os.environ[name] for name in RECIPE.b12x_environment} == dict(RECIPE.b12x_environment)
    assert os.environ["B12X_PCIE_ALLREDUCE_ALGORITHM"] == "island_rs"


def test_flashinfer_workspaces_separate_checkouts_without_importing_them(monkeypatch, tmp_path):
    import importlib

    monkeypatch.delitem(sys.modules, "flashinfer", raising=False)
    roots = [tmp_path / name for name in ("source_a", "source_b")]
    for root in roots:
        package = root / "flashinfer"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text('raise AssertionError("preparation imported too early")\n')
    monkeypatch.syspath_prepend(str(roots[0]))
    first = environment.flashinfer_workspace(tmp_path / "kernels")
    assert first == environment.flashinfer_workspace(tmp_path / "kernels")
    assert "flashinfer" not in sys.modules

    monkeypatch.syspath_prepend(str(roots[1]))
    importlib.invalidate_caches()
    second = environment.flashinfer_workspace(tmp_path / "kernels")
    assert second != first
    assert second.parent == first.parent == tmp_path / "kernels" / "flashinfer"
    assert "flashinfer" not in sys.modules

    alias = tmp_path / "source_alias"
    alias.symlink_to(roots[1], target_is_directory=True)
    monkeypatch.syspath_prepend(str(alias))
    assert environment.flashinfer_workspace(tmp_path / "kernels") == second


def test_tensor_parallel_run_preserves_rank_selection_in_native_launch(host):
    assert run("qwen3.8-flash", "--gpus", "0,1") == 0
    assert host["plan"].world_size == 2
    assert host["exec"][3:5] == ["--gpus", "0,1"]


def test_run_approves_only_its_export_signer_and_namespace(host):
    assert run("qwen3.8-flash") == 0
    signer = host["publisher"]
    args = host["exec"]
    assert args[args.index("--dangerously-approve-key") + 1] == signer.fingerprint.hex()
    assert args[args.index("--dangerously-accept-publisher") + 1] == signer.namespace
    profile = json.loads(Path(args[args.index("--publisher-keys") + 1]).read_text())[0]
    assert bytes(profile["public_key"]) == signer.public_key
    assert profile["namespace"] == signer.namespace


@pytest.mark.parametrize('command', ['export', 'serve'])
def test_local_tuning_gpus_cannot_change_multihost_sharding(tmp_path, capsys, command):
    arguments = [command, 'qwen3.8-flash', '--ranks', '2', '--tuning-gpus', '0,1']
    if command == 'export':
        arguments += ['--output', str(tmp_path / 'unused.safetensors')]
    assert main(arguments) != 0
    assert 'multi-host ranks retain their sharding' in capsys.readouterr().err
