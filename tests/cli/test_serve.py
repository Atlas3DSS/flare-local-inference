"""`flare run` against a recorded recipe bootstrap and native server."""

import os
import socket
import sys
import types
from dataclasses import replace

import pytest

from flare import planning
from flare.cli import environment, main, serve
from flare.recipes import ResolvedModel, resolve
from flare.recipes.declaration import RankUsage

GIB = 1 << 30
FREE = 95 * GIB


class Server:
    address = "127.0.0.1:8000"

    def __init__(self, waits):
        self.waits, self.dashboard = iter(waits), 0

    def start_dashboard(self):
        self.dashboard += 1

    def wait(self, timeout):
        outcome = next(self.waits)
        if outcome is KeyboardInterrupt:
            raise KeyboardInterrupt
        return outcome


class Prepared:
    """Stands in for a prepared model whose graphs hold `extra` bytes beyond the plan."""

    def __init__(self, plan, *, extra=0, pages=None, waits=(True, KeyboardInterrupt), start_error=None):
        self.plan, self.extra, self.pages, self.waits, self.start_error = plan, extra, pages, waits, start_error
        self.closed, self.started = False, None
        self.memory_report = {"tuning": {"requests": 3, "cache_hits": 2, "benchmarked_candidates": 5,
                                         "selection_counts": {"cached": 7, "tuned": 1}}}

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.closed = True

    def manifest(self):
        return {}

    def usage(self):
        plan = self.plan
        state = (plan.live_rows + plan.cache_states) * plan.costs.state_row_bytes
        held = plan.weights_bytes + plan.graph_bytes + self.extra + state + plan.page_bytes
        return (RankUsage(self.pages or plan.pages, FREE - held),) * plan.world_size

    def start_server(self, *, bind, options):
        if self.start_error is not None:
            raise self.start_error
        self.started = (bind, options)
        self.server = Server(self.waits)
        return self.server


@pytest.fixture
def host(monkeypatch, tmp_path):
    """Give the commands free GPUs, a cached checkpoint, and a recording bootstrap."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    gpus = tuple(planning.Gpu(index, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{index:04d}", 96 * GIB, FREE, (12, 0))
                 for index in range(2))
    monkeypatch.setattr(planning, "discover_gpus", lambda: gpus)
    monkeypatch.setattr(environment, "apply", lambda configuration, selected: calls.setdefault("gpus", selected))
    monkeypatch.setattr(ResolvedModel, "cached", lambda self, hub_cache: True)
    signals = types.SimpleNamespace(entered=0, exited=0)

    class CliSignals:
        def __enter__(self):
            signals.entered += 1

        def __exit__(self, *exception):
            signals.exited += 1

    monkeypatch.setitem(sys.modules, "flare_helios", types.SimpleNamespace(CliSignals=CliSignals))
    calls = {"signals": signals, "behavior": {}}

    def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress):
        calls.update(snapshot=snapshot, plan=plan, compile_forwards=compile_forwards, kernel_cache=kernel_cache)
        behavior = dict(calls["behavior"])
        if "raises" in behavior:
            raise behavior["raises"]
        calls["prepared"] = Prepared(plan, **behavior)
        return calls["prepared"]

    model = resolve("qwen3.8-flash")
    recipe = replace(model.recipe, prepare=prepare, usage=lambda prepared: prepared.usage())
    monkeypatch.setattr(serve, "resolve_model", lambda configuration, name: ResolvedModel(recipe, model.checkpoint))
    return calls


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def run(*arguments):
    return main(["run", *arguments, "--port", str(free_port())])


def phases(capsys):
    return [line for line in capsys.readouterr().err.splitlines() if line.startswith("[    OK    ]")]


def test_run_selects_gpus_and_prepares_with_the_recipe_settings(host, tmp_path, capsys):
    assert run("qwen3.8-flash", "--no-compile") == 0
    plan, prepared = host["plan"], host["prepared"]
    assert host["compile_forwards"] is False and host["kernel_cache"] == tmp_path / ".cache/flare/kernels"
    assert [gpu.index for gpu in host["gpus"]] == [0] and prepared.closed
    assert plan.speculation == plan.recipe.speculation and plan.concurrency == 4
    lines = phases(capsys)
    assert lines[0].startswith("[    OK    ] Resolve checkpoint.  qwen3.8-flash · dc5b8ab2 · cached")
    assert lines[1].startswith("[    OK    ] Select GPUs.  GPU 0 · concurrency 4 · up to 3 draft tokens  ")
    assert lines[2].startswith("[    OK    ] Start Helios.") and len(lines) == 3
    assert host["signals"].entered == 1 and host["signals"].exited >= 1


def test_command_line_settings_reach_the_plan(host, tmp_path, monkeypatch, capsys):
    path = tmp_path / ".config/flare/config.toml"
    path.parent.mkdir(parents=True)
    path.write_text('[defaults]\nmax_concurrency = 2\n')
    assert run("qwen3.8-flash") == 0
    assert host["plan"].concurrency == 2 and host["plan"].world_size == 1
    assert run("qwen3.8-flash", "--max-concurrency", "3", "--gpus", "0-1", "--prefill-chunk-size", "1024") == 0
    plan = host["plan"]
    assert (plan.concurrency, plan.world_size, plan.prefill_chunk) == (3, 2, 1024)
    assert "Select GPUs.  GPUs 0-1 · concurrency 3 · " in phases(capsys)[-2]

    assert run("qwen3.8-flash", "--mtp-fixed-draft-tokens", "2", "--history-drafts") == 0
    assert host["plan"].speculation.fixed_drafts == 2 and host["plan"].speculation.history_drafts
    assert "· concurrency 2 · exactly 2 draft tokens with history drafts  " in phases(capsys)[1]
    assert run("qwen3.8-flash", "--no-mtp") == 0
    assert host["plan"].speculation is None and host["plan"].live_rows == 1 + 2 + 6
    assert "· concurrency 2 · speculation off  " in phases(capsys)[1]

    # A caller's restriction renumbers the GPUs that --gpus indexes.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    assert run("qwen3.8-flash", "--gpus", "0") == 0
    assert host["plan"].gpus[0].uuid == "GPU-0001"
    assert "Select GPUs.  GPU 0 of CUDA_VISIBLE_DEVICES=1,0 · concurrency 2 · " in phases(capsys)[1]


def test_run_serves_with_derived_options_until_interrupted(host, tmp_path, capsys):
    port = free_port()
    assert main(["run", "qwen3.8-flash", "--max-concurrency", "2", "--port", str(port)]) == 0
    prepared = host["prepared"]
    bind, options = prepared.started
    assert bind == f"127.0.0.1:{port}"
    assert options["queue_capacity"] == 8 and options["frontend_workers"] == 2
    assert options["max_body_bytes"] == 4 * (32 << 20) + (1 << 20)
    assert options["model_id"] == "local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD"
    assert options["served_names"] == ["qwen3.8-flash"] and options["api_key"] == ""
    assert options["context_hint"] == "this is the checkpoint's maximum context"
    # A pool smaller than one full context limits the longest request.
    host["behavior"] = {"pages": 4000}
    assert main(["run", "qwen3.8-flash", "--port", str(free_port())]) == 0
    assert host["prepared"].started[1]["context_hint"] == (
        "a lower --max-concurrency leaves more KV pages for each request")
    host["behavior"] = {}
    assert main(["run", "qwen3.8-flash", "--max-concurrency", "2", "--port", str(port)]) == 0
    prepared = host["prepared"]
    assert host["compile_forwards"] is True and prepared.closed and prepared.server.dashboard == 1
    lines = capsys.readouterr().err.splitlines()
    assert lines[-1].startswith("[    OK    ] Start Helios.  listening on 127.0.0.1:8000")
    assert not any(line.startswith("listening on") for line in lines)
    written = (tmp_path / ".config/flare/config.toml").read_text()
    assert '# model = "qwen3.8-flash"' in written and "# max_concurrency = 2" in written
    assert all(line.startswith(("#", "[")) or not line for line in written.splitlines())
    from flare.cli.config import Configuration

    assert Configuration.load().values == {"server": {}, "defaults": {}, "memory": {}, "paths": {}}


def test_run_uses_the_configured_address_and_key(host, tmp_path):
    port = free_port()
    path = tmp_path / "config.toml"
    path.write_text(f'[server]\nhost = "127.0.0.1"\nport = {port}\napi_key = "secret"\n')
    assert main(["run", "qwen3.8-flash", "--config", str(path)]) == 0
    bind, options = host["prepared"].started
    assert bind == f"127.0.0.1:{port}" and options["api_key"] == "secret"


def test_a_stopped_server_is_a_runtime_failure(host, capsys):
    host["behavior"] = {"waits": (True, False)}
    assert main(["run", "qwen3.8-flash", "--port", str(free_port())]) == 6
    assert "stopped unexpectedly" in capsys.readouterr().err and host["prepared"].closed


def test_an_address_in_use_fails_before_preparation(host, capsys):
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        assert main(["run", "qwen3.8-flash", "--port", str(occupied.getsockname()[1])]) == 2
    assert "cannot listen on 127.0.0.1" in capsys.readouterr().err and "plan" not in host
    assert main(["run", "qwen3.8-flash", "--host", "localhost"]) == 2
    assert "--host must be an IP address" in capsys.readouterr().err


@pytest.mark.parametrize("behavior, code, message", [
    ({"raises": RuntimeError("kernel tuning failed")}, 5, "preparation failed: RuntimeError: kernel tuning failed"),
    ({"raises": MemoryError("insufficient memory for requested KV pages")}, 4, "fewer than the minimum 512 pages"),
    ({"start_error": RuntimeError("cannot bind HTTP listener: Address already in use")}, 2, "cannot listen on"),
    ({"start_error": RuntimeError("invalid HTTP admission limits")}, 6, "did not start"),
])
def test_failures_map_to_exit_codes(host, capsys, behavior, code, message):
    host["behavior"] = behavior
    assert main(["run", "qwen3.8-flash", "--port", str(free_port())]) == code
    assert message in capsys.readouterr().err
    assert host["signals"].exited >= 1


def test_exhausted_gpu_memory_is_a_capacity_error(host, monkeypatch, capsys):
    class OutOfMemoryError(RuntimeError):
        pass

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(OutOfMemoryError=OutOfMemoryError))
    host["behavior"] = {"raises": OutOfMemoryError(
        "CUDA out of memory. Tried to allocate 1.64 GiB. GPU 0 has a total capacity of 95.01 GiB of which"
        " 157.06 MiB is free. Including non-PyTorch memory, this process has 94.85 GiB memory in use.")}
    assert run("qwen3.8-flash") == 4
    lines = capsys.readouterr().err.splitlines()
    error = lines[next(index for index, line in enumerate(lines) if line.startswith("error: ")):]
    assert error == [
        "error: preparation ran out of GPU memory: CUDA out of memory. Tried to allocate 1.64 GiB. GPU 0 has a total"
        " capacity of 95.01 GiB of which 157.06 MiB is free.",
        "       Fewer concurrent requests, --no-mtp, or more GPUs need less memory.",
    ]
    assert run("qwen3.8-flash", "--no-mtp") == 4
    assert "       Fewer concurrent requests, or more GPUs need less memory." in capsys.readouterr().err
    host["behavior"] = {"raises": RuntimeError("CUDA error: device-side assert")}
    assert run("qwen3.8-flash") == 5
    capsys.readouterr()


def test_preparation_that_consumes_the_reserve_is_a_capacity_error(host, tmp_path, capsys):
    path = tmp_path / ".config/flare/config.toml"
    path.parent.mkdir(parents=True)
    path.write_text('[memory]\nreserve = "2G"\n')
    host["behavior"] = {"extra": 2 * GIB}
    assert run("qwen3.8-flash") == 4
    error = capsys.readouterr().err
    assert "error: preparation left 1.0 GiB free;" in error and "[memory] reserve requires 2.0 GiB." in error
    assert host["prepared"].closed and host["prepared"].started is None
    host["behavior"] = {}
    assert run("qwen3.8-flash") == 0


def test_an_uncached_checkpoint_downloads_before_preparation(host, monkeypatch, capsys):
    fetched = []
    monkeypatch.setattr(ResolvedModel, "cached", lambda self, hub_cache: False)
    monkeypatch.setattr(serve, "fetch",
                        lambda model, hub_cache, progress_bars: fetched.append((model.checkpoint.alias, progress_bars)))
    assert run("qwen3.8-flash") == 0
    assert fetched == [("qwen3.8-flash", False)]
    error = capsys.readouterr().err
    assert "Resolve checkpoint.  qwen3.8-flash · dc5b8ab2 · not cached" in error
    assert "[    OK    ] Download checkpoint." in error


def test_environment_selects_gpus_by_uuid_and_places_caches(monkeypatch, tmp_path):
    from flare.cli.config import Configuration

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
