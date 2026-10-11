"""Command output and exit codes with injected GPUs and hub cache."""

import json

import pytest

from flare import planning
from flare.cli import main

GIB = 1 << 30


def gpu(index, free_gib):
    return planning.Gpu(index, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{index:04d}", 96 * GIB, free_gib * GIB, (12, 0))


@pytest.fixture
def host(monkeypatch, tmp_path):
    """Give commands four free GPUs, an empty hub cache, and no configuration file."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    gpus = [gpu(index, 95) for index in range(4)]
    monkeypatch.setattr(planning, "discover_gpus", lambda: tuple(gpus))
    return gpus


def configure(tmp_path, text):
    path = tmp_path / ".config/flare/config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_plan_prints_the_table_and_json(host, capsys):
    assert main(["plan", "qwen3.8-flash"]) == 0
    output = capsys.readouterr().out
    assert output.startswith("qwen3.8-flash\n  checkpoint      local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-"
                             "CSF-QAD@dc5b8ab2\n")
    assert "tensor-parallel 1" in output and "guarantees 4 × " in output
    assert not any(line != line.rstrip() for line in output.splitlines()) and "\x1b" not in output
    assert main(["plan", "qwen3.8-flash", "--gpus", "2-3", "--max-concurrency", "2", "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert [entry["index"] for entry in record["gpus"]] == [2, 3]
    assert record["tensor_parallel"] == 2 and record["max_concurrency"] == 2
    assert record["cuda_visible_devices"] is None


def test_plan_uses_the_configured_model_and_concurrency(host, tmp_path, capsys):
    assert main(["plan"]) == 2
    assert "no MODEL given" in capsys.readouterr().err
    configure(tmp_path, '[defaults]\nmodel = "glm-5.3-flash"\nmax_concurrency = 2\n')
    assert main(["plan", "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["alias"] == "glm-5.3-flash" and record["max_concurrency"] == 2
    assert main(["plan", "--max-concurrency", "3", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["max_concurrency"] == 3


def test_plan_reports_a_caller_restriction(host, monkeypatch, capsys):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    assert main(["plan", "qwen3.8-flash", "--gpus", "0", "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["gpus"][0]["uuid"] == "GPU-0003" and record["cuda_visible_devices"] == "3,1"
    assert main(["plan", "qwen3.8-flash"]) == 0
    assert "\n                  indices follow CUDA_VISIBLE_DEVICES=3,1\n" in capsys.readouterr().out


@pytest.mark.parametrize("arguments, code, message", [
    (["plan", "nope"], 2, "supported models: qwen3.8-flash, glm-5.3-flash"),
    (["plan", "qwen3.8-flash", "--gpus", "x"], 2, "--gpus expects indices"),
    (["plan", "qwen3.8-flash", "--gpus", "9"], 2, "does not have"),
    (["plan", "qwen3.8-flash", "--max-concurrency", "65"], 2, "1 through 64"),
    (["plan", "glm-5.3-flash", "--gpus", "0"], 4, "needs 2 or 4 GPUs; 1 selected; use --gpus 0-1"),
    (["pull", "/"], 2, "supported models"),
])
def test_failures_use_the_documented_exit_codes(host, capsys, arguments, code, message):
    assert main(arguments) == code
    error = capsys.readouterr().err
    assert error.startswith("error: ") and message in error


def test_capacity_errors_exit_four_with_aligned_continuation_lines(host, tmp_path, capsys):
    configure(tmp_path, '[memory]\nreserve = "18G"\n')
    assert main(["plan", "qwen3.8-flash", "--gpus", "0"]) == 4
    lines = capsys.readouterr().err.splitlines()
    assert lines[0].startswith("error: needs ")
    assert all(line.startswith("       ") for line in lines[1:])
    assert lines[-1].lstrip().startswith("Options: ")


def test_unknown_flags_and_abbreviations_are_usage_errors(host, capsys):
    for arguments in (["plan", "qwen3.8-flash", "--context", "4096"], ["plan", "qwen3.8-flash", "--gpu", "0"],
                      ["plan", "qwen3.8-flash", "--max", "2"], ["serve"]):
        with pytest.raises(SystemExit) as exit:
            main(arguments)
        assert exit.value.code == 2
    capsys.readouterr()


def listed(output):
    """Split a models listing into {alias: {label: value}}, joining continuation lines to their label."""
    models = {}
    for line in output.splitlines():
        if line and not line.startswith(" "):
            fields = models.setdefault(line, {})
        elif line.startswith("  ") and line[2] != " ":
            label, value = line[2:17].rstrip(), line[17:]
            fields[label] = value
        elif line:
            fields[label] += "\n" + line.strip()
    return models


def test_models_describes_every_recipe(host, tmp_path, monkeypatch, capsys):
    configure(tmp_path, '[defaults]\nmodel = "glm-5.3-flash"\n')
    assert main(["models"]) == 0
    models = listed(capsys.readouterr().out)
    assert list(models)[:2] == ["qwen3.8-flash", "glm-5.3-flash  defaults.model"]
    assert "org/name[@revision]" in list(models)[2]
    qwen, glm = models["qwen3.8-flash"], models["glm-5.3-flash  defaults.model"]
    assert qwen["checkpoint"] == "local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD@dc5b8ab2"
    assert qwen["architecture"] == "Qwen4ExpForConditionalGeneration"
    assert qwen["size"] == "95.1 GiB stored · 68.3 GiB on GPUs · 26.8 GiB PLE table in host memory"
    assert (qwen["context"], qwen["input"]) == ("262,144 tokens", "text and up to 4 images per request")
    assert (qwen["speculation"], qwen["GPUs"]) == ("up to 7 draft tokens per step", "1, 2, or 4")
    assert qwen["concurrency"] == "4 by default · at most 64"
    assert qwen["cached"] == "no · downloaded by the first run"
    assert qwen["fits here"].startswith("1 GPU · 4 requests of ")
    assert glm["checkpoint"] == "local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD@dec48abd"
    assert glm["size"] == "165.5 GiB stored · 165.1 GiB on GPUs"
    assert (glm["input"], glm["speculation"], glm["GPUs"]) == ("text", "none", "2 or 4")
    assert glm["fits here"].startswith("2 GPUs · 4 requests of ")

    monkeypatch.setattr(planning, "discover_gpus", lambda: (gpu(0, 95),))
    assert main(["models"]) == 0
    glm = listed(capsys.readouterr().out)["glm-5.3-flash  defaults.model"]
    assert glm["fits here"] == "no\nglm-5.3-flash needs 2 or 4 SM120 or SM121 GPUs; this host has 1"
    assert main(["models", "--json"]) == 0
    entries = {entry["alias"]: entry for entry in json.loads(capsys.readouterr().out)}
    qwen, glm = entries["qwen3.8-flash"], entries["glm-5.3-flash"]
    assert qwen["plan"]["tensor_parallel"] == 1 and qwen["error"] is None and not qwen["default"]
    assert (qwen["recipe"], qwen["revision"]) == ("qwen38", "dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2")
    assert (qwen["context"], qwen["max_images"], qwen["speculative_tokens"]) == (262144, 4, 7)
    assert (qwen["tensor_parallel_sizes"], qwen["default_concurrency"], qwen["concurrency_limit"]) == ([1, 2, 4], 4, 64)
    assert qwen["device_bytes"] + qwen["mapped_host_bytes"] <= qwen["stored_bytes"] and qwen["prefill_chunk"] == 4096
    assert glm["plan"] is None and "this host has 1" in glm["error"] and glm["default"]


def test_config_labels_each_value_with_its_source(host, tmp_path, capsys):
    configure(tmp_path, '[server]\nport = 9000\napi_key = "secret"\n')
    assert main(["config"]) == 0
    output = capsys.readouterr().out
    assert "secret" not in output and "(set)" in output
    assert any(line.split() == ["server.port", "9000", "[server]"] for line in output.splitlines())
    assert any(line.split() == ["server.host", "127.0.0.1", "default"] for line in output.splitlines())
    assert any(line.split() == ["setting", "value", "source"] for line in output.splitlines())
    assert f"configuration   {tmp_path / '.config/flare/config.toml'}" in output
    assert main(["config", "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["settings"]["server.port"] == {"value": 9000, "source": "[server]"}
    assert record["locations"]["kernel cache"] == str(tmp_path / ".cache/flare/kernels")


def test_pull_downloads_the_pinned_revision_into_the_hub_cache(host, tmp_path, monkeypatch, capsys):
    import huggingface_hub

    calls = []

    def download(repository, *, revision, cache_dir):
        calls.append((repository, revision, cache_dir))
        return "/snapshots/x"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert main(["pull", "glm-5.3-flash"]) == 0
    assert capsys.readouterr().out == "/snapshots/x\n"
    assert calls == [("local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD",
                      "dec48abd33efa73c3bb7c95b74eee10cad34f9be", tmp_path / ".cache/huggingface/hub")]

    def fail(*args, **kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail)
    assert main(["pull", "glm-5.3-flash"]) == 3
    assert "cannot download" in capsys.readouterr().err
