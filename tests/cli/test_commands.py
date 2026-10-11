"""Command output and exit codes with injected GPUs and hub cache."""

import json
import socket

import pytest

from flare import planning
from flare.cli import _parser, commands, main
from flare.recipes import resolve

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


def run(*arguments):
    """Run until the first failure; these cases fail before preparation would load weights."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return main(["run", *arguments, "--port", str(port)])


def failure(capsys):
    lines = capsys.readouterr().err.splitlines()
    return lines[next(index for index, line in enumerate(lines) if line.startswith("error: ")):]


def test_run_uses_the_configured_model(host, tmp_path, capsys):
    assert run() == 2
    assert "no MODEL given and defaults.model is not set" in failure(capsys)[0]
    configure(tmp_path, '[defaults]\nmodel = "nope"\n')
    assert run() == 2
    assert "unknown model 'nope'" in failure(capsys)[0]


@pytest.mark.parametrize("arguments, code, message", [
    (["run", "nope"], 2, "supported models: qwen3.8-flash, glm-5.3-flash"),
    (["run", "qwen3.8-flash", "--gpus", "x"], 2, "--gpus expects indices"),
    (["run", "qwen3.8-flash", "--gpus", "9"], 2, "does not have"),
    (["run", "qwen3.8-flash", "--max-concurrency", "65"], 2, "1 through 64"),
    (["run", "glm-5.3-flash", "--gpus", "0"], 4, "needs 2 or 4 GPUs; 1 selected; use --gpus 0-1"),
    (["pull", "/"], 2, "supported models"),
])
def test_failures_use_the_documented_exit_codes(host, capsys, arguments, code, message):
    assert (run(*arguments[1:]) if arguments[0] == "run" else main(arguments)) == code
    assert message in failure(capsys)[0]


def test_capacity_errors_exit_four_with_aligned_continuation_lines(host, tmp_path, capsys):
    configure(tmp_path, '[memory]\nreserve = "18G"\n')
    assert run("qwen3.8-flash", "--gpus", "0") == 4
    lines = failure(capsys)
    assert lines[0].startswith("error: needs ")
    assert all(line.startswith("       ") for line in lines[1:])
    assert lines[-1].lstrip().startswith("Options: ")


def test_unknown_commands_flags_and_abbreviations_are_usage_errors(host, capsys):
    for arguments in (["run", "qwen3.8-flash", "--context", "4096"], ["run", "qwen3.8-flash", "--gpu", "0"],
                      ["run", "qwen3.8-flash", "--max", "2"], ["serve"], ["plan", "qwen3.8-flash"],
                      ["prepare", "qwen3.8-flash"], ["models", "--no-mtp"]):
        with pytest.raises(SystemExit) as exit:
            main(arguments)
        assert exit.value.code == 2
    capsys.readouterr()


def speculation(*arguments, model="qwen3.8-flash"):
    """Resolve the speculation settings `run` would prepare for a command line."""
    return commands.speculation(_parser().parse_args(["run", model, *arguments]), resolve(model))


def test_speculation_flags_override_the_recipe():
    default = speculation()
    assert default == resolve("qwen3.8-flash").recipe.speculation
    assert (default.draft_tokens, default.adaptive, default.widths, default.fixed_width) == (
        3, True, (1, 2, 3, 4), None)
    assert (default.graph_budget, default.proposal_source, default.history_drafts) == (8, "mtp", False)
    assert speculation("--no-mtp") is None
    # Verification adapts among every count up to the draft-token setting unless it is fixed.
    deeper = speculation("--mtp-draft-tokens", "5")
    assert (deeper.draft_tokens, deeper.adaptive, deeper.widths, deeper.fixed_width) == (
        5, True, (1, 2, 3, 4, 5, 6), None)
    fixed = speculation("--mtp-draft-tokens", "5", "--no-mtp-adaptive-verification")
    assert (fixed.draft_tokens, fixed.adaptive, fixed.fixed_width) == (5, False, 6)
    assert speculation("--no-mtp-adaptive-verification").fixed_width == 4
    assert speculation("--mtp-draft-tokens", "0").widths == (1,)
    assert speculation("--mtp-draft-tokens", "7").draft_tokens == 7
    assert speculation("--mtp-graph-budget", "12").graph_budget == 12
    copied = speculation("--mtp-proposal-source", "copy")
    assert copied.proposal_source == "copy" and not copied.learned and not copied.history_drafts
    assert speculation("--history-drafts").history_drafts
    assert speculation(model="glm-5.3-flash").draft_tokens == 5
    assert speculation("--no-mtp", model="glm-5.3-flash") is None


@pytest.mark.parametrize("arguments, message", [
    (["--no-mtp", "--mtp-draft-tokens", "2"], "--no-mtp disables speculation; --mtp-draft-tokens cannot"),
    (["--no-mtp", "--no-mtp-adaptive-verification"],
     "--no-mtp disables speculation; --no-mtp-adaptive-verification cannot accompany it"),
    (["--no-mtp", "--history-drafts"], "--no-mtp disables speculation; --history-drafts cannot accompany it"),
    (["--mtp-draft-tokens", "8"], "--mtp-draft-tokens must be 0 through 7; --no-mtp disables speculation"),
    (["--mtp-draft-tokens", "-1"], "--mtp-draft-tokens must be 0 through 7"),
    (["--mtp-graph-budget", "2"], "--mtp-graph-budget must be 3 through 56 at --max-concurrency 4"),
    (["--max-concurrency", "64", "--mtp-graph-budget", "6"], "--mtp-graph-budget must be 7 through 56"),
])
def test_invalid_speculation_flags_are_usage_errors(host, capsys, arguments, message):
    assert run("qwen3.8-flash", *arguments) == 2
    assert message in failure(capsys)[0]


def test_speculation_flags_have_one_spelling(host, capsys):
    for arguments in (["--mtp-draft-tokens", "1,3"], ["--mtp-max-draft-tokens", "5"],
                      ["--mtp-fixed-draft-tokens", "2"], ["--mtp-draft-tokens", "x"]):
        with pytest.raises(SystemExit) as exit:
            main(["run", "qwen3.8-flash", *arguments])
        assert exit.value.code == 2
    capsys.readouterr()


def test_history_copy_flags_apply_only_to_recipes_that_support_them(host, capsys):
    assert run("glm-5.3-flash", "--mtp-proposal-source", "copy") == 2
    assert "glm-5.3-flash drafts only with its MTP module; --mtp-proposal-source copy does not apply" in (
        failure(capsys)[0])
    assert run("glm-5.3-flash", "--history-drafts") == 2
    assert "--history-drafts does not apply" in failure(capsys)[0]
    with pytest.raises(SystemExit):
        main(["run", "qwen3.8-flash", "--mtp-proposal-source", "ngram"])
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
    assert (qwen["speculation"], qwen["GPUs"]) == ("up to 3 draft tokens per round · MTP module", "1, 2, or 4")
    assert qwen["concurrency"] == "4 by default · at most 64"
    assert qwen["cached"] == "no · downloaded by the first run"
    assert qwen["fits here"].startswith("1 GPU · 4 requests of ")
    assert glm["checkpoint"] == "local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD@dec48abd"
    assert glm["size"] == "165.5 GiB stored · 165.1 GiB on GPUs"
    assert (glm["input"], glm["GPUs"]) == ("text", "2 or 4")
    assert glm["speculation"] == "up to 5 draft tokens per round · MTP module"
    assert glm["fits here"].startswith("2 GPUs · 4 requests of ")

    monkeypatch.setattr(planning, "discover_gpus", lambda: (gpu(0, 95),))
    assert main(["models"]) == 0
    glm = listed(capsys.readouterr().out)["glm-5.3-flash  defaults.model"]
    assert glm["fits here"] == "no\nglm-5.3-flash needs 2 or 4 SM120 or SM121 GPUs; this host has 1"
    assert main(["models", "--json"]) == 0
    entries = {entry["alias"]: entry for entry in json.loads(capsys.readouterr().out)}
    qwen, glm = entries["qwen3.8-flash"], entries["glm-5.3-flash"]
    assert qwen["estimate"]["tensor_parallel"] == 1 and qwen["error"] is None and not qwen["default"]
    assert (qwen["recipe"], qwen["revision"]) == ("qwen38", "dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2")
    assert (qwen["context"], qwen["max_images"]) == (262144, 4)
    assert qwen["speculation"] == {"draft_tokens": 3, "adaptive": True, "graph_budget": 8, "proposal_source": "mtp",
                                   "history_drafts": False, "history_copy": True}
    assert glm["speculation"]["draft_tokens"] == 5 and not glm["speculation"]["history_copy"]
    assert (qwen["tensor_parallel_sizes"], qwen["default_concurrency"], qwen["concurrency_limit"]) == ([1, 2, 4], 4, 64)
    assert qwen["device_bytes"] + qwen["mapped_host_bytes"] <= qwen["stored_bytes"] and qwen["prefill_chunk"] == 4096
    assert glm["estimate"] is None and "this host has 1" in glm["error"] and glm["default"]


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
