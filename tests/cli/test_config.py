"""Configuration file validation and precedence."""

import pytest

from ignition.cli.config import Configuration, Setting, parse_size
from ignition.errors import UsageError


def load(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text)
    return Configuration.load(path)


def test_absent_default_file_yields_built_in_values(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    configuration = Configuration.load()
    assert not configuration.exists and configuration.path == tmp_path / ".config/flare/config.toml"
    assert configuration.get("server.host") == Setting("127.0.0.1", "default")
    assert configuration.get("server.port") == Setting(8000, "default")
    assert configuration.reserve_bytes == 0 and configuration.model is None
    assert configuration.directory("hub_cache") == tmp_path / ".cache/huggingface/hub"
    assert configuration.directory("kernel_cache") == tmp_path / ".cache/flare/kernels"
    assert configuration.directory("cuda") is None


def test_an_explicit_missing_file_is_a_usage_error(tmp_path):
    with pytest.raises(UsageError, match="does not exist"):
        Configuration.load(tmp_path / "absent.toml")


def test_flags_override_the_file_which_overrides_the_recipe(tmp_path):
    configuration = load(tmp_path, """
[server]
host = "0.0.0.0"
port = 9000

[defaults]
model = "qwen3.8-flash"
max_concurrency = 6

[memory]
reserve = "16G"

[models."qwen3.8-flash"]
max_concurrency = 2
""")
    assert configuration.get("server.host") == Setting("0.0.0.0", "[server]")
    assert configuration.get("server.port") == Setting(9000, "[server]")
    assert configuration.get("server.port", 7000) == Setting(7000, "flag")
    assert configuration.model == "qwen3.8-flash" and configuration.reserve_bytes == 16 << 30

    assert configuration.max_concurrency("qwen3.8-flash", 4) == Setting(2, '[models."qwen3.8-flash"]')
    assert configuration.max_concurrency("glm-5.3-flash", 4) == Setting(6, "[defaults]")
    assert configuration.max_concurrency("qwen3.8-flash", 4, 9) == Setting(9, "flag")


def test_recipe_supplies_concurrency_when_the_file_is_silent(tmp_path):
    configuration = load(tmp_path, "[server]\nport = 8001\n")
    assert configuration.max_concurrency("qwen3.8-flash", 4) == Setting(4, "recipe")


@pytest.mark.parametrize("text, message", [
    ("[memory]\nfraction = 0.9\n", "unknown key fraction"),
    ("[engine]\npages = 4\n", "unknown section"),
    ("[server]\nport = \"8000\"\n", "port in .* must be int"),
    ("[server]\nport = 70000\n", "1 through 65535"),
    ("[defaults]\nmax_concurrency = true\n", "must be int"),
    ("[defaults]\nmax_concurrency = 0\n", "at least 1"),
    ("[memory]\nconversation_cache = 16\n", "unknown key conversation_cache"),
    ("[memory]\nreserve = \"lots\"\n", "memory size"),
    ("[memory]\nreserve = 16\n", "must be str"),
    ("[models.\"qwen3.8-flash\"]\ncontext = 4096\n", "unknown key context"),
    ("server = 3\n", "must be a table"),
    ("[server\n", "config.toml"),
])
def test_invalid_files_are_usage_errors(tmp_path, text, message):
    with pytest.raises(UsageError, match=message):
        load(tmp_path, text)


@pytest.mark.parametrize("text, expected", [
    ("0", 0), ("4096", 4096), ("16G", 16 << 30), ("16g", 16 << 30), ("1.5G", 3 << 29), ("512M", 512 << 20),
    ("8 GiB", 8 << 30), ("2T", 2 << 40), ("64K", 64 << 10),
])
def test_sizes_use_binary_units(text, expected):
    assert parse_size(text) == expected


@pytest.mark.parametrize("text", ["", "G", "-1G", "16X", "1e3"])
def test_malformed_sizes_are_usage_errors(text):
    with pytest.raises(UsageError):
        parse_size(text)
