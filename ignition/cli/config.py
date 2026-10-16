"""The configuration file: its schema, built-in values, and precedence under flags.

A flag overrides the configuration file, which overrides the recipe. Every
resolved setting carries the source that supplied it.
"""

import re
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ignition.errors import UsageError

DEFAULT_PATH = Path("~/.config/flare/config.toml")

_SECTIONS = {
    "publishing": {"signing_key": str, "namespace": str, "official_source": str},
    "server": {"host": str, "port": int, "api_key": str},
    "defaults": {"model": str, "max_concurrency": int, "speculative_token_capacity": int},
    "memory": {"reserve": str},
    "prefix_cache": {"memory": str, "disk_limit": str, "disk_transport": str},
    "paths": {"hub_cache": str, "kernel_cache": str, "package_cache": str, "prefix_cache": str, "cuda": str},
}
_MODEL_KEYS = {"max_concurrency": int, "speculative_token_capacity": int}
# Empty prefix-cache values leave the decision to Flare, which derives them from the host.
_BUILT_IN = {
    "server.host": "127.0.0.1", "server.port": 8000, "server.api_key": "",
    "memory.reserve": "0",
    "prefix_cache.memory": "", "prefix_cache.disk_limit": "", "prefix_cache.disk_transport": "",
    "paths.hub_cache": "~/.cache/huggingface/hub", "paths.kernel_cache": "~/.cache/flare/kernels",
    "paths.package_cache": "~/.cache/flare/packages",
    "paths.prefix_cache": "",
    "paths.cuda": "",
}
_TRANSPORTS = ("staged", "direct")
_SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:([KMGT])(?:iB|B)?)?", re.IGNORECASE)


def parse_size(text: str, what: str = "memory size") -> int:
    """Parse a byte count with an optional binary K, M, G, or T suffix."""
    match = _SIZE.fullmatch(text.strip())
    if match is None:
        raise UsageError(f'{what} {text!r} must be a number with an optional K, M, G, or T suffix, such as "16G"')
    return int(float(match[1]) * (1 << 10 * " KMGT".index((match[2] or " ").upper())))


def parse_memory_limit(text: str) -> str:
    """Validate a native serving limit without importing preparation dependencies."""
    value = text.strip()
    number = value.removeprefix("-").strip()
    size = _SIZE.fullmatch(number)
    if size and size[2] and size[1].isascii():
        amount = float(size[1]) * (1 << 10 * " KMGT".index(size[2].upper()))
        valid = math.isfinite(amount) and 1 <= amount < 1 << 64
    else:
        valid = False
        if not value.startswith("-") and re.fullmatch(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)", value):
            valid = 0 < float(value) <= 1
    if not valid:
        raise UsageError("memory limit must be a positive size (30G), a reserve (-30G),"
                         " or a fraction greater than zero and at most one (0.8)")
    return value


@dataclass(frozen=True)
class Setting:
    value: object
    source: str


class Configuration:
    """Validated configuration file contents; an absent default file is empty."""

    def __init__(self, path: Path, values: dict, exists: bool):
        self.path, self.values, self.exists = path, values, exists

    @classmethod
    def load(cls, path: Path | None = None) -> "Configuration":
        resolved = (DEFAULT_PATH if path is None else Path(path)).expanduser()
        if not resolved.is_file():
            if path is not None:
                raise UsageError(f"configuration file {resolved} does not exist")
            return cls(resolved, {}, False)
        try:
            values = tomllib.loads(resolved.read_text())
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
            raise UsageError(f"{resolved}: {error}") from error
        for section, table in values.items():
            if section == "models" and isinstance(table, dict):
                for alias, overrides in table.items():
                    cls._validate(resolved, f'models."{alias}"', overrides, _MODEL_KEYS)
            elif section in _SECTIONS:
                cls._validate(resolved, section, table, _SECTIONS[section])
            else:
                raise UsageError(f"{resolved}: unknown section [{section}]")
        configuration = cls(resolved, values, True)
        parse_size(configuration.get("memory.reserve").value)
        for key in ("prefix_cache.memory", "prefix_cache.disk_limit"):
            if configuration.get(key).value:
                parse_size(configuration.get(key).value, key.replace(".", " "))
        transport = configuration.get("prefix_cache.disk_transport").value
        if transport and transport not in _TRANSPORTS:
            raise UsageError(f"{resolved}: disk_transport in [prefix_cache] must be one of {', '.join(_TRANSPORTS)}")
        return configuration

    @staticmethod
    def _validate(path, section, table, schema):
        if not isinstance(table, dict):
            raise UsageError(f"{path}: [{section}] must be a table")
        for key, value in table.items():
            if key not in schema:
                raise UsageError(f"{path}: unknown key {key} in [{section}]")
            if type(value) is not schema[key]:
                raise UsageError(f"{path}: {key} in [{section}] must be {schema[key].__name__}")
            if key == "port" and not 1 <= value <= 65535:
                raise UsageError(f"{path}: port in [{section}] must be 1 through 65535")
            if key in ("max_concurrency", "speculative_token_capacity") and value < 1:
                raise UsageError(f"{path}: {key} in [{section}] must be at least 1")

    def get(self, key: str, flag=None) -> Setting:
        """Resolve `section.key` from a flag, the file, or the built-in value."""
        if flag is not None:
            return Setting(flag, "flag")
        section, name = key.split(".")
        if name in self.values.get(section, {}):
            return Setting(self.values[section][name], f"[{section}]")
        return Setting(_BUILT_IN[key], "default")

    @property
    def model(self) -> str | None:
        return self.values.get("defaults", {}).get("model")

    def max_concurrency(self, alias: str, recipe_default: int, flag: int | None = None) -> Setting:
        return self.model_setting(alias, "max_concurrency", recipe_default, flag)

    def model_setting(self, alias: str, key: str, recipe_default, flag=None) -> Setting:
        if flag is not None:
            return Setting(flag, "flag")
        for section, table in ((f'[models."{alias}"]', self.values.get("models", {}).get(alias, {})),
                               ("[defaults]", self.values.get("defaults", {}))):
            if key in table:
                return Setting(table[key], section)
        return Setting(recipe_default, "recipe")

    @property
    def reserve_bytes(self) -> int:
        return parse_size(self.get("memory.reserve").value)

    def prefix_cache_flags(self) -> list[str]:
        """The `flare run` flags carrying the host prefix cache settings that are set."""
        flags = []
        directory = self.directory("prefix_cache")
        if directory is not None:
            flags += ["--prefix-cache-disk", str(directory)]
        for key, flag in (("memory", "--prefix-cache-memory"), ("disk_limit", "--prefix-cache-disk-limit"),
                          ("disk_transport", "--prefix-cache-disk-transport")):
            value = self.get(f"prefix_cache.{key}").value
            if value:
                flags += [flag, value]
        return flags

    def directory(self, name: str) -> Path | None:
        """Return a `[paths]` entry with the home directory expanded, or None when it is empty."""
        value = self.get(f"paths.{name}").value
        return Path(value).expanduser() if value else None

    def write_initial(self, model: str, max_concurrency: int) -> None:
        """Create the file with every resolved value present as a comment."""
        lines = [
            "# Flare configuration. Every key is optional; remove a leading # to set it.",
            "", "[server]",
            f'# host = "{self.get("server.host").value}"',
            f'# port = {self.get("server.port").value}',
            '# api_key = ""                    # empty: no authentication',
            "", "[defaults]",
            f'# model = "{model}"',
            f"# max_concurrency = {max_concurrency}",
            '# speculative_token_capacity = 8  # wider target graphs require batch * width within this limit',
            "", "[memory]",
            f'# reserve = "{self.get("memory.reserve").value}"'
            "                   # GPU memory left untouched for processes that start later",
            "", "[prefix_cache]",
            '# memory = "16G"                 # host RAM retaining prefix checkpoints; 0 disables the memory tier',
            '# disk_limit = "200G"            # record files kept under paths.prefix_cache; 0 disables the disk tier',
            '# disk_transport = "staged"      # staged, or direct for GPUDirect Storage',
            "", "[paths]",
            f'# hub_cache = "{self.get("paths.hub_cache").value}"',
            f'# kernel_cache = "{self.get("paths.kernel_cache").value}"',
            f'# package_cache = "{self.get("paths.package_cache").value}"',
            '# prefix_cache = "~/.cache/flare/prefix-cache"   # prefix checkpoint records; empty: no disk tier',
            '# cuda = "/opt/cuda"              # used only when the toolkit is not found in installed packages',
        ]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(lines) + "\n")
