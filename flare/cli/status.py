"""`flare status`: summarize a running server from `/v1/models` and `/metrics`."""

import json
import re
import urllib.error
import urllib.request

from flare.errors import FlareError, UsageError
from flare.recipes import recipes

from .config import Configuration

_SAMPLE = re.compile(r"(flare_\w+)\s+(\S+)")


def server_url(configuration: Configuration, url: str | None) -> str:
    if url is not None:
        if not url.startswith(("http://", "https://")):
            raise UsageError(f"--url must begin with http:// or https://; got {url!r}")
        return url.rstrip("/")
    host = configuration.get("server.host").value
    return f"http://{f'[{host}]' if ':' in host else host}:{configuration.get('server.port').value}"


def fetch(configuration: Configuration, url: str, path: str) -> bytes:
    request = urllib.request.Request(url + path)
    if key := configuration.get("server.api_key").value:
        request.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        reason = "the server rejected server.api_key" if error.code == 401 else f"HTTP {error.code}"
        raise FlareError(f"cannot read {url}{path}: {reason}") from error
    except (OSError, ValueError) as error:
        raise FlareError(f"cannot reach a server at {url}: {getattr(error, 'reason', error)}") from error


def summarize(names: list[str], metrics: str) -> dict:
    """Reduce served names and Prometheus samples to the figures `status` prints."""
    values = {name: float(value) for name, value in map(
        lambda line: _SAMPLE.fullmatch(line).groups(),
        (line for line in metrics.splitlines() if _SAMPLE.fullmatch(line)))}

    def count(name):
        return int(values.get(f"flare_{name}", 0))

    page_tokens = next((recipe.page_tokens for recipe in recipes() for checkpoint in recipe.checkpoints
                        if {checkpoint.alias, checkpoint.repository} & set(names)), None)
    uptime, lookups, rounds = count("uptime_seconds"), count("cache_lookups_total"), count("draft_request_rounds_total")
    return {
        "models": names, "uptime_seconds": uptime,
        "active_requests": count("active_requests"),
        "queued_requests": max(0, count("http_inflight_requests") - count("active_requests")),
        "decode_tokens_per_second": count("generated_tokens_last_minute") / max(1, min(60, uptime)),
        "prefix_cache_hit_rate": count("cache_hits_total") / lookups if lookups else None,
        "cached_tokens": None if page_tokens is None else count("cached_kv_pages") * page_tokens,
        "free_pages": count("free_kv_pages"), "free_snapshots": count("free_snapshots"),
        "accepted_drafts_per_round": count("accepted_draft_tokens_total") / rounds if rounds else None,
    }


def _duration(seconds: int) -> str:
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = [(days, "d"), (hours, "h"), (minutes, "m"), (seconds, "s")]
    shown = [f"{value}{unit}" for value, unit in parts if value][:2]
    return " ".join(shown) or "0s"


def status(args) -> int:
    configuration = Configuration.load(args.config)
    url = server_url(configuration, args.url)
    try:
        names = [entry["id"] for entry in json.loads(fetch(configuration, url, "/v1/models"))["data"]]
    except (ValueError, KeyError, TypeError) as error:
        raise FlareError(f"{url}/v1/models is not a Flare model listing") from error
    summary = summarize(names, fetch(configuration, url, "/metrics").decode())
    if args.json:
        print(json.dumps({"url": url, **summary}, indent=2))
        return 0
    hit_rate, cached = summary["prefix_cache_hit_rate"], summary["cached_tokens"]
    rows = [
        ("model", "  ".join(reversed(names))),
        ("uptime", _duration(summary["uptime_seconds"])),
        ("requests", f"{summary['active_requests']} active  {summary['queued_requests']} queued"),
        ("decode", f"{summary['decode_tokens_per_second']:.1f} tokens/s over the last minute"),
        ("prefix cache", ("no lookups" if hit_rate is None else f"{hit_rate:.0%} hit rate")
         + ("" if cached is None else f"  {cached:,} cached tokens")),
        ("free", f"{summary['free_pages']:,} pages  {summary['free_snapshots']:,} snapshots"),
    ]
    if summary["accepted_drafts_per_round"] is not None:
        rows.append(("speculation", f"{summary['accepted_drafts_per_round']:.2f} accepted drafts per round"))
    for label, value in rows:
        print(f"{label:<15}{value}")
    return 0
