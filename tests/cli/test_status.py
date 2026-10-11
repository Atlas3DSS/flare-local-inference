"""`flare status` against a recorded metrics page."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from flare.cli import main
from flare.cli.status import summarize

METRICS = """# HELP flare_active_requests Active requests.
# TYPE flare_active_requests gauge
flare_active_requests 2
flare_http_inflight_requests 5
flare_uptime_seconds 11520
flare_generated_tokens_last_minute 2478
flare_cache_lookups_total 8
flare_cache_hits_total 6
flare_cached_kv_pages 300
flare_free_kv_pages 5871
flare_free_snapshots 34
flare_accepted_draft_tokens_total 380
flare_draft_request_rounds_total 200
flare_speculative_rounds_total{batch="1",width="4"} 6
"""
NAMES = ["local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD", "qwen3.8-flash"]


@pytest.fixture
def server(monkeypatch, tmp_path):
    """Serve the recorded pages on a loopback port and require the configured key."""
    monkeypatch.setenv("HOME", str(tmp_path))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("Authorization") != "Bearer secret":
                self.send_response(401)
                self.end_headers()
                return
            body = (json.dumps({"object": "list", "data": [{"id": name} for name in NAMES]})
                    if self.path == "/v1/models" else METRICS).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *arguments):
            pass

    listener = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=listener.serve_forever, daemon=True).start()
    yield listener.server_address[1]
    listener.shutdown()


def configure(tmp_path, port, key="secret"):
    path = tmp_path / ".config/flare/config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'[server]\nport = {port}\napi_key = "{key}"\n')


def test_summary_figures():
    summary = summarize(NAMES, METRICS)
    assert summary["active_requests"] == 2 and summary["queued_requests"] == 3
    assert summary["decode_tokens_per_second"] == pytest.approx(41.3)
    assert summary["prefix_cache_hit_rate"] == 0.75 and summary["cached_tokens"] == 300 * 64
    assert summary["free_pages"] == 5871 and summary["free_snapshots"] == 34
    assert summary["accepted_drafts_per_round"] == 1.9
    idle = summarize(["unknown"], "flare_uptime_seconds 20\nflare_generated_tokens_last_minute 100\n")
    assert idle["decode_tokens_per_second"] == 5 and idle["prefix_cache_hit_rate"] is None
    assert idle["cached_tokens"] is None and idle["accepted_drafts_per_round"] is None


def test_status_reads_the_configured_server(server, tmp_path, capsys):
    configure(tmp_path, server)
    assert main(["status"]) == 0
    lines = dict(line.split(None, 1) for line in capsys.readouterr().out.splitlines())
    assert lines["model"].split() == list(reversed(NAMES))
    assert lines["uptime"] == "3h 12m"
    assert lines["requests"] == "2 active  3 queued"
    assert lines["decode"] == "41.3 tokens/s over the last minute"
    assert lines["prefix"] == "cache   75% hit rate  19,200 cached tokens"
    assert lines["free"] == "5,871 pages  34 snapshots"
    assert lines["speculation"] == "1.90 accepted drafts per round"
    assert main(["status", "--url", f"http://127.0.0.1:{server}/", "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["url"] == f"http://127.0.0.1:{server}" and record["uptime_seconds"] == 11520


def test_status_failures_are_runtime_errors(server, tmp_path, capsys):
    configure(tmp_path, server, key="wrong")
    assert main(["status"]) == 6
    assert "rejected server.api_key" in capsys.readouterr().err
    configure(tmp_path, 9)
    assert main(["status"]) == 6
    assert "cannot reach a server at http://127.0.0.1:9" in capsys.readouterr().err
    assert main(["status", "--url", "127.0.0.1:9"]) == 2
