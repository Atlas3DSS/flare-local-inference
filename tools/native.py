#!/usr/bin/env python3
"""Run native SM70 benchmarks with explicit package, executable and input paths.

Run only on an externally allocated GPU selected by CUDA_VISIBLE_DEVICES.
No site-specific lease service, model, signing key or private trace is bundled.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request


DEST = None
GPU = None


def query(*command, timeout=30):
    return subprocess.check_output(command, text=True, timeout=timeout).strip()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


TAG = ""   # --tag prefixes every artifact name (e.g. "32k-" for the S1L package at --context 32768)


def artifacts(mode, depth):
    stem = f"{TAG}{mode}-depth{depth}"
    return {name: DEST / (stem + suffix) for name, suffix in {
        "report": ".json", "requests": "-requests.json", "server": "-server.log",
        "client": "-client.log", "telemetry": "-telemetry.jsonl",
        "before": "-metrics-before.json", "after": "-metrics-after.json",
    }.items()}


def port_for(args, depth):
    # One port per depth: the previous depth's server leaves TIME_WAIT sockets that make a plain bind probe fail.
    return args.port + (depth - 3)


def prefill_srpt(args):
    """`flare run --prefill-srpt` (opt-in shortest-remaining prefill ordering) when requested; callers that build
    their own argument namespace without the option keep the default ordering."""
    return ["--prefill-srpt"] if getattr(args, "prefill_srpt", False) else []


def server_command(args, depth):
    if args.mode == "real":
        # Exactly the leaderboard's standard server (agentbench/prep-mtp/run_fixed.sh).
        return [str(args.executable), "run", str(args.package), "--publisher-keys",
            str(args.publisher_keys), "--trust-store", str(args.trust_store),
            "--verify-weights", "--gpus", "0", "--memory-limit", args.memory_limit, *(["--scheduler-policy", args.scheduler_policy] if args.scheduler_policy else []), "--max-concurrency", "4",
            "--mtp-draft-tokens", str(depth), "--no-mtp-adaptive-verification", *prefill_srpt(args),
            "--prefix-cache-memory", args.prefix_cache, "--host", "127.0.0.1", "--port", str(port_for(args, depth))]
    return [str(args.executable), "run", str(args.package), "--publisher-keys",
        str(args.publisher_keys), "--trust-store", str(args.trust_store),
        "--verify-weights", "--gpus", "0", "--prefix-cache-memory", args.prefix_cache,
        "--embedding-table-storage", "ram", "--memory-limit", args.memory_limit, *(["--scheduler-policy", args.scheduler_policy] if args.scheduler_policy else []),
        "--kv-cache-limit", "concurrency", "--max-concurrency", "4",
        "--mtp-draft-tokens", str(depth), "--no-mtp-adaptive-verification", *prefill_srpt(args),
        "--host", "127.0.0.1", "--port", str(port_for(args, depth))]


def client_command(args, depth, paths, model):
    base = f"http://127.0.0.1:{port_for(args, depth)}"
    if args.mode == "native":
        return [sys.executable, str(args.client),
            "--base-url", base, "--model", model, "--depth", str(depth),
            "--prompts", str(args.prompts),
            "--prompt-ids", *args.prompt_ids,
            "--output-tokens", str(args.tokens), "--repetitions", str(args.repetitions),
            "--seed", "7035", "--concurrencies", *map(str, args.concurrencies), "--output", str(paths["requests"])]
    return [sys.executable, str(args.real_client),
        "--base-url", base + "/v1", "--model", model, "--traces-dir", str(args.traces_dir),
        "--agents", "4", "--duration", str(args.duration), "--timeout", "1800", "--stagger", "5",
        "--exactness", "4", "--temperature", "0", "--save-chunks", "--out", str(paths["requests"])]


def require_supervisor():
    # Allocation/lease ownership is an external prerequisite, not a portable
    # property inferable from a machine name or a particular wrapper pathname.
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU:
        raise RuntimeError('--gpu-uuid must equal the single CUDA_VISIBLE_DEVICES allocation')
    capability = query('nvidia-smi', '-i', GPU, '--query-gpu=compute_cap', '--format=csv,noheader,nounits')
    if capability != '7.0':
        raise RuntimeError('native SM70 benchmark requires compute capability 7.0')
    return {'allocation': 'externally managed; not verified by this harness', 'parent_pid': os.getppid()}


def processes():
    raw = query("nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits")
    return [{"pid": int(pid), "gpu": uuid.strip(), "mib": int(mib)}
            for pid, uuid, mib in csv.reader(raw.splitlines()) if uuid.strip() == GPU]


def telemetry():
    fields = ("memory.total", "memory.used", "memory.free", "utilization.gpu",
              "temperature.gpu", "clocks.sm", "clocks.mem")
    raw = query("nvidia-smi", "-i", GPU, "--query-gpu=" + ",".join(fields),
                "--format=csv,noheader,nounits")
    values = next(csv.reader(raw.splitlines()))
    return {"unix": time.time(), "gpu": dict(zip(fields, map(int, values), strict=True)),
            "processes": processes(), "loadavg": Path("/proc/loadavg").read_text().strip(),
            "mem_available_kib": int(re.search(r"^MemAvailable:\s+(\d+)",
                Path("/proc/meminfo").read_text(), re.M).group(1))}


def fetch(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode()


def metrics(base):
    result = {"observed_at": datetime.now(timezone.utc).isoformat(), "errors": []}
    for name in ("json", "prometheus"):
        try:
            raw = fetch(base + "/metrics?format=" + name)
            result[name] = json.loads(raw) if name == "json" else raw
        except Exception as error:
            result["errors"].append({"endpoint": name, "error": str(error)})
    return result


def prometheus_values(raw):
    values = {}
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"(\S+)\s+([-+0-9.eE]+)", line)
        if match:
            value = float(match[2]) if any(c in match[2] for c in ".eE") else int(match[2])
            if math.isfinite(value):
                values[match[1]] = value
    return values


def metric_delta(before, after):
    left = prometheus_values(before.get("prometheus", ""))
    right = prometheus_values(after.get("prometheus", ""))
    counters = {key: right[key] - value for key, value in left.items()
                if key in right and key.split("{", 1)[0].endswith("_total")}
    return {"counters": counters,
            "counter_reset": any(value < 0 for value in counters.values()),
            "host_restores": counters.get("flare_host_cache_fetched_total"),
            "host_restore_failures": counters.get("flare_host_cache_fetch_failed_total"),
            "restore_graph_launches": counters.get('flare_graph_launches_total{kind="restore"}'),
            "interval_note": "Includes client execution and the real harness's solo exactness requests; no throughput is inferred from this interval."}


def summarize_real(raw):
    """Use server token usage only; retain replay.py's approximate arrival window."""
    requests = raw["requests"]
    def median(values):
        return statistics.median(values) if values else None
    agents = []
    all_rates = []
    missing_usage = 0
    for agent in range(4):
        selected = [row for row in requests if row["agent"] == agent]
        good = [row for row in selected if not row["error"]]
        rates = []
        for row in good:
            count = (row.get("usage") or {}).get("completion_tokens")
            if type(count) is not int:
                missing_usage += 1
                continue
            if count >= 8 and row["t_first"] is not None and row["t_last"] > row["t_first"]:
                rates.append((count - 1) / (row["t_last"] - row["t_first"]))
        all_rates.extend(rates)
        agents.append({"agent": agent, "requests": len(selected), "errors": len(selected) - len(good),
            "timeouts": sum(bool(row["timeout"]) for row in selected),
            "median_decode_tokens_per_second": median(rates), "decode_samples": len(rates),
            "median_ttft_seconds": median([row["t_first"] - row["t_start"]
                for row in good if row["t_first"] is not None]),
            "median_turn_seconds": median([row["t_end"] - row["t_start"] for row in good])})
    good = [row for row in requests if not row["error"]]
    return {"wall_seconds": raw["wall_seconds"], "requests": len(requests),
        "errors": sum(bool(row["error"]) for row in requests),
        "incomplete_requests": sum(not row["error"] and row.get("finish_reason") is None for row in requests),
        "timeouts": sum(bool(row["timeout"]) for row in requests),
        "requests_missing_native_usage": missing_usage, "per_agent": agents,
        "median_individual_decode_tokens_per_second": median(all_rates),
        "median_ttft_seconds": median([row["t_first"] - row["t_start"]
            for row in good if row["t_first"] is not None]),
        "median_turn_seconds": median([row["t_end"] - row["t_start"] for row in good]),
        "bad_tool_json": sum(row.get("tool_calls_bad_json", 0) for row in requests),
        "truncated_tool_calls": sum(row.get("tool_calls_truncated", 0) for row in requests),
        "exactness": raw.get("exactness"),
        "exactness_interpretation": "Standard agentbench requests temperature=0. The package's declared ranges clamp it to temperature=.7, top_k=20, top_p=.95; this is a fixed-width concurrent/solo reproducibility check, not deterministic argmax generation.",
        "decode_basis": "(server completion_tokens - 1) / first-to-last nonempty SSE arrival; replies below eight tokens omitted. SSE chunks are never counted as tokens. First-chunk batching and detokenization make this an estimate, unlike native token timing.",
        "duration_note": "The requested 1200-second workload is not the generic agentbench analyzer's 1800-second soak criterion; final in-flight requests and solo checks can extend wall time."}


def stop_child(child, timeout=180):
    if child is None:
        return None
    started = time.monotonic()
    forced = False
    try:
        if child.poll() is None:
            child.terminate()
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Only this driver's separately created child process group is owned.
            forced = True
            os.killpg(child.pid, signal.SIGKILL)
            code = child.wait(timeout=30)
    except Exception as error:
        return {"error": str(error), "returncode": child.poll(), "forced_kill": forced,
                "seconds": time.monotonic() - started}
    return {"returncode": code, "seconds": time.monotonic() - started, "forced_kill": forced}


def run_depth(args, depth, supervisor, help_text):
    paths = artifacts(args.mode, depth)
    report = {"status": "starting", "mode": args.mode, "depth": depth,
        "started_at": datetime.now(timezone.utc).isoformat(), "hostname": os.uname().nodename,
        "gpu_uuid": GPU, "supervisor": supervisor, "server_command": server_command(args, depth),
        "executable_help": help_text, "artifacts": {key: str(path) for key, path in paths.items()},
        "client_protocol": "benchmark_mtp.py protocol; explicit input file" if args.mode == "native" else "external replay client",
        "package": str(args.package), "executable": str(args.executable)}
    with args.executable.open('rb') as executable:
        report['executable_sha256'] = hashlib.file_digest(executable, 'sha256').hexdigest()
    with paths["report"].open("x"):
        pass
    write_json(paths["report"], report)
    server = client = watcher = None
    stop = threading.Event()
    samples, monitor_errors = [], []
    base = f"http://127.0.0.1:{port_for(args, depth)}"
    before = after = None

    def monitor():
        with paths["telemetry"].open("x") as log:
            while not stop.is_set():
                try:
                    row = telemetry()
                    samples.append(row)
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                except Exception as error:
                    monitor_errors.append(str(error))
                stop.wait(2)

    try:
        report["compute_before"] = processes()
        if report["compute_before"]:
            raise RuntimeError("Selected V100 has a foreign compute process before server launch")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port_for(args, depth)))
        started = time.monotonic()
        with paths["server"].open("x") as log:
            server = subprocess.Popen(report["server_command"], stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        report["server_pid"] = server.pid
        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        deadline = started + args.ready_timeout
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"server exited before readiness: {server.returncode}")
            try:
                fetch(base + "/health", timeout=1)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("native server readiness timeout")
                time.sleep(1)
        report["readiness_seconds"] = time.monotonic() - started
        live = processes()
        if len(live) != 1 or live[0]["pid"] != server.pid:
            raise RuntimeError(f"expected only the owned native server on the selected V100: {live}")
        if Path(f"/proc/{server.pid}/exe").resolve() != args.executable.resolve():
            raise RuntimeError("unexpected serving executable")
        report["cpu_spread"] = query("ps", "-L", "-p", str(server.pid), "-o", "pid,tid,psr,pcpu,comm")
        report["models"] = json.loads(fetch(base + "/v1/models"))
        card = report["models"]["data"][0]
        report["ready_telemetry"] = telemetry()
        before = metrics(base)
        write_json(paths["before"], before)
        if before["errors"]:
            raise RuntimeError(f"cannot record complete baseline metrics: {before['errors']}")
        snapshot = before["json"]
        report["capacity"] = {"served_context_tokens_per_request": card.get("max_model_len"),
            "max_concurrency": card.get("flare", {}).get("max_concurrency"),
            "prefill_chunk_size": card.get("flare", {}).get("prefill_chunk_size"),
            "kv_pages": snapshot.get("total_pages"), "tokens_per_page": snapshot.get("page_tokens"),
            "kv_pool_tokens": snapshot.get("total_pages", 0) * snapshot.get("page_tokens", 0),
            "note": "KV pool includes shared/private backing; served per-request context comes from the native model card, not pool tokens divided by four."}
        report["client_command"] = client_command(args, depth, paths, card["id"])
        report["status"] = "running"
        write_json(paths["report"], report)
        measured = time.monotonic()
        with paths["client"].open("x") as log:
            client = subprocess.Popen(report["client_command"], stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            while client.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError("native server exited during the benchmark")
                time.sleep(1)
        report["client_seconds"] = time.monotonic() - measured
        report["client_exit_code"] = client.returncode
        after = metrics(base)
        write_json(paths["after"], after)
        report["metrics_delta"] = metric_delta(before, after)
        if paths["requests"].exists():
            raw = json.loads(paths["requests"].read_text())
            report["summary"] = raw["summary"] if args.mode == "native" else summarize_real(raw)
            if args.mode == "real":
                exact = raw.get("exactness") or {}
                if (not raw["requests"] or report["summary"]["errors"]
                        or report["summary"]["incomplete_requests"] or raw["wall_seconds"] < args.duration
                        or any(agent["requests"] == 0 for agent in report["summary"]["per_agent"])
                        or exact.get("checked", 0) != 4 or exact.get("mismatches")):
                    raise RuntimeError("real workload has request failures or incomplete/failed exactness checks")
        else:
            raise RuntimeError("benchmark did not write its raw request JSON")
        if client.returncode or after["errors"] or report["metrics_delta"]["counter_reset"]:
            raise RuntimeError("client failed, metrics collection failed, or counters reset")
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        report["client_cleanup"] = stop_child(client, timeout=30)
        try:
            if before is not None and after is None and server is not None and server.poll() is None:
                after = metrics(base)
                write_json(paths["after"], after)
                report["metrics_delta"] = metric_delta(before, after)
        except Exception as error:
            report["cleanup_metrics_error"] = str(error)
        finally:
            report["server_cleanup"] = stop_child(server)
        stop.set()
        if watcher is not None:
            watcher.join(timeout=65)
            if watcher.is_alive():
                monitor_errors.append("telemetry collector did not drain within 65 seconds")
        report["monitor_errors"] = monitor_errors
        if samples:
            report["vram"] = {"peak_device_used_mib": max(s["gpu"]["memory.used"] for s in samples),
                "minimum_device_free_mib": min(s["gpu"]["memory.free"] for s in samples),
                "peak_process_used_mib": max(sum(p["mib"] for p in s["processes"]) for s in samples),
                "peak_temperature_c": max(s["gpu"]["temperature.gpu"] for s in samples),
                "sample_count": len(samples), "sampling_seconds": 2}
        try:
            report["compute_after"] = processes()
        except Exception as error:
            report["postflight_error"] = str(error)
        if paths["server"].exists():
            report["capacity_log_lines"] = [line for line in paths["server"].read_text(errors="replace").splitlines()
                if re.search(r"cache|contexts?|tokens/request|GiB|concurrency", line, re.I)]
        cleanup = report.get("server_cleanup") or {}
        if (cleanup.get("forced_kill") or cleanup.get("error") or cleanup.get("returncode", 0) != 0
                or (report.get("client_cleanup") or {}).get("error")
                or report.get("compute_after") or report.get("postflight_error")
                or report.get("cleanup_metrics_error") or monitor_errors):
            report["status"] = "failed"
        report["ended_at"] = datetime.now(timezone.utc).isoformat()
        write_json(paths["report"], report)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    if report["status"] != "complete":
        raise RuntimeError("benchmark or server cleanup failed; see " + str(paths["report"]))


def require_qualification(qualification, package):
    measurement = qualification.get('measurement', {})
    if (qualification.get('status') != 'exported'
            or qualification.get('package') != str(package)
            or any(not qualification.get(key, {}).get('passed') for key in
                   ('validation', 'long_sequence', 'consumer_identity', 'commit_capacity_growth'))
            or qualification.get('components', {}).get('status') != 'complete'
            or qualification.get('accuracy', {}).get('status') != 'complete'
            or measurement.get('status') != 'complete' or measurement.get('qualification_passed') is not True
            or not measurement.get('runs') or not measurement.get('summary')
            or any(run.get('status') != 'complete' for run in measurement.get('runs', []))):
        raise RuntimeError('native measurement requires all final package qualification gates')
    with Path(package).open('rb') as primary:
        digest = hashlib.file_digest(primary, 'sha256').hexdigest()
    if qualification.get('package_primary_sha256') != digest:
        raise RuntimeError('qualified signed package primary hash mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "real"), required=True)
    parser.add_argument("--depths", type=int, choices=(2, 3, 4), nargs="+")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--traces-dir", type=Path, help="external traces for --mode real; not bundled")
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--publisher-keys', type=Path, required=True)
    parser.add_argument('--trust-store', type=Path, required=True)
    parser.add_argument('--memory-limit', default='29.5G')
    parser.add_argument('--scheduler-policy')
    parser.add_argument('--prefill-srpt', action='store_true',
                        help='serve with flare run --prefill-srpt (opt-in shortest-remaining prefill ordering)')
    parser.add_argument('--real-client', type=Path, help='external agent replay client; not bundled')
    parser.add_argument("--port", type=int, default=8185)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--duration", type=float, default=1200)
    parser.add_argument("--ready-timeout", type=float, default=900)
    parser.add_argument("--dry-run", action="store_true", help="print exact commands; no server, GPU or files")
    parser.add_argument("--tag", default="", help="artifact name prefix (for example '32k-' for the S1L package)")
    parser.add_argument("--prefix-cache", default="64G", help="host prefix cache (--prefix-cache-memory); 24G fallback when cuMemCreate fails")
    parser.add_argument("--qualification", type=Path, help="prepare report of the package (default: <package dir>/prepare.json)")
    # Native client, prompt file/ids and concurrency order are parameters (defaults = the short-prompt protocol)
    parser.add_argument("--client", type=Path, default=Path(__file__).resolve().with_name("benchmark_mtp.py"))
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--prompt-ids", nargs="+", default=["code-00", "code-01", "code-02", "code-03"])
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4])
    args = parser.parse_args()
    global TAG, DEST, GPU
    TAG, DEST, GPU = args.tag, args.output_dir, args.gpu_uuid
    if not DEST.is_dir():
        parser.error('--output-dir must exist')
    if args.mode == 'real' and not (args.real_client and args.traces_dir):
        parser.error('--mode real requires external --real-client and --traces-dir')
    args.depths = args.depths or ([3, 4] if args.mode == "native" else [3])
    if (len(set(args.depths)) != len(args.depths) or args.tokens < 2 or args.repetitions < 1
            or args.duration <= 0 or args.ready_timeout <= 0 or not 1 <= args.port <= 65535):
        parser.error("invalid repeated depth, capacity, duration or port")
    if args.dry_run:
        print(json.dumps([{ "depth": depth, "server": server_command(args, depth),
            "client": client_command(args, depth, artifacts(args.mode, depth), "<native model ID>"),
            "artifacts": {key: str(path) for key, path in artifacts(args.mode, depth).items()}}
            for depth in args.depths], indent=2))
        return
    supervisor = require_supervisor()
    for path in (args.package, args.executable):
        if not path.is_file():
            raise FileNotFoundError(path)
    qualification = json.loads((args.qualification or (args.package.parent / 'prepare.json')).read_text())
    require_qualification(qualification, args.package)
    for depth in args.depths:
        for path in artifacts(args.mode, depth).values():
            if path.exists():
                raise FileExistsError(f"preserve existing benchmark artifact: {path}")
    help_text = query(str(args.executable), "run", "--help")
    missing = [flag for flag in server_command(args, args.depths[0])
               if flag.startswith("--") and flag not in help_text]
    if missing:
        raise RuntimeError(f"serving executable lacks required flags: {missing}")

    def interrupted(signum, frame):
        # The outer observer may repeat SIGTERM while graceful shutdown drains.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise KeyboardInterrupt(signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    for depth in args.depths:
        run_depth(args, depth, supervisor, help_text)


if __name__ == "__main__":
    main()
