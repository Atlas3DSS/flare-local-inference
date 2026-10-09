"""Qualify a startup-selected speculative policy through native HTTP serving."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import sys
import time
from urllib.request import Request, urlopen

from qualify.provenance import file_record, utc, write_json


def http(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=180) as response:
        value = response.read().decode()
    return value if path == "/metrics" else json.loads(value)


def metrics(base):
    raw = http(base, "/metrics")
    values = {}
    for line in raw.splitlines():
        if line and not line.startswith("#"):
            key, value = line.rsplit(" ", 1)
            values[key] = float(value)
    return raw, values


def idle(base):
    deadline = time.monotonic() + 30
    while True:
        raw, values = metrics(base)
        if (values.get("sparkgap_active_requests", -1) == 0
                and values.get("sparkgap_pending_requests", -1) == 0
                and values.get("sparkgap_free_request_slots", -1) == 4):
            return raw, values
        if time.monotonic() >= deadline:
            raise TimeoutError("native server did not reclaim all requests within 30 seconds")
        time.sleep(.01)


def literal_request(model, text, *, seed=42, budget=256):
    return {"model": model, "messages": [
        {"role": "system", "content": "Copy the supplied text exactly."},
        {"role": "user", "content": text}],
        "chat_template_kwargs": {"enable_thinking": False},
        "grammar": "start: " + json.dumps(text),
        "temperature": 1.0, "top_k": 20, "top_p": .95,
        "seed": seed, "max_tokens": budget,
        "return_token_ids": True, "return_token_timestamps": True}


def run(args, model, server):
    base = "http://" + server.address
    identity = http(base, "/v1/models")["data"][0]["id"]
    libraries = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                        if "/flashinfer/" in line and line.split()[-1].endswith(".so")})
    receipt = {"started_at": utc(), "source": args.source,
        "script": file_record(Path(__file__)), "memory": model.memory_report,
        "loaded_sampling_libraries": [file_record(path) for path in libraries],
        "server_config": model.server_config, "requests": [], "checks": [],
        "scope": "Exact-checkpoint HTTP correctness with forced literal outputs; timings are diagnostic, not representative serving performance."}
    destination = args.output / "service.json"
    text = " ".join(["alpha beta gamma delta epsilon zeta eta theta"] * 8)
    def check(name, passed, **details):
        receipt["checks"].append(dict(name=name, passed=bool(passed), **details))

    def complete(body):
        started = time.monotonic_ns()
        response = http(base, "/v1/chat/completions", body)
        return {"request": body, "response": response,
                "elapsed_ns": time.monotonic_ns() - started}

    try:
        before_raw, before = idle(base)
        (args.output / "before.metrics").write_text(before_raw)
        for concurrency in (1, 2, 4):
            requests = [literal_request(identity, text, seed=42 + lane) for lane in range(concurrency)]
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                results = list(executor.map(complete, requests))
            receipt["requests"].append({"name": "literal", "concurrency": concurrency, "results": results})
            for lane, result in enumerate(results):
                choice = result["response"]["choices"][0]
                actual = choice["message"]["content"]
                name = f"concurrency-{concurrency}/lane-{lane}"
                check(name + "/literal", actual == text, actual=actual, expected=text)
                check(name + "/stop", choice["finish_reason"] == "stop", actual=choice["finish_reason"])
            write_json(destination, receipt)
        for budget in (1, 2, 3, 7, 8, 9, 15):
            result = complete(literal_request(identity, text, budget=budget))
            receipt["requests"].append({"name": f"budget-{budget}", "concurrency": 1, "results": [result]})
            response = result["response"]
            choice = response["choices"][0]
            check(f"budget-{budget}/count", response["usage"]["completion_tokens"] == budget,
                  actual=response["usage"]["completion_tokens"], expected=budget)
            check(f"budget-{budget}/length", choice["finish_reason"] == "length", actual=choice["finish_reason"])
            check(f"budget-{budget}/prefix", text.startswith(choice["message"]["content"]))
            write_json(destination, receipt)
        after_raw, after = idle(base)
        (args.output / "after.metrics").write_text(after_raw)
        receipt["metrics_delta"] = {key: value - before.get(key, 0) for key, value in after.items()
                                    if "_total" in key}
        if args.source == "copy" and not args.no_copy and args.fixed_width != 1:
            accepted_copy = sum(value for key, value in receipt["metrics_delta"].items()
                                if key.startswith("sparkgap_speculative_drafts_total{")
                                and 'source="copy"' in key and 'outcome="accepted"' in key)
            check("grammar-accepted-copy", accepted_copy > 0, accepted_drafts=accepted_copy)
        receipt["passed"] = all(check["passed"] for check in receipt["checks"])
        if not receipt["passed"]:
            raise AssertionError("speculative service checks failed; inspect service.json")
    except BaseException as error:
        receipt.update(passed=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["finished_at"] = utc()
        write_json(destination, receipt)


def main():
    import flashinfer
    import sparkgap_native
    from sparkgap.recipes.qwen38.bootstrap import PreparationConfig, prepare_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--source", choices=("copy", "mtp"), default="copy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bind", default="127.0.0.1:8000")
    parser.add_argument("--fixed-width", type=int, choices=range(1, 9))
    parser.add_argument("--no-copy", action="store_true")
    parser.add_argument("--keep-serving", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "service.json").exists():
        raise FileExistsError(args.output / "service.json")
    root = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__)]
    source_paths += sorted((root / "sparkgap").rglob("*.py"))
    source_paths += sorted((root / "native" / "src").rglob("*.rs"))
    source_paths += sorted(Path(sparkgap_native.__file__).parent.glob("*.so"))
    source_paths += sorted((Path(flashinfer.__file__).parent / "experimental" / "b12x").rglob("*.py"))
    source_paths += [Path(flashinfer.__file__).parent / "sampling.py",
                     Path(flashinfer.__file__).parent.parent / "include/flashinfer/sampling.cuh"]
    write_json(args.output / "provenance.json", {
        "started_at": utc(), "argv": sys.argv, "python": sys.executable,
        "pid": os.getpid(), "cwd": str(Path.cwd()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_home": os.environ.get("CUDA_HOME"),
        "snapshot": str(args.snapshot.resolve()),
        "sources": [file_record(path) for path in source_paths]})
    config = PreparationConfig(mtp=True, speculative_proposal_source=args.source,
        speculative_fixed_width=args.fixed_width, speculative_copy=not args.no_copy)
    with prepare_model(args.snapshot, config, progress=lambda text: print(text, flush=True)) as model:
        write_json(args.output / "memory.json", model.memory_report)
        server = model.start_server(bind=args.bind)
        print(f"Listening on {server.address}", flush=True)
        run(args, model, server)
        if args.keep_serving:
            def stop(signum, frame):
                raise KeyboardInterrupt
            previous = signal.signal(signal.SIGTERM, stop)
            try:
                while server.wait(1000):
                    pass
                raise RuntimeError("native server stopped unexpectedly")
            except KeyboardInterrupt:
                pass
            finally:
                signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
