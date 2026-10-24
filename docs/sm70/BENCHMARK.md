# Reproduce the SM70 numbers

This page explains how every number in [README-SM70.md](../../README-SM70.md)
and [BENCHMARKS-SM70.md](../../BENCHMARKS-SM70.md) was measured, what was
actually run, and how to run the same measurement on your own card.

## Two kinds of number

The fork publishes two kinds of result and keeps them in separate tables:

- **End to end.** Generated tokens over wall time for a whole workload,
  including prefill, queueing, scheduling and HTTP. On real coding-agent traces
  this is what a user of one card gets. Real traces are prefill-bound on one
  V100: an analysis put prefill at about 85 % of busy time at real contexts
  (STATUS.md line 37). So end-to-end numbers are much lower than decode rates.
- **Decode-only.** A request's decode rate after its first token, or Flare's
  own GPU time per speculative round. These isolate the decode kernels that this
  port changes, and they compare with decode figures published for other
  engines and GPUs.

Both are published because each misleads alone. A decode rate says nothing
about a prefill-bound agent workload; an end-to-end trace number hides whether
the kernels improved. Never put a number from one table next to a number from
the other as if they measured the same thing.

## What was run

The numbers come from the fork's lab harness. It is not in this repository. It
consists of:

- a chain script that runs the gates in [VALIDATE.md](VALIDATE.md) in order;
- a server driver that starts `flare run`, waits for `/health`, snapshots
  `/metrics` before and after, and checks clean shutdown;
- two streaming clients, one for the short prompts and one for the long prompts.

The client below reproduces the clients' requests and statistics.

**Package.** `model-32k-tonight-a`, prepared from the release tree `0e4c81f`:
`--context 32768`, 4 requests, draft depth 4 (widths 1 to 5), 128-token
prefill chunks (STATUS.md line 62).

**Server flags**, identical for every serving leg:

```sh
flare run model-32k-tonight-a.safetensors --publisher-keys development-publisher.json --trust-store trust.toml \
  --verify-weights --gpus 0 --prefix-cache-memory 8G --embedding-table-storage ram --memory-limit 31G \
  --kv-cache-limit concurrency --max-concurrency 4 --mtp-draft-tokens 4 --no-mtp-adaptive-verification \
  --host 127.0.0.1 --port PORT
```

- `--embedding-table-storage ram` has no effect here; this package has no
  embedding tables.
- `--kv-cache-limit concurrency` caps the KV pool at one prepared context per
  request, plus each request's private draft page: 4 x 32,768 + 4 x 128 =
  131,584 tokens. The release runs hit that cap: the startup line reads
  `KV capped at 4 contexts` (STATUS.md line 63), and the pool is 131,584 tokens
  (lines 65, 66). The baseline before replay-commit ran with the same cap but
  stopped at 98,304 tokens, `limited by memory` in its server log. So +33,280 tokens is what
  replay-commit freed **up to the cap**. The fork's plan estimated about
  +39.5k to +40.1k tokens without a cap. The pool at the default
  `--kv-cache-limit memory` was not measured.
- `--memory-limit 31G` comes from the chain script, which sets it for every leg.
  Lines 64 and 66 of the run log do not restate it.

**Serving binary.** The run log identifies the `flare` binary only by its
SHA-256 prefix `2c792073` (STATUS.md line 63). The same binary served the
baseline runs at 10:20 on 2026-10-21 (lines 2, 3). That is before any of this
branch's ten Helios and Flare commits (`07ba807` to `718a187`), the first of
which was committed at 10:24 that day. So the published serving numbers were
measured with packages from this tree, served by a binary without this branch's
scheduler changes or hang fix. A binary built from this tree has not been
benchmarked in the cited sources.

**Hardware.** One Tesla V100 PCIe 32 GB per run, on a shared host. The harness
refused to start if any other process was on the card.

**Sampling.** The short-prompt client requests temperature 0.7, top-p 0.95,
top-k 20, seeded. The long-prompt client requests greedy decoding (temperature 0,
top-k 1). The server clamps requests to the package's model-author ranges, and
the binary had no `--unclamped-sampling`. So the long runs sampled at
temperature 0.7, top-p 0.95 and top-k 20 as well (STATUS.md line 19 on the
clamp). Every request sets `ignore_eos`, so it generates exactly its token
budget. Seeds are `7035 + repetition x 4 + index`.

## The client

Paste this block at the repository root. It writes `bench_sm70.py`, which needs
only the Python standard library.

```sh
cat > bench_sm70.py <<'EOF'
#!/usr/bin/env python3
"""Fixed-prompt Flare benchmark: per-request decode rate, TTFT and aggregate output (standard library only).

Statistics match the fork's benchmark clients (docs/sm70/BENCHMARK.md). For each request:
decode rate = output tokens after the first observation / (last observation - first observation),
where an observation is a streamed chunk whose cumulative completion_tokens grew; TTFT = time to the
first observation. For each concurrency: aggregate output = output tokens / total batch wall time,
which includes prefill and HTTP. Prompts run in groups of `concurrency`, released together.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import threading
import time
import urllib.request

SAMPLING = {"short": {"temperature": 0.7, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
            "greedy": {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "min_p": 0.0}}


def request(base, model, prompt, tokens, seed, sampling, barrier, timeout):
    body = {"model": model, "messages": prompt["messages"], "max_tokens": tokens, "stream": True,
            "ignore_eos": True, "seed": seed, **SAMPLING[sampling], "repetition_penalty": 1.0,
            "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "chat_template_kwargs": {"enable_thinking": bool(prompt.get("thinking", False))},
            "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
    if prompt.get("tools"):
        body.update(tools=prompt["tools"], tool_choice="auto")
    http = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"})
    result = {"id": prompt["id"], "seed": seed, "ok": False}
    barrier.wait()
    start = time.perf_counter()
    count = first_count = 0
    first = last = None
    done = False
    try:
        with urllib.request.urlopen(http, timeout=timeout) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    done = True
                    break
                event = json.loads(payload)
                if "error" in event:
                    raise RuntimeError(event["error"])
                seen = (event.get("usage") or {}).get("completion_tokens", 0)
                if seen > count:
                    now = time.perf_counter() - start
                    if first is None:
                        first, first_count = now, seen
                    last, count = now, seen
        if not done or count != tokens or first is None or last <= first:
            raise RuntimeError(f"incomplete stream: {count} of {tokens} tokens, [DONE] {done}")
        result.update(ok=True, ttft=first, decode=(count - first_count) / (last - first))
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result.update(tokens=count, elapsed=time.perf_counter() - start)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8300")
    parser.add_argument("--prompts", type=Path, required=True, help="JSONL: one {id, messages[, thinking, tools]} per line")
    parser.add_argument("--prompt-ids", nargs="+", help="run these ids, in this order (default: all)")
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--sampling", choices=sorted(SAMPLING), default="short")
    parser.add_argument("--seed", type=int, default=7035)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--output", type=Path, help="write every request and the summary as JSON")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    if args.prompt_ids:
        by_id = {row["id"]: row for row in rows}
        rows = [by_id[name] for name in args.prompt_ids]
    base = args.base_url.rstrip("/")
    with urllib.request.urlopen(base + "/v1/models", timeout=30) as response:
        model = json.load(response)["data"][0]["id"]
    report = {"model": model, "args": {k: str(v) for k, v in vars(args).items()}, "requests": [], "summary": []}
    for concurrency in args.concurrencies:
        done, wall = [], 0.0
        for repetition in range(args.repetitions):
            for offset in range(0, len(rows), concurrency):
                group = rows[offset:offset + concurrency]
                barrier = threading.Barrier(len(group) + 1)
                with ThreadPoolExecutor(len(group)) as pool:
                    futures = [pool.submit(request, base, model, prompt, args.output_tokens,
                                           args.seed + repetition * len(rows) + offset + index,
                                           args.sampling, barrier, args.timeout)
                               for index, prompt in enumerate(group)]
                    started = time.perf_counter()
                    barrier.wait()
                    results = [future.result() for future in futures]
                    wall += time.perf_counter() - started
                for result in results:
                    result.update(concurrency=concurrency, repetition=repetition)
                    print(json.dumps({k: result.get(k) for k in ("concurrency", "repetition", "id", "ok", "tokens",
                                                                    "ttft", "decode", "error") if k in result}), flush=True)
                done.extend(results)
        report["requests"].extend(done)
        ok = [r for r in done if r["ok"]]
        summary = {"concurrency": concurrency, "requests_ok": f"{len(ok)}/{len(done)}",
                   "median_decode_tok_s": statistics.median(r["decode"] for r in ok) if ok else None,
                   "per_prompt_median_decode_tok_s": {row["id"]: statistics.median(
                       [r["decode"] for r in ok if r["id"] == row["id"]] or [float("nan")]) for row in rows},
                   "ttft_p50_s": statistics.median(r["ttft"] for r in ok) if ok else None,
                   "aggregate_output_tok_s": sum(r["tokens"] for r in ok) / wall if wall else None}
        report["summary"].append(summary)
        print(json.dumps(summary), flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if all(r["ok"] for r in report["requests"]) else 1)


if __name__ == "__main__":
    main()
EOF
```

The prompt file is JSON Lines, one prompt per line:

```json
{"id": "code-00", "thinking": false, "messages": [{"role": "system", "content": "You are a senior software engineer working as a coding agent. Be concise and output working code."}, {"role": "user", "content": "..."}]}
```

**The fork's prompt files are not in this repository.** They are:

- Short prompts `code-00` to `code-03`: a coding-agent system prompt and a user
  message that contains one source file of this repository with a coding task.
- Prompts `l24-00` to `l24-03`: four prompts of 16.6k to 18.6k tokens.
- Prompts `long-00` to `long-03`: 30,194, 22,388, 29,487 and 27,953 prompt
  tokens, 110,022 in total (STATUS.md line 53).

With your own prompts of similar lengths, expect similar decode rates but not
the same numbers. Decode rates depend on how many draft tokens the model
accepts, which depends on the text.

## Reproduce each row

Start the server with the flags under [What was run](#what-was-run), on port
8990, and wait for `/health`. Then:

| Row | Command | Report |
| --- | --- | --- |
| Short prompts, median decode, 1 and 4 agents (90.4 and 70.7 tok/s); aggregate output (238.0 tok/s) | `.venv/bin/python bench_sm70.py --base-url http://127.0.0.1:8990 --prompts short.jsonl --concurrencies 1 4 --repetitions 3 --output-tokens 128 --sampling short` | `median_decode_tok_s` at concurrency 1 and 4; `aggregate_output_tok_s` at 4; requests 24/24 |
| 16.6k-18.6k prompts, median decode, 1 and 4 agents (82.6 and 42.6 tok/s) | `.venv/bin/python bench_sm70.py --base-url http://127.0.0.1:8990 --prompts long24.jsonl --concurrencies 4 1 --repetitions 3 --output-tokens 512 --sampling greedy` | `median_decode_tok_s`, per-prompt medians, and the GPU step per round below |
| 22.4k-30.2k prompts, median decode, 1 and 4 agents (82.2 and 37.2 tok/s); TTFT p50 at 4 agents (90.7 s); residency | `.venv/bin/python bench_sm70.py --base-url http://127.0.0.1:8990 --prompts long26.jsonl --concurrencies 4 1 --repetitions 3 --output-tokens 512 --sampling greedy` | `median_decode_tok_s`, per-prompt medians, `ttft_p50_s` at 4; `Allocate cache` KV tokens; zero restores |
| KV pool (131,584 tokens) | the server's `Allocate cache` line | KV tokens and the limiting factor |

Restart the server between legs, as the harness did, so that one leg's prefix
cache does not serve the next.

**Medians.** `median_decode_tok_s` is the median over all successful requests
at that concurrency, 12 per concurrency. "Per agent" at 4 agents means this
median over the requests that ran four at a time. It is not the sum of four
rates.

### GPU step per round

The 1 x 5 round time (49.5-49.6 ms) is Flare's own CUDA-event time per
speculative graph at batch 1 and width 5. It covers drafting, verification,
commit and draft absorption. Read it from the Prometheus counters before and
after a run at concurrency 1:

```sh
curl -s http://127.0.0.1:8990/metrics > before.prom
.venv/bin/python bench_sm70.py --base-url http://127.0.0.1:8990 --prompts long24.jsonl --concurrencies 1 --repetitions 3 --output-tokens 512 --sampling greedy
curl -s http://127.0.0.1:8990/metrics > after.prom
python3 - before.prom after.prom <<'PY'
import re, sys
pattern = re.compile(r'^flare_speculative_(rounds|gpu_nanoseconds)_total\{batch="(\d+)",width="(\d+)"\} (\d+)$', re.M)
def read(path):
    return {(kind, int(b), int(w)): int(v) for kind, b, w, v in pattern.findall(open(path).read())}
before, after = read(sys.argv[1]), read(sys.argv[2])
for (kind, b, w), value in sorted(after.items()):
    rounds = value - before.get((kind, b, w), 0)
    if kind == "rounds" and rounds:
        gpu = after.get(("gpu_nanoseconds", b, w), 0) - before.get(("gpu_nanoseconds", b, w), 0)
        print(f"batch {b} width {w}: {rounds} rounds, {gpu / rounds / 1e6:.2f} ms GPU per round")
PY
```

Report the `batch 1 width 5` line. For the 4 x 5 round, take the
`batch 4 width 5` line from a concurrency-4 run, and only if all four agents
decoded together for most of the run. On the release package they never did,
so no 4 x 5 figure exists for it ([VALIDATE.md](VALIDATE.md#gates-3-to-5-native-serving)).

### Residency

At 4 agents with the 22.4k-30.2k prompts, all four prompts must stay resident:
no eviction or restore. Check that the restore graph counter did not move:

```sh
grep 'flare_graph_launches_total{kind="restore"}' before.prom after.prom
```

Also check the server log for eviction or restore lines. The release run had
none (STATUS.md line 65). 110,022 prompt tokens alone exceed the 98,304-token
pool of the package before replay-commit, so that package could not hold them.

### End to end on real traces

The 47.23 tok/s figure came from the fork's replay of real coding-agent traces,
on a different package (`best24`), not the release (STATUS.md line 68). Neither
the traces nor the replay tool is in this repository; the traces are not public. The run log gives no memory limit or concurrency for that
figure. To measure end to end on your own traces, replay recorded agent
conversations against the server with their real tool-call delays. Report
generated tokens over wall time, with the number of agents, the memory limit,
the package and the date.

## Reporting your numbers

For each number, give the GPU model and memory, `--memory-limit`, concurrency,
`--mtp-draft-tokens`, the package (tree commit, context, depth), the `flare`
binary's commit, the date, and whether it is end to end or decode-only. Use the
same table columns as [BENCHMARKS-SM70.md](../../BENCHMARKS-SM70.md).

## Sources

Fragments of the fork's run log (`STATUS.md`) cited in these documents. Times
are local, dates 2026-10-22 unless noted. Internal host and card names are
replaced with `[card]`, and other internal labels with a bracketed description.

- Lines 2-3 (2026-10-21, 10:20 and 10:28): "baseline [...] runs with binary
  2c79207"; "baseline (binary 2c79207, package mtp-tc = BF16 draft head, depth
  3, [card], no trace)".
- Line 8 (2026-10-21, 10:41): "build OK on [workstation] (binary c802060e,
  rustc 1.97.1)".
- Line 10 (2026-10-21, 10:52): "gain <=0.1 %, launch was never on the critical
  path".
- Line 19 (2026-10-21, 11:24): "package sampling ranges clamp temperature to
  [0.7,1.0], top_p [0.8,0.95] and pin top_k=20".
- Line 21 (2026-10-21, 11:28): "fixed-width-4 [agent-trace benchmark] ([trace set], old
  binary): 44.9 emitted tok/GPU-s vs adaptive 39.8 (+13 %), and exactness 4/4
  EQUAL (adaptive: 2/2 differ)".
- Line 25 (2026-10-21, 12:28): "checkpoint can be evicted -> generated tokens
  re-prefilled with prefill arithmetic (!= verification) -> persistent logit
  change".
- Line 27 (2026-10-21, 12:53): "emitted tok per GPU-s b4 53.5 vs 47.1 old vs
  52.0 fixed-w4 [...] sampled exactness 1/2 (adaptive) -> fixed-width default
  for exactness".
- Line 30 (2026-10-21, 13:24): "agent+pipe rerun 53.6 tok/GPU-s at b4 [...]
  balanced+new estimator [...] 50.8 tok/GPU-s [...] recommend
  --scheduler-policy agent for agents".
- Line 32 (2026-10-21, 15:58): "[agent-trace benchmark] with the NEW DEFAULT ([...] no width
  flag -> fixed width 4) [...] at preemption free pages 0 -> terminal snapshot
  cannot be taken [...] pinning is infeasible under real page pressure".
- Line 34 (2026-10-21, 22:48): "21 execution rows x 166.1 MiB = 3.41 GiB (1
  null + 4 agents x 5 verify positions), replay-commit -> 5 rows [...] recurrence
  5.51 ms + commit copy 1.29 ms per 4x5 round today".
- Line 37 (09:38): "one card is prefill-bound on real traces (85 % of busy
  time at real contexts)".
- Line 38 (11:38): "retained width-4 round with the destination row logically
  excluded must be rejected by the batch validator (status 3/code 32) with
  destination + null row bytes unchanged".
- Line 39 (12:4x): "the attention tuning test failure is cache-order dependent
  (fails on the untouched base with the shared b12x selection cache, passes with
  a fresh one)".
- Line 43 (14:46): "replay-commit TESTS LEG PASSED [...] group A exit 0 = every
  test passed [...] group B exit 1 with only its 2 known failures".
- Lines 44-45 (15:0x): "delta replay-r5-gdn-capacity logged without a
  reviewer"; "the replay-r5 delta record is NOT logged yet (the ledger requires
  a reviewer per record)".
- Line 46 (14:52): "tests-leg RERUN [...] PASSED [...]: A exit 0 [...], B only
  its 2 known failures".
- Line 47 (15:1x): "25 PASS / 22 ERROR, every error a B4 case with [...] code
  72 [...] Diagnosis: HARNESS, not integration".
- Line 49 (15:34): "PREPARE LEG PASSED [...] 25.6 min, peak alloc 20.8 GiB
  [...] `Allocate cache. 9.4 GiB · 131584 KV tokens · 32 shared snapshot slots`
  at --memory-limit 31G / 4 agents = +33,280 lossless KV tokens vs the 22:20
  baseline's 98,304 (+33.9 %)".
- Line 51 (16:1x): "90 package_protection + 10 package_snapshot errors".
- Line 52 (15:56): "d4 rounds 1x5 55.1 / 4x5 75.7 ms ([...] baseline 76.6
  ms [...])".
- Line 53 (16:21): "long-00 30,194 / long-01 22,388 / long-02 29,487 / long-03
  27,953 prompt tokens = 110,022 total".
- Line 54 (16:42): "REFERENCE CHANGE [...] Exact identity is still required
  (gate not relaxed; [...])".
- Line 55 (17:05): "PREPARE LEG PASSED [...] 23.9 min".
- Line 58 (17:3x): "all_active_steps = 0 (the four agents' decode windows never
  fully overlap under chunked prefill, so no 4x5 period from this run [...])
  [...] the slow agent rotates with prefill interleaving, not a fixed slot".
- Line 59 (17:45): "+33,280 lossless KV tokens (131,584; 16 x 166 MiB of
  candidate rows returned to the cache) [...] one validation-only change
  (replay-r5.diff) unreviewed by choice".
- Line 60 (18:09): "chain launch #9 from PREPARE".
- Line 61 (18:3x): "16 failed / 1544 passed / 860 skipped / 100 errors vs the
  untouched 718a187 baseline 16 / 1536 / 855 / 100 [...] rerun in isolation 3/3
  PASS".
- Line 62 (19:04): "TONIGHT-A PREPARE LEG PASSED [...] 54.6 min incl. CPU
  contention [...] strict 47/47 incl. every forced_prefix_*, long-sequence
  invariance 28/28, commit growth 4/4 (incl. the negative capacity round),
  consumer identity, components complete, accuracy/KL complete, tau/coding-token
  goldens qualification_passed with prior identity [...] Memory report:
  state_rows 5, 1 state column, replay operands 30.4 MiB, continuation 1,045
  MiB".
- Line 63 (19:19): "native server readiness timeout [...] same binary (sha
  2c792073) [...] Verify stored data 735.1 s (normally ~114) + Load weights
  158.7 s because three ~20 GB packages were loading from the spinning [disk] at
  once [...] (21.15 GiB loaded, 9.76 GiB cache budget, KV capped at 4 contexts)
  [...] replay_chain.sh passes --ready-timeout ${READY_TIMEOUT:-1800} [...] The
  package manifest (958 KB) carries no weight paths".
- Line 64 (19:36): "1x5 GPU step 49.5-49.6 ms per round ([...] 50.4, pre-S1L
  replay 55.1); 1 agent median decode 82.6 tok/s; 4 agents per-agent median
  42.6 tok/s [...] 4-agent reps mean GPU step 58.1 / 58.1 / 49.6 ms, no
  fully-overlapped decode window); KV pool 131,584".
- Line 65 (19:57): "all four resident at 31G (KV 131,584, zero
  evict/restore); 4 agents per-agent median decode 37.2 tok/s (long-00 36.2,
  long-01 13.0 [shortest prompt under the others' prefill], long-02 39.9,
  long-03 53.5), 1 agent 82.2 tok/s [...] TTFT p50 90.7 s at 4 agents".
- Line 66 (20:07): "NATIVE d4 LEG PASSED on rerun (chain #10, READY_TIMEOUT=1800;
  server load 389.7 s verify + 101.8 s weights): 1 agent 90.4 tok/s median
  decode; 4 agents 70.7 tok/s per agent, 238.0 tok/s aggregate output, 24/24
  ok; KV pool 131,584 tokens".
- Line 68 (23:35): "REALITY remains best24 47.23 tok/s end-to-end;
  PUBLIC-COMPARABLE remains native90.4 single /70.7 median per agent
  at4,238.0 aggregate".
