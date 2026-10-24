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

The numbers come from the fork's lab harness. Its portable helpers now ship under `tools/`; the original orchestration and
private trace inputs do not. The workflow consists of:

- a chain script that runs the gates in [VALIDATE.md](VALIDATE.md) in order;
- a server driver that starts `flare run`, waits for `/health`, snapshots
  `/metrics` before and after, and checks clean shutdown;
- two streaming clients, one for the short prompts and one for the long prompts.

The checked-in clients preserve the requests and statistics. New public fixtures
are different workloads, so historical scores below are context, not expected results.

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

Use [tools/native.py](../../tools/native.py) to start and cleanly stop a server,
collect telemetry and before/after native metrics, and run the original streaming
clients. It requires all preparation qualification results before measurement.
`--dry-run` prints the commands without loading a model or touching a GPU.
Allocate a V100 externally first; the tool checks the selected UUID and SM70
capability, not your site's lease authority. Signing keys and trust files remain
external (see [EXPORT.md](EXPORT.md#signing-keys-and-the-trust-store)).

Run from the checkout root after the fully gated export. All output directories
must already exist; existing benchmark files are refused. The prefix-cache and
GPU memory limits below match the historical serving configuration.

```sh
CUDA_VISIBLE_DEVICES="$V100_GPU_UUID" .venv/bin/python tools/native.py \
  --mode native --depths 4 --gpu-uuid "$V100_GPU_UUID" \
  --executable "$PWD/flare/target/release/flare" \
  --package /data/flare-run/model.safetensors --qualification /data/flare-run/prepare.json \
  --publisher-keys /data/flare-keys/development-publisher.json \
  --trust-store /data/flare-keys/trust.toml --memory-limit 31G --prefix-cache 8G \
  --prompts /data/flare-run/prompts.jsonl --output-dir /data/flare-bench \
  --prompt-ids code-00 code-01 code-02 code-03 --concurrencies 1 4 \
  --tokens 128 --repetitions 3 --ready-timeout 1800
```

This is **new public input data**, not the unpublished historical prompt set.
There are 24 requests total: 12 at concurrency one and 12 at concurrency four.
The client requires the exact output-token budget, complete SSE termination and
consistent native token observations. Partial streams fail the measurement.

The release-binary benchmark chain ran `tools/native.py` at SHA-256 `32f1d6de`,
before its GPU labels were renamed (behaviour identical), with
`benchmark_prefill.py` at `c6478679`, `benchmark_mtp.py` at `a4656afb` and
`benchmark_long.py` at `0f3ee504`. Later label-only edits change the current
hashes of `native.py` and `benchmark_long.py`; match these prefixes against the
chain manifest.

## Reproduce each row

To measure a long-context workload, first generate deterministic example modules
rendered to approximately 28,000 tokens with the checkpoint's actual chat template:

```sh
.venv/bin/python tools/generate_prompts.py --mode long --target-tokens 28000 \
  --checkpoint /data/models/Qwen3.8-27B-NVFP4 --output /data/flare-run/long-prompts.jsonl
```

Run the same `native.py` command with a fresh output directory and these changes:

```sh
--client "$PWD/tools/benchmark_long.py" --prompts /data/flare-run/long-prompts.jsonl \
--prompt-ids long-00 long-01 long-02 long-03 --concurrencies 4 1 --tokens 512
```

The generator records rendered prompt lengths. These four inputs are not the
historical 16.6–18.6k or 22.4–30.2k sets. Report the measured lengths and establish
a new result table. The long client requests greedy sampling, but the server's
model-author clamps still apply; inspect the package and native configuration
before describing actual sampling as greedy. Each driver invocation starts a
fresh server so the previous leg's prefix cache is not retained.

Read each concurrency's `summary` entry in the request JSON:

| Quantity | Field |
| --- | --- |
| Median request decode rate after first observed tokens | `median_per_agent_decode_tokens_per_second` |
| Output throughput including prefill and HTTP | `aggregate_output_tokens_per_second` |
| Median for each prompt | `per_agent[].median_decode_tokens_per_second` |
| Time to first token | `first_token_seconds.p50` and `.p95` |
| Request completeness | `successful_requests` and `failed_requests` |

A concurrency-four median is per request, not the sum of four rates. Keep
aggregate wall-time throughput separate from decode-only rates. Exact tau is
measured by `measure_mtp.py` during preparation; the HTTP client deliberately
reports tau as unavailable rather than inferring it from graph counters.

### GPU step per round

The 1 x 5 round time (49.5-49.6 ms) is Flare's own CUDA-event time per
speculative graph at batch 1 and width 5. It covers drafting, verification,
commit and draft absorption. Read it from the Prometheus counters before and
after a run at concurrency 1:

```sh
curl -s http://127.0.0.1:8990/metrics > before.prom
.venv/bin/python tools/benchmark_long.py --base-url http://127.0.0.1:8990 --depth 4 --prompts /data/flare-run/long-prompts.jsonl --prompt-ids long-00 long-01 long-02 long-03 --concurrencies 1 --repetitions 3 --output-tokens 512 --output /data/flare-bench/gpu-round-requests.json
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

## Fixed-length prefill and TTFT

[tools/benchmark_prefill.py](../../tools/benchmark_prefill.py) supplies the new
fixed-length workload required before publishing a prefill comparison. It is a
client for an already running, exclusively allocated native server; it never
starts one. Run the release binary with the package/flags above, wait for health,
and exclude every other client for the entire benchmark. Use the exact served
checkpoint's `tokenizer.json` and the model ID returned by `/v1/models`.

```sh
.venv/bin/python tools/benchmark_prefill.py \
  --base-url http://127.0.0.1:8990 --model YOUR_NATIVE_MODEL_ID \
  --tokenizer /data/models/Qwen3.8-27B-NVFP4/tokenizer.json \
  --exclusive-endpoint --run-id public-prefill-run-001 \
  --warmups 1 --repetitions 3 --output-tokens 32 \
  --output /data/flare-bench/prefill.json
```

The complete matrix is **2,048, 8,192, 22,528 and 30,720 prompt tokens**, each at
**concurrency 1 and 4**. Raw `/v1/completions` token-ID inputs avoid chat-template
length ambiguity, but every final server `usage.prompt_tokens` must still match
the requested length exactly. Completions requests carry no token observation
extensions (`return_token_ids`, `return_token_timestamps`): the server accepts
those only on `/v1/chat/completions` and rejects them on `/v1/completions` with
HTTP 400. The client requests continuous usage statistics instead and derives
first-output time and output-token counts from the SSE stream itself. These
are deterministic, newly authored generic coding inputs, not the historical
prompts. A new `--run-id` changes the early prompt tokens for every
warmup/repetition/slot; reuse of a namespace is permitted only if cache checks
still prove zero reuse. The tool does not flush or change server caches.

There is one separately recorded warmup batch before three measured batches for
each matrix cell: 20 warmup requests and 60 measured requests. Warmups never enter
the summary. Each request must complete its exact output-token budget and SSE
termination. Cache credit in usage fails the request. Before/after native metrics
must show idle endpoints, exact completed/prefill/generated counts, no failures
and zero reused tokens; missing metrics, resets or extra traffic fail the batch.
The report retains raw token IDs, SSE events, per-request timing and both metric
snapshots, including failed batches. A failed report is not a comparison result.

Keep these fields in separate columns:

- `transport_ttft_seconds.p50` / `.p95`: client POST start to the first SSE chunk
  carrying output (non-empty `choices[0].text`, or continuous usage showing at
  least one completion token); includes queueing, transport and first sampling.
- `effective_prefill_tokens_per_second_median`: native prefill-token counter
  delta divided by the batch's first-token wall window. This includes queueing,
  first-token generation and transport. It is an effective wall-time rate,
  **not pure GPU prefill throughput**.
- `median_request_decode_tokens_per_second`: tokens after each request's first
  observation divided by its subsequent decode window, counted from per-chunk
  usage. A request whose whole output arrives in one chunk has no decode window;
  it is counted in `decode_rate_unavailable_requests` instead.

The current release exposes prefill-token counters but no isolated prefill GPU
execution-duration counter. Therefore `pure_gpu_prefill_tokens_per_second` stays
`null`; do not substitute prompt tokens / TTFT and label it a CUDA-only rate.
Record the exact binary/package identities and server flags beside the table.
Only the CPU stub-server tests have passed for this added client. Real V100 results
and all prefill comparison claims remain pending until the complete matrix passes.

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
