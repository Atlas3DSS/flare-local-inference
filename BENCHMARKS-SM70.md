# SM70 benchmarks (Tesla V100 32 GB)

Measurements for the SM70 port described in [README-SM70.md](README-SM70.md).
End-to-end numbers and decode-only numbers are reported in separate tables. Do
not compare a number from one table with a number from the other.

- **End-to-end** numbers include prefill, scheduling and every request in the
  workload.
- **Decode-only** numbers are per-request decode rates (tokens per second while
  the request is decoding) or GPU step times. They exclude prefill and queueing.

## Release binary d4ffb1fa (built from 71029458; production source unchanged through c6fdc120)

Measured on 2026-10-23 from 21:54 to 22:48 PDT in one benchmark chain on one
Tesla V100 PCIe 32 GB. Receipts are in [docs/sm70/receipts/release-binary/](docs/sm70/receipts/release-binary/); each table names its file.

- **Server binary.** SHA-256
  `d4ffb1fa3453bf37f2017a7b58d09559010881311b2d9afcdfb33b4011caa667`, built from
  `71029458`; production source is unchanged through `c6fdc120`.
- **Package.** Primary SHA-256
  `fd7c24dce58c4897a2ab5299bcd875e3e6ab1e2a1f714e5f09724bc6619a7382`, the release
  export qualified on 2026-10-23
  ([validation record](docs/sm70/ARCHITECTURE.md#export-qualification-of-the-release-package)).
- **Server flags.** `--memory-limit 31G --prefix-cache-memory 8G
  --kv-cache-limit concurrency --max-concurrency 4 --mtp-draft-tokens 4
  --no-mtp-adaptive-verification --scheduler-policy balanced`, the same for all
  three legs. The KV pool is 131,584 tokens, 1,028 pages of 128 tokens
  ([chain/native-chain-manifest.json](docs/sm70/receipts/release-binary/chain/native-chain-manifest.json),
  [short/native-depth4.json](docs/sm70/receipts/release-binary/short/native-depth4.json)).
- **Hardware.** One Tesla V100 PCIe 32 GB, used only by the chain. Other jobs on
  the host could share CPU, memory and PCIe bandwidth
  ([chain/native-chain-run.json](docs/sm70/receipts/release-binary/chain/native-chain-run.json)).
- **Inputs.** Newly authored public prompts, generated as described in
  [BENCHMARK.md](docs/sm70/BENCHMARK.md). They are not the historical prompt
  sets, so these rows start a new baseline and do not compare row for row with
  the historical measurements below.

### Prefill and time to first token (end to end)

Fixed prompt lengths of 2,048, 8,192, 22,528 and 30,720 tokens, each at
concurrency 1 and 4. Each cell has 1 warmup batch and 3 measured repetitions,
with 32 output tokens per request. TTFT runs from client POST start to the first
SSE chunk carrying output. The effective prefill rate is the server's prefill
tokens divided by the batch's first-token wall window, which includes queueing,
transport and first sampling. Pure-GPU prefill duration is `null` by design: the
release exposes no isolated prefill GPU-time counter
([BENCHMARK.md](docs/sm70/BENCHMARK.md#fixed-length-prefill-and-ttft)). Every
request was uncached: no cached tokens in usage, and no reused tokens in the
server counters
([prefill/prefill-summary.json](docs/sm70/receipts/release-binary/prefill/prefill-summary.json)).

| Prompt tokens | Concurrency | TTFT p50 / p95 (s) | Effective prefill tok/s (first-token wall window) | Pure-GPU prefill (null) | Repetitions |
| ---: | ---: | ---: | ---: | --- | ---: |
| 2,048 | 1 | 2.54 / 2.54 | 807 | null | 3 |
| 2,048 | 4 | 9.89 / 10.30 | 796 | null | 3 |
| 8,192 | 1 | 10.43 / 10.44 | 785 | null | 3 |
| 8,192 | 4 | 41.44 / 41.88 | 782 | null | 3 |
| 22,528 | 1 | 30.39 / 30.39 | 741 | null | 3 |
| 22,528 | 4 | 121.11 / 121.67 | 741 | null | 3 |
| 30,720 | 1 | 42.75 / 42.75 | 719 | null | 3 |
| 30,720 | 4 | 170.49 / 170.95 | 719 | null | 3 |

At concurrency 4 the rate is the aggregate over the four requests. Concurrent
prefills do not overlap: prefill graphs run one request at a time, so four
requests prefill no faster in total than one. The server interleaves their
chunks, so all four first tokens arrive within 0.9 s of each other, at about
four times the single-request TTFT (170.2-171.0 s against 42.7 s at 30,720
tokens). The effective rate falls from 807 to 719 tok/s as the prompt grows from
2,048 to 30,720 tokens.

#### Opt-in prefill ordering (shortest remaining prompt first)

A scheduler option added to this tree after the release binary orders
concurrent prompts by remaining prefill tokens, shortest first, instead of
interleaving their chunks round-robin. It is off by default, and the release
binary `d4ffb1fa` does not contain it. It was measured in the prefill lab on
2026-10-24 with a lab build of the option (`dd7533f1`), the release package,
one Tesla V100 PCIe 32 GB and the release server flags
([v3-off.json](docs/sm70/receipts/prefill-srpt/v3-off.json), [v3-on.json](docs/sm70/receipts/prefill-srpt/v3-on.json)). Each row is 1
warmup and 1 measured batch with 2 output tokens, not the 3 measured batches
and 32 tokens of the table above.

| Prompt tokens | Concurrency | Option | First token per request (s) | Mean first token (s) | Effective prefill tok/s |
| ---: | ---: | --- | --- | ---: | ---: |
| 2,048 | 1 | off / on | n/a | n/a | 805.6 / 806.2 |
| 30,720 | 1 | off / on | n/a | n/a | 718.2 / 718.5 |
| 2,048 | 4 | off | 9.66 / 9.82 / 9.98 / 10.20 | 9.92 | 803.3 |
| 2,048 | 4 | on | 2.54 / 5.08 / 7.68 / 10.26 | 6.39 (36 % lower) | 798.2 |
| 30,720 | 4 | off | 170.11 / 170.31 / 170.52 / 170.78 | 170.43 | 719.5 |
| 30,720 | 4 | on | 42.73 / 85.49 / 128.24 / 171.00 | 106.86 (37 % lower) | 718.6 |

With the option on, the four prompts finish prefill one after another. The
first 30,720-token prompt gets its first token after 42.7 s instead of about
170 s, and the last at 171.0 s as before. The mean first-token time falls 36-37
%. In this 2-output-token protocol, the worst case, the single-request rate and
the aggregate rate stay unchanged within 0.7 %. Greedy outputs matched the release binary on 4 of 4
prompts with the option off and on. With the option off, the build reproduces
the release: 805.6 and 718.2 tok/s at one request against 807 and 719, and
170.11-170.78 s for four 30,720-token prompts against 170.15-170.96 s.

The public `--prefill-srpt` flag was checked on a build of this commit: four
2,048-token prompts got their first tokens at 2.56 / 5.09 / 7.67 / 10.26 s
against 9.67-10.20 s with the flag off, and greedy outputs matched on 3 of 3
prompts ([smoke-public-on.json](docs/sm70/receipts/prefill-srpt/smoke-public-on.json),
[smoke-public-off.json](docs/sm70/receipts/prefill-srpt/smoke-public-off.json)). Tests on this commit:
Helios library 796 passed with 19 ignored, and `flare` 112 passed
([public-helios-lib-tests.log](docs/sm70/receipts/prefill-srpt/public-helios-lib-tests.log),
[public-flare-tests.log](docs/sm70/receipts/prefill-srpt/public-flare-tests.log)). On the lab tree, the GPU
groups passed (A: 223 and 17 subtests, B: 245;
[groups-A](docs/sm70/receipts/prefill-srpt/groups-A-release-srpt3-tree.log),
[groups-B](docs/sm70/receipts/prefill-srpt/groups-B-release-srpt3-tree.log)), and so did the export gates
with the option set: strict 47/47 and invariance 28/28
([gates.log](docs/sm70/receipts/prefill-srpt/gates.log)). The export gates run in Python and never enter
the Helios scheduler, so they confirm the package path, not the ordering.

The same option measured with the public protocol, `tools/benchmark_prefill.py`
at `c6478679` with 1 warmup and 3 measured batches and 32 output tokens. The
server is a build of the release Rust sources plus this change (`79ae7b1f`),
run as `flare run --prefill-srpt` with the release package and flags
([public-prefill-srpt-on-summary.json](docs/sm70/receipts/prefill-srpt/public-prefill-srpt-on-summary.json)).
The default column repeats the release table above:

| Prompt tokens | Concurrency | TTFT p50 / p95 (s), `--prefill-srpt` | TTFT p50 / p95 (s), default | Effective prefill tok/s, `--prefill-srpt` / default |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 1 | 2.54 / 2.54 | 2.54 / 2.54 | 806 / 807 |
| 2,048 | 4 | 6.40 / 10.82 | 9.89 / 10.30 | 758 / 796 |
| 8,192 | 1 | 10.44 / 10.44 | 10.43 / 10.44 | 785 / 785 |
| 8,192 | 4 | 26.21 / 42.41 | 41.44 / 41.88 | 773 / 782 |
| 22,528 | 1 | 30.37 / 30.37 | 30.39 / 30.39 | 742 / 741 |
| 22,528 | 4 | 76.03 / 122.24 | 121.11 / 121.67 | 737 / 741 |
| 30,720 | 1 | 42.70 / 42.71 | 42.75 / 42.75 | 719 / 719 |
| 30,720 | 4 | 106.94 / 171.74 | 170.49 / 170.95 | 716 / 719 |

Single-request rates match the release. At four requests the p50 falls by 35-37
%. With 32 output tokens, the earlier requests decode while the last prompt is
still in prefill, so the last first token arrives slightly later than with
round-robin: 10.72-10.85 s against at most 10.31 s at 2,048 tokens, and
171.68-171.76 s against at most 170.96 s at 30,720 tokens. For the same reason
the effective rate over the first-token window is lower at four requests.

To enable the option, start the server with `flare run --prefill-srpt`. The
flag sets the top-level server configuration key `prefill_srpt` (a JSON
boolean), which a configuration file can also set. When a flag is given, it
overrides the configuration file's key. Round-robin stays the default.

Aging bounds the wait of a long prompt behind a stream of shorter arrivals. By
default a waiting prompt gains 32 tokens of credit for every other prompt's
chunk scheduled, and is served first-come after 256 such chunks.
`--prefill-srpt-max-wait-chunks` and `--prefill-srpt-aging-tokens-per-chunk`
change these, as do the configuration keys `prefill_srpt_max_wait_chunks` and
`prefill_srpt_aging_tokens_per_chunk`. A prompt whose prefill has started is not
preempted.

### Short-prompt native decode, MTP depth 4

128 output tokens per request, concurrency 1 then 4, 3 repetitions. Tok/s per
agent p50 is decode-only: the median request decode rate after its first token.
Aggregate tok/s divides output tokens by batch wall time, including prefill and
HTTP, so it is an end-to-end figure. Do not compare the two columns. Prompts are
the four public `code-00` to `code-03` prompts of 578-582 tokens
([short/native-depth4.json](docs/sm70/receipts/release-binary/short/native-depth4.json),
[short/native-depth4-requests-summary.json](docs/sm70/receipts/release-binary/short/native-depth4-requests-summary.json)).

| Concurrency | Output tokens | Tok/s per agent, p50 | Aggregate tok/s | Repetitions |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 83.0 | 67.6 | 3 |
| 4 | 128 | 64.8 | 220.6 | 3 |

At concurrency 4 the aggregate over the decode window (tokens after each first
token, earliest first to latest last token) is 212.9 tok/s. Request latency p50
was 1.80 s at concurrency 1 and 1.98 s at concurrency 4; first-token p50 was
12 ms and 18 ms. Repeated prompts reuse their prefixes, so those first tokens
include prefix-cache credit and are not prefill measurements. Of the 3,072
generated tokens, 2,234 were accepted MTP drafts. All 24 requests completed.

### Long-context native serving

Four public coding prompts of 28,654 to 28,671 tokens (114,664 in total), 512
output tokens per request, concurrency 4 first and then 1, 3 repetitions. Each
row is labelled cold or cache-credited. A row is cold only when its requests
completed, their usage and stream events carry no prefix-cache credit, and the
server's reused-token counter did not move for the batch. First-token times are
given for cold rows only; decode rates are given for every row
([long/native-depth4.json](docs/sm70/receipts/release-binary/long/native-depth4.json),
[long/native-depth4-requests-summary.json](docs/sm70/receipts/release-binary/long/native-depth4-requests-summary.json),
[long/native-chain-long-cache-labels.json](docs/sm70/receipts/release-binary/long/native-chain-long-cache-labels.json)).

| Concurrency | Prompt tokens | Output tokens | Cold or cache-credited | First-token s (cold rows only) | Decode tok/s per agent | KV tokens resident | Evictions |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 4 | 28,654-28,671 | 512 | first batch: no request credited, batch not cold (a) | 157.96-159.80 (a) | 40.2 / 41.1 / 42.7 / 46.1 (median 41.9) | four contexts, 114,664 prompt tokens (b) | not exported (c) |
| 4 | 28,654-28,671 | 512 | batches 2 and 3: cache-credited, 5 of 8 requests credited | n/a | 25.0-76.8 (median 39.3) | four contexts (b) | not exported (c) |
| 1 | 28,654-28,671 | 512 | cold, 6 requests | 39.83-39.87 | 70.0-81.2 (median 72.4) | one context | not exported (c) |
| 1 | 28,654-28,671 | 512 | cache-credited, 6 requests | n/a | 62.7-79.4 (median 71.5) | one context | not exported (c) |

(a) In the first four-agent batch no request reports cached tokens or reused
prompt tokens, but the server's reused-token counter rose by 28,664 during the
batch, so the chain's labels mark it not cold. Its first-token times are given
because they are the only four-agent long-context times without request-level
cache credit; read them with that caveat. All four first tokens arrived between
157.96 and 159.80 s. The server interleaves the concurrent prefills, and none of
the four emits a token before all four have finished prefill. Its four decode
rates sum to 170 tok/s; the receipt's aggregate over the decode window is
140.5 tok/s.

(b) In that batch the four requests decoded at the same time (from 159.80 s to
at least 170.17 s), so four 28.7k-token contexts were resident simultaneously
within the 131,584-token pool.

(c) The server exports no KV-page eviction counter. The long leg recorded 52
prefix-cache snapshot-slot evictions (`flare_snapshot_evictions_total`); those
are snapshot slots, not KV pages.

Over the whole long leg the server counted 373,063 prefill tokens, 12,288
generated tokens, of which 9,093 were accepted MTP drafts, and 344,346 tokens
reused from the prefix cache, 28,664 of them during the first batch. All 24
requests completed. Over all 12 requests at each concurrency, the per-agent
decode median is 41.5 tok/s at 4 and 72.4 tok/s at 1. A cold single request
reaches its first token after 39.8 s at 28.7k tokens, consistent with the
prefill table.

## Measurement history (2026-10-20 to 2026-10-23)

The best run is one point of a distribution. This section gives the
distribution of every completed run of each protocol, compiled on 2026-10-23
from the fork's run reports, with the best run marked. The per-run sources are
held with the fork. Read it with these limits:

- The runs used six different Tesla V100 32 GB cards in one host, one card per
  run.
- Binaries, packages, memory limits, prefix-cache sizes and KV pools differ
  between runs; each best run names its own.
- The internal short-context protocol uses four prompts of 1,128 to 1,658
  tokens. The release run above used the public prompts of 578 to 582 tokens.
  The release's 83.0 tok/s against the best recorded 90.4 tok/s is therefore not
  a like-for-like comparison and is not a regression claim.
- Decode-only, round-time and end-to-end figures are kept in separate tables.

### Decode-only, short context (internal prompts, MTP depth 4, 128 output tokens)

| Concurrency | Metric | Runs | Min | Median | Mean | Best | Best run |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | Median decode per request | 11 | 80.1 | 87.6 | 87.2 | 90.4 | 2026-10-22 19:57 PDT, binary `2c792073`, package `model-32k-tonight-a` |
| 4 | Median decode per agent | 11 | 36.1 | 64.3 | 59.3 | 70.7 | same run |
| 4 | Median decode per agent, runs on the final 131,584-token KV pool | 3 | 67.7 | 70.5 | 69.6 | 70.7 | same run |

### Decode-only, long context (16.6k-18.6k-token prompts, MTP depth 4, 512 output tokens)

| Concurrency | Metric | Runs | Min | Median | Mean | Best | Best run |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4 | Median decode per agent (tok/s) | 8 | 38.4 | 44.0 | 43.1 | 46.0 | 2026-10-22 09:41 PDT, binary `2c792073`, an earlier 32k package (98,304-token KV pool) |
| 1 | GPU time per 1 x 5 speculative round (ms, lower is better) | 8 | 49.5 | 52.8 | 53.1 | 49.5 | 2026-10-22 19:19 PDT, binary `2c792073`, package `model-32k-tonight-a` |

### End to end, real coding-agent traces (4 agents, MTP depth 3)

Generated tokens over wall time in a 20-minute replay of the fork's real
coding-agent traces, including prefill, tool turns and every request. The traces
are not public.

| Scope | Runs | Min | Median | Mean | Best | Best run |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| All runs | 23 | 27.0 | 43.4 | 41.2 | 47.23 | 2026-10-22 19:08 PDT, run `best24`: package `model-32k-tonight-a`, binary `2f9128e4`, `--memory-limit 31G`, 32G prefix cache |
| Runs not flagged as confounded, test, stall or regression | 16 | 35.1 | 43.7 | 43.4 | 47.23 | same run |

## Historical measurements (older binary 2c792073, 2026-10-22)

The decode-only and fixed-prompt rows for package `model-32k-tonight-a` were
measured with the older `flare` binary `2c792073`, built before this branch's
Helios commits, on the fork's historical prompt sets (STATUS.md line 63;
[BENCHMARK.md](docs/sm70/BENCHMARK.md#what-was-run)). The real-trace row used
another pre-release binary (note d). Rows for other packages or dates, such as
`model-32k-replay` and the 2026-10-21 baseline, are older context and name
their package. None of these are this release's numbers;
the release-binary section above gives those.

Every row names its hardware, memory limit (`flare run --memory-limit`),
package, concurrency, date and the line of the fork's run log (`STATUS.md`) that
records it. That log is held with the fork; quoted fragments are listed under
[Sources](#sources). All runs used Qwen3.8-27B NVFP4 (`qwen3.8-27b`), TP1,
prepared for a 32,768-token context, with MTP draft depth 4 (verification width
5) except the real-trace row, which ran depth 3.

`model-32k-tonight-a` is the package of the 2026-10-22 release tree (`0e4c81f`).

### End-to-end, real coding-agent traces

| Measurement | Value | Hardware | Memory limit | Package | Concurrency | Date | Source |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| Generated tokens per second, end to end | 47.23 tok/s | V100 PCIe 32 GB | 31G (d) | `model-32k-tonight-a`, run `best24`, binary `2f9128e4`, MTP depth 3 (d) | 4 (d) | 2026-10-22 | STATUS.md line 68 (23:35) |
| Release binary `d4ffb1fa` | not measured | V100 PCIe 32 GB | 31G | release export | 4 | n/a | none |

(d) The run log names only the run tag `best24`. Its run report, compiled in
the [measurement history](#measurement-history-2026-10-20-to-2026-10-23), gives
the package `model-32k-tonight-a`, a binary `2f9128e4` built before the release
binary, MTP depth 3, `--memory-limit 31G`, a 32G prefix cache and 4 agents.
Earlier versions of this page called `best24` a different package; it is a run
of the 2026-10-22 release-tree package. No real-trace end-to-end run of the
release binary `d4ffb1fa` exists.

### End-to-end, fixed short prompts (not real traces)

| Measurement | Value | Hardware | Memory limit | Package | Concurrency | Date | Source |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| Short-context screening, aggregate output | 238.0 tok/s | V100 PCIe 32 GB | 31G (a) | `model-32k-tonight-a` | 4 | 2026-10-22 | STATUS.md line 66 (20:07) |

The aggregate rate divides output tokens by batch wall time, including prefill
and HTTP overhead. It is not a decode-only throughput measurement.

### End-to-end, fixed long prompts (not real traces)

Four fixed prompts of 22.4k to 30.2k tokens (110,022 prompt tokens in total),
512 output tokens each, 3 repetitions at each of 1 and 4 agents, 24 requests
across both concurrency settings.

| Measurement | Value | Hardware | Memory limit | Package | Concurrency | Date | Source |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| TTFT, p50 | 90.7 s | V100 PCIe 32 GB | 31G | `model-32k-tonight-a` | 4 | 2026-10-22 | STATUS.md line 65 (19:57) |
| Requests completed | 24/24 across the combined runs | V100 PCIe 32 GB | 31G | `model-32k-tonight-a` | 1 and 4 | 2026-10-22 | STATUS.md line 65 (19:57) |
| Residency | all four prompts resident, no eviction or restore | V100 PCIe 32 GB | 31G | `model-32k-tonight-a` | 4 | 2026-10-22 | STATUS.md line 65 (19:57) |

### Decode-only

| Workload | Value | Hardware | Memory limit | Package | Concurrency | Date | Source |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| Short-context screening, median decode | 90.4 tok/s | V100 PCIe 32 GB | 31G (a) | `model-32k-tonight-a` | 1 | 2026-10-22 | STATUS.md line 66 (20:07) |
| Short-context screening, decode per agent | 70.7 tok/s | V100 PCIe 32 GB | 31G (a) | `model-32k-tonight-a` | 4 | 2026-10-22 | STATUS.md line 66 (20:07) |
| 4 x 16.6k-18.6k prompts, median decode | 82.6 tok/s | V100 PCIe 32 GB | 31G (a) | `model-32k-tonight-a` | 1 | 2026-10-22 | STATUS.md line 64 (19:36) |
| 4 x 16.6k-18.6k prompts, median decode per agent | 42.6 tok/s | V100 PCIe 32 GB | 31G (a) | `model-32k-tonight-a` | 4 | 2026-10-22 | STATUS.md line 64 (19:36) |
| 4 x 16.6k-18.6k prompts, GPU step per 1 x 5 round | 49.5-49.6 ms | V100 PCIe 32 GB | 31G (a) | `model-32k-tonight-a` | 1 | 2026-10-22 | STATUS.md line 64 (19:36) |
| 4 x 22.4k-30.2k prompts, median decode | 82.2 tok/s | V100 PCIe 32 GB | 31G | `model-32k-tonight-a` | 1 | 2026-10-22 | STATUS.md line 65 (19:57) |
| 4 x 22.4k-30.2k prompts, median decode per agent | 37.2 tok/s | V100 PCIe 32 GB | 31G | `model-32k-tonight-a` | 4 | 2026-10-22 | STATUS.md line 65 (19:57) |

(a) Lines 64 and 66 do not restate the memory limit. Both report the same
131,584-token KV pool as the 31G run on line 65.

Per-agent decode rates at 4 agents vary with prefill interleaving. In the
22.4k-30.2k run the four agents' medians were 36.2, 13.0, 39.9 and 53.5 tok/s.
The 13.0 tok/s agent had the shortest prompt and decoded while the other three
were still in chunked prefill (STATUS.md line 65).

### Memory and capacity

| Measurement | Value | Hardware | Memory limit | Package | Concurrency | Date | Source |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| KV pool | 131,584 tokens | V100 PCIe 32 GB | 31G | `model-32k-tonight-a` | 4 | 2026-10-22 | STATUS.md lines 65 (19:57) and 66 (20:07) |
| KV pool, replay-commit package before the release merge | 131,584 tokens | V100 PCIe 32 GB | 31G | `model-32k-replay` | 4 | 2026-10-22 | STATUS.md line 49 (15:34) |
| KV pool, baseline before replay-commit | 98,304 tokens | V100 PCIe 32 GB | 31G | baseline package of 2026-10-21 22:20 | 4 | 2026-10-21 | as cited on STATUS.md line 49 (15:34) |
| KV pool change from replay-commit | +33,280 tokens (+33.9 %) | V100 PCIe 32 GB | 31G | `model-32k-replay` vs baseline | 4 | 2026-10-22 | STATUS.md line 49 (15:34) |
| Speculative state rows (preparation report) | 5 (1 null + 4), 1 state column | V100 PCIe 32 GB | n/a (preparation) | `model-32k-tonight-a` | 4 | 2026-10-22 | STATUS.md line 62 (19:04) |
| Replay operand storage (preparation report) | 30.4 MiB | V100 PCIe 32 GB | n/a (preparation) | `model-32k-tonight-a` | 4 | 2026-10-22 | STATUS.md line 62 (19:04) |

Replay-commit keeps one committed recurrent-state row per request instead of
one per verification position. The candidate rows it returns
(16 x 166 MiB) become KV pages (STATUS.md line 59, 17:45).

### Sources

Quoted fragments of the cited `STATUS.md` lines (2026-10-22 unless noted):

- Line 49 (15:34), replay-commit package: "`Allocate cache. 9.4 GiB · 131584 KV
  tokens · 32 shared snapshot slots` at --memory-limit 31G / 4 agents =
  +33,280 lossless KV tokens vs the 22:20 baseline's 98,304 (+33.9 %)".
- Line 59 (17:45): "+33,280 lossless KV tokens (131,584; 16 x 166 MiB of
  candidate rows returned to the cache)".
- Line 62 (19:04), release package preparation: "strict 47/47 incl. every
  forced_prefix_*, long-sequence invariance 28/28, commit growth 4/4 ...
  Memory report: state_rows 5, 1 state column, replay operands 30.4 MiB,
  continuation 1,045 MiB".
- Line 64 (19:36), release package: "1x5 GPU step 49.5-49.6 ms per round ...
  1 agent median decode 82.6 tok/s; 4 agents per-agent median 42.6 tok/s ...
  KV pool 131,584".
- Line 65 (19:57), release package: "all four resident at 31G (KV 131,584,
  zero evict/restore); 4 agents per-agent median decode 37.2 tok/s (long-00
  36.2, long-01 13.0 ..., long-02 39.9, long-03 53.5), 1 agent 82.2 tok/s ...
  TTFT p50 90.7 s at 4 agents".
- Line 66 (20:07), release package: "1 agent 90.4 tok/s median decode;
  4 agents 70.7 tok/s per agent, 238.0 tok/s aggregate output, 24/24 ok; KV
  pool 131,584 tokens".
- Line 68 (23:35): "REALITY remains best24 47.23 tok/s end-to-end;
  PUBLIC-COMPARABLE remains native90.4 single /70.7 median per agent
  at4,238.0 aggregate".
