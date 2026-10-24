# SM70 benchmarks (Tesla V100 32 GB)

Measurements for the SM70 port described in [README-SM70.md](README-SM70.md).
End-to-end numbers and decode-only numbers are reported in separate tables. Do
not compare a number from one table with a number from the other.

- **End-to-end** numbers include prefill, scheduling and every request in the
  workload.
- **Decode-only** numbers are per-request decode rates (tokens per second while
  the request is decoding) or GPU step times. They exclude prefill and queueing.

Every row names its hardware, memory limit (`flare run --memory-limit`),
package, concurrency, date and the line of the fork's run log (`STATUS.md`) that
records it. That log is held with the fork; quoted fragments are listed under
[Sources](#sources). All runs used Qwen3.8-27B NVFP4 (`qwen3.8-27b`) with MTP
draft depth 4 (verification width 5), TP1, prepared for a 32,768-token context.

`model-32k-tonight-a` is the package of the release tree (`0e4c81f`).

## End-to-end, real coding-agent traces

| Measurement | Value | Hardware | Memory limit | Package | Concurrency | Date | Source |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| Generated tokens per second, end to end | 47.23 tok/s | V100 PCIe 32 GB | not stated in the source line | `best24`, a different package, **not** the release tree | not stated in the source line | 2026-10-22 | STATUS.md line 68 (23:35) |
| Release package | not measured | V100 PCIe 32 GB | 31G | `model-32k-tonight-a` | 4 | n/a | none in the cited sources |

No real-trace end-to-end measurement of the release package appears in the
cited sources. The `best24` row is listed only because it is the real-trace
figure the run log reports for the fleet at that time. It does not describe
this release.

## End-to-end, fixed short prompts (not real traces)

| Measurement | Value | Hardware | Memory limit | Package | Concurrency | Date | Source |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| Short-context screening, aggregate output | 238.0 tok/s | V100 PCIe 32 GB | 31G (a) | `model-32k-tonight-a` | 4 | 2026-10-22 | STATUS.md line 66 (20:07) |

The aggregate rate divides output tokens by batch wall time, including prefill
and HTTP overhead. It is not a decode-only throughput measurement.

## End-to-end, fixed long prompts (not real traces)

Four fixed prompts of 22.4k to 30.2k tokens (110,022 prompt tokens in total),
512 output tokens each, 3 repetitions at each of 1 and 4 agents, 24 requests
across both concurrency settings.

| Measurement | Value | Hardware | Memory limit | Package | Concurrency | Date | Source |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| TTFT, p50 | 90.7 s | V100 PCIe 32 GB | 31G | `model-32k-tonight-a` | 4 | 2026-10-22 | STATUS.md line 65 (19:57) |
| Requests completed | 24/24 across the combined runs | V100 PCIe 32 GB | 31G | `model-32k-tonight-a` | 1 and 4 | 2026-10-22 | STATUS.md line 65 (19:57) |
| Residency | all four prompts resident, no eviction or restore | V100 PCIe 32 GB | 31G | `model-32k-tonight-a` | 4 | 2026-10-22 | STATUS.md line 65 (19:57) |

## Decode-only

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

## Memory and capacity

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

## Sources

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
