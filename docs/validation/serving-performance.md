# HTTP serving and graph replay performance

The pinned vLLM server records higher standalone decode throughput at both
prefill limits. At limit 4,096, median delivered rates at requested concurrency
1/2/4 are **78.329/128.105/200.651 tokens/s for Sparkgap** and
**111.262/197.400/353.295 tokens/s for vLLM**. Sparkgap has lower short-prompt
and prefix-hit TTFT; vLLM has lower 4K/64K cold-prompt TTFT at limit 4,096.
At limit 512, Sparkgap has lower TTFT for every measured single-output workload.
All 35 measured 256-token sequences differ between implementations at each
limit, so paired decode speed ratios are omitted. All 25 single-output token
IDs match at each limit, including prefix seeds.

The Qwen3.8 native server completes **64 requests in 41 groups at each prefill
limit, 512 and 4,096**, with all 832 request predicates and all group checks
passing in both runs. Every request paired between the two Sparkgap runs has identical bytes and
generates the same complete token sequence. At prefill limit 4,096 and requested concurrency
1/2/4, median delivered decode throughput is **78.329/128.105/200.651 tokens/s**.
Median native token-ready intervals are **12.763/15.163/18.507 ms**. Client
delivery p99 intervals are **40.503/40.815/40.919 ms**, exceeding the corresponding
native p99 intervals of **12.904/15.338/18.842 ms**.

Accepted-socket TCP_NODELAY then passes the same 64-request corpus with all raw
IDs and 16,711 selected configurations unchanged. Normal-client delivery p99 at
requested concurrency 1/2/4 falls to **12.889/15.338/18.832 ms**; core p99 remains
**12.890/15.332/18.814 ms**. The backend comparison tables retain the measured
baseline runs rather than combining measurements from different native builds.

The 4,096-token prefill configuration's batch-1
native core latency exceeds the autoregressive replay floor by **0.095–0.104%**
across five paired workloads; the 512-token configuration exceeds its replay
floor by **0.714–0.742%**. Software Nsight traces identify steady GPU batches 1/2/4 at both limits.
At limit 4,096 their median inter-graph gaps are **15.040/16.256/16.480 µs**;
at limit 512 they are **16.288/16.192/16.352 µs**. Requested concurrency alone
does not establish the captured GPU batch used for each step. Independent
[numerical acceptance](text-serving-oracle.md) remains open, so these timings
do not satisfy the complete [performance acceptance](../core-design.md#performance-acceptance)
contract.

## Conditions and metric definitions

The following execution and memory values describe prefill limit 4,096;
the 512-token configuration is compared separately below.

| Property | Recorded value |
|---|---|
| Model | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD` |
| Revision | `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2` |
| GPU | RTX PRO 6000 Blackwell Max-Q Workstation Edition; UUID `GPU-afd4b1ad-8a64-7057-4bde-241822724c7f` |
| Driver / power limit | 610.57.04 / 325 W |
| CPU | Ryzen Threadripper PRO 9965WX, 24 cores / 48 logical CPUs |
| Software | PyTorch 2.13.0+cu130; Transformers 5.19.0.dev0; FlashInfer 0.7.1; CuTe DSL 4.7.1; Triton 3.7.1 |
| Model execution | TP1; prefill limit 4,096; 43 captured variants; greedy sampling; no speculative decoding |
| Arithmetic policy | QSA 16 attention partitions; GDN sequential recurrence with one key split; quantized MXFP8 dense projections; NVFP4 experts |
| KV capacity | 8,844 pages × 64 tokens = 566,016 cached tokens; 37 immutable snapshots |
| Allocated / reserved device memory | 95,578,595,328 / 96,517,226,496 bytes |
| Operator scratch / free device memory | 297,959,168 / 4,312,858,624 bytes |
| HTTP measurement | 2026-10-09 17:40:30–17:42:28 UTC; PID 2849343; loopback SSE |

The hardware snapshot was taken during the workload. It reports P1 state,
2,265 MHz graphics and 13,365 MHz memory clocks; it does not establish locked
clocks. Source and hardware records are linked from the
[compact JSON receipt](serving-performance.json).

Five repetitions cover each measured workload. There are 35 measured groups
containing 55 requests; one four-request warmup and five prefix-seed requests
bring the executed total to 41 groups and 64 requests. The corpus contains
411,703 prompt tokens and produces 10,009 tokens. Requests explicitly select
temperature 0, neutral penalties, ordinary EOS handling, and an output budget
of either one or 256 tokens. Every request reaches its exact requested count
with `finish_reason="length"`; early EOS would fail the check. Native rendered
prompt IDs must equal the frozen tokenizer-derived IDs.

- **Core TTFT:** native chat-handler entry, after JSON extraction, to the first
  CPU token observation after scheduler completion. It includes frontend work,
  queueing, graph execution, completion polling and scheduler processing.
- **Delivery TTFT:** client request send to receipt of the first raw token ID.
  Connections are opened before the simultaneous-send barrier; connection time
  is recorded separately.
- **Core ITL:** successive native CPU token-observation timestamps. These are
  token-ready timestamps, not GPU kernel completion timestamps.
- **Delivery ITL:** successive singleton token-ID SSE arrivals. All observed
  groups contain exactly one token, so no per-token timing is inferred from
  coalesced ID groups. Several arrivals can still be buffered and delivered
  close together; the observed minima are 7.8–9.2 µs for the decode workloads.
- **Decode throughput:** sum of generated tokens after each request's first
  token, divided by the interval from the earliest first delivery to the latest
  final delivery in the group. Each request contributes 255 tokens.
- **HTTP prefill rate:** prompt length divided by delivery TTFT. It includes
  frontend, sampling and delivery rather than measuring isolated GPU prefill.

Token observation adds a bounded native channel event and metadata SSE chunk
per token. Its overhead has not yet been isolated. Native and client clocks
have separate origins; durations are computed within each clock domain.
Percentiles use nearest rank `ceil(0.99 × n)`. Lower TTFT/ITL and higher token
throughput are better.

## Sparkgap and vLLM at prefill limit 512

Both backends pass all 64 requests and 41 groups in the frozen shape corpus.
The comparator revalidates wire requests, rendered prompt IDs, raw SSE payloads,
usage, finish reasons, cache metrics and delivery timing. All 394 indexed raw
artifacts are verified. The corpus uses numbered-list and apple prompts; it is
not a task-quality or MTP acceptance corpus.

The vLLM server is revision `64fc1eec593397cbd765cc082e9ec90fbeb5ea66`, TP1,
with BF16 KV/conv state, FP32 recurrent state, host PLE and no MTP. It uses
b12x linear/MoE/GDN paths, quantized MXFP8 activations, asynchronous scheduling,
aligned prefix caching and a 512-token scheduling budget. Its isolated FlashInfer
source is revision `98fb87a461183f98d6a7049ca947fb136cc6fb59`. Exact launch
arguments, environment and source hashes are indexed by the compact receipt;
backend arithmetic and selected kernel configurations are not assumed identical.
No causal timing attribution to individual kernel policies has been established.

All 25 single-output cases generate identical IDs. All 39 requests generating
256 tokens differ, including four warmups and all 35 measured decode requests.
Their first mismatch occurs after 2–210 common tokens, median 52. Each backend
still returns the requested 256 tokens with `finish_reason="length"`. Those
workloads therefore retain standalone rates and timing distributions, but no
paired decode speed ratios. Low native scheduling overhead does not establish
that model execution matches vLLM throughput.

| Requested concurrency | Sparkgap delivered decode median, tok/s | vLLM delivered decode median, tok/s | Sparkgap delivery ITL median / p99, ms | vLLM delivery ITL median / p99, ms |
|---|---:|---:|---:|---:|
| 1 | 77.782 | 114.221 | 12.852 / 40.525 | 8.767 / 9.401 |
| 2 | 128.216 | 199.772 | 15.139 / 40.736 | 9.953 / 10.823 |
| 4 | 201.040 | 310.043 | 18.486 / 40.917 | 11.647 / 19.769 |

Each throughput entry is the median of five group measurements; ITL entries
pool 1,275/2,550/5,100 intervals. Both backends emit singleton token-ID chunks
throughout, including the first chunk. No token timings are inferred from
coalesced chunks. Sparkgap's observation path also timestamps tokens in the
native worker; vLLM receives only `return_token_ids=true`. Native core timings
remain separate from these client delivery measurements.

For the following single-output measurements, all five repetitions produce the
same output ID on both backends. Cold requests reuse zero tokens. Both prefix
paths restore exactly 4,096 tokens and process the same 11-token suffix.

| Single-request workload | Sparkgap delivery TTFT median / p99, ms | vLLM delivery TTFT median / p99, ms |
|---|---:|---:|
| 512-token cold prompt | 105.852 / 106.037 | 142.965 / 153.484 |
| 4,096-token cold prompt | 783.026 / 784.490 | 1,118.328 / 1,131.027 |
| 65,536-token cold prompt | 12,054.800 / 12,067.749 | 17,886.788 / 17,996.643 |
| 4,096 cached tokens plus 11-token suffix | 45.358 / 46.227 | 163.308 / 170.759 |

The measured result is mixed: Sparkgap has lower prefill/prefix TTFT on these
single-output inputs and lower standalone decode rates. Different generated
sequences and the open numerical gate prevent complete performance acceptance.

## Sparkgap and vLLM at prefill limit 4,096

Both backends pass all 64 requests and 41 groups. The comparator verifies all
394 indexed artifacts, identical rendered inputs and semantic request settings,
exact output counts, cache reuse and timing distributions. The vLLM scheduling
budget is 4,096; its model and library revisions and state dtypes are the same
as specified above. Selected backend algorithms need not match Sparkgap.

All 25 single-output IDs match. All 39 complete 256-token sequences differ,
including the four warmups; first mismatches occur after 2–131 common tokens,
median 52. The measured 35 decode sequences therefore retain standalone rates
without paired speed ratios. Neither backend coalesces token-ID chunks.

| Requested concurrency | Sparkgap delivered decode median, tok/s | vLLM delivered decode median, tok/s | Sparkgap delivery ITL median / p99, ms | vLLM delivery ITL median / p99, ms |
|---|---:|---:|---:|---:|
| 1 | 78.329 | 111.262 | 12.763 / 40.503 | 9.001 / 9.670 |
| 2 | 128.105 | 197.400 | 15.160 / 40.815 | 10.099 / 10.936 |
| 4 | 200.651 | 353.295 | 18.507 / 40.919 | 11.385 / 12.521 |

Throughput entries are five-group medians; ITL pools contain 1,275/2,550/5,100
intervals. The single-output measurements below have equal output IDs for every
repetition. Cold requests reuse zero tokens, and both prefix paths reuse exactly
4,096 tokens before processing the same 11-token suffix.

| Single-request workload | Sparkgap delivery TTFT median / p99, ms | vLLM delivery TTFT median / p99, ms |
|---|---:|---:|
| 512-token cold prompt | 105.796 / 105.997 | 147.294 / 159.370 |
| 4,096-token cold prompt | 511.713 / 512.668 | 281.197 / 281.669 |
| 65,536-token cold prompt | 8,642.687 / 8,655.267 | 4,964.189 / 4,971.702 |
| 4,096 cached tokens plus 11-token suffix | 45.293 / 45.645 | 177.144 / 179.176 |

Sparkgap's small measured host overhead coexists with lower full-backend decode
throughput and slower long-prompt prefill at this scheduling budget. The timings
do not isolate a specific kernel or arithmetic policy as the cause. Independent
numerical acceptance remains open.


## Operator policies and selection evidence

The 512-token source/selection audit verifies 17 Sparkgap/main-FlashInfer files
against the launch manifest and reconstructs 13 vLLM cache keys using the pinned
library API. Sparkgap configurations come directly from its measured selection
receipt. The vLLM configurations come from matching immutable cache entries and
historical worker plans, not a fresh actor dump from the HTTP benchmark.

| Component | Sparkgap | Pinned vLLM path |
|---|---|---|
| MoE decode B1/B2 | Micro; deterministic output; fast math disabled | Dynamic M16; direct/internal routing; nondeterministic output; fast math enabled |
| MoE decode B4 | Micro; same numerical policy | Dynamic M16; grouped/Triton routing; 128 active clusters |
| MoE prefill512 | Dynamic M32; grouped/internal | Dynamic M16; grouped/internal |
| GDN prefill | Sequential/key split 1 policy; capacity 512 window 33 | Capacity512 window 36; cached capacity 16 winner key split 2 |
| QSA attention | Fixed 16 partitions, attention chunks at most 64 rows | 64/32/16 partitions for 1/2–4/5–64 rows; direct above 64 |
| Hyperconnection | FP32 checkpoint gain; separate 320+4 projections; combine then norm | BF16 zero-centered norm weight; merged 336 projection; delayed combine/norm |
| PLE norm | FP32 checkpoint norm weights | BF16 norm weights |

For each decode capacity 1/2/4, Sparkgap measures all 17 legal MoE candidates and
selects micro without an override. Determinism excludes direct and
Triton-planned routes while retaining 16 grouped/internal dynamic candidates.
All 48 layers agree within each capacity. The smaller search is a semantic
eligibility constraint, not an interrupted tuning pass. Both systems use BF16
GDN intermediates, BF16 KV/conv state, FP32 recurrent state, quantized MXFP8
activations and NVFP4-CSF experts.

The tuning input distributions and route-weight precision also differ. These
operator and tuning differences are concrete candidates for controlled tests;
the end-to-end timings do not identify their individual costs. The pinned
prepared vLLM HC workspace path executes a merged projection followed by public
scaled SiLU; an enabled module flag does not establish execution of its fused
down-SiLU helper. vLLM's 4,096-token startup performs additional GDN searches and
is outside the 512-token reconstructed-selection audit. Full audit hashes and
source excerpts are indexed in the compact receipt.

## Decode at prefill limit 4,096: 512-token prompts and 256 generated tokens

ITL distributions pool all token intervals across five repetitions. Throughput
is the median of five whole-group rates.

| Requested concurrency | Requests / ITL samples | Core ITL median / p99, ms | Delivery ITL median / p99, ms | Delivered throughput median, tok/s |
|---|---:|---:|---:|---:|
| 1 | 5 / 1,275 | 12.763 / 12.904 | 12.763 / 40.503 | 78.329 |
| 2 | 10 / 2,550 | 15.163 / 15.338 | 15.160 / 40.815 | 128.105 |
| 4 | 20 / 5,100 | 18.507 / 18.842 | 18.507 / 40.919 | 200.651 |

| Requested concurrency | Core TTFT median / p99, ms | Delivery TTFT median / p99, ms | Maximum core / delivery ITL, ms |
|---|---:|---:|---:|
| 1 | 105.738 / 105.865 | 105.887 / 106.026 | 12.925 / 41.382 |
| 2 | 158.005 / 210.342 | 158.205 / 210.517 | 117.362 / 117.347 |
| 4 | 269.288 / 447.833 | 269.472 / 448.104 | 138.066 / 138.138 |

Native metrics record 256/257/259 decode launches per concurrency-1/2/4 group,
respectively, in every repetition. Those counts include the scheduler's actual
submission behavior and do not determine each launch's active batch. The large
delivery tail is investigated by the TCP diagnostic below; occasional core
outliers still require timeline attribution.

## TCP delivery diagnostic

Four separate requests repeat `decode-b1-r0-0` on the prepared replay process,
PID 2930618, in normal/QUICKACK/QUICKACK/normal order. The HTTP request bytes,
complete 256 generated IDs, usage and finish reason are identical across all
four arms; each reports zero reused prompt tokens. The client keeps its
ordinary `HTTPResponse.readline` and SSE parser. Both arms record socket reads;
the QUICKACK arm additionally requests immediate TCP acknowledgements after
connection and every successful `recv_into`.

| Arm, execution order | Delivery ITL p99, ms | Core ITL p99, ms | Delivery gaps between 39 and 42 ms |
|---|---:|---:|---:|
| Normal, 0 | 40.451 | 12.912 | 4 |
| QUICKACK, 1 | 12.898 | 12.893 | 0 |
| QUICKACK, 2 | 12.908 | 12.913 | 0 |
| Normal, 3 | 40.822 | 12.888 | 4 |

Changing receiver acknowledgement behavior removes the approximately 40 ms
delivery tail while core token timing remains similar; the tail returns in the
final normal arm. This identifies a transport contribution rather than a model
execution pause. The baseline sockets have no explicit TCP_NODELAY setting. These four requests
are separate from the 64-request baseline and do not replace its timings.

## Accepted-socket TCP_NODELAY

The native listener sets `TCP_NODELAY` on every accepted connection before
Axum receives the stream. A failed `setsockopt` logs the peer/error and drops
that connection; it does not serve a socket with an unconfirmed option. The
same listener covers ordinary and qualification routes. Focused loopback tests
verify the accepted-server option independently from the client socket;
100 Rust tests pass with one checkpoint-dependent test ignored, and 42 Python
tests pass. Formatting and Clippy checks pass.

Native extension SHA-256
`586907fc1cb14cd5bcffa4af2f5865fb97012fa7dcd74d81266e638800fe7cdc`
serves the full unchanged corpus through `python -m sparkgap` on loopback port 8000,
PID 3455550. All 64 requests/41 groups pass with zero errors. An independent CPU
comparison revalidates request bytes, raw SSE, prompt IDs, complete generated
IDs, usage, finish reasons, reused prefixes and all 16,711 selected
configurations. All agree with the 4,096-token baseline. The ordinary client
uses its original socket settings; it does not enable QUICKACK.

| Requested concurrency | Delivery ITL p99 before / after, ms | Core ITL p99 before / after, ms | Delivered throughput median before / after, tok/s |
|---|---:|---:|---:|
| 1 | 40.503 / 12.889 | 12.904 / 12.890 | 78.329 / 78.365 |
| 2 | 40.815 / 15.338 | 15.338 / 15.332 | 128.105 / 128.225 |
| 4 | 40.919 / 18.832 | 18.842 / 18.814 | 200.651 / 200.585 |

The approximately 40 ms delivery tail disappears from p99 while model throughput
and core timing remain similar. The full record still contains approximately
117/138 ms maximum intervals at concurrency 2/4 during overlapping request
startup; TCP_NODELAY does not remove those core pauses. The fixed-token receiver
crossover above independently supports transport attribution for the 40 ms tail.

The process-to-first-healthy-listener duration is **156.207 seconds**, excluding
source fingerprinting and without a profiler. This uses retained compiled
kernels and 51 selection-cache hits with zero measured candidates; its tuning
call takes 4.091 seconds. It is a warm selection-cache startup, not directly
comparable to the independently cold-tuned preparations above.

The model retains 43 graphs, 297,959,168 bytes of operator scratch and 37 snapshots.
Its automatic KV pool has 8,892 pages/569,088 tokens, versus 8,844 pages/566,016
tokens in the baseline; allocated/reserved/free device memory is
95,629,189,120/96,529,809,408/4,340,121,600 bytes. This capacity difference is
retained in the receipt rather than treated as an identical memory layout.
The request workloads and observed reused-token counts remain equal.

All 31 Python runtime files and 8,159 FlashInfer files remain byte-identical.
The native source differences are exactly `native/src/server.rs` and the added
`native/src/server/listener.rs`; changed qualification helpers/tests are listed
separately in the source comparison. No model arithmetic, sampling policy or
CUDA graph definition changes. Replay-floor measurements are not rerun for this
socket change. The independent comparison recomputes every aggregate timing
summary and preserves the two native extension hashes, warm startup conditions
and complete source differences.

A separate ordinary SSE request, with token observation disabled, generates 256
tokens and matches the baseline's complete content, usage and length finish.
This verifies presentation without returning raw token IDs. Ordinary JSON and
SSE chat requests both return `Paris`, 25 prompt tokens, 2 completion tokens and
`finish_reason="stop"`. Health and model discovery return 200. These checks were
recorded at 2026-10-09 18:52:17 UTC for the same process and port; their raw
requests/responses and readiness records are indexed separately from the
64-request performance corpus.

## Prefill and prefix reuse at prefill limit 4,096

Each row contains five sequential single-request measurements, each generating
one token.

| Prompt / cache condition | Core TTFT median / p99, ms | Delivery TTFT median / p99, ms | HTTP prefill rate median, tok/s |
|---|---:|---:|---:|
| 512 tokens, cold | 105.666 / 105.877 | 105.796 / 105.997 | — |
| 4,096 tokens, cold | 511.494 / 512.477 | 511.713 / 512.668 | 8,004.489 |
| 65,536 tokens, cold | 8,642.313 / 8,654.900 | 8,642.687 / 8,655.267 | 7,582.827 |
| 4,107 tokens, 4,096-token checkpoint plus 11-token suffix | 45.053 / 45.496 | 45.293 / 45.645 | — |

All cold requests report zero reused tokens. Every prefix-hit request reports
exactly 4,096 reused tokens. Per-group native metrics confirm admission and
completion counts, no failures or cancellations, exact cache counters, and
conservation of free-plus-cached KV pages and snapshots after requests drain.

## Prefill-limit comparison

The 512-token configuration runs the identical frozen corpus in PID 2989319.
All 64 request bodies and complete generated token sequences match the
4,096-token run. All raw timing distributions and group spans/rates are
independently recomputed for both runs. Both use the same model, native binary,
31 Python runtime files, 42 native source/build files and 8,159 FlashInfer files.

| Single-request workload | Delivery TTFT median, ms: limit 512 / 4,096 | Delivery TTFT p99, ms: limit 512 / 4,096 |
|---|---:|---:|
| 512-token cold prompt | 105.852 / 105.796 | 106.037 / 105.997 |
| 4,096-token cold prompt | 783.026 / 511.713 | 784.490 / 512.668 |
| 65,536-token cold prompt | 12,054.800 / 8,642.687 | 12,067.749 / 8,655.267 |
| 4,096 cached tokens plus 11-token suffix | 45.358 / 45.293 | 46.227 / 45.645 |

The measured 4,096-token configuration has 34.65% lower median 4K TTFT and
28.31% lower median 64K TTFT. The corresponding HTTP prefill rates are
5,230.992 versus 8,004.489 tokens/s at 4K and 5,436.507 versus 7,582.827 tokens/s
at 64K, for limits 512 and 4,096 respectively.

| Requested concurrency | Core ITL median / p99, ms: limit 512 | Delivery ITL median / p99, ms: limit 512 | Delivered throughput median, tok/s: limit 512 / 4,096 |
|---|---:|---:|---:|
| 1 | 12.852 / 12.990 | 12.852 / 40.525 | 77.782 / 78.329 |
| 2 | 15.144 / 15.351 | 15.139 / 40.736 | 128.216 / 128.105 |
| 4 | 18.488 / 18.814 | 18.486 / 40.917 | 201.040 / 200.651 |

Both preparations start with empty selection caches while retaining compiled
kernel caches. Of 12,646 common named plans, 12,333 have identical selected
configurations and 313 differ: 216 GDN-prefill bindings and 97 hyperconnection
bindings at prefill capacity 64. These are named bindings, including shared-plan
aliases, not 313 independent searches. All decode plan configurations match.
The 4,096-token catalog contains another 4,065 declarations. Small decode
differences are not isolated effects of prefill chunk size; the runs use
separate processes and independently selected prefill plans.

| Preparation / memory | Prefill limit 512 | Prefill limit 4,096 |
|---|---:|---:|
| Captured graph variants | 37 | 43 |
| Process-to-ready, s | 123.636 | 175.300 |
| Source recording included above, s | 3.944 | 4.264 |
| Batched tuning, s | 13.598 | 20.210 |
| Measured candidates / selection hits | 1,667 / 0 | 1,883 / 0 |
| Selected operator scratch, bytes | 166,879,232 | 297,959,168 |
| KV pages / cached-token capacity | 9,847 / 630,208 | 8,844 / 566,016 |
| Allocated device memory, bytes | 95,710,476,288 | 95,578,595,328 |
| Free device memory, bytes | 4,455,464,960 | 4,312,858,624 |

The 512-token catalog reserves more available memory for KV pages, so its final
allocated device memory is not an operator-scratch comparison. Its 163 recorded
Sparkgap source files, 8,159 FlashInfer files and installed extension all match
the post-run source scan. Full plan differences and both HTTP file indexes are
referenced from the compact receipt.

## Autoregressive replay floor

The captured decode graphs run 255 consecutive steps on one native CUDA stream,
with preallocated pages and fixed packed lanes. Timing covers native enqueue
through synchronized completion, including 255 driver graph-launch calls. It
excludes prefill, uploads, per-token completion downloads, scheduling, cache
publication and HTTP delivery. It is a wall-clock replay measurement, not a
sum of GPU kernel durations.

Each prefill configuration passes all **15 groups** of final-record and
bytewise state comparisons against isolated Python replay: **1,645 tensor
comparisons per configuration**, including live/candidate
hybrid rows, request state, touched QSA cache pages/rings and final logits.
All 30 timed blocks preserve PyTorch CUDA allocation/free and segment counters.
Within each configuration, isolated replay and HTTP produce the same complete
256-token sequence for all **35 decode requests**. Queued native replay overwrites its completion
frame and verifies final records/state; it does not independently record each
intermediate token ID. Resident admission state is reset after native drain,
and the same prepared model subsequently starts its HTTP server successfully.

Within each prefill configuration, cold HTTP and warm replay preparations select
identical configurations for every recorded plan: 16,711 at limit 4,096 and
12,646 at limit 512. Their Python runtime, native and FlashInfer source records
are identical. The floor preparations find 51 and 42 selection-cache hits,
respectively, and measure no candidates.

| Prefill limit | Fixed replay batch | Median of five mean step times, ms | Minimum / maximum mean step time, ms |
|---|---:|---:|---:|
| 4,096 | 1 | 12.753973 | 12.753385 / 12.755529 |
| 4,096 | 2 | 15.059848 | 14.987038 / 15.067606 |
| 4,096 | 4 | 18.393967 | 18.033949 / 18.434353 |
| 512 | 1 | 12.763767 | 12.760755 / 12.765331 |
| 512 | 2 | 15.035038 | 14.964200 / 15.047311 |
| 512 | 4 | 18.381741 | 18.015382 / 18.422261 |

The following pairs compare **means** for the identical batch-1 token workload:
HTTP `(last core timestamp − first core timestamp) / 255` versus replay
`elapsed / 255`. They do not compare HTTP median ITL with replay mean latency.

| Prefill limit | Repetition | Native replay mean, ms | HTTP core mean, ms | Core latency excess over replay |
|---|---:|---:|---:|---:|
| 4,096 | 0 | 12.753385 | 12.766094 | 0.099654% |
| 4,096 | 1 | 12.755529 | 12.768853 | 0.104463% |
| 4,096 | 2 | 12.755275 | 12.767587 | 0.096525% |
| 4,096 | 3 | 12.753973 | 12.766556 | 0.098659% |
| 4,096 | 4 | 12.753844 | 12.765988 | 0.095213% |
| 512 | 0 | 12.760755 | 12.855417 | 0.741820% |
| 512 | 1 | 12.764844 | 12.857239 | 0.723829% |
| 512 | 2 | 12.763767 | 12.856767 | 0.728623% |
| 512 | 3 | 12.765331 | 12.856465 | 0.713917% |
| 512 | 4 | 12.763220 | 12.855630 | 0.724035% |

All ten batch-1 pairs meet the less-than-5% replay-overhead target. This ratio
is a measured latency difference, not an isolated CPU-time measurement.
Concurrency-2/4 HTTP requests have staggered admission and prefill. The 512-token
timeline below identifies their steady GPU batches; a paired unprofiled core
interval comparison must exclude those admission and terminal transitions.

## GPU graph timeline at prefill limit 512

Nsight Systems 2026.1.3 software tracing records ordinary HTTP requests with
`return_token_ids` and `return_token_timestamps` disabled. All requests reproduce
the observed baseline's complete presentation, usage and `length` finish reason;
raw token IDs are not independently observed during profiling. One captured
request group covers each requested concurrency, with 512 prompt tokens and
256 output tokens per request.

Every GPU graph executable is identified by the exact CUPTI executable ID
queried from the already-loaded Nsight library. All **792 GPU graph records**
correlate with host launches and match native per-kind and sampling-mode counters.
The batch-1 capture has 259 graph launches; the sequential, drained batch-2/4
capture has 263 and 270. The combined capture is partitioned by those exact
ordered counts, without equating client and Nsight clock origins.

Steady measurements start after the final prefill and two consecutive matching
batch-capacity decode transitions. Global decode ordinals 255 and above are
excluded. Each request begins decode with one prefill sample, so none can reach
its 256-token budget before 255 decode launches; all requests finish by length
without cancellation or failure. The two transitions clear the scheduler's
at-most-two-frame prefill transition. This retains 252/251/250 full-cohort decode
executions and excludes mixed admissions and terminal no-op launches, without
filtering by measured duration.

| GPU batch | Steady executions | GPU graph median / p99, ms | Inter-graph gap median / p99, µs | Gap without other observed GPU work median / p99, µs |
|---|---:|---:|---:|---:|
| 1 | 252 | 12.778950 / 12.916197 | 16.288 / 18.816 | 12.928 / 15.392 |
| 2 | 251 | 15.107330 / 15.258882 | 16.192 / 18.528 | 13.088 / 15.360 |
| 4 | 250 | 18.524870 / 18.779333 | 16.352 / 18.592 | 13.152 / 15.232 |

Graph time is the complete GPU graph envelope, including internal dependencies.
The remaining-gap column subtracts the union of other observed GPU activity in
the same process/device. It does not establish device-wide idle time or attribute
all remaining time to CPU work. Nsight reports possible dropped CUDA events in
both software exports. Exact graph counts establish graph-launch completeness;
they do not establish completeness of every copy, event or API record.

Steady submissions retain six host-to-device input copies totaling 80 bytes and
one 256-byte completion download per step; page-table updates add occasional
4-byte uploads. Capture windows include boundary effects in their exact copy
counts. For batches 2/4, median H2D overlap with inter-graph gaps is 2.752/2.912 µs
and D2H overlap is 0.480/0.512 µs. Per-direction overlaps are not additive when
activities overlap. Captured dependency APIs include H2D/D2H copies, two event
records and a stream wait per step.

| GPU batch | Traced `cuGraphLaunch` median / p99, µs | Recorded `cuEventQuery` calls | Sum of traced query API durations, s |
|---|---:|---:|---:|
| 1 | 929.356 / 1,215.988 | 6,524,058 | 1.842 |
| 2 | 933.825 / 1,058.116 | 7,920,631 | 2.258 |
| 4 | 936.916 / 1,201.596 | 9,948,355 | 2.904 |

These API durations include profiler instrumentation and cannot replace the
unprofiled replay-floor measurements. All exported event queries report
`returnValue=0`; that field is not treated as evidence that every poll completed.
The analyzer retains aggregate query/synchronization counts and durations rather
than millions of individual poll records. Dependency records remain explicit.
There is one profile capture per batch, so repeat consistency has not been measured.

Both profile processes use all 12,646 configurations from the 512-token cold
preparation, with 42 selection-cache hits and no measured candidates. Python and
FlashInfer runtime source hashes match. Both load the baseline native extension
`7a1d8469…`; the separately prepared TCP_NODELAY Rust source edits were not built
or installed for these runs.

The retained hardware-tracing export lacks graph/kernel GPU activity tables despite
259 matching host graph-launch calls. It supplies no GPU timing result. A second
collection in one software-tracing process loses its Nsight agent connection;
the batch-2/4 evidence uses a separate process with one collection. Analyzer logs
preserve the SQL-variable-limit failure and an incorrect-PID invocation before
the successful bounded analysis. Their raw files remain indexed alongside the
completed results.

## Software Nsight timeline at prefill limit 4,096

One completed software-tracing range in PID 3391293 contains three drained HTTP
groups, in batch 1/2/4 order. All **791** graph executions have exact CUPTI
manifest identity and correlated host launches; their per-kind and sampling-mode
counts match native metrics. Group counts are 259/263/269. Batch 4 combines two
admissions into one `init2` graph, accounting for its one-launch difference from
the 512-token trace. All 43 model graphs are prepared, but these 512-token prompts
select `prefill512_greedy` before decode.

The same budget-based steady selection retains 252/251/250 graphs: after the
last prefill, omit two matching-capacity transitions and keep global decode
ordinals below 255. Early, mismatched-capacity and potentially terminal work is
excluded without filtering on duration.

| GPU batch | Steady graphs | GPU graph median / p99, ms | Inter-graph gap median / p99, µs | Gap without other observed GPU work median / p99, µs |
|---|---:|---:|---:|---:|
| 1 | 252 | 12.761 / 12.896 | 15.040 / 18.944 | 12.256 / 15.616 |
| 2 | 251 | 15.146 / 15.288 | 16.256 / 18.720 | 13.280 / 15.424 |
| 4 | 250 | 18.536 / 18.822 | 16.480 / 19.296 | 13.312 / 15.424 |

The usual per-step transfers are six H2D copies totaling 80 bytes and one
256-byte D2H completion copy, plus page-table updates when needed. Median H2D
overlap with inter-graph gaps is 2.624/2.688/2.944 µs for batches 1/2/4;
D2H overlap is 0.416/0.480/0.512 µs. Two event records and one stream wait occur
per step. Copy-direction overlaps must not be summed when they overlap each
other; the gap metric subtracts their union.

| GPU batch | Traced `cuGraphLaunch` median / p99, µs | Recorded `cuEventQuery` calls | Sum of query API durations, s |
|---|---:|---:|---:|
| 1 | 899.740 / 1,011.555 | 6,552,384 | 1.856 |
| 2 | 927.224 / 1,135.116 | 7,883,957 | 2.289 |
| 4 | 933.910 / 1,085.005 | 9,989,236 | 2.958 |

Host API durations include profiler overhead. Exported event-query return codes
are not interpreted as completed polls. Nsight again warns that some CUDA
events might be missing: exact graph counts establish graph completeness, not
completeness of every API or copy activity. Gaps without other observed work do
not prove device-wide idleness or CPU causation. There is one capture per batch
at each limit; cross-limit agreement does not substitute for repeated identical
captures.

All 16,711 selected configurations match the 4,096-token cold preparation, with
51 selection-cache hits and no measured candidates. The 31 Python runtime files,
8,159 FlashInfer files and installed native 7a1d8469… extension match the cold
run. Source manifests explicitly retain the uninstalled two-file TCP_NODELAY
change. The profiled requests omit raw-ID/timestamp instrumentation; exact
request controls, complete presentation and usage match the observed baseline,
but this does not supply a second profiled raw-token trace.

## Preparation and evidence

For prefill limit 4,096, with an empty b12x selection cache and retained compiled-kernel caches,
process-to-listener readiness takes **175.300 seconds**, including **4.264
seconds** of source fingerprinting. The batched tuning call takes **20.210
seconds**, measures **1,883 candidates**, and records **zero selection-cache
hits** across 16,711 declarations. The report records zero parent CuTe/Triton
compilations. Compiled-cache and filesystem cold-start cost are not measured.

The HTTP baselines, replay floors and Nsight captures load the native extension
with SHA-256
`7a1d846930ca76f2c86ffe6ea8ce49c4a5d10ecf6893206519248d4b2fe23cc4`.
Both working trees were dirty: Sparkgap HEAD was
`9d70e7f25310df4957a953f6000b089c7a4c60c2` and FlashInfer HEAD was
`98fb87a461183f98d6a7049ca947fb136cc6fb59`. The launch receipt records every
source file's hash, so the HEADs alone do not identify the executed source.

An independent post-run scan verifies all 31 recorded Python package files,
42 native source/build files, 8,159 FlashInfer files and the installed native
extension unchanged. Of all 160 Sparkgap files in the launch manifest, 158 are
unchanged. Only `qualify/performance_server.py` and `qualify/replay_floor.py`
differ; their before/after hashes are retained. The measured HTTP process had
already loaded its preparation wrapper and had `replay_corpus=null`.

The HTTP receipt itself reports unchanged client source and launch/memory JSON
hashes, plus the same server process throughout. An independent collector
reconstructs all 64 requests' raw IDs and token timestamps, verifies every
reported metric distribution and all 41 group decode spans/rates, and indexes
280 raw HTTP files by size and SHA-256. Full SSE, requests, observations,
metrics, source maps and hardware records remain under
`/data/sparkgap-artifacts/performance/`; the repository retains the compact
[receipt and artifact references](serving-performance.json).

The replay-floor attempt recorded in `sg-4096-replay.log` fails during
final-state byte comparison: a singleton strided int64 view retains stride
14,456,032 and cannot be reinterpreted as uint8. It supplies no completed
replay-floor result. Its inputs, launch and log remain preserved alongside the
successful packed-readback results above. An external comparison watcher also
read an incremental vLLM receipt before its final observations existed and
raised `KeyError: 'observations'`. That failed comparison receipt is retained;
both completed HTTP backend comparisons pass. Complete numerical acceptance
and unprofiled full-cohort batch 2/4 mean-to-mean floor attribution remain open.
