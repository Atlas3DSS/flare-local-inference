# SM70 benchmarks (Tesla V100 32 GB)

Measurements for the SM70 port described in [README-SM70.md](README-SM70.md).
End-to-end numbers and decode-only numbers are reported in separate tables. Do
not compare a number from one table with a number from the other.

- **End-to-end** numbers include prefill, scheduling and every request in the
  workload.
- **Decode-only** numbers are per-request decode rates (tokens per second while
  the request is decoding) or GPU step times. They exclude prefill and queueing.

## Package f21b9abb (512-token prefill chunks) on server binary 48b69ed4

Measured on 2026-10-24 from 02:32 to 03:27 PDT in two benchmark
chains that ran at the same time on two Tesla V100 PCIe 32 GB cards: one with
the default prefill ordering and one with `--prefill-srpt`. Each chain ran a
prefill/TTFT leg, a short-prompt leg and a long-context leg. Receipts are in
[docs/sm70/receipts/g128/](docs/sm70/receipts/g128/); each table names its
file. The previous package's results of 2026-10-23 follow in their own section
and are repeated in the tables here for comparison.

- **Server binary.** SHA-256
  `48b69ed46f5183ff0b62f1906b5e80819e370408f035a3643be4db68c83d7342`, built
  from `4528c220`: the release Rust sources plus the opt-in prefill ordering.
  The previous package was measured with the release binary `d4ffb1fa`.
- **Package.** Primary SHA-256
  `f21b9abbe3e9d5a1c033e5fccdcdd0b5317c295af52345c983ad1264534c5d0d`, prepared
  with 512-token prefill chunks from production source identical to
  `4acbecf0`, and qualified on 2026-10-24
  ([validation record](docs/sm70/ARCHITECTURE.md#export-qualification-of-package-f21b9abb)).
  Its coding-token goldens and export-accuracy logits are bit-identical to the
  previous package's.
- **Server flags.** The previous package's flags, unchanged; the ordering chain
  adds `--prefill-srpt`. The KV pool is 124,928 tokens, 976 pages of 128
  tokens, and the server reports it as `limited by memory`. The 512-token
  prefill graphs take more device memory, so the pool stops 6,656 tokens
  (5.1 %) below the 131,584-token `--kv-cache-limit concurrency` cap that the
  previous package reached ([default-short-server.log](docs/sm70/receipts/g128/default-short-server.log), [default-short-native-depth4.json](docs/sm70/receipts/g128/default-short-native-depth4.json)).
- **Hardware.** One Tesla V100 PCIe 32 GB per chain, used only by that chain.
  Other jobs on the host could share CPU, memory and PCIe bandwidth
  ([default-chain-identity.txt](docs/sm70/receipts/g128/default-chain-identity.txt), [srpt-chain-identity.txt](docs/sm70/receipts/g128/srpt-chain-identity.txt), [srpt-public-tool-identity.txt](docs/sm70/receipts/g128/srpt-public-tool-identity.txt)).
- **Inputs.** The public prompts of the previous package's rows.

### Prefill and time to first token (end to end), package f21b9abb

The previous package's protocol: fixed prompt lengths of 2,048, 8,192, 22,528
and 30,720 tokens at concurrency 1 and 4, 1 warmup batch and 3 measured
repetitions per cell, and 32 output tokens per request. TTFT runs from client
POST start to the first SSE chunk carrying output. The effective rate is the
server's prefill tokens over the batch's first-token wall window. Pure-GPU
prefill time is `null`, as before. Every request was uncached
([default-prefill-summary.json](docs/sm70/receipts/g128/default-prefill-summary.json)).

| Prompt tokens | Concurrency | TTFT p50 / p95 (s) | Effective prefill tok/s | Previous package: TTFT p50 / p95 (s) | Previous package: tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 1 | 2.06 / 2.06 | 994 | 2.54 / 2.54 | 807 |
| 2,048 | 4 | 7.82 / 8.60 | 954 | 9.89 / 10.30 | 796 |
| 8,192 | 1 | 8.53 / 8.64 | 961 | 10.43 / 10.44 | 785 |
| 8,192 | 4 | 33.95 / 34.70 | 945 | 41.44 / 41.88 | 782 |
| 22,528 | 1 | 25.08 / 25.12 | 898 | 30.39 / 30.39 | 741 |
| 22,528 | 4 | 99.92 / 100.87 | 893 | 121.11 / 121.67 | 741 |
| 30,720 | 1 | 35.46 / 35.50 | 866 | 42.75 / 42.75 | 719 |
| 30,720 | 4 | 141.38 / 142.36 | 863 | 170.49 / 170.95 | 719 |

At one request the effective rate is 23.1 % higher than the
previous package's at 2,048 tokens and 20.5 % higher at
30,720 tokens. Concurrent prefills still do not overlap. The server interleaves
the four requests' 512-token chunks, so all four first tokens arrive within
2.7 s of each other, at about four times the
single-request TTFT. Greedy outputs of three short prompts (64 tokens) and one
28.7k-token prompt (32 tokens) equal the previous package's on the release
binary, token for token, on 4 of 4 prompts in both chains ([default-prefill-greedy.json](docs/sm70/receipts/g128/default-prefill-greedy.json), [srpt-prefill-greedy.json](docs/sm70/receipts/g128/srpt-prefill-greedy.json)).

#### With `--prefill-srpt`, package f21b9abb

The ordering chain ran the same protocol on the same package with
`flare run --prefill-srpt` ([srpt-prefill-summary.json](docs/sm70/receipts/g128/srpt-prefill-summary.json)). The default columns repeat
the table above.

| Prompt tokens | Concurrency | TTFT p50 / p95 (s), `--prefill-srpt` | TTFT p50 / p95 (s), default | Effective prefill tok/s, `--prefill-srpt` / default |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 1 | 2.12 / 2.13 | 2.06 / 2.06 | 964 / 994 |
| 2,048 | 4 | 5.27 / 9.05 | 7.82 / 8.60 | 906 / 954 |
| 8,192 | 1 | 8.53 / 8.65 | 8.53 / 8.64 | 961 / 961 |
| 8,192 | 4 | 21.47 / 35.02 | 33.95 / 34.70 | 937 / 945 |
| 22,528 | 1 | 25.60 / 25.66 | 25.08 / 25.12 | 880 / 898 |
| 22,528 | 4 | 63.14 / 102.10 | 99.92 / 100.87 | 886 / 893 |
| 30,720 | 1 | 36.18 / 36.23 | 35.46 / 35.50 | 849 / 866 |
| 30,720 | 4 | 89.71 / 145.73 | 141.38 / 142.36 | 844 / 863 |

The two chains ran at the same time on different cards. The single-request
rows, where the option changes nothing, differ from the default chain's by up
to 3 %. At four requests the prompts finish prefill one after another. Four
30,720-token prompts get their first tokens at 36.2 / 71.9 / 107.6 / 145.1 s
(mean over the 3 measured batches, by order of arrival) instead of all at
139.7-142.4 s, and the p50 falls by
37 %. As with the previous package, the earlier requests
decode while the last prompt is still in prefill, so the last first token
arrives slightly later than with round-robin: 143.74-145.88 s
against at most 142.39 s at 30,720 tokens. At four long agents the earlier
first tokens cost decode speed, because the early finishers decode while the
others' 512-token prefill chunks run
([long-context section](#long-context-native-serving-package-f21b9abb)).

### Short-prompt native decode, MTP depth 4, package f21b9abb

The previous package's protocol: 128 output tokens per request, concurrency 1
then 4, 3 repetitions, on the four public `code-00` to `code-03` prompts of
578-582 tokens. Tok/s per agent p50 is decode-only; aggregate tok/s divides
output tokens by batch wall time and is end to end. Do not compare the two
columns ([default-short-native-depth4.json](docs/sm70/receipts/g128/default-short-native-depth4.json), [default-short-native-depth4-requests-summary.json](docs/sm70/receipts/g128/default-short-native-depth4-requests-summary.json), [srpt-short-native-depth4.json](docs/sm70/receipts/g128/srpt-short-native-depth4.json)).

| Concurrency | Chain | Tok/s per agent, p50 | Aggregate tok/s | Previous package: per agent p50 / aggregate | Repetitions |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | default | 83.2 | 69.3 | 83.0 / 67.6 | 3 |
| 4 | default | 64.9 | 220.5 | 64.8 / 220.6 | 3 |
| 1 | `--prefill-srpt` | 83.2 | 69.3 | n/a | 3 |
| 4 | `--prefill-srpt` | 64.7 | 220.4 | n/a | 3 |

In the default chain at concurrency 4 the aggregate over the decode window is
213.1 tok/s (previous package 212.9). Of the
3,072 generated tokens, 2,234 were accepted MTP
drafts, the same counts as with the previous package. 24 of 24 requests
completed. Repeated prompts reuse their prefixes, so first-token times in this
leg include prefix-cache credit and are not prefill measurements.

### Long-context native serving, package f21b9abb

The previous package's protocol: four public coding prompts of 28,654 to 28,671
tokens (114,664 in total), 512 output tokens per request, concurrency 4 first
and then 1, 3 repetitions. Each row is labelled cold or cache-credited by the
same rule, and first-token times are given for cold rows only
([default-long-native-depth4.json](docs/sm70/receipts/g128/default-long-native-depth4.json),
[default-long-native-depth4-requests-summary.json](docs/sm70/receipts/g128/default-long-native-depth4-requests-summary.json),
[default-long-native-chain-long-cache-labels.json](docs/sm70/receipts/g128/default-long-native-chain-long-cache-labels.json);
the ordering chain's files start with `srpt-long-`).

| Chain | Concurrency | Cold or cache-credited | First-token s (cold rows only) | Decode tok/s per agent |
| --- | ---: | --- | ---: | ---: |
| default | 4 | first batch: cold, 4 requests | 123.65 / 125.49 / 138.65 / 142.08 (p50 132.07) | 27.6 / 32.6 / 53.6 / 63.4 (median 43.1) |
| default | 4 | batches 2 and 3: cache-credited, 5 of 8 requests credited | n/a | 21.8-77.1 (median 38.2) |
| default | 1 | cold, 5 requests | 33.03-33.41 (p50 33.04) | 64.9-77.0 (median 72.9) |
| default | 1 | cache-credited, 7 requests | n/a | 62.9-80.8 (median 74.1) |
| `--prefill-srpt` | 4 | first batch: cold, 4 requests | 33.45 / 66.72 / 99.77 / 145.80 (p50 83.24) | 6.5 / 6.5 / 10.6 / 70.3 (median 8.5) |
| previous package | 4 | first batch: no request credited, batch not cold | 157.96-159.80 | 40.2 / 41.1 / 42.7 / 46.1 (median 41.9) |
| previous package | 1 | cold, 6 requests | 39.83-39.87 (p50 39.85) | 70.0-81.2 (median 72.4) |

A cold 28.7k-token request reaches its first token after 33.04 s (p50),
against 39.85 s with the previous package, consistent with the prefill table.
In the default chain's cold four-agent batch the first tokens arrive between
123.65 and 142.08 s (p50 132.07 s), against 157.96-159.80 s for the previous
package's first batch, which is not labelled cold. The four decode windows
overlap only partly: at 141.16 s, when the first request finished, two others
were decoding and the fourth was 0.92 s from its first token.

With `--prefill-srpt` the cold four-agent batch gets its first tokens one after
another, at 33.45, 66.72, 99.77 and 145.80 s (p50 83.24 s against 132.07 s by
default). The cost is decode speed: a request that has finished prefill decodes
while the others' 512-token prefill chunks run, so in that batch the three
early finishers decoded at 6.5-10.6 tok/s. Over all 12 four-agent requests the
per-agent decode median is 21.0 tok/s with the option and 38.2 tok/s by
default (previous package 41.5). At one agent the medians over 12 requests are
73.5 tok/s by default and 71.1 tok/s in the ordering chain (previous package
72.4).

Over the whole default long leg the server counted 344,458 prefill tokens and
12,288 generated tokens, of which 9,097 were accepted MTP drafts. All 24
requests completed in each chain. The server exports no KV-page eviction
counter; the default long leg recorded 43 prefix-cache snapshot-slot evictions
(`flare_snapshot_evictions_total`), which are snapshot slots, not KV pages.
Four 28.7k-token contexts fit in the 124,928-token pool; four contexts at the
32,768-token maximum (131,072 tokens) would not, and that case was not
measured.

### Refined prefill ordering (opt-in, not in the release binaries)

The opt-in ordering above gets four long prompts their first tokens earlier,
but it slows the decoding of requests that have finished prefill (21.0 against
38.2 tok/s per agent). The cause is in how the shipped option prices a prefill
step: it values the step like the best-priced waiting prompt. Under
shortest-first ordering the waiting prompts are not served, so their price
keeps rising and every prefill step outvalues decoding. A request that has
finished prefill then gets about one speculative round per 512-token chunk
(about 0.52 s), which is the 6.5 tok/s seen above. Its boundary snapshot step
loses the same way. Separately, a prompt whose prefill has started is not
preempted, so in mixed traffic short prompts wait behind a started long prompt
([LIMITATIONS.md](docs/sm70/LIMITATIONS.md#not-validated-or-validated-only-in-part)).

This tree adds two knobs that refine the option. Both act only with
`--prefill-srpt`, both default to 0, and 0 keeps the shipped behaviour:

- `--prefill-srpt-short-tokens S` (configuration key
  `prefill_srpt_short_tokens`): a prompt with at most S uncached tokens left
  takes one prefill chunk ahead of the prompt being served, alternating chunk
  by chunk, and its first step is not held behind prompt chunks. The served
  prompt still gets at least every other chunk, so it finishes within twice its
  own chunk count, and a prompt with more than S tokens left never preempts it.
- `--prefill-srpt-decode-priority-tokens B` (configuration key
  `prefill_srpt_decode_priority_tokens`): while a request that has emitted
  fewer than B tokens is decoding, or in the boundary snapshot that resumes its
  decoding, no prompt chunk is offered. Each request therefore delays waiting
  prompts by at most one boundary snapshot plus B output tokens. A non-zero B
  also values each prefill step by the prompt that runs. When nothing else is
  runnable, prefill runs, so the GPU is never held idle.

As for the other prefill-ordering options, a flag given on the command line
overrides the configuration file's key. With both knobs at 0, a Helios test
replays the shipped option's schedules (default, `--prefill-srpt`, and
`--prefill-srpt` with an aging bound of 16 chunks) and requires every committed
step and outcome to match. On the GPU, a lab build of this change with both
knobs at 0 reproduced the first tokens of four 30,720-token prompts: 127.39 /
137.29 / 141.61 / 142.52 s without a flag and 35.78 / 72.09 / 107.59 /
143.11 s with `--prefill-srpt`, with greedy outputs equal to the previous
package on 4 of 4 prompts in both runs
([equiv-off.json](docs/sm70/receipts/prefill-srpt-r2/equiv-off.json),
[equiv-srpt.json](docs/sm70/receipts/prefill-srpt-r2/equiv-srpt.json)).
Helios library tests pass, 810 with 19 ignored, and `flare` 104 + 4 + 4
([public-helios-lib-tests.log](docs/sm70/receipts/prefill-srpt-r2/public-helios-lib-tests.log),
[public-flare-tests.log](docs/sm70/receipts/prefill-srpt-r2/public-flare-tests.log)).
The scheduler change does not touch arithmetic, graphs or chunk cuts. It was
reviewed independently (Argon, `gemini-4-argon`) and cleared in two rounds.

The configuration measured here, Y, is `--prefill-srpt
--prefill-srpt-short-tokens 2048 --prefill-srpt-decode-priority-tokens 4096`:
up to four 512-token chunks count as short, and the decode priority covers the
512-token and 2,048-token outputs of these runs. It ran on 2026-10-24 with a
lab build of this change (`83b71a1c`, the sources of `48b69ed4` plus the
change; this commit differs from it only in a configuration-parsing helper
with the same parsing and in the name of the test module), package `f21b9abb`
and the release server flags, on one Tesla V100 PCIe 32 GB per run. The
default and shipped-option columns repeat the package `f21b9abb` chains above,
which ran earlier on other cards.

| Measurement, package `f21b9abb` | Default | `--prefill-srpt` | Y |
| --- | ---: | ---: | ---: |
| Prefill, 1 request, 2,048 / 30,720 tokens (tok/s) | 994 / 866 | 964 / 849 | 965 / 867 |
| Four 30,720-token prompts, TTFT p50 / p95 (s) | 141.38 / 142.36 | 89.71 / 145.73 | 89.98 / 144.21 |
| Short prompts, decode per agent at 1 / 4 agents (tok/s) | 83.2 / 64.9 | 83.2 / 64.7 | 83.3 / 64.8 |
| Long, cold four-agent batch, first tokens (s) | 123.65 / 125.49 / 138.65 / 142.08 | 33.45 / 66.72 / 99.77 / 145.80 | 33.36 / 74.30 / 114.84 / 154.19 |
| Long, cold four-agent batch, last token mean / worst (s) | 145.4 / 150.1 | 139.6 / 153.1 | 101.1 / 160.6 |
| Long, decode per agent at 4 agents, median of 12 requests (tok/s) | 38.2 | 21.0 | 73.3 |
| Long, decode per agent at 1 agent, median of 12 requests (tok/s) | 73.5 | 71.1 | 73.4 |
| Greedy outputs equal to the previous package | 4 of 4 | 4 of 4 | 4 of 4 |

Y's receipts: [y-pub-prefill-summary.json](docs/sm70/receipts/prefill-srpt-r2/y-pub-prefill-summary.json)
and [y-pub-short-native-depth4.json](docs/sm70/receipts/prefill-srpt-r2/y-pub-short-native-depth4.json)
(public prefill protocol, 1 warmup and 3 measured batches, and the short leg),
[y-long-native-depth4.json](docs/sm70/receipts/prefill-srpt-r2/y-long-native-depth4.json)
and [y-long-native-depth4-requests-summary.json](docs/sm70/receipts/prefill-srpt-r2/y-long-native-depth4-requests-summary.json)
(long leg; the first four-agent batch has no cache credit in usage and no
change in the reused-token counter), and
[y-chain-prefill.json](docs/sm70/receipts/prefill-srpt-r2/y-chain-prefill.json)
(greedy check).

In the cold long batch with Y, each agent decodes at 70-80 tok/s once its
prefill ends, and the batch's mean last token comes 38.5 s earlier than with
the shipped option. Prefill rates and the four-prompt TTFT stay those of the
shipped option, and the short-prompt rates those of the release.

The mixed-traffic soaks used seeded Poisson arrivals, one every 6 s on average
for 15 minutes, 50 % 512, 30 % 2,048, 15 % 8,192 and 5 % 30,720-token
prompts, 4 client workers, and 8 or 128 output tokens, with two seeds (157 and
154 arrivals). TTFT "from send" starts when a client worker sends the request;
"from arrival" adds the wait for a worker. Every soak completed every request
with no failure and no stall.

| Soak | Measurement | Default | `--prefill-srpt` | Y |
| --- | --- | ---: | ---: | ---: |
| 8 tokens, seed 2 | 512-token TTFT p95 from send (s) | 2.81 | 39.14 | 1.29 |
| 8 tokens, seed 2 | 8,192-token TTFT p95 from send (s) | 37.53 | 45.92 | 56.82 |
| 8 tokens, seed 2 | Mean TTFT from arrival, all requests (s) | 15.94 | 24.20 | 10.58 |
| 8 tokens, seed 2 | 30,720-token TTFT max from arrival (s) | 134.7 | 91.1 | 116.8 |
| 128 tokens, seed 1 | Mean TTFT from arrival, all requests (s) | 18.16 | 22.80 | 12.96 |
| 128 tokens, seed 1 | Decode per request, p50 (tok/s) | 38.6 | 42.5 | 79.2 |
| 128 tokens, seed 1 | 30,720-token TTFT max from arrival (s) | 138.3 | 86.4 | 200.4 |
| 128 tokens, seed 2 | Mean TTFT from arrival, all requests (s) | 27.71 | 35.27 | 23.29 |
| 128 tokens, seed 2 | Decode per request, p50 (tok/s) | 32.5 | 34.7 | 77.6 |
| 128 tokens, seed 2 | 30,720-token TTFT max from arrival (s) | 173.1 | 122.2 | 184.8 |
| 128 tokens, seed 2 | 8,192-token TTFT max from arrival (s) | 125.4 | 135.0 | 187.3 |

Receipts: `s2-*`, `soakdec-*` and `sd2-*` in
[docs/sm70/receipts/prefill-srpt-r2/](docs/sm70/receipts/prefill-srpt-r2/).
On the 8-token soak of the first seed Y itself was not run; the default gave a
512-token p95 from send of 2.72 s and the shipped option 35.90 s, against
1.05 s with short-tokens 1024 and 1.15 s with short-tokens 2048, each with a
decode priority of 1024 tokens (`soak-*-g128.json`).

**Costs.** These come with the benefits above:

- In the cold four-agent long batch, the second to fourth agents get their
  first tokens 7.6, 15.1 and 8.4 s later than with the shipped option, because
  each waits while the agents before it decode; the batch's last token arrives
  at 160.6 s, against 153.1 s with the shipped option and 150.1 s by default.
- Under decode-heavy mixed load (128 output tokens), the longest prompts wait
  longer in the worst case: the 30,720-token TTFT maximum is 184.8-200.4 s with
  Y, against 86.4-122.2 s with the shipped option and 138.3-173.1 s by default.
  On the second seed the 8,192-token maximum was 187.3 s with Y. On the
  8-token soak of the second seed, the 8,192-token p95 rose to 56.8 s.
- The option's aging bound of 256 chunks was sized when prefill ran in
  128-token chunks, where 256 chunks take about 41 s. At this package's
  512-token chunks they take about 133 s at short context, and time a prompt
  spends held behind decoding does not count toward it. A tighter bound
  (`--prefill-srpt-max-wait-chunks 64`, with short-tokens and decode priority
  of 1024) moved the tail onto 8,192-token prompts (maximum 173.5 s) and raised
  the mean TTFT from arrival from 13.56 to 20.44 s on the 128-token soak
  (`soakdec-x2-g128.json`, `soakdec-x2mw64-g128.json`), so it is not
  recommended.

The decode-priority budget matters when outputs are long. With 2,048 output
tokens on the four long prompts (short-tokens 1024 in both refined runs), the
cold batch's mean last token was 159.8 s with a budget of 4,096, 175.2 s with
1,024, 195.6 s by default and 242.5 s with the shipped option (`l2k-*`).

The refinements stay opt-in, and round-robin stays the default.

## Previous package fd7c24dc on release binary d4ffb1fa (128-token prefill chunks)

These are the results of the previous release package, which prefills in
128-token chunks, on the release binary. They are kept for comparison with the
section above. Measured on 2026-10-23 from 21:54 to 22:48 PDT in one benchmark chain on one
Tesla V100 PCIe 32 GB. Receipts are in [docs/sm70/receipts/release-binary/](docs/sm70/receipts/release-binary/); each table names its file.

- **Server binary.** SHA-256
  `d4ffb1fa3453bf37f2017a7b58d09559010881311b2d9afcdfb33b4011caa667`, built from
  `71029458`; production source is unchanged through `c6fdc120`.
- **Package.** Primary SHA-256
  `fd7c24dce58c4897a2ab5299bcd875e3e6ab1e2a1f714e5f09724bc6619a7382`, the release
  export qualified on 2026-10-23
  ([validation record](docs/sm70/ARCHITECTURE.md#export-qualification-of-the-previous-package-fd7c24dc)).
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

### Prefill and time to first token (end to end), previous package

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

### Short-prompt native decode, MTP depth 4, previous package

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

### Long-context native serving, previous package

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
  tokens. The release runs above, for both packages, used the public prompts of
  578 to 582 tokens. Their 83.2 tok/s (package `f21b9abb`) and 83.0 tok/s
  (previous package) against the best recorded 90.4 tok/s are therefore not
  like-for-like comparisons and are not regression claims.
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
the package `f21b9abb` section above gives those.

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
| Server binary `48b69ed4`, package `f21b9abb` | not measured | V100 PCIe 32 GB | 31G | release export | 4 | n/a | none |
| Release binary `d4ffb1fa`, previous package `fd7c24dc` | not measured | V100 PCIe 32 GB | 31G | release export | 4 | n/a | none |

(d) The run log names only the run tag `best24`. Its run report, compiled in
the [measurement history](#measurement-history-2026-10-20-to-2026-10-23), gives
the package `model-32k-tonight-a`, a binary `2f9128e4` built before the release
binary, MTP depth 3, `--memory-limit 31G`, a 32G prefix cache and 4 agents.
Earlier versions of this page called `best24` a different package; it is a run
of the 2026-10-22 release-tree package. No real-trace end-to-end run of the
release binary `d4ffb1fa` or of the server binary `48b69ed4` exists.

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
