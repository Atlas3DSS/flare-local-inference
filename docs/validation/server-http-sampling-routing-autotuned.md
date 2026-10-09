# HTTP sampler routing with autotuned prepared kernels

The unchanged saved routing fixture records **118/120 passing checks**, zero
case errors, **11 model requests** and 54 total HTTP requests. The two failures
remain `grammar-mixed.stream_complete` and `grammar-mixed.exact_literal`.
The [complete receipt](/data/sparkgap-artifacts/validation-receipts/server-http-sampling-routing-autotuned.json) preserves
both failed assertions and every request, response and metric delta.

All 11 model-request files are byte-identical to the
[fixed-QSA routing qualification](server-http-sampling-routing-fixed16.md);
all 120 check names and their order are unchanged. The saved `fixtures.json`
SHA-256 is
`b8f529f1522343d88e3b97874748f1a1c300ca08e0e37c3f30dd0b4853a56f6e`.
The qualifier SHA-256 remains
`53d6924f743d97c5b65c0a69801e9a330aae50c51a8ca61e0ede987acff21a25`.

| Routing contract | Observation |
| --- | --- |
| Zero-temperature cold request | Greedy prefill and decode |
| Aligned full-prompt reuse | 128 cached tokens, greedy resume, no prefill |
| Positive temperature with top-k 1 | General sampling |
| Repeated seeded full-prompt hits | Identical output at the same cached geometry |
| Mixed zero/positive-temperature requests | Both complete 64 tokens; 65 decode launches versus an unbatched minimum of 126 |
| Grammar/plain mixed requests | 63 shared general decode launches; constrained maximum one submission in flight |
| Resources, process/source/fixture identity and health | All original checks pass |

The mixed-temperature window has exactly two admissions and completions, no
failure or cancellation, and idle resources before and after. Its 64 general
and one greedy decode launches establish shared execution under the recorded
exclusive-client conditions. The grammar window also records the direct mixed
request counter; after the plain request finishes, the remaining constrained
lane uses greedy graphs.

The constrained SSE response ends with `length` after 1,024 nonempty content
deltas. Its text is **3,066 bytes**, an exact prefix of the required
**3,074-byte** JSON string, missing the eight-byte suffix ` alpha "`. The output
is byte-identical to the fixed-QSA routing result, with SHA-256
`005386c317cbf750560719a60ed54c0d11db6905ba65077f00eb0800353cba1d`.
The archive includes an independent CPU comparison of the recorded text and
original fixture. SSE fragments do not expose GPU token IDs. The earlier
[tokenization diagnostic](server-http-sampling-routing.md) retains its analysis
of budget exhaustion; the request budget, required literal and failed aggregate
result are unchanged.

The server uses main-source CLI process 1417418, the exact Qwen checkpoint
revision `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`, sequential single-key-split
BF16 GDN, fixed 16-way QSA and all 37 general/greedy graph variants. All 155
recorded main-source hashes match the qualified implementation. Native extension
SHA-256 is
`6048fb01a15dc0059debfd8ec50876891c8d6948ed62083a45a4402d6b364c9f`;
the process-mapped sampler binary is
`1a17a3858b3dc8bdf189ccaff28dbccb1e10c2472b36fee035b4d6b5fd453757`.
The [default HTTP receipt](server-http-autotuned.md) records the shared launch,
30-file FlashInfer patch, memory sizing and selection-cache conditions. The
[44-check vision/grammar suite](server-http-vision-grammar-autotuned.md) retains
its separate successful original mixed-grammar prompt.

The [evidence archive](/data/sparkgap-artifacts/validation-receipts/server-http-sampling-routing-autotuned.tar.gz) contains
all raw JSON/SSE requests and responses, exact saved fixtures and baseline input
comparison, metrics, client log, source files, native binary, process sampler
mapping/binary and build metadata. SIGTERM produces server exit 0, closes the
listener and releases the GPU allocation; the final log, shutdown receipt and
unchanged source/mapping checks are included. Archive SHA-256:
`ab3f0e89a41d39ba832984d663b306d53d7b579799be79a1201ab9fcaa1d305f`.
All 833 payload members were re-read and verified against `MANIFEST.json`.
The qualifier exits 1 for its two preserved assertions. Inference latency,
independent canonical numerical accuracy, and changed-budget structured-output
behavior are outside this qualification.

For reproduction, first extract `artifacts/fixtures.json` from the archive into
a fresh artifact directory. Do not generate another nonce. Use the PID and memory
report of the actual prepared server:

```bash
ARTIFACTS=/tmp/sparkgap-sampling-routing-autotuned-reproduction
mkdir "$ARTIFACTS"
tar -xOf /data/sparkgap-artifacts/validation-receipts/server-http-sampling-routing-autotuned.tar.gz \
  artifacts/fixtures.json > "$ARTIFACTS/fixtures.json"
.venv/bin/python scripts/qualify_sampling_routing.py \
  --base-url http://127.0.0.1:38761 --server-pid "$SERVER_PID" \
  --memory-report /tmp/sparkgap-autotuned-server-memory.json \
  --artifacts "$ARTIFACTS" --output "$ARTIFACTS.json"
```
