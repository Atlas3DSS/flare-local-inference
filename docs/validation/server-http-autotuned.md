# HTTP qualification with autotuned prepared kernels

The unchanged default HTTP suite and its original mixed-grammar case pass
**152/152 checks**, with zero case errors, across **39 model requests** and
113 total HTTP requests. The server loads
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD` revision
`dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2` through the main repository CLI.
It uses BF16 GDN intermediates with sequential single-key-split recurrence,
16-way QSA attention, and FlashInfer sampling.

All 39 request files are byte-identical to the
[default-preparation fixed-QSA qualification](server-http-fixed16.md).
The qualifier SHA-256 is
`dacdfec000f3b74d3179cd21f54da922a039d36cbedb535aa27aed3614cf8466`;
all 152 check names and their order are unchanged. The number of metric polls
differs between runs. Every original response, SSE stream, metric scrape and
assertion is preserved in the [receipt](/data/sparkgap-artifacts/validation-receipts/server-http-autotuned.json) and
[evidence archive](/data/sparkgap-artifacts/validation-receipts/server-http-autotuned.tar.gz).

| Contract | Result |
| --- | --- |
| Canonical tool-result prompt, cold JSON versus cached SSE | Complete message and finish reason agree |
| PNG prompt, cold JSON versus cached SSE | Complete message and finish reason agree |
| Full-prompt, aligned-prefix and branched cache reuse | Original restore, suffix-graph and output assertions pass |
| Four simultaneous requests | All four active slots observed; outputs match individual baselines |
| Thinking, tools, structured literal tags, JSON and SSE | Original content, token-count and serialization assertions pass |
| Mixed constrained/unconstrained generation | Exact constrained output and plain answer; shared decode and one constrained submission in flight |
| Cancellation, subsequent admission and resource accounting | All original checks pass; final health is healthy |

The process is PID 1417418, launched from `/home/luke/projects/sparkgap` using
`.venv/bin/python -u -m serve --bind 127.0.0.1:38761` with
`CUDA_VISIBLE_DEVICES=GPU-afd4b1ad-8a64-7057-4bde-241822724c7f`,
`CUDA_HOME=/opt/cuda`, and no `PYTHONPATH` override. All 155 recorded main-source
files match the qualified implementation. Native extension SHA-256 remains
`6048fb01a15dc0059debfd8ec50876891c8d6948ed62083a45a4402d6b364c9f`.

The installed [recurrence-policy patch](recurrence-policy-install.json) applies
30 files to FlashInfer revision `98fb87a461183f98d6a7049ca947fb136cc6fb59`;
its patch SHA-256 is
`4b78c54a1f3078c2e1c67a322ada8ef8eb3223bb6db8770e5b10db7e5ed91d0e`.
All 21 installed runtime-file hashes match that receipt. The serving process's
`/proc/1417418/maps` identifies the sampler file and inode; the retained binary
has SHA-256
`1a17a3858b3dc8bdf189ccaff28dbccb1e10c2472b36fee035b4d6b5fd453757`.
The archive retains the mapping, binary, Ninja metadata and native extension.
[Sampler qualification](sampling-recurrence-policy.md) covers this binary
separately.

| Prepared resource | Value |
| --- | ---: |
| Graph variants | 37 |
| KV pages / cached-token capacity | 9,708 / 621,312 |
| Immutable snapshot slots / active request slots | 37 / 4 |
| Required GPU headroom | 4,294,967,296 bytes |
| Free GPU memory after preparation | 4,329,635,840 bytes |
| Selection-cache hits / measured candidates | 42 / 0 |
| Compiler workers / batched preparation time | 8 / 2.370500 seconds |
| Allocator-visible preparation storage released | 1,957,580,800 bytes |

The selected configurations come from completed searches under the constrained
numerical policy; this server startup reuses their cache entries. These timing
observations do not characterize cold startup or sustained inference latency.
The separate [prefix gate](prefix-replay-sequential-k1.md) and
[Python/Rust catalog parity](graph-execution-parity-autotuned.md) retain their
bitwise state and execution checks. Independent canonical text/vision accuracy
and the complete-runtime replay-floor performance bound remain unresolved.

The client exited zero. After all three HTTP suites, SIGTERM closed the server
with exit 0. Process 1417418 and the port-38761 listener are absent; GPU4 returns
to 2 MiB with no compute process. All 155 source hashes and the sampler mapping
remain unchanged. The archive contains the final server log, shutdown receipt
and final process mapping. Archive SHA-256:
`a266ae57a73dcf0daaf196d3eeb81725eca3b4eabd45af21cb560a9bc6ce9483`.
All 942 payload members were re-read and verified against `MANIFEST.json`.

With a prepared server using the recorded configuration, reproduce into fresh
output destinations:

```bash
.venv/bin/python scripts/qualify_server.py \
  --base-url http://127.0.0.1:38761 \
  --gdn-intermediate-dtype bfloat16 \
  --memory-report /tmp/sparkgap-autotuned-server-memory.json \
  --require-metrics --mixed-grammar \
  --artifacts /tmp/sparkgap-http-autotuned-reproduction \
  --output /tmp/sparkgap-http-autotuned-reproduction.json
```
