# Four-image aggregate vision capacity

The exact Qwen checkpoint completed three HTTP requests containing four distinct
2048 × 2048 RGB images each. All 30 checks passed, including four vision graph
launches per request, canonical JSON token usage, strict cold/repeated JSON/SSE
output equality, and resource conservation. The qualification ran on
2026-10-09 from 08:45:52 to 08:46:01 UTC.

The [result receipt](/data/sparkgap-artifacts/validation-receipts/server-http-vision-aggregate.json) contains every assertion,
response, metric delta, canonical token ID, mRoPE position, and source hash. Its
SHA-256 is `c499acb9b8a1e26b5abe29db0aeb7b1691f9304fa4cf7f4a6818789ad31dee51`.
The [artifact archive](/data/sparkgap-artifacts/validation-receipts/server-http-vision-aggregate-raw.json) preserves 26 files
byte-for-byte, including all PNGs, generated requests, raw HTTP/SSE responses,
processor configuration, both client scripts, and the server memory report.
Its SHA-256 is `379a66732e4318451df05472a7164cb0e67c3193c198b0c4c1b6e04bb2aee775`.
Every archived byte count and hash was verified against its original file.

## Model and process

The server loaded
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD` at revision
`dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`. Requests used
`http://127.0.0.1:38761`; PID 563919 retained its process identity and mapped
native extension throughout the qualification. The extension SHA-256 was
`735bf767999b26affeacc5ae574b6e52c6a5e1dc18f42cc208ee8b9f0b2a18ab`.
The server selected GPU UUID `GPU-afd4b1ad-8a64-7057-4bde-241822724c7f` and
`CUDA_HOME=/opt/cuda`. The CPU client disabled CUDA initialization.

The server memory report declared 25 captured graphs, four request slots,
9,795 KV pages, and 37 cached-checkpoint slots. Native and Python model sources
were hashed before the requests and verified unchanged afterward. The
[single-image and grammar receipt](server-http-vision-grammar.json) identifies
the same native extension and memory report, with BF16 GDN intermediates.

## Input and canonical geometry

Each image consists of a 128 × 128 array of RGB blocks expanded into 16 × 16
pixel squares. For block coordinates `(x, y)` and seed `s`, the channels are
`(3*x + 5*y + 17*s) % 256`, `(11*x + 7*y + 29*s) % 256`, and
`(13*x + 19*y + 31*s) % 256`. Seeds are 13, 29, 47, and 71. PNG decoding was
checked against the generated RGB bytes. The images retain their full spatial
resolution and require no resize.

| Image seed | PNG bytes | Canonical grid | Patches | Features | Feature offset |
| --- | ---: | --- | ---: | ---: | ---: |
| 13 | 24,396 | `[1,128,128]` | 16,384 | 4,096 | 0 |
| 29 | 24,290 | `[1,128,128]` | 16,384 | 4,096 | 4,096 |
| 47 | 24,413 | `[1,128,128]` | 16,384 | 4,096 | 8,192 |
| 71 | 24,364 | `[1,128,128]` | 16,384 | 4,096 | 12,288 |

Pinned Transformers commit `4cc2aa84301c9aa210b5513dab9fefae03981f6e`,
Torch `2.13.0+cu130`, torchvision `0.28.0+cu130`, Pillow `12.3.0`, and NumPy
`2.5.3` produced 65,536 patches with pixel tensor shape `[65536,1536]`,
16,384 image features, and 16,412 prompt tokens. The cumulative patch offsets
were `[0,16384,32768,49152,65536]`; mRoPE delta was `-16128`. The result receipt
preserves full token IDs, modality IDs and mRoPE positions, plus hashes of the
pixel tensor, vision positions, and interpolation indices and weights.

The JSON request body was 130,564 bytes and the SSE request body 130,563 bytes,
both below the server's 32 MiB limit. These coordinate-pattern inputs exercise
aggregate capacity and indexing; they have no caption-quality oracle.

## Observed execution

All requests used temperature 0, seed 1234, a 32-token output limit, and disabled
thinking. There were three model requests and eight health, model-list, or
metrics requests.

| Request | Vision graphs | Prefill graphs | Decode graphs | Reused tokens | Finish |
| --- | ---: | ---: | ---: | ---: | --- |
| Cold JSON | 4 | 33 | 32 | 0 | `length` |
| Repeated JSON | 4 | 1 | 32 | 16,384 | `length` |
| Repeated SSE | 4 | 1 | 32 | 16,384 | `length` |

Each model request was admitted and completed once, with no failure or
cancellation. The cold request missed the prefix cache and published one
checkpoint. Each repeated request hit that checkpoint and submitted one
restore graph. Both JSON responses reported 16,412 prompt tokens, 32 completion
tokens, and 16,444 total tokens. SSE terminated with one `length` event and one
`[DONE]` marker, with no incomplete event or client error. Native SSE does not
emit usage counts.

Content and finish reason were identical across all three responses. The
content ended at the configured token limit:

```text
Based on the visual information, here is a brief description of the four coordinate-pattern images:

All four images display a similar abstract pattern composed of smooth, diagonal
```

The final metrics showed zero active, pending, retiring, frontend, or HTTP
requests and four free request slots. KV page ownership was conserved:
9,721 free + 74 cached before, and 9,465 free + 330 cached after, each totaling
9,795. Snapshot ownership was also conserved: 33 free + 4 cached before,
32 free + 5 cached after, each totaling 37. The retained checkpoint owns the
additional 256 pages for its 16,384-token prefix.

Client elapsed times were 5.262, 1.794, and 1.812 seconds respectively, under a
300-second absolute deadline per HTTP request. These observations do not
constitute performance acceptance measurements.

Feature offsets follow the canonical geometry and the native
[`pack_vision` contract](../../native/src/model/inputs.rs). HTTP does not expose
GPU upload buffers or intermediate feature values. This qualification does not
resolve the two cold/cache image-output mismatches retained in
[server-http-vision-grammar.json](server-http-vision-grammar.json), or the
independent model numerical gates.

## Reproduction

Run [qualify_vision_aggregate.py](../../scripts/qualify_vision_aggregate.py)
against an already prepared server, supplying its PID, mapped extension hash,
memory report, and exact checkpoint snapshot. Use fresh artifact and receipt
paths. The recorded command is in the receipt; original artifacts are under
`/tmp/sparkgap-http-vision-aggregate`. The generator/client SHA-256 is
`beac7459a377d3c67ac98a091c76edc3f409ab30b4e53c15e24d6994dbbc0290`.
The script preserves raw requests and responses and checks idle resource
accounting even when a model assertion fails. It neither starts nor stops the
server.
