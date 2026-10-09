# Shared hyperconnection priming and measured KV budgeting

The 4,096-token graph catalog prepares all hyperconnection operators using the
decoder's existing zero-initialized residual storage. Each mixer borrows a view
matching its token capacity. Private autotuning fixtures retain seed 65537,
their original shapes and strides, and the checkpoint normalization gain.

There are 97 mixers per decoder and 8,152 aggregate rows across nine prefill
and three decode capacities. Removing their separate BF16 `[tokens,10240]`
priming allocations eliminates **16,194,437,120 bytes (15.08 GiB)** by source
accounting. The borrowed storage remains separate from every scratch output;
preparation callbacks retain its owner, and decoder execution begins after
priming and tuning finish.

The exact checkpoint is
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`, revision
`dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`. The actual main CLI on GPU
`GPU-afd4b1ad-8a64-7057-4bde-241822724c7f` uses its default 4,096-token limit,
context 262,144 and 4 GiB headroom. It captures **43 graphs**, allocates **8,154
KV pages / 521,856 cached-token capacity**, and passes **98/98 HTTP checks**.
All twelve cold/restored responses have the same content, finish reason and
usage as the saved [4,096-token baseline](prefill-capacity.md), with byte-identical
request payloads. Prompt lengths are 1,024, 2,048, 4,043, 4,096, 4,113 and 8,209.

[Results and provenance](hyperconnection-priming.json) ·
[Evidence archive](/data/sparkgap-artifacts/validation-receipts/hyperconnection-priming.tar.gz)

## Preparation memory

| Measurement, bytes | Separate priming buffers | Shared residual priming |
|---|---:|---:|
| Allocated before tuning | 98,209,292,800 | 82,010,835,456 |
| Allocated before trial release | 99,478,464,512 | 82,695,086,080 |
| Allocated after trial release | 82,284,889,600 | 82,251,614,208 |
| Freed during trial cleanup | 17,193,574,912 | 443,471,872 |

The measured pre-tuning reduction is 16,198,457,344 bytes. It exceeds the removed
priming-buffer payload by 4,020,224 bytes; that difference is not attributed to
the shared-source change. The baseline records 43 selection-cache hits and 208
measured candidates; shared priming records 51 hits and zero measured candidates.
Both collect 14,983 declarations. These counters bracket preparation stages;
they do not measure peak memory or isolate startup latency.

## KV allocation headroom

KV budgeting uses actual device consumption before rebinding any graph. An
automatic capacity that consumes configured headroom is discarded after a
device fence; preparation releases the rejected pools and unused allocator
cache, reduces the capacity using the measured deficit plus the 64 MiB
allocation allowance, and allocates again. Explicit page counts fail if they
cannot fit. Final graph capture retains its independent headroom check.

Two failed startup executions are preserved. The instrumented failure isolates
the deficit to allocation of the 36 KV tensors:

| Condition | Free device bytes | Allocated bytes | Reserved bytes |
|---|---:|---:|---:|
| Complete sizing catalog, before KV allocation | 17,657,036,800 | 82,251,658,240 | 83,152,076,800 |
| 8,196 KV pages allocated | 4,293,984,256 | 95,545,701,376 | 96,515,129,344 |
| Final catalog with 8,196 pages | 4,293,984,256 | 95,545,701,376 | 96,515,129,344 |

The logical KV payload is 13,294,043,136 bytes; reserved memory grows by
13,363,052,544 bytes. The 69,009,408-byte overhead exceeds the fixed 64 MiB
allowance, leaving free memory 983,040 bytes below the required 4,294,967,296.
Final capture adds no further consumption in this execution.

Automatic sizing reproduces the same rejected 8,196-page allocation, releases
it, and selects 8,154 pages. Preparation finishes with **4,419,813,376 free
bytes**, above the unchanged headroom requirement. No request, graph capacity,
numerical tolerance or explicit page count is reduced by this sizing policy.

## Verification and limits

- Complete preparation/recipe CPU suite: **271 passed, 54 CUDA skips**. Focused
  tests verify borrowed-buffer lifetime, unchanged private fixtures, rejected
  storage aliases, release before KV retry, and explicit-capacity failure.
- Exact-checkpoint hyperconnection: **16 native graph comparisons** across
  capacities 1, 4, 16 and 64, seed 1607, unchanged `rtol=.02`, `atol=.02`.
  Maximum relative L2 is 0.00247705. `torch.cuda.memory_allocated` stays unchanged
  over each native replay.
- HTTP: **98/98 checks**, 31 HTTP requests and twelve model requests, with
  configured chunk counts, prefix restoration and reconciled resources.
- Native binary `0dcc9ca245ed83bcbf7c0ea8147640a09c3ea2f7de6a8e6bc0559826058a0dca`
  and mapped sampler `1a17a3858b3dc8bdf189ccaff28dbccb1e10c2472b36fee035b4d6b5fd453757`
  retain their qualified identities. Sources remain unchanged throughout the
  successful server execution. SIGTERM exits zero; no listener or GPU compute
  process remains, and GPU usage returns to 2 MiB.

Independent model numerical accuracy remains unqualified. The complete-catalog
device-state parity and strict prefix bitwise receipts retain their own source
snapshots. The shared-priming HTTP receipt does not replace either check.

The archive retains all 216 payloads with SHA-256 and length metadata, including
failed startup logs, before/after sources, test logs, requests, responses,
memory reports, mapped sampler and shutdown receipt. Independent readback
verifies every member. Archive SHA-256:
`1cdb033189fd42c01085cd6ce992114ad61208189de7614a52690116743c450d`.
