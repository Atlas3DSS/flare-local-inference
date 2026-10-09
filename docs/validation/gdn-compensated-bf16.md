# Compensated BF16 GDN continuation diagnostic

BF16 residual corrections let the exact-checkpoint GDN test pass its initial
65-token prefill, checkpoint and following decode comparisons. The complete test
still fails on its continued 3+2-token prefill. The candidate remains isolated
from the serving stack.

The checkpoint is `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`,
revision `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`. Both executions run
`tests/recipes/test_qwen38_gdn.py::test_actual_checkpoint_prefill_decode_continuation_graphs[65]`
on GPU `GPU-afd4b1ad-8a64-7057-4bde-241822724c7f`. Seed 741, canonical Transformers
reference, request lengths, graph capacities and every assertion are unchanged.

| Output comparison | Relative L2 | Limit | Pointwise failures | Maximum absolute error |
|---|---:|---:|---:|---:|
| Packed 65+1-token prefill | 0.0120601691 | 0.015 | 0 | 0.001220703125 |
| Two-request single-token decode | 0.0090385191 | 0.015 | 0 | 0.000518798828125 |
| Continued 3+2-token prefill | 0.0155812744 | 0.015 | 3 | 0.00152587890625 |

The first eleven numerical assertions pass: initial output, both recurrent and
convolution states, the token-16 recurrent checkpoint, decode output and both
decode states. The twelfth assertion fails on continued-prefill output. Its
pointwise tolerance is `rtol=.025`, `atol=.000846947265625`. Continued-state and
zero-active assertions do not execute after this failure. All three executed
graph replays preserve `allocation.all.allocated`.

For an FP32 intermediate, the candidate stores BF16 high and residual terms:
`hi=BF16(x)`, `lo=BF16(x-FP32(hi))`. The recurrence uses both terms for V′ and U,
adds first-order residual products to the intra-chunk attention result, and
includes the U residual in the persistent FP32 state update. Normalized Q/K,
decayed Q/K, triangular inverse and BF16 state shadow retain their existing
arithmetic. Every matrix contraction uses BF16 tensor cores with FP32
accumulation; kernel launch count is unchanged.

An explicit `GdnWorkspaceRecord.MQK_LO` plane grows each GDN record by 512 bytes.
Workspace sizing, shared-memory checks, copy extents and candidate/cache identities
include the layout. KDA retains its existing layout and arithmetic. The exact
65-token configuration adds 393,216 ring-workspace bytes and 1,536 shared-memory
bytes. Source counting gives 22 additional MMA instructions per unsplit-key
value-group warp/token tile over 52 existing instructions; latency is unmeasured.

Both GPU executions produce identical twelve-comparison metric records. The
extended recorder retains source, scratch, metadata, native state and canonical
state at each step after replay and the allocation assertion, then calls the
original numerical assertion. It records all operands through the failure;
the installed b12x sources remain unchanged.

[Machine-readable results](gdn-compensated-bf16.json) and the
[evidence archive](/data/sparkgap-artifacts/validation-receipts/gdn-compensated-bf16.tar.gz) retain the four-file patch, baseline
and candidate sources, CPU layout checks, both GPU logs and receipts, the complete
step tensor artifact, test and recorders. All 30 payload hashes and lengths pass
independent verification. Archive SHA-256:
`f2b39c8f17397267b4f4ec0eea6159f94b3ace43e07f95c7039edf547986cfc3`.

Generic summary modes, other key splits, full-model accuracy and runtime latency
remain unqualified.
