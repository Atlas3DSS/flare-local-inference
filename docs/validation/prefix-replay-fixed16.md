# Whole-model prefix replay with fixed QSA partitions

The exact 459-token tool-result prompt passed **595/595 bitwise equality
predicates across seven comparisons** with 16 QSA attention partitions at every
decoder capacity. The native executor completed 12 submissions with the expected
completion records and unchanged cumulative CUDA allocation counts. The run
prepared all 37 graph variants and completed in 178.36 seconds, including model
loading and preparation.

[Full receipt](prefix-replay-fixed16.json) and
[raw receipt, source, log, request, and CPU verification archive](/data/sparkgap-artifacts/validation-receipts/prefix-replay-fixed16.tar.gz)
preserve the original observations. CPU verification independently recomputed
all 595 predicates from the saved tensors and checked bytewise equality of the
compared regions, complete logits, and recorded QSA operands.

| Comparison | State regions | Complete logits | Layer-3 QSA tensors | Total predicates |
| --- | ---: | ---: | ---: | ---: |
| Cold 459-token prefill vs identical cold replay | 77 | 1 | 13 over 459 rows | 91 |
| Aligned boundary candidate vs immutable snapshot | 77 | — | — | 77 |
| Immutable snapshot vs restored live state | 77 | — | — | 77 |
| Boundary candidate vs independently executed 448-token prefix | 77 | — | — | 77 |
| Cold prefill vs restored 448-token prefix plus 11-token tail at capacity 16 | 77 | 1 | 13 over 11 rows | 91 |
| Cold prefill vs restored prefix plus identical tail at capacity 512 | 77 | 1 | 13 over 11 rows | 91 |
| Restored prefix plus tail vs independently split prefix plus tail | 77 | 1 | 13 over 11 rows | 91 |

Every recorded relative L2 error, maximum absolute error, and QSA differing-element
count is zero. The complete 248,320-element final logit vectors match; each has
argmax 9,419. Sampling is disabled for these submissions, so generated counts
remain zero and completion token IDs remain `-1`.

The 77 continuation-state regions comprise 36 GDN recurrent states, 36 GDN
convolution histories, PLE convolution history, last hidden vector, consumed-token
count, two-token PLE history, and rotary-position delta. Layer-3 QSA recording
compares the input, Q/K/V and index projections, normalized/rotated query and key,
attended output, selected positions, gated output, output projection, index query,
and raw index key on common logical rows. The full tensor artifact also retains
cache pages and pre/post raw-state observations; those additional tensors are
inventoried by hash rather than added to the 595-predicate gate.

## Input and implementation identity

The checkpoint is
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`, revision
`dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`. The request is structurally identical
to the preserved tool-result HTTP case and both prior prefix-replay receipts.
Canonical tokenizer rendering reproduces all 459 token IDs. Their uint32
little-endian SHA-256 is
`d787632f8fe5fdb40b299e728307651391d5f4fac13bd7d4eeeeea5beadc1cc3`.
The snapshot boundary remains 448, with full and tail graph capacities 512 and 16.

CPU verification matched all 533 recorded b12x Python source hashes, all 17 recipe
source hashes, the executed diagnostic script, and the native extension. All nine
recorded QSA capacity contracts use 16 partitions and cover all twelve QSA layers.
GDN prefill retains BF16 intermediate arithmetic. The installed QSA contract hash
is `5727e5beb75d19d4fdb51783019403e46b0b9ccb6fecb6293719f941e1ed7094`;
the native extension hash is
`6048fb01a15dc0059debfd8ec50876891c8d6948ed62083a45a4402d6b364c9f`.

The archived script checks the exact completion record and
`allocation.all.allocated` before recording each successful submission. It closes
and requires the native executor to drain before writing comparisons and the
final receipt. The run used `--require-bitwise`, which returns nonzero after saving
evidence if any recorded equality fails. The process exited zero before GPU4 was
released for the HTTP server.

## Artifacts and scope

The external tensor artifact `/tmp/sparkgap-prefix-fixed16.pt` contains 249 tensors
and is 1,229,191,869 bytes. Its verified SHA-256 is
`82bbc260582513469a930296000fba89c398d8caea79453462a53e467fe1ec76`.
The archive includes its complete tensor inventory and hashes; the large tensor
payload remains at that external path. Original JSON observations are preserved
without rescoring or tolerance changes.

The run executes `init1`, `snapshot1`, `restore1`, `prefill512`, and `prefill16`
through the native executor. Its numerical scope is whole-model continuation
state, final logits, and recorded layer-3 QSA operands for this exact text prompt
and replay sequence. Independent canonical model accuracy, generated-response
quality, the other graph variants, and HTTP scheduling retain separate acceptance
gates. Vision graphs are prepared but not submitted.

Reproduction requires the recorded installed package and recipe sources, the
pinned checkpoint, and an exclusively assigned GPU4. Choose fresh output paths:

```bash
cd /home/luke/projects/sparkgap
ARTIFACTS=$(mktemp -d /tmp/sparkgap-prefix-fixed16-replay.XXXXXX)
tar -xzf /data/sparkgap-artifacts/validation-receipts/prefix-replay-fixed16.tar.gz -C "$ARTIFACTS" request.json
env -u PYTHONPATH \
  CUDA_VISIBLE_DEVICES=GPU-afd4b1ad-8a64-7057-4bde-241822724c7f \
  CUDA_HOME=/opt/cuda .venv/bin/python scripts/diagnose_prefix_replay.py \
  --request-json "$ARTIFACTS/request.json" \
  --output "$ARTIFACTS/result.json" --tensor-output "$ARTIFACTS/observations.pt" \
  --record-qsa --require-bitwise
```
