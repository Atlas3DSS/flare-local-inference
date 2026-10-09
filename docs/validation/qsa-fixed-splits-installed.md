# Installed fixed-partition QSA qualification

The installed FlashInfer package passed **19/21 checks**: all nine fixed-partition
operator cases, seven of nine selected existing default-policy regressions, and
all three exact-checkpoint attention recipe tests. Two existing default-policy
tests failed before kernel launch. An isolated baseline reproduced both failures
with identical exception messages.

[Machine-readable receipt](qsa-fixed-splits-installed.json) and
[raw logs, JUnit reports, launch scripts, and source archive](/data/sparkgap-artifacts/validation-receipts/qsa-fixed-splits-installed.tar.gz)
preserve every outcome. The installation is identified by
[qsa-partitions-install.json](qsa-partitions-install.json).

| Suite | Result | Pytest time |
| --- | --- | --- |
| Fixed-partition public API and existing default-policy tests | 16 passed, 2 failed | 27.15 s |
| Exact-checkpoint attention recipe | 3 passed | 12.58 s |
| Isolated historical baseline, two unchanged failing tests | 2 failed with identical exceptions | 5.98 s |

The installed runs removed `PYTHONPATH`, imported QSA before pytest collection,
asserted its location under the repository `.venv`, and checked the installed
contract and tuning hashes. Pytest used importlib mode for external operator
tests, preventing their source checkout from replacing the installed package.
All 16 installed runtime file hashes match the combined patch manifest. Native
extension hash `6048fb01a15dc0059debfd8ec50876891c8d6948ed62083a45a4402d6b364c9f`
was unchanged.

The nine [fixed-partition operator cases](qsa-fixed-splits-public-api.md) passed
from the final formatted test source
`f0845e747ab09783342a0fcd57570937543843bfdb3b26a9701a54144a109499`.
They cover partitions 1/16/64, capacity-invariant outputs, selected support of
2,051 positions, independent attention/LSE references, draft invalidation, and
allocation-stable `torch.CUDAGraph` replay.

The successful default-policy cases cover paged large-prefill addressing,
full-selection direct/split attention, four-way DCP output and LSE merging,
speculative rollback across a page boundary, draft anchor reuse, and handoff
of an incomplete compression group.

## Prepared ABI failures

Both parameterizations of
`test_qsa_run_prewarm_covers_bound_capacity_and_graph_replay[66-2048-False|True]`
prepare the canonical `int64` request-ID ABI and then iterate over runtime request
IDs beginning with `int32`. Preparation specializes this dtype. The first runtime
call raises:

```text
ValueError: QSA runtime request_ids ABI (torch.int32, (1,)) differs from prepared FrozenMapping(_items=(('dtype', 'int64'), ('strides', (1,))))
```

The isolated baseline checkout `/tmp/sparkgap-qsa-default-abi-baseline` contains
FlashInfer `98fb87a461183f98d6a7049ca947fb136cc6fb59` plus the historical 18-file
serving patch, SHA-256
`b9ea72148ed6c0c05a5deacc6d88610627cd83b1d1b40b4dac73a9365ea988cd`.
All 18 patched file hashes were verified. Its QSA contract hash is
`bd6fee66788ef4b9f144e5f3af52fdb3af99d677e62ceaf476850623cd29aa8f`.
The two test names and exception messages match the installed run exactly.
The baseline and installed `_canonical_abi` and `_require_runtime_abi` function
ASTs are identical; both use the unchanged test source hash
`e28e7c56c33abc903d7e60590782b7bfc50466fb235c2c28e1d2cc4dcb3b4f6e`.

These two prewarm cases remain failed. Their intended default-policy capacity
replay checks were not reached. No fixture, reference, tolerance, or production
ABI validation was changed. Raw logs retain the empty-graph warnings emitted
when capture unwinds after these exceptions, along with PyTorch/CuTe deprecations.

## Exact-checkpoint recipe

`tests/recipes/test_qwen38_attention.py` loaded layer 3 attention weights from
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`, revision
`dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`. The recipe requested 16 attention
partitions. Its native worker replayed two-request prefill lengths 17/9, decode
1/1, prefill 46/12, and decode 1/1. This crosses the 64-token page boundary and
includes multimodal rotary positions, padded lanes, and an unused state slot.

The unchanged exact-weight canonical reference gates passed: relative L2 below
2%, elementwise `rtol=0.035`, and absolute tolerance equal to 2% of the reference
maximum magnitude. Replay allocation counts remained stable. CPU geometry and
canonical rotary-table tests also passed.

All subprocesses exited and GPU4 had no compute process at release to the
whole-model replay qualification. Whole-model prefix equivalence, HTTP cache
replay behavior, and serving performance require their separate gates.
