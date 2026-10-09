# FlashInfer chain acceptance at the uniform endpoint

FlashInfer chain sampling incorrectly rejected a positive probability-one acceptance ratio when `curand_uniform` returned exactly `1`. Correcting both acceptance loops passes four exact GPU reproducer fixtures and 72 targeted FlashInfer tests. The [machine-readable receipt](flashinfer-chain-endpoint.json) retains input comparisons, raw results, source hashes, loaded-library hashes, and process mappings.

In [`sampling.cuh`](/home/luke/projects/flashinfer/include/flashinfer/sampling.cuh:1917), `p` is the draft probability and `q` is the target probability. Both sequential prefix acceptance and the independent acceptance counter now use:

```cpp
(p > 0 && q >= p) || u * p < q
```

The random draw remains in place. Positive ratios at least one always accept; ratios below one retain the strict comparison; two zero probabilities still reject. At subsequence and offset zero, Philox seed `16390493` produces FP32 uniform `1` on draw zero, and seed `2871038` produces it on draw two. These seeds exercise both loops without changing the distributions or oracle between runs.

The [baseline GPU receipt](/data/sparkgap-artifacts/mtp-implementation/chain-endpoint-baseline/receipt.json) passes 13 of 20 predicates. The [corrected GPU receipt](/data/sparkgap-artifacts/mtp-implementation/chain-endpoint-fixed/receipt.json) passes all 20:

| Fixture | Baseline result | Corrected result |
| --- | --- | --- |
| Identical point masses | IDs `[0, -1]`, accepted prefix length 0 | IDs `[7, 11]`, accepted prefix length 1 |
| Identical two-token distributions | IDs `[0, -1]`, accepted prefix length 0 | IDs `[7, 11]`, accepted prefix length 1 |
| Certain independent acceptance after prefix rejection | Independent count 0 | Independent count 1; IDs `[11, -1, -1]` unchanged |
| Acceptance ratio below one | Rejection, IDs `[11, -1]` | Identical rejection and counts |

All four fixtures use vocabulary size 248320 and pass the zero-replay-allocation check in both runs. Eager execution and PyTorch CUDA graph replay agree. CPU readback verifies all 12 paired input tensors are byte-identical and both raw tensor files agree with their receipts. Execution used PyTorch `2.13.0+cu130` on GPU `GPU-afd4b1ad-8a64-7057-4bde-241822724c7f`, an RTX PRO 6000 Blackwell Max-Q Workstation Edition.

The separate [FlashInfer regression log](/data/sparkgap-artifacts/mtp-implementation/chain-endpoint-fi-tests.log) records **72 passed, 118 deselected, three deprecation warnings, 0.59 seconds**. Its 32 endpoint cases cover four fixtures, vocabularies 257 and 248320, scalar and device seed/offset arguments, and both deterministic settings. The other 40 tests cover per-row chain RNG and sampler CUDA graph replay. The four standalone reproducer fixtures are not included in that pytest count.

Run from `/home/luke/projects/flashinfer`:

```bash
CUDA_HOME=/opt/cuda \
CUDA_VISIBLE_DEVICES=GPU-afd4b1ad-8a64-7057-4bde-241822724c7f \
/home/luke/projects/sparkgap/.venv/bin/python -m pytest -q \
  tests/utils/test_sampling_seed_offset.py \
  -k 'uniform_endpoint or chain_sampling_row_seed_offset or sampling_rng_tensor_cuda_graph'
```

The FlashInfer checkout has base revision `98fb87a461183f98d6a7049ca947fb136cc6fb59` with local serving changes. Fresh-process qualification records these SHA-256 identities:

| Artifact | Baseline | Corrected |
| --- | --- | --- |
| `include/flashinfer/sampling.cuh` | `e297a6a05581b7bb3328ca8b839ea480f9ff4c3463f86a824565655ed2cd8d6f` | `289412563ad3435ce9bd1548101cdc5d8a33a90f37d35d947505722bae6738f1` |
| Loaded `sampling.so` | `e0b9edf1df3223dcaadcda91cc09369d08a121b75464bc94366dbca4457a9c91` | `7e9777ab864b81d8bddbabf5a55b016da7c70255e1ebdc2beabd602b903433b3` |

The JIT cache path is reused across builds; the receipts retain the loaded hashes and mapping rows for each process. Raw logs and tensors remain alongside the external receipts. This component qualification does not establish full-model speculative serving correctness or performance; graph evidence uses PyTorch CUDA graphs.
