# Fixed-partition QSA public API qualification

The isolated FlashInfer candidate at `/tmp/sparkgap-qsa-fixed-splits`, based on
`98fb87a461183f98d6a7049ca947fb136cc6fb59`, passed **9/9 GPU tests in 11.76 seconds**.
The tests use deterministic synthetic operands, public QSA preparation/binding/run
interfaces, and `torch.CUDAGraph` replay on GPU
`GPU-afd4b1ad-8a64-7057-4bde-241822724c7f`.

[Machine-readable receipt](qsa-fixed-splits-public-api.json) and
[raw log, JUnit, test source, and receipt archive](/data/sparkgap-artifacts/validation-receipts/qsa-fixed-splits-public-api.tar.gz)
record the command, source hashes, package versions, and device identity.

| Contract | Checked values |
| --- | --- |
| Caller-fixed attention partitions | 1, 16, 64 |
| Combined selector/attention graph capacities | 16, 65, 128 rows |
| Rank-local DCP selection then attention | Rank 1 of 2, interleave 4, capacities 65 and 128 |
| Attention geometry | 24 query heads, 2 KV heads, dimension 256 |
| Selected support | Exactly 2,051 positions at query positions 2,050 and 2,054 |
| Reference output gate | Existing sparse paged GQA reference, `rtol=0`, `atol=2e-2` |
| Independent DCP LSE gate | Gathered FP32 QK scores followed by `logsumexp`, existing `rtol=1e-5`, `atol=1e-5` |
| Capacity invariance | Bitwise active output and selected positions for identical operands and fixed partition count; DCP LSE also bitwise |
| Replay allocation stability | 87 replays across 18 captures; cumulative allocation count and allocated bytes unchanged |
| Draft reuse | Anchor at 2,050; causal tails through 2,051 and 2,053; inactive/out-of-range sources and requests, excessive speculative position |

Changed device row counts exercise active row 64 and an empty second attention
chunk. Padding produces zero output, empty selections, and negative-infinite DCP
LSE. Selected positions are checked for causality independently of the reference
attention mask. Draft replay preserves all selector state and recorded anchors.
DCP local group IDs are converted to global group IDs before `attend()`; distributed
candidate merging and cross-rank output reduction are outside this fixture.

The executed test source hash was
`9c3dbc6157b052a069904c5b676805615fcf8f84ec4cff464c879f1634851045`.
Ruff 0.16.2 subsequently formatted the file and passed lint; the archived source
hash is `f0845e747ab09783342a0fcd57570937543843bfdb3b26a9701a54144a109499`.
Production candidate source hashes remained unchanged throughout qualification.
Installed FlashInfer and the main working checkout were not modified.

These operator tests do not measure checkpoint inference, whole-model prefix
replay equivalence, Rust graph execution, or serving performance. The exact
459-token whole-model replay gate and HTTP cache replay gates remain separate.

Reproduction from the isolated candidate checkout:

```bash
cd /tmp/sparkgap-qsa-fixed-splits
CUDA_VISIBLE_DEVICES=GPU-afd4b1ad-8a64-7057-4bde-241822724c7f \
CUDA_HOME=/opt/cuda \
PYTHONPATH="$PWD/flashinfer/experimental/b12x/_compat:$PWD" \
/home/luke/projects/sparkgap/.venv/bin/python -m pytest \
  --confcutdir=tests/experimental/b12x -x -vv -s \
  tests/experimental/b12x/attention/test_qsa_fixed_splits_gpu.py
```
