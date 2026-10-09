# ABI-2 native startup rejection

All 21 constructor-rejection cases pass against native library SHA256
`6048fb01a15dc0059debfd8ec50876891c8d6948ed62083a45a4402d6b364c9f`.
Each isolated process calls the installed `sparkgap_native.Server` with invalid
manifest or configuration metadata. Every call rejects with the expected error
and releases its borrowed owner. Syscall traces record zero CUDA-library or
NVIDIA-device opens, zero socket/bind/listen/connect attempts, and zero thread
creation. Native and harness source hashes remain unchanged during execution.

The [receipt](native-startup-validation-abi2.json) identifies the installed build,
source hashes, exact errors and each trace hash. The
[artifact archive](/data/sparkgap-artifacts/validation-receipts/native-startup-validation-abi2-artifacts.tar.gz) includes all
input descriptors, subprocess output, syscall traces and the qualified harness.
The [ABI-1 receipt](native-startup-validation.md) retains its original evidence.

The otherwise valid fixture declares all 37 ABI-2 variants: three each for
initialization, restoration and snapshot, six each for resume and decode,
twelve prefill variants, and four vision variants. Prefill, decode and resume
require both `general` and `greedy`; other graph kinds require `general`.
Addresses and handles are synthetic integers that are never dereferenced.

Fourteen cases cover ABI/context/span/layout validation, absent general graph
capacities, absent semantic inputs, model geometry, frontend assets/checksums,
and HTTP/native queue limits. Seven sampling-mode cases cover a missing mode,
an unknown mode, a greedy non-sampling graph, missing greedy variants separately
for resume/decode/prefill, and an ambiguous exact variant with distinct name and
graph handle. The valid-shape control reaches asset validation; the queue-limit
cases load the exact checkpoint frontend assets before rejecting.

The constructor tests allocate no model weights or GPU graphs and do not create
a Python `PreparedModel`. CUDA execution faults, fatal worker health propagation,
and checkpoint inference are outside these checks.

```sh
.venv/bin/python scripts/qualify_startup_validation.py \
  --artifacts /tmp/sparkgap-startup-validation-abi2-reproduction
```
