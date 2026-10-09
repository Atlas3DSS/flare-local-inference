# Native decode host work

This source audit covers the implementation packaged in
[native-conversation-build.json](native-conversation-build.json). It records
operations and bounds; allocator counts, CPU time, CUDA submission gaps, and
latency significance require profiling. Numerical qualification remains open.

The production model thread owns `GraphExecutor` directly. Python's
`ExecutionWorker` qualification wrapper adds a mutex, a reply channel, and a
thread roundtrip, which are absent from HTTP serving.

## Ordinary unconstrained decode

With unchanged page assignments, input packing updates four reusable byte
buffers: physical slots, request generations, live row count, and sampling
flags. These total 68 bytes. The transport adds a four-byte frame index and an
eight-byte sequence number, then issues six H2D copies, one graph launch, two
event records, one stream wait, and a 256-byte D2H completion copy. Successful
submission does not acquire an application mutex or switch CUDA contexts.

Relevant implementations are [input packing](../../native/src/model/inputs.rs),
[CUDA transport](../../native/src/execution/transport.rs), and the
[model loop](../../native/src/server/runtime.rs).

| Host work | Bound and behavior |
|---|---|
| Scheduler selection | Request, selection, and lane vectors hold at most four entries. Submission repeats selection for validation and clones lanes into a submission map with at most two entries. |
| Short allocations | Selection, submission, and upload descriptors construct roughly nine short vectors per successful submission at source level. Completion and stop handling construct additional token vectors. Actual allocator activity is unmeasured. |
| Graph and buffer lookup | Graph lookup scans 25 variants. Decode bindings contain 12 names, and the registered input catalog contains 41 buffers. Upload validation checks overlaps pairwise; ordinary decode has four uploaded buffers. |
| Prefix references | Each cached lane traverses and allocates its ancestor chain when marking GPU use. Chain depth is bounded by the snapshot capacity, 37 with the default budget. |
| Cache maintenance | Completion scans checkpoint readiness. Progress reporting computes cache statistics with three further node scans. |
| Pending work | Poll iterations check queued cancellations, four active requests, and two transport frames, then yield the host thread if no progress occurred. The queue defaults to 16 entries and permits at most 1,024. |

The relevant policies are in [scheduler.rs](../../native/src/scheduler.rs),
[cache.rs](../../native/src/cache.rs), and
[stops.rs](../../native/src/server/stops.rs).

## Work that depends on request content

Ordinary decode performs no vocabulary scan or full page-table rebuild. A page
boundary adds a four-byte page ID per affected lane. Admission can upload the
complete matched prefix page table, up to 16 KiB at 262,144 context tokens.

Prefix hashing accumulates tokens and hashes one 64-token block at a boundary.
The token, prefix-boundary, and private-page vectors can occasionally grow and
copy their accumulated contents. The admitted prompt vector does not reserve
the maximum output capacity in advance.

Grammar decoding computes a 7,760-word vocabulary mask and copies 31,040 bytes
per constrained lane through host staging. It also limits the submission window
to one graph. The 993,280-byte token histogram is uploaded only at admission.
See [grammar.rs](../../native/src/grammar.rs) and
[inputs.rs](../../native/src/model/inputs.rs).

Output parsing and JSON/SSE formatting execute outside the model thread.
Unfinished tool calls repeatedly search their accumulated buffer for the closing
tag; the buffer is bounded at 1 MiB. Total scanning can therefore grow
quadratically with the length of a streamed tool call. An incremental search
cursor retaining the closing tag's overlap would remove repeated prefix scans;
its benefit requires a trace with long tool arguments. See
[output.rs](../../native/src/server/output.rs).

## Measurement targets

Use the same graph sequence and generation parameters to measure CUDA driver
submission time, unsuccessful polling iterations, and the gap between GPU
executions. Attribute allocator activity and scheduler/cache work separately.
Measure grammar masks and long tool-argument parsing as distinct workloads;
their costs do not characterize ordinary unconstrained decode.
