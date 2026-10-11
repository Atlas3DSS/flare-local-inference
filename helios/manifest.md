# Prepared execution manifest

ABI versions `2` and `3` describe graph execution and transport over one existing CUDA
context. A resource owner retains the context, graphs and allocations through
shutdown. `NativeResources` reconstructs these resources from a runtime package written
by Ignition before native serving starts. The model recipe defines the numerical behavior of each executable.

The Python representation is a dictionary with these fields:

```python
{
    "abi_version": 2,
    "context": int,                 # Existing CUcontext handle.
    "device": int,                  # Visible device ordinal.
    "max_active": int,              # 1 through 64.
    "input_frame_bytes": int,       # Positive, multiple of 8.
    "buffers": [
        {
            "name": str,
            "address": int,
            "nbytes": int,
            "dtype": str,
            "shape": [int, ...],
            "strides": [int, ...],  # Element strides; contiguous row-major.
        },
    ],
    "graphs": [
        {
            "name": str,
            "kind": str,
            "sampling_mode": str,   # general.
            "handle": int,          # Instantiated CUgraphExec handle.
            "batch_capacity": int,
            "token_capacity": int,
            "input_bindings": [
                {
                    "buffer_name": str,
                    "host_offset": int,
                    "nbytes": int,  # Entire named buffer extent.
                },
            ],
        },
    ],
}
```

Supported dtypes are `bool`, `uint8`, `int32`, `int64`, `float32`, `float16`, and
`bfloat16`. Shapes are nonempty and every extent is positive. `nbytes` equals
the contiguous element count times dtype size. Device addresses and host
offsets are aligned to that size. Names are nonempty and unique within their
lists. Graph executable handles are nonzero and unique. Registered device
spans do not overlap; captured weights and scratch need not appear in this I/O
manifest but must remain in the resource owner.

Graph kinds are `init`, `restore`, `snapshot`, `prefill`, `decode`, `resume`,
`vision`, and, in ABI3, `speculative`. Decode and speculative batch capacities are powers of two through 64.
Decode has one token per packed lane. Speculative graphs have an integer
verification width from 1 through 8 in `token_capacity`.
Prefill has batch capacity 1. Token capacities are 1, 16, or a multiple of 64
within 64..=262144. The model configuration separately requires the captured
catalog for its `prefill_chunk_size`; transport validation does not infer a
scheduling policy or substitute a missing graph.
State operations
and resume use batch capacities 1, 2, or 4. Vision batch capacity is 1, and its
token capacity is one of 256, 1024, 4096, or 16384 patches. Batch capacity can
exceed `max_active` for padding, but must not exceed its next power of two.
Request storage capacity is `max_active.next_power_of_two()`; immutable
state slots follow the null slot and this many live slots.

The sampling mode is `general` for every graph. Temperature zero is represented
by a one-hot distribution in the same stochastic sampling executable used for
positive temperatures. Unsupported mode strings reject manifest parsing before
CUDA initialization. A model contract defines the required graph capacities.

Two graph inputs are reserved and required in every executable's bindings:

| Buffer | Dtype | Shape | Writer |
|---|---|---|---|
| `frame_index` | int32 | `[1]` | Native executor |
| `submission_sequence` | int64 | `[1]` | Native executor |
| `completion_records` | int64 | ABI2 `[2,capacity,8]`; ABI3 `[2,capacity,24]` | Captured graph finalizer |

`completion_records` is an output and cannot appear in input bindings. Each
buffer has one fixed host offset shared across all graph bindings that name
it. Different buffers' host spans do not overlap. A command can upload any
subset of its graph's non-reserved bindings, including aligned subranges;
omitted ranges preserve resident device values. Preparation initializes all
device inputs before handoff. The native executor always uploads the reserved
frame index and positive, monotonic submission sequence. Command callers
cannot override those two values.

Each completion lane begins with eight signed 64-bit fields in this order:

1. Positive request generation, bounded by `INT64_MAX`.
2. Submission sequence.
3. Consumed-token count.
4. Generated-token count.
5. Sampled token ID, or `-1` when no token was sampled.
6. Status: `0` inactive, `1` running, `2` finished, `3` error.
7. Error code, `0` on success.
8. Reserved, `0`.

ABI3 appends these fields at zero-based indices:

| Index | Meaning |
|---|---|
| 8 | Emitted token count, 0 through verification width |
| 9 | Offered draft count, 0 through width minus one |
| 10 | Accepted drafts within the emitted prefix |
| 11 | Bitmask of offered drafts supplied by copying; bit zero is the first draft |
| 12 | Model-reserved field; the `qwen4exp.graph-inputs.v2` contract requires `-1` |
| 13 | Committed draft-cache length, `max(consumed_tokens-1,0)`; `-1` for a copy-only speculative round with no draft cache |
| 14 | Reserved, `0` |
| 15 | Verification width, an integer from 1 through 8 |
| 16–23 | Emitted token IDs in order, with unused entries `-1` |

The scalar token in field 4 equals the last vector token, or `-1` for an empty
vector. Accepted drafts number either `emitted_count-1` or `emitted_count`;
the latter occurs when EOS or the output budget clips inside the accepted draft
prefix. It never exceeds the offered count. Ordinary sampling emits at most
one token and reports width one. Maintenance and non-sampling prefill emit
zero tokens. Every graph in an ABI3 catalog writes the 24-field layout.
The executor validates vector bounds, padding, epochs and transport-reserved
fields before delivering records. The model contract validates field 12 before
request state advances.

The lane index is the physical request slot within the configured capacity. A captured finalizer scatters
packed model rows through the request-slot mapping. Every launch writes the
selected frame's records, marking unused lanes inactive. Records for
active lanes carry the submitted sequence and the corresponding request
generation. Finished requests suppress further mutation inside subsequent
graphs. The model scheduler requires an active completion for every submitted
live maintenance or vision lane; an inactive row cannot confirm a state copy.

The executor allocates two pinned input frames and two pinned completion
frames. Input updates, graph execution and production events use the model
stream. A copy stream waits for each production event, transfers the selected
completion row, and records its completion event. A frame remains occupied
until that transfer completes. Ordinary work allows at most two submissions
in flight. Every speculative graph uses a submission barrier, including direct
executor calls without an explicit grammar flag. Grammar-constrained work and
host token stops also run with one submission in flight,
so host token processing finishes before the next generation step.

Speculative submissions use preallocated CUDA timing events around the entire
captured graph. `FrameCompletion.elapsed_ns` excludes input uploads and the
completion transfer. Events and frames remain occupied through the transfer
fence. Other graph kinds return no timing measurement.

Submission sequences also identify the ordered completion fence. Resource
retirement waits for the completed fence covering every operation that can
reference the resource. Shutdown retains graph owners, context owners, and
pinned host frames until both streams drain. An unconfirmed drain preserves
ownership and reports failure; it cannot authorize memory reuse.

The manifest describes transport and graph capacities. The model recipe's
request-state, page-table, positional and sampling input names are additional
named buffers and bindings. A scheduler adapter checks those semantic names
against the recipe before serving requests.

## Native runtime package

`ignition export MODEL --gpus N --output DIR` prepares a TP1 package for
`flare run DIR`. Ignition exits or replaces its Python process through `execv` before the native server starts. The
server loads CUDA machine code and data without Python, PyTorch, compilation,
or graph capture. The CUDA driver creates fresh context, module, function,
graph, and graph-executable handles.

`package.json` version `1` contains:

- GPU compute capability and CUDA context flags, limits, and cache preference.
- Device address reservations and allocations, including mapped host regions.
  Each allocation names a binary snapshot or a zero initializer.
- Cubin files, kernel symbols, initialized globals and constants, parameter
  sizes, and function attributes.
- Graph nodes, dependency edges with CUDA edge data, kernel arguments and node
  attributes, and graph instantiation flags.
- An execution manifest with allocation-relative buffer references and graph
  names in place of process-local handles.

`server.json` contains the model contract, prepared capacities, request
defaults, and package-local frontend assets. It excludes the API key;
authentication and listener address are supplied when launching the native
server. Package files are trusted executable input.

Allocation references are an allocation name plus a byte offset. Explicit
relocations support 64-bit pointers and 128-byte CUDA tensor maps in both
kernel arguments and persistent device storage. Tensor maps are rebound with
`cuTensorMapReplaceAddress`. Relocatable allocations require complete binding
metadata; scanning bytes for values resembling pointers cannot supply it.

The Qwen exporter preserves virtual addresses and copies opaque arguments and
persistent allocation contents, including model weights and pointer tables.
Preparation uses two PyTorch memory pools backed by one native VMM allocator:
one for ordinary allocations and one for captured graphs. CUPTI records loaded
cubins and driver allocations. Device allocations outside the controlled pool
are grouped into complete VMM pages; only their live spans are copied and the
remaining bytes are zero. Mapped host tables use explicitly owned registered
host storage. Native restore reserves the recorded ranges before mapping or
loading resources and rejects address collisions.

Supported graph nodes are kernels, empty nodes, memset operations, and
contiguous device-to-device copies. Unsupported node types and device-updatable
kernel nodes reject export. TP, IPC resources, CUDA arrays, and texture objects
require additional resource descriptions. The native server accepts the Qwen
model contract. Resources remain owned until replay drains; a failed drain
retains them until process termination.
