# Copy speculation replay state

The Qwen checkpoint at revision `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`
passes identical-command Python/native replay on SM120 with the 4,096-token
prefill catalog and copy proposals. The catalog contains 39 graphs, including
all 14 legal speculative batch/width combinations.

The [state qualifier](../../qualify/speculative_state_parity.py) executes 30
commands through each executor. All 2,252 checks pass: completion records,
persistent state, untouched lanes/pages, request epochs, and explicit RNG state
match; replay allocations remain unchanged. Covered transitions include W8→W1,
inactive/permuted lanes, a 64-token snapshot and restore, slot reuse, output-budget
clipping, terminal replay, and two-token stochastic grammar verification.

The [run result](/data/sparkgap-artifacts/mtp-implementation/speculative-state-parity-reset/receipt.json)
records zero state mismatches. Python replay exists only in qualification tooling.
The native service owns production execution.

The [transaction comparison](../../qualify/copy_transaction_comparison.py)
separately executes W1 continuations and wider copy-verification graphs across
all 14 shapes. Its [exact-checkpoint run](/data/sparkgap-artifacts/mtp-implementation/exact-checkpoints/copy-selected.json)
passes 475 commands and 175,529 checks. All 44,640 compared state tensors are
bitwise equal. Coverage includes a shared 64-token prefix, cross-width execution,
and selected-column snapshots at endpoints 65–72 after accepting widths 1–8.
Snapshots restore into different physical slots and private tail pages before
resume sampling and a following W1 step. Replay allocations, source pages, and
immutable snapshots remain unchanged. The inputs use forced literal grammar
continuations with all copied tokens accepted.

These comparisons do not cover the ordinary decoder, every rejection prefix,
or learned MTP. Bitwise agreement on these inputs does not establish a general
cross-capacity numerical tolerance. Ordinary-decoder comparisons, rejection and
termination clipping, and the complete agentic workload remain required.
