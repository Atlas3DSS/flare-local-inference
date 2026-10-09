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

Both executors replay identical graph handles and inputs. This establishes
executor parity; it does not establish equivalence between verification widths,
ordinary decoding, or learned MTP. Those comparisons require independent target
execution and calibrated component checks. The
[transaction comparison](../../qualify/copy_transaction_comparison.py) implements
independent W1/Wn and restored-continuation arms; its GPU run remains pending.
