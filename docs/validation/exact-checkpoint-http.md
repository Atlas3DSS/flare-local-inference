# Exact checkpoint retention through native HTTP

The Qwen checkpoint `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`
at revision `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2` passes exact instruction,
prompt, and image-conditioned prefix checks through the installed Rust server
on SM120. The server uses copy speculation, a 4,096-token prefill limit,
262,144-token context, 39 captured graphs, 6,779 KV pages, and 34 snapshot slots.
Native errors are typed and the model thread uses private worker phases.

| Case | Observed behavior |
|---|---|
| Full prompt retry | The complete 160-token prompt restores and samples through Resume without prefill. Output matches the initial request. |
| Prompt edits | Edits before a retained endpoint do not reuse a rounded page prefix. Exact retries restore the edited prompt. |
| Short instructions | A 12-token instruction prefix is published and reused after changing user content. Changed instructions create a distinct 13-token prefix. |
| Tool-bearing instructions | The 254-token instruction prefix is reused after changing user content. Changing the tool name misses the retained prefix. |
| Endpoint retention | Instruction, prompt, and terminal publications match the expected distinct endpoints. A one-token output coalesces prompt and terminal state. |
| Snapshot pressure | Independent prefixes exceed the 34-slot capacity, evict the oldest checkpoint, and preserve resource totals and cold-replay output. |
| Conversation pressure | 37 edited prompts exceed snapshot capacity; 731 complete pages remain associated with the retained checkpoints. The final 2,464-token prompt restores without prefill. |
| Cancellation | A disconnected stream releases its active request; four subsequent serial generations complete with correct resource accounting. |
| KV pressure | One context-sized request reserves 4,096 pages. Excess admission returns JSON 503 or one terminal SSE error; the admitted request continues. Cancellation releases its reservations and a follow-up succeeds. |
| Image identity | Equal text with distinct pinned images does not share an image-conditioned checkpoint. Repeating the matching image restores state and preserves the output. |

The system/tool cases use temperature 1, top-k 20, top-p 0.95, seed 41873,
and a 32-token output budget. The other fixtures retain their point-mass sampling
through the common FlashInfer implementation and their explicit request budgets.
Every run finishes with healthy service and unchanged native sources.

## Run results

- [Pressure, prompt, and cancellation run](/data/sparkgap-artifacts/mtp-implementation/exact-checkpoints/http-pressure/result.json):
  518 checks passed, zero failed checks, one fixture-construction error.
  The four-character nonce `9485` produces a 256-token tool instruction prefix;
  the fixture requires a nonaligned endpoint and stopped before sending a tool
  request. The run therefore reports failure despite passing its executed checks.
- [System/tool-prefix run](/data/sparkgap-artifacts/mtp-implementation/exact-checkpoints/http-system-prefix/result.json):
  all six requests and 45 checks passed. The fixture bounds its hexadecimal
  nonce to three characters, preserving the nonalignment requirement. Sampling,
  seeds, budgets, and cache assertions are unchanged.
- [Image-identity run](/data/sparkgap-artifacts/mtp-implementation/exact-checkpoints/http-image-identity/result.json):
  all 26 checks passed.

Each result directory retains HTTP bodies, metrics, the qualifier source, and
its console output. [Fixture tests](../../tests/reference/test_exact_checkpoint_http.py)
cover numeric and mixed-character nonces, exact prefill geometry, and the pressure
reservation contract.

## Scope

These checks establish cache identity, graph selection, publication, and resource
lifetime for the exercised HTTP cases. Four serial follow-ups do not establish
reuse of every physical slot. Image identity does not establish accepted
image-conditioned speculative drafts or vision accuracy.

An HTTP endpoint-residue sweep, injected snapshot-copy failure, and explicit
reuse of generated terminal prefixes remain open. GPU transfer and selected-state
tests cover additional endpoints independently. Learned MTP cache state and
complete-model numerical acceptance remain open; BF16 GDN arithmetic can differ
between arbitrary prefill partitions even when same-partition restore is exact.
