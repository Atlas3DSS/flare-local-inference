# Python/Rust graph replay with 4096-token prefill capacity

**PASS: 3,693/3,693 recorded checks**; raw harness status `True`, supervisor exit code 0. Overall status also requires complete execution and successful provenance/archive verification. 413,708/413,708 native tensor observations are byte-identical to Python, with 0 recorded mismatches. Error: `None`.

The catalog contains 43 graph variants (28 general and 15 greedy). The fixture scripts 124 commands in 117 observed stage groups. Python records 124 commands and Rust records 124. Six added cases cover 1024-, 2048-, and 4096-token prefills in both sampling modes: full-chunk consumption without sampling, immutable snapshot, 65-token continuation, decode, restoration into another physical slot, and the same continuation/decode sequence.

The original 76-command fixture is compared against its archived 512-capacity execution. Runtime completion observations are excluded from fixture identity. Token IDs, mRoPE positions, and image-index uploads retain the original byte prefixes and add only declared inactive trailing padding; other uploads must match completely. Comparison result: `True`. Detailed comparisons and historical upload files are archived. The separate CPU scenario proof retains exact complete upload identity when configured with capacity 512.

Allocation checks pass 234/234; unrelated-state checks pass 234/234. Reset equality: `True`. Before the harness resets mutable state, the pristine-state check scans all 5,319,819,776 backing bytes and finds exactly 270 permitted nonzero bytes with valid PLE history. Native streams drained: `True`. Readbacks retain all 77 hybrid regions, state backing padding, request metadata/RNG/histogram, logits, touched QSA pages, raw QSA state, feature rows, and canaries. The deduplicated host baseline is 15,716,186,164 bytes, below the unchanged 32 GiB limit. Supervisor cleanup confirms process exit 0, no GPU compute process, and 2 MiB GPU memory use.

The harness records 599 source hashes, including all 31 main Rust files. The archived native build includes 34 build inputs; its extension hashes to `0dcc9ca245ed83bcbf7c0ea8147640a09c3ea2f7de6a8e6bc0559826058a0dca`. The sampler mapping receipt, copied binary, build files, and process maps are archived. The main runtime snapshot and installed b12x sources are included. Kernel policy retains fixed 16 QSA partitions and sequential GDN recurrence with one key split.

The preceding qualification process completed model preparation but failed before reset/replay: a full-backing `torch.count_nonzero` requested a 4.96 GiB temporary with 4.03 GiB free. It recorded zero checks and commands, then exited 1. Its raw receipt, log, script, process exit and sampler mapping are retained. The verifier now sums counts over every contiguous 64 MiB byte slice, preserving the complete backing/EOS-history predicate. Six CPU tests cover the exact predicate, late corruption, a partial final slice, and the per-call bound; AST comparison confirms all other qualifier code is unchanged.

Failed recorded checks:

None.

Failed evidence verification checks:

None.

This run compares the same captured graphs across execution engines. [The 4,043-token prefix replay](prefix-replay-prefill4096.md) separately checks cold/restored state and logits across graph capacities. [The configurable-prefill HTTP receipt](prefill-capacity.md) records the 512- and 4,096-token text runs and the default-4,096 image/grammar run. Their frozen summaries and reports are archived alongside this evidence. Canonical model accuracy and performance have separate qualification requirements.

[Summary](graph-execution-parity-prefill4096.json), [complete raw receipt](/data/sparkgap-artifacts/validation-receipts/graph-execution-parity-prefill4096.json.gz), and [source/upload evidence](/data/sparkgap-artifacts/validation-receipts/graph-execution-parity-prefill4096.tar.gz). Archive SHA-256: `467b227b64c860ccfcde4d1fd3d2e38e54c888a9695b3ac030422fc5be1d5240`.
