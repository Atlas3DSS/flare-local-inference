# Validate an SM70 tree and package

The fork qualifies a tree in five gates, run in this order. A failed gate stops
the chain; nothing after it runs.

1. Unit and regression tests, with CUDA tests on, on a V100.
2. Preparation with every export gate. The package is exported only if all pass.
3. Native serving, MTP depth 4, short prompts, at 1 and 4 agents.
4. Native serving, four 16.6k to 18.6k-token prompts, at 1 and 4 agents.
5. Native serving, four 22.4k to 30.2k-token prompts at `--memory-limit 31G`, at
   4 and 1 agents.

All five gates now have checked-in helpers under `tools/`. Follow the explicit
commands in [EXPORT.md](EXPORT.md) and [BENCHMARK.md](BENCHMARK.md). The public
prompt fixtures are newly authored inputs; they require new baseline results.
CPU CLI/gate tests do not qualify a model, export, or GPU execution.

Release results below are for the release tree `0e4c81f` and its package
`model-32k-tonight-a`, on one Tesla V100 PCIe 32 GB, 2026-10-22. Sources are
lines of the fork's run log (`STATUS.md`), quoted in
[BENCHMARK.md](BENCHMARK.md#sources).

## Gate 1: unit and regression tests

Run two pytest groups at once, on disjoint halves of the CPUs, with CUDA tests
enabled on the V100. Group A holds the SM70 component tests, group B the
sampling, recipe, contract and planning tests.

```sh
# OUTPUT_DIR and EMPTY_CACHE_DIR must already exist; outputs must be new.
# Set disjoint CPU_SET_A/CPU_SET_B covering allocated cores except one physical core.
CUDA_VISIBLE_DEVICES="$V100_GPU_UUID" bash tools/tests_replay.sh \
  "$PWD/.venv/bin/python" /data/flare-tests /data/flare-test-cache \
  "$CPU_SET_A" "$CPU_SET_B" "$COMPILER_THREADS_PER_GROUP"
```

Choose the compiler thread budget explicitly for each disjoint group; the historical
86-thread host used 40 compiler workers per group.

`B12X_COMPILE_CACHE_DIR` gives the run an empty b12x selection cache. With a
warm shared cache, one attention tuning test reports no benchmarked candidates
and fails (STATUS.md line 39).

**Pass criteria** (the fork's harness checks all of them):

- Both groups finish with a pytest summary line that includes a pass count,
  for example `3 failed, 213 passed, ... in 201.77s`. Exit status 0 or 1 only.
  Exit 2 to 5, a collection error, `INTERNALERROR` or an interruption fails the
  gate.
- Every `FAILED` line is one of the four failures that predate the SM70
  replay work:
  - `tests/recipes/test_qwen38_dense_speculation.py::test_reject_invalid_catalog[widths4]`
  - `tests/preparation/test_contract.py::test_dense_qwen_preserves_device_decode_and_page_storage_in_typed_contract`
  - `tests/recipes/test_declarations.py::test_fp8_cache_views_and_planned_bytes[True-kimi-k3-mxfp4-csf-16]` and `[False-kimi-k3-mxfp4-csf-16]`
- These five replay-commit V100 tests report `PASSED`, not `SKIPPED`:
  `test_replay_commit_matches_candidate_columns_byte_for_byte[1]`, `[4]` and
  `[5]`, `test_replay_commit_captures_without_allocation_and_skips_zero_counts`
  and `test_replay_rejections`.

Check them with:

```sh
grep -E "^FAILED" tests-A.log tests-B.log
grep -cE "^PASSED tests/components/test_sm70_gdn_replay.py::(test_replay_commit_matches_candidate_columns_byte_for_byte\[(1|4|5)\]|test_replay_commit_captures_without_allocation_and_skips_zero_counts|test_replay_rejections)" tests-A.log
```

The second command must print 5.

**Release result.** The last GPU run of both groups was on the replay-commit
branch before it was rebased and merged: group A passed every test, and group B
failed only its 2 known failures (STATUS.md lines 43, 46). On the release tree
itself these groups were not rerun on the GPU; its chain started at gate 2
(line 60).

**CPU suites.** The fork also runs the broad CPU suites and compares their
failure set with the base commit:

```sh
.venv/bin/python -m pytest -q -rA -p no:cacheprovider tests/components/test_sm70_gdn_replay.py tests/recipes tests/preparation tests/cli > cpu.log 2>&1
grep -E "^(FAILED|ERROR)" cpu.log | sort > cpu-failures.txt
```

Run the same on the base, `git worktree add ../flare-base 718a187`, and compare
the two `cpu-failures.txt` files. The gate passes when nothing fails on the tree
that does not fail on the base. On the release tree: 16 failed, 1,544 passed,
860 skipped and 100 errors, against 16, 1,536, 855 and 100 on `718a187`. Every
failure and error also occurs on the base, apart from one cluster-export test
that passed 3/3 alone and failed only while the suite shared the CPU with a
preparation run (STATUS.md line 61). The 100 errors are package-protection and
package-snapshot tests in that environment (line 51).

**Native tests.** The Helios scheduler tests that guard the hang fix are Rust
tests ([BUILD.md](BUILD.md#build-the-native-server)):

```sh
cargo test --locked --release --manifest-path helios/Cargo.toml --target-dir flare/target --features runtime-package,server --lib progress_tests
```

The commit messages of `e81534a` and `718a187` record that the regression tests
stall without the fix and pass with it. The run log does not record a
`cargo test` run on the release tree. On `fe058dc`, the command above reports
`4 passed; 0 failed; 1 ignored`.

## Gate 2: preparation with every export gate

Prepare with `--context 32768`, 4 requests and draft depth 4, as in
[EXPORT.md](EXPORT.md). The checked-in harness runs the following gates on retained graphs (accuracy is measured before strict validation):

| Gate | Pass criterion | Release result (STATUS.md line 62) |
| --- | --- | --- |
| Strict validation | 47/47, including every `forced_prefix_*` | 47/47 |
| Long-sequence invariance | 28/28 | 28/28 |
| Commit growth | 4/4, including the negative capacity round | 4/4 |
| Tau / coding-token goldens | token IDs and summary identical to the golden file | identical to the S1L-draft goldens |
| Consumer identity | pass | pass |
| Components | complete | complete |
| Accuracy (full-vocabulary KL, top-1) | complete; no threshold | complete |

The package is exported only after all of them pass. The release preparation
took 54.6 min, including CPU contention from the CPU suites (line 62).

**How to read a failure.** A strict-check failure names the check and the first
mismatching bytes of each field. Read it this way:

- A `forced_prefix_*` failure means speculative verification left a state that
  k greedy decodes would not.
- A `learned_greedy_*` failure means speculation changed the emitted tokens.
- An `ERROR` with `code 72` is a failed maintenance command, usually the harness
  addressing a row the pool does not back. During the replay-commit work that
  was a harness bug, not an integration bug (STATUS.md line 47).

A tau-golden failure has two possible causes. Either the rebuild changed
arithmetic, or the reference predates a reviewed change to the draft. Compare
against the goldens of the commit that introduced that change before you
conclude anything (line 54).

Do not skip the byte-level checks in favor of native serving. The first public
fixture run establishes a new tau baseline; subsequent runs must match the
preserved baseline exactly. Neither a first baseline nor an `accuracy: complete`
measurement establishes an unreported numerical tolerance or the historical scores.

## Gates 3 to 5: native serving

These gates start the real server on the package and drive it over HTTP. Start
the server with the benchmark flags:

```sh
./flare/target/release/flare run /data/flare-runtime/qwen38-27b-sm70-c4-d4.safetensors \
  --publisher-keys /data/flare/keys/development-publisher.json --trust-store /data/flare/keys/trust.toml \
  --verify-weights --gpus 0 --prefix-cache-memory 8G --memory-limit 31G --kv-cache-limit concurrency \
  --max-concurrency 4 --mtp-draft-tokens 4 --no-mtp-adaptive-verification \
  --host 127.0.0.1 --port 8990 > server.log 2>&1 &
until curl -sf http://127.0.0.1:8990/health; do sleep 5; done
```

Allow up to 1800 s for readiness. Then run the client from
[BENCHMARK.md](BENCHMARK.md#the-client) for each gate:

| Gate | Prompts | Output tokens | Sampling | Repetitions | Concurrency |
| --- | --- | --- | --- | --- | --- |
| 3, native depth 4 | 4 short coding prompts | 128 | `short` | 3 | 1, then 4 |
| 4, long 16.6k-18.6k | 4 prompts of 16.6k-18.6k tokens | 512 | `greedy` | 3 | 4, then 1 |
| 5, long 22.4k-30.2k | 4 prompts of 22.4k-30.2k tokens | 512 | `greedy` | 3 | 4, then 1 |

Each gate is 4 prompts x 3 repetitions x 2 concurrencies = 24 requests.

**Pass criteria** (the fork's harness checks all of them):

- The server reaches `/health` before the readiness timeout and stays up.
- 24/24 requests complete, each with exactly the requested number of output
  tokens (`ignore_eos`), and the stream ends with `[DONE]`.
- Metrics before and after the run show no counter reset.
- The server exits 0 on SIGTERM, without a forced kill.
- Gates 4 and 5: no eviction or restore during the run. Check the server log for
  eviction and restore lines, and check that
  `flare_graph_launches_total{kind="restore"}` did not grow.

**Report** for each gate: median decode at 1 agent and per agent at 4 agents,
per-prompt medians, TTFT p50, aggregate output, the `Allocate cache` line (KV
tokens) and the GPU step per round ([BENCHMARK.md](BENCHMARK.md)).

**Release results:**

| Gate | Requests | KV pool | Decode | Round times | Source |
| --- | --- | --- | --- | --- | --- |
| 3 | 24/24 ok | 131,584 tokens | 90.4 tok/s at 1 agent; 70.7 tok/s per agent at 4; 238.0 tok/s aggregate output | not reported | STATUS.md line 66 |
| 4 | 24/24 ok, zero evict/restore | 131,584 tokens | 82.6 tok/s at 1 agent; 42.6 tok/s per agent at 4 | 1 x 5 GPU step 49.5-49.6 ms | STATUS.md line 64 |
| 5 | 24/24 ok, all four resident, zero evict/restore | 131,584 tokens | 82.2 tok/s at 1 agent; 37.2 tok/s per agent at 4; TTFT p50 90.7 s at 4 | 4 x 5: not obtained | STATUS.md line 65 |

Gate 3 first failed on the readiness timeout, not the package. Three ~20 GB
packages were loading from one spinning disk; it passed on rerun with a 1800 s
timeout (STATUS.md lines 63, 66).

**The 4 x 5 round time.** The gate asks for the GPU time of a round with all
four agents decoding at width 5. On the release package no run produced one: the
four agents' decode windows never fully overlapped under chunked prefill, so
the 4-agent repetitions averaged mixed batch shapes (58.1, 58.1 and 49.6 ms mean
GPU step, STATUS.md line 64). The package before the release merge showed the
same (line 58). For comparison, the
replay-commit package before the S1L merge measured 75.7 ms per 4 x 5 round at
4 x 16.6k-18.6k tokens, against 76.6 ms for the baseline before replay-commit
(line 52).

**How to read the 4-agent decode medians.** Per-agent rates at 4 agents vary
with prefill interleaving. In gate 5 the four prompts' medians were 36.2, 13.0,
39.9 and 53.5 tok/s. The 13.0 tok/s prompt was the shortest. It finished
prefill first and decoded while the other three were still in chunked prefill
(STATUS.md line 65). On the package before the release merge, the slow agent
rotated between repetitions rather than staying on one prompt (line 58). Report the per-prompt medians, not only
the overall median.
