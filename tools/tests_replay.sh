#!/usr/bin/env bash
# Two-group release regressions; external GPU allocation required.
set -uo pipefail
if [ "$#" != 6 ]; then
  echo "Usage: tests_replay.sh PYTHON OUTPUT_DIR EMPTY_CACHE_DIR CPU_SET_A CPU_SET_B COMPILER_THREADS_PER_GROUP" >&2
  exit 2
fi
PY=$1; D=$2; CACHE=$3; CPUS_A=$4; CPUS_B=$5; COMPILER_THREADS=$6
cd "$(dirname "$0")/.." || exit 2
[ -d "$D" ] && [ -d "$CACHE" ] || exit 2
[ ! -e "$D/tests-A.log" ] && [ ! -e "$D/tests-B.log" ] || exit 2
[ -z "$(ls -A "$CACHE")" ] || { echo "Test compile cache must be empty" >&2; exit 2; }
[[ "$COMPILER_THREADS" =~ ^[1-9][0-9]*$ ]] || exit 2
export FLARE_RUN_CUDA_TESTS=1 B12X_COMPILE_CACHE_DIR="$CACHE" TORCHINDUCTOR_COMPILE_THREADS="$COMPILER_THREADS"
A=(tests/components/test_sm70_gdn_replay.py tests/components/test_sm70_gdn_cuda.py
   tests/components/test_sm70_build.py tests/components/test_sm70_dense.py tests/components/test_sm70_precision.py
   tests/components/test_sm70_projection_cuda.py tests/components/test_sm70_sampling_cuda.py
   tests/components/test_sm70_attention_cuda.py tests/components/test_sm70_prefill_attention_cuda.py
   tests/components/test_sm70_gdn_prefill_cuda.py tests/components/test_sm70_decode_attention_cuda.py
   tests/components/test_sm70_embedding_cuda.py tests/components/test_conditional_scores_cuda.py
   tests/components/test_sampling_scratch.py)
B=(tests/components/test_sampling_flashinfer.py tests/components/test_draft_candidate_ties_cuda.py
   tests/recipes/test_qwen38_speculation_cuda.py tests/recipes/test_qwen38_copy_speculation.py
   tests/recipes/test_qwen38_sampling.py tests/recipes/test_qwen38_dense_mtp.py
   tests/recipes/test_qwen38_dense_speculation.py tests/preparation/test_contract.py tests/cli/test_planning.py)
taskset -c "$CPUS_A" "$PY" -m pytest -q -rA -p no:cacheprovider "${A[@]}" > "$D"/tests-A.log 2>&1 & pa=$!
taskset -c "$CPUS_B" "$PY" -m pytest -q -rA -p no:cacheprovider "${B[@]}" > "$D"/tests-B.log 2>&1 & pb=$!
echo "A pid $pa  B pid $pb  $(date +%T)"
wait $pa; ra=$?; echo "A exit $ra $(date +%T)"
wait $pb; rb=$?; echo "B exit $rb $(date +%T)"
tail -n 3 "$D"/tests-A.log; tail -n 3 "$D"/tests-B.log
# Both groups must finish without failures. Required V100 replay tests must run,
# rather than skip; historical fixture failures are not exempted.
ok=0
for g in A B; do
  log="$D"/tests-$g.log; rc=$([ $g = A ] && echo $ra || echo $rb)
  if [ "$rc" != 0 ]; then echo "GROUP $g: pytest exit $rc (not a clean pass)"; ok=1; fi
  # pytest -q ends with e.g. "3 failed, 213 passed, 2 warnings, 17 subtests passed in 201.77s (0:03:21)" (no leading =).
  summary=$(grep -E "^(=+ )?[0-9]+ (failed|passed|skipped|error|errors)(, [0-9]+ [a-z ]+)* in [0-9.]+s" "$log" | tail -1)
  if [ -z "$summary" ] || ! echo "$summary" | grep -qE "(^|, )[0-9]+ passed"; then echo "GROUP $g: no completed pytest summary with a pass count"; ok=1; fi
  # Infrastructure failures: collection errors ("ERROR tests/..." in the -rA summary, "N errors in"), internal errors,
  # interruptions. "E   ..." assertion detail lines belong to ordinary test failures and are judged by the FAILED filter.
  if grep -qE "^ERROR tests/|^=+ ERRORS|INTERNALERROR|^!!+ .*(Interrupted|KeyboardInterrupt|stopping)|error during collection|(^|, )[0-9]+ errors? in [0-9.]+s" "$log"; then echo "GROUP $g: collection/internal error or interruption in the log"; ok=1; fi
  if grep -qE "^FAILED" "$log"; then echo "GROUP $g: FAILURES"; grep -E "^FAILED" "$log"; ok=1; fi
done
for t in "test_replay_commit_matches_candidate_columns_byte_for_byte\[1\]" "test_replay_commit_matches_candidate_columns_byte_for_byte\[4\]" \
         "test_replay_commit_matches_candidate_columns_byte_for_byte\[5\]" "test_replay_commit_captures_without_allocation_and_skips_zero_counts" \
         "test_replay_rejections"; do
  grep -qE "^PASSED tests/components/test_sm70_gdn_replay.py::$t" "$D"/tests-A.log || { echo "required V100 test did not PASS: $t"; ok=1; }
done
[ $ok = 0 ] && echo "unit + regression A/B: PASS (A exit $ra, B exit $rb)" || echo "unit + regression A/B: FAIL"
exit $ok
