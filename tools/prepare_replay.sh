#!/usr/bin/env bash
# Pass explicit paths/resources after PYTHON. Every release export gate stays on, except that
# --skip-reference-gates drops the accuracy gate, the only one that needs the FP32 references.
set -euo pipefail
PYTHON=$1; shift
ACCURACY=(--measure-accuracy)
for arg in "$@"; do
  if [ "$arg" = --skip-reference-gates ]; then ACCURACY=(); fi
done
exec "$PYTHON" "$(dirname "$0")/prepare_replay.py" \
  --context 32768 --concurrency 4 --depth 4 --lease-gib 30 --chunk 512 \
  --validate --check-invariance --check-commit-growth --check-consumers \
  --measure-components ${ACCURACY[@]+"${ACCURACY[@]}"} --measure-batches 1 4 \
  --measure-prompt-ids code-00 code-01 code-02 code-03 "$@"
