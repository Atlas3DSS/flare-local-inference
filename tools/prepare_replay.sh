#!/usr/bin/env bash
# Pass explicit paths/resources after PYTHON. Every release export gate stays on.
set -euo pipefail
PYTHON=$1; shift
exec "$PYTHON" "$(dirname "$0")/prepare_replay.py" \
  --context 32768 --concurrency 4 --depth 4 --lease-gib 30 \
  --validate --check-invariance --check-commit-growth --check-consumers \
  --measure-components --measure-accuracy --measure-batches 1 4 \
  --measure-prompt-ids code-00 code-01 code-02 code-03 "$@"
