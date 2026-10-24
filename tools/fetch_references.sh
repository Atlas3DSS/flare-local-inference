#!/usr/bin/env bash
# Download the published FP32 accuracy references and verify every SHA-256 before use.
# Usage: tools/fetch_references.sh [DESTINATION]   (default /data/flare-refs)
# Source: https://huggingface.co/datasets/Atlas3D/flare-sm70-references at a pinned revision.
# Needs curl and sha256sum; no Hugging Face account or token.
set -euo pipefail
DEST=${1:-/data/flare-refs}
REPO=https://huggingface.co/datasets/Atlas3D/flare-sm70-references
REVISION=878913932bae27959f5a1c11f13146226d9348ca
FILES=(
  "30dd669b1baf6a1ec517b635c083b275dd72417ca07521bf33cecdd954f02a3f reference.safetensors"
  "277f47825bac44feba200dfab6ab555d637b66a30d31c2bfdfa2f0f09afcc5a3 reference.json"
  "d4479957d5ec101c8274d7a7fe59032d0171423ebacd8607a6aeb61e00f1afc9 long-code-reference.safetensors"
  "663ab9262b33ae26a4f07dbbd08ac02ea188990dbc1cc7c8df1effa474e0f15a long-code-reference.json"
)
mkdir -p "$DEST"
for entry in "${FILES[@]}"; do
  sha=${entry%% *}; name=${entry#* }
  target="$DEST/$name"
  if [ -f "$target" ] && echo "$sha  $target" | sha256sum -c --quiet - >/dev/null 2>&1; then
    echo "$name: already present, sha256 OK"; continue
  fi
  partial="$target.partial"
  curl -fL -# --retry 5 --retry-delay 5 -C - -o "$partial" "$REPO/resolve/$REVISION/$name" || {
    # A stale partial file that cannot be resumed is discarded and fetched once more from the start.
    rm -f "$partial"; curl -fL -# --retry 5 --retry-delay 5 -o "$partial" "$REPO/resolve/$REVISION/$name"; }
  if ! echo "$sha  $partial" | sha256sum -c --quiet -; then
    rm -f "$partial"; echo "$name: sha256 mismatch; nothing kept" >&2; exit 1
  fi
  mv -f "$partial" "$target"
  echo "$name: sha256 OK"
done
echo "References verified in $DEST"
