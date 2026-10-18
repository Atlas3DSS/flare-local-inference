#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: build_container.sh [--builder NAME] [--tag IMAGE:TAG]

Build and push Flare for linux/amd64 and linux/arm64, then display the manifest.
Requires an existing Buildx builder supporting both platforms and registry login.

Options:
  --builder NAME    Buildx builder (default: flare-multi)
  --tag IMAGE:TAG    Published image (default: ghcr.io/local-inference-lab/flare:latest)
  -h, --help        Show this help
EOF
}

builder=flare-multi
tag=ghcr.io/local-inference-lab/flare:latest

while (($#)); do
    case "$1" in
        --builder|--tag)
            if (($# < 2)) || [[ -z "$2" || "$2" == -* ]]; then
                printf 'Missing value for %s\n' "$1" >&2
                exit 2
            fi
            case "$1" in
                --builder) builder=$2 ;;
                --tag) tag=$2 ;;
            esac
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

printf 'Building and pushing %s for linux/amd64 and linux/arm64 with builder %s...\n' "$tag" "$builder"
docker buildx build \
    --builder "$builder" \
    --platform linux/amd64,linux/arm64 \
    --file "$repo_root/Dockerfile" \
    --tag "$tag" \
    --progress plain \
    --push \
    "$repo_root"

docker buildx imagetools inspect "$tag"
