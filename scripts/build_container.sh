#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: build_container.sh [--builder NAME] [--beta | --release | --tag IMAGE:TAG]

Build and push Flare for linux/amd64 and linux/arm64, then display the manifest.
Creates the selected Buildx builder if missing and starts it before building.
Requires Docker Buildx and registry login.

Options:
  --builder NAME    Buildx builder (default: flare-multi)
  --beta            Publish ghcr.io/local-inference-lab/flare:beta
  --release         Publish ghcr.io/local-inference-lab/flare:release
  --tag IMAGE:TAG    Published image (default: ghcr.io/local-inference-lab/flare:latest)
  -h, --help        Show this help
EOF
}

builder=flare-multi
tag=

while (($#)); do
    case "$1" in
        --builder)
            if (($# < 2)) || [[ -z "$2" || "$2" == -* ]]; then
                printf 'Missing value for %s\n' "$1" >&2
                exit 2
            fi
            builder=$2
            shift 2
            ;;
        --beta|--release|--tag)
            if [[ -n "$tag" ]]; then
                printf 'Specify only one of --beta, --release, or --tag\n' >&2
                exit 2
            fi
            if [[ "$1" == --tag ]]; then
                if (($# < 2)) || [[ -z "$2" || "$2" == -* ]]; then
                    printf 'Missing value for --tag\n' >&2
                    exit 2
                fi
                tag=$2
                shift 2
            else
                tag="ghcr.io/local-inference-lab/flare:${1#--}"
                shift
            fi
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

tag=${tag:-ghcr.io/local-inference-lab/flare:latest}
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

if ! docker buildx inspect "$builder" >/dev/null 2>&1; then
    printf 'Creating Buildx builder %s...\n' "$builder"
    docker buildx create --name "$builder" --driver docker-container
fi
docker buildx inspect --bootstrap "$builder"

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
