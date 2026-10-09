"""Inspect the pinned Qwen checkpoint without loading tensor payloads."""

import argparse
import json
import sys

from .checkpoint import CheckpointError, inspect_qwen38_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", help="Local directory containing config.json and the safetensors index")
    args = parser.parse_args(argv)
    try:
        manifest = inspect_qwen38_checkpoint(args.snapshot)
    except (CheckpointError, OSError) as exc:
        print(f"checkpoint inspection failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
