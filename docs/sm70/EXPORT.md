# Export a V100 package

A Flare runtime package holds everything the server loads: weights, compiled
kernels, captured CUDA graphs and frontend assets, signed by a publisher key.
Prepare it on a V100, the architecture that will serve it. Preparation checks
the compute capability, and packages record it.

Do the [build](BUILD.md) first.

## Why `ignition export` does not work for SM70

The upstream command `ignition export qwen3.8-27b ...` cannot prepare this
package yet, for two reasons:

- The `qwen3.8-27b` recipe declares a 262,144-token context. The SM70 S1L
  attention admits at most 32,768 tokens and refuses a wider page table at
  preparation.
- The CLI does not pass a CUTLASS include directory, and SM70 MTP preparation
  needs one.

The fork prepared its packages by calling
`ignition.recipes.qwen38_dense.bootstrap.prepare` with a `PreparationConfig`
directly. The script below does the same.

## 1. Download the checkpoint

```sh
.venv/bin/hf download nvidia/Qwen3.8-27B-NVFP4 --revision 482ca0f3832238542f8f5295dde86b5f22711d80 --local-dir /data/models/Qwen3.8-27B-NVFP4
```

The recipe pins this revision; its stored weights are 21,921,428,072 bytes
(`ignition/recipes/qwen38_dense/recipe.py`). If the download asks for
authentication, run `.venv/bin/hf auth login` and accept the model's terms on
its Hugging Face page first.

## 2. Write the preparation script

Paste this block into a shell at the repository root. It writes
`prepare_sm70.py`.

```sh
cat > prepare_sm70.py <<'EOF'
#!/usr/bin/env python3
"""Prepare and export the SM70 qwen3.8-27b package (one Tesla V100 32 GB, TP1, MTP).

The same PreparationConfig and export calls as the fork's release preparation,
without its export gates (docs/sm70/VALIDATE.md). Run from the repository root
with the repository's .venv, on one V100 selected with CUDA_VISIBLE_DEVICES,
with CUDA_HOME pointing at CUDA 12.x.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

REPOSITORY = "nvidia/Qwen3.8-27B-NVFP4"
REVISION = "482ca0f3832238542f8f5295dde86b5f22711d80"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help=f"local {REPOSITORY} snapshot at {REVISION}")
    parser.add_argument("--cutlass-include", type=Path, required=True, help="CUTLASS include directory")
    parser.add_argument("--signing-dir", type=Path, required=True,
                        help="development-signing-key.pem and development-publisher.json are created or reused here")
    parser.add_argument("--kernel-cache", type=Path, required=True, help="writable SM70 kernel cache")
    parser.add_argument("--output", type=Path, required=True, help="new .safetensors primary; payload blobs go beside it")
    parser.add_argument("--report", type=Path, help="optional JSON report")
    parser.add_argument("--context", type=int, default=32768, help="tokens per request; SM70 admits at most 32768")
    parser.add_argument("--concurrency", type=int, default=4, help="requests; SM70 MTP admits at most 4")
    parser.add_argument("--depth", type=int, choices=(2, 3, 4), default=4, help="MTP draft depth: widths 1..depth+1")
    parser.add_argument("--chunk", type=int, default=128, help="prefill chunk tokens")
    parser.add_argument("--compile-workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--compiler-context-mib", type=int, default=400,
                        help="device memory budgeted per compiler worker")
    parser.add_argument("--pages", type=int, default=86, help="KV pages backed during preparation")
    parser.add_argument("--device-budget-gib", type=int, default=30,
                        help="preparation leaves the rest of the card, plus 256 MiB, free")
    args = parser.parse_args()
    if args.output.suffix != ".safetensors":
        parser.error("--output must end in .safetensors")
    if args.output.exists():
        parser.error(f"{args.output} exists; choose a new name")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from cuda.pathfinder import load_nvidia_dynamic_lib
    from ignition.development import publisher
    from ignition.preparation import compile_pool
    from ignition.preparation.compilation import _OPTIONS
    from ignition.preparation.model_card import model_license
    from ignition.preparation.model_package import export_package
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.package_storage import PackageStorage
    from ignition.recipes.qwen38_dense.bootstrap import PreparationConfig, prepare

    def progress(message):
        print(time.strftime("%H:%M:%S"), message, flush=True)

    started = time.monotonic()
    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability() != (7, 0):
        raise SystemExit("this script prepares SM70 packages; GPU 0 of this process is not sm_70")
    _OPTIONS["compile_threads"] = args.compile_workers
    compile_pool._DISCRETE_WORKER_DEVICE_BYTES = args.compiler_context_mib << 20
    total = torch.cuda.get_device_properties(0).total_memory
    headroom = max(0, total - (args.device_budget_gib << 30)) + (256 << 20)
    config = PreparationConfig(context=args.context, max_active=args.concurrency, prefill_chunk_size=args.chunk,
        compile_workers=args.compile_workers, snapshot_budget_bytes=172 << 20, headroom_bytes=headroom,
        preparation_pages=args.pages, mtp=True, speculative_widths=tuple(range(1, args.depth + 2)),
        retain_graphs=True, kernel_cache=args.kernel_cache, cutlass_include=args.cutlass_include)
    recorder = ModuleImages(load_nvidia_dynamic_lib("cupti").abs_path)
    with recorder:
        storage = PackageStorage(device=0)
        with storage:
            prepared = prepare(args.checkpoint, config, progress=progress, package_storage=storage)
    memory = prepared.memory_report
    signer = replace(publisher(SimpleNamespace(path=args.signing_dir / "config.toml")),
                     license=model_license(args.checkpoint, REPOSITORY, REVISION))
    with prepared:
        export_package(prepared, recorder, storage, args.output, options={}, publisher=signer, progress=progress)
    result = {"package": str(args.output), "signer_fingerprint": signer.fingerprint.hex(),
              "publisher_profile": str(args.signing_dir / "development-publisher.json"),
              "memory": memory, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "elapsed_seconds": time.monotonic() - started, "torch": torch.__version__}
    text = json.dumps(result, indent=2, default=str)
    if args.report:
        args.report.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
EOF
```

The quickstart in [README-SM70.md](../../README-SM70.md) extracts the same block
with `sed -n '/^cat > prepare_sm70.py/,/^EOF$/p' docs/sm70/EXPORT.md | sh`.

The values match the fork's release preparation: `--context 32768`,
4 requests, draft depth 4 (verification widths 1 to 5), 128-token prefill
chunks, 86 preparation pages, a 172 MiB snapshot budget, a 30 GiB device budget
and 400 MiB per compiler worker. The fork ran 86 compiler workers on an
88-thread host (STATUS.md line 61). The default here is the host's CPU count
minus two.

## 3. Prepare and export

```sh
export CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH
CUDA_VISIBLE_DEVICES=0 .venv/bin/python prepare_sm70.py \
  --checkpoint /data/models/Qwen3.8-27B-NVFP4 \
  --cutlass-include /data/src/cutlass/include \
  --signing-dir /data/flare/keys \
  --kernel-cache /data/flare/kernel-cache \
  --output /data/flare-runtime/qwen38-27b-sm70-c4-d4.safetensors \
  --report /data/flare-runtime/qwen38-27b-sm70-c4-d4.prepare.json
```

Use a GPU with no other processes on it. The script prints a timestamped line
per preparation phase. At the end it prints a JSON summary with the signer
fingerprint and the memory report. A package prepared like the release package
reports `state_rows` 5 (one null row and one per request), one speculative state
column and 30.4 MiB of replay operands (STATUS.md line 62).

### Sizes and times

| Item | Value | Source |
| --- | --- | --- |
| Preparation time, one card, no other load | 23.9 min, 25.6 min | STATUS.md lines 55, 49 |
| Preparation time with a CPU-heavy test suite on the same host | 54.6 min | STATUS.md line 62 |
| Peak device allocation during preparation | 20.8 GiB | STATUS.md line 49 |
| Loaded footprint at server startup (`Plan cache` line) | 21.15 GiB | STATUS.md line 63 |
| Package primary (`.safetensors`, metadata only) | 958 KB | STATUS.md line 63 |
| Whole package (primary plus payload blobs) | about 20 GB | STATUS.md line 63 |

The primary file is small. Weights, kernels and graphs are immutable payload
blobs named by their SHA-256, written beside it. Copy the primary and every blob
together, and never edit them; signature verification fails on any change. See
[runtime packages](../model-preparation.md#runtime-packages).

## 4. Check the package

```sh
./flare/target/release/flare inspect /data/flare-runtime/qwen38-27b-sm70-c4-d4.safetensors --publisher-keys /data/flare/keys/development-publisher.json
./flare/target/release/flare inspect /data/flare-runtime/qwen38-27b-sm70-c4-d4.safetensors --publisher-keys /data/flare/keys/development-publisher.json --graphs
```

`inspect` verifies the signed index without a GPU. Check that it reports a
32,768-token context, concurrency 4 and speculative widths up to 5.

## Signing keys and the trust store

Flare loads only signed packages, and only from publishers you trust.

- **Signing key.** The first run creates `development-signing-key.pem`
  (Ed25519, owner-only permissions) in `--signing-dir`. Later runs reuse it.
  Keep it private.
- **Publisher profile.** The same run writes `development-publisher.json` beside
  the key. It holds only the public verification key, for the namespace
  `local-development`. Pass it to `flare run --publisher-keys`. It is safe to
  copy to serving hosts.
- **Trust store.** `trust.toml` records which signing-key fingerprints and
  namespaces you have accepted. The first time Flare loads a package from a new
  key, it shows the publisher disclosure and asks you to type the namespace
  (`local-development`). It then writes the acceptance to the trust store passed
  with `--trust-store`, so later loads do not ask.

For a service with no terminal, accept the key once unattended. Compute the
fingerprint from the profile, then pass it with the two explicit approval flags:

```sh
FP=$(python3 -c "import hashlib, json, sys; print(hashlib.sha256(bytes(json.load(open(sys.argv[1]))[0]['public_key'])).hexdigest())" /data/flare/keys/development-publisher.json)
./flare/target/release/flare run /data/flare-runtime/qwen38-27b-sm70-c4-d4.safetensors --publisher-keys /data/flare/keys/development-publisher.json --trust-store /data/flare/keys/trust.toml --dangerously-approve-key "$FP" --dangerously-accept-publisher local-development --gpus 0 --memory-limit 31G
```

Stop it once it is ready (Ctrl-C). The acceptance is now in
`/data/flare/keys/trust.toml`, and the [serve command](SERVE.md) needs neither
flag. The fingerprint printed by `prepare_sm70.py` must match `$FP`. To withdraw
trust, run `flare trust revoke FINGERPRINT --trust-store /data/flare/keys/trust.toml`.

Development keys are for your own packages. To publish packages for others,
create a production key with `ignition keygen` ([signing](../signing.md)).

## Export gates

The fork exported a package only after every gate below passed. They run on the
retained production CUDA graphs inside the preparation process, after `prepare`
and before `export_package`. They are part of the fork's preparation harness,
which is **not in this repository**. `prepare_sm70.py` runs the same
preparation and export without them. A package it produces has the same
configuration as the release package but has not been through these gates.

| Gate | What it checks | Pass criterion | Release package (`0e4c81f`, STATUS.md line 62) |
| --- | --- | --- | --- |
| Strict validation | 47 checks at batch 1 and 4 (below) | All 47 pass, byte for byte | 47/47, including every `forced_prefix_*` |
| Long-sequence invariance | 28 identities over 512 greedy tokens per coding prompt, 128-token prefill chunks | All 28 identical | 28/28 |
| Commit growth | Recurrent commit into a row admitted after logical capacity growth | All 4 checks pass | 4/4, including the negative capacity round |
| Consumer identity | Fused projection outputs equal their member projections | All pass | pass |
| Components | Isolated MTP draft and absorb/save graphs measured | Measurement completes | complete |
| Accuracy | Full-vocabulary KL(reference ‖ candidate) and top-1 on fixed short and long references | Measurement completes; there is no threshold | complete |
| Coding-token (tau) goldens | Fixed coding prompts at batch 1 and 4, draft depths 3 and 4 | Token IDs and summary identical to the golden file | identical |

**The 47 strict checks.** For batch 1 and batch 4 each:

- `ordinary_greedy_B{b}`: the reference greedy decode;
- `learned_greedy_B{b}_width{w}` for widths 1 to 5: speculative decoding emits
  the same token IDs as the reference;
- `actual_prefix_B{b}_width{w}`: a production round's commit equals the
  captured ordinary steps;
- `forced_prefix_B{b}_P{127,128}` at widths 4 and 5, and at batch 4 also with an
  inactive lane: one verification round, then every accepted count k from 1 to
  the width. The committed state (KV prefix, recurrent and convolution state,
  last hidden state) must equal k ordinary single-token decodes, compared as
  bytes. The verification itself must leave the committed rows untouched;
- `snapshot_restore_branch_B{b}_P{128,129,130}`: a snapshot taken at a page
  boundary and restored continues identically;
- `all_inactive_B{b}`; at batch 4 also `padded_slot_alias_B4` and
  `stale_generation_slot_alias_B4`: ignored lanes change nothing.

Then `sm70_prefill_attention_injected_status` (a poisoned K page must fail the
prefill with an error code, and the prompt must prefill cleanly again after the
page is restored), `sm70_decode_attention_nonfinite_probe` (NaN K/V in a visible
page must fail the request with non-finite code 1), and
`restore_pre_export_state`. That is 19 checks at batch 1, 25 at batch 4 and
3 more: 47. The `forced_prefix_*` checks are the ones that prove speculative
decoding is exact: accepting k verification rows leaves the same state as
k greedy decodes.

**The 28 long-sequence identities.** Four coding prompts each generate
512 greedy tokens alone and in a batch of four, with no MTP, draft depth 3 and
draft depth 4. Per prompt: alone equals co-scheduled for each of the three
modes, and speculative equals no-MTP at depth 3 and 4, alone and co-scheduled.
That is 7 per prompt.

**Commit growth.** The pool admits one more row (N-1 to N) inside already-backed
memory. Then the retained width-5 round, the retained width-4 round, and a
direct replay commit of every accepted count from 1 to 5 must commit into the
new row exactly like the teacher-forced reference. The fourth check is
negative: a width-4 round whose destination row is logically excluded must be
rejected by the batch validator (status 3, code 32), with the destination and
null rows unchanged (STATUS.md line 38).

**What the goldens are.** The tau goldens are a measurement file: the
token IDs and summary statistics the retained graphs produce on fixed coding
prompts (`code-00` to `code-03`) at batch 1 and 4 and draft depths 3 and 4. The
first qualified package of a given draft arithmetic writes the file. Every later
rebuild must reproduce it exactly, which shows the rebuild changed no
arithmetic. A reviewed change that legitimately changes draft bits needs a new
reference. When the S1L draft attention (`0aea71a`) changed draft tokens, the
fork kept the old file for older trees. It measured S1L-draft trees against the
goldens of the package that introduced `0aea71a`, and still required exact
identity (STATUS.md line 54). The release package matched those goldens
(line 62).

**The accuracy references.** These are reference logits for the same checkpoint
revision, on fixed short and long token contexts. The gate records
full-vocabulary KL and top-1 agreement and sets no pass threshold. For the
component packages before the release merge, the commit messages record short
`kld_mean` 0.0011443 and long 0.0024010, bit for bit equal to the previous
package (`8ad725c`, `ca16f00`).

[VALIDATE.md](VALIDATE.md) lists every gate in run order, including the serving
checks after export.

## Loading several packages at once

Flare reads and, with `--verify-weights`, checksums about 20 GB per package at
startup. One package loading from a spinning disk took 389.7 s to verify and
101.8 s to load weights (STATUS.md line 66). Three packages loading from the
same spinning disk at once took 735.1 s and 158.7 s, which exceeded the 900 s
readiness timeout of the fork's harness. The fork then passes
`READY_TIMEOUT=1800` to its serving legs (STATUS.md line 63). Allow at least
1800 s in any supervisor that waits for `/health`, or stagger the starts. The
primary file carries no weight paths, so moving it alone to faster storage does
not speed up the load (line 63).
