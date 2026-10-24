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
directly. The checked-in harness does the same, then runs the export gates.

## 1. Download the checkpoint

```sh
.venv/bin/hf download nvidia/Qwen3.8-27B-NVFP4 --revision 482ca0f3832238542f8f5295dde86b5f22711d80 --local-dir /data/models/Qwen3.8-27B-NVFP4
```

The recipe pins this revision; its stored weights are 21,921,428,072 bytes
(`ignition/recipes/qwen38_dense/recipe.py`). If the download asks for
authentication, run `.venv/bin/hf auth login` and accept the model's terms on
its Hugging Face page first.

## 2. Prepare public inputs and FP32 references

The checked-in [tools/prepare_replay.py](../../tools/prepare_replay.py) is the
release replay-commit preparation harness, with its numerical helpers in the
same flat `tools/` directory. It retains the strict, invariance, commit-growth,
consumer, component, full-vocabulary accuracy and token/tau gates. It replaces
the earlier documentation-only export example, which omitted those gates.

Use explicit paths. Create the following output directories once, outside the
checkout; keep keys and generated model/reference data out of Git:

```sh
mkdir -p /data/flare-run /data/flare-refs /data/flare-keys /data/flare-cache
.venv/bin/python tools/generate_prompts.py --output /data/flare-run/prompts.jsonl
.venv/bin/python tools/generate_prompts.py --output /data/flare-refs/prompts.jsonl \
  --checkpoint /data/models/Qwen3.8-27B-NVFP4 --target-tokens 4096
```

These are **new benchmark inputs**, labelled `public-generated-v1-new-baseline`.
They contain deterministic, newly authored coding tasks and example modules.
They are not the unpublished historical prompts or agent traces. Their results
must establish a new baseline, not be presented as reproduction of the old scores.
The four hardcoded strict-validator prompts remain unchanged.

Accuracy references are generated locally from the pinned checkpoint; no binary
reference download is required. The short reference streams FP32 layers using
canonical Transformers arithmetic. The long-reference helper retains its original
GPU-resident checkpoint implementation and PRO 6000 device check: use a separately
allocated RTX PRO 6000 96 GB for that step. It is not a V100-memory-fit claim.
Neither reference script imports Flare kernels. Use the pinned Transformers
preparation environment, and allocate the hardware before running these commands.
Set `THREADS` to the allocated CPU thread count, reserving one physical core.

```sh
CUDA_VISIBLE_DEVICES="$REFERENCE_GPU_UUID" .venv/bin/python tools/reference_model.py \
  --root /data/flare-refs --checkpoint /data/models/Qwen3.8-27B-NVFP4 \
  --device cuda:0 --threads "$THREADS"
CUDA_VISIBLE_DEVICES="$REFERENCE_GPU_UUID" .venv/bin/python tools/gpu_reference.py long \
  --root /data/flare-refs --out /data/flare-refs --calib-out /data/flare-refs \
  --checkpoint /data/models/Qwen3.8-27B-NVFP4 --prompts /data/flare-refs/prompts.jsonl \
  --threads "$THREADS" --count 6 --new-tokens 32 --positions 48
```

Retain both `.safetensors` files and their companion JSON files. Accuracy checks
bind the model ID, revision, checkpoint config/index hashes and exact token
contexts. They report FP64-reduced full-vocabulary KL and top-1 differences;
`complete` means the measurement completed, not an invented numerical threshold.
Reference generation is expensive GPU work and has not been rerun for this port.

## 3. Prepare and export

Complete [Gate 1](VALIDATE.md#gate-1-unit-and-regression-tests) first. Use an
exclusively allocated V100 and an explicit compiler budget. The shell helper
turns on every release export gate; it does not allocate hardware for you.

```sh
export CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH
CUDA_VISIBLE_DEVICES="$V100_GPU_UUID" bash tools/prepare_replay.sh .venv/bin/python \
  --root /data/flare-run --checkpoint /data/models/Qwen3.8-27B-NVFP4 \
  --prompts /data/flare-run/prompts.jsonl --compile-workers "$THREADS" \
  --cutlass-include /data/src/cutlass/include --kernel-cache /data/flare-cache \
  --signing-dir /data/flare-keys \
  --short-reference /data/flare-refs/reference.safetensors \
  --long-reference /data/flare-refs/long-code-reference.safetensors \
  --measure /data/flare-run/tau.json \
  --output /data/flare-run/model.safetensors --report /data/flare-run/prepare.json
```

The helper prepares 512-token prefill chunks: prefill graphs at capacities
128, 256 and 512, 48 graphs in all. To prepare the previous 128-token catalog
(44 graphs) instead, add `--chunk 128` after the helper's arguments; the later
value wins. At `--memory-limit 31G` the 512-token package leaves a
124,928-token KV pool, limited by memory, against 131,584 tokens (the
`--kv-cache-limit concurrency` cap) for the 128-token package
([LIMITATIONS.md](LIMITATIONS.md#not-validated-or-validated-only-in-part)).
Both prefill the same tokens
([ARCHITECTURE.md](ARCHITECTURE.md#5-512-token-prefill-exact-fusions-and-the-g128-split-plan)).

The first successful run creates `tau.json` for the new public inputs. Preserve
it. A subsequent preparation using that same file must match its prompt-file
hash, per-run token IDs and complete tau summary exactly before export. Use a
new package output path for the rebuild: existing packages are not overwritten.
An initial baseline is not a prior-golden match; the latter is reported explicitly
as `measurement.prior_measurement_identity: true` only after comparison.

The report records the SHA-256 of the exported signed primary; native benchmarking
re-hashes that exact file and still asks the server to verify weight blobs.
It records every gate and fails export if any required gate is missing
or failed. Raw per-check progress remains in `validation-attempts.jsonl`, including
failed attempts. Review the report before [native qualification](BENCHMARK.md#the-client).
The port has CPU-only validation so far; it does not assert a new GPU pass.

### Sizes and times

| Item | Value | Source |
| --- | --- | --- |
| Preparation time, one card, no other load | 23.9 min, 25.6 min | STATUS.md lines 55, 49 |
| Preparation time with a CPU-heavy test suite on the same host | 54.6 min | STATUS.md line 62 |
| Peak device allocation during preparation | 20.8 GiB | STATUS.md line 49 |
| Loaded footprint at server startup (`Plan cache` line) | 21.15 GiB | STATUS.md line 63 |
| Package primary (`.safetensors`, metadata only) | 958 KB | STATUS.md line 63 |
| Whole package (primary plus payload blobs) | about 20 GB | STATUS.md line 63 |
| Preparation time with every export gate, package `f21b9abb` | 36.2 min, with the rebuild running on another card of the same host; the rebuild 29.4 min | [gate-summary.json](receipts/g128/gate-summary.json) |
| Peak device allocation during preparation, package `f21b9abb` | 21.22 GiB (previous package 20.77 GiB) | [gate-summary.json](receipts/g128/gate-summary.json) |
| Loaded footprint at server startup (`Plan cache` line), package `f21b9abb` | 21.93 GiB, KV pool 124,928 tokens | [default-short-server.log](receipts/g128/default-short-server.log) |

The primary file is small. Weights, kernels and graphs are immutable payload
blobs named by their SHA-256, written beside it. Copy the primary and every blob
together, and never edit them; signature verification fails on any change. See
[runtime packages](../model-preparation.md#runtime-packages).

## 4. Check the package

```sh
./flare/target/release/flare inspect /data/flare-run/model.safetensors --publisher-keys /data/flare-keys/development-publisher.json
./flare/target/release/flare inspect /data/flare-run/model.safetensors --publisher-keys /data/flare-keys/development-publisher.json --graphs
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
FP=$(python3 -c "import hashlib, json, sys; print(hashlib.sha256(bytes(json.load(open(sys.argv[1]))[0]['public_key'])).hexdigest())" /data/flare-keys/development-publisher.json)
./flare/target/release/flare run /data/flare-run/model.safetensors --publisher-keys /data/flare-keys/development-publisher.json --trust-store /data/flare-keys/trust.toml --dangerously-approve-key "$FP" --dangerously-accept-publisher local-development --gpus 0 --memory-limit 31G
```

Stop it once it is ready (Ctrl-C). The acceptance is now in
`/data/flare-keys/trust.toml`, and the [serve command](SERVE.md) needs neither
flag. The `fingerprint` in `/data/flare-run/prepare.json` must match `$FP`. To withdraw
trust, run `flare trust revoke FINGERPRINT --trust-store /data/flare-keys/trust.toml`.

Package `f21b9abb` and its rebuild `ad848aaa…` are signed with the
development key that signed the previous package `fd7c24dc` (fingerprint
`ef39a0be…`), so a trust store that accepted that key accepts them.

Development keys are for your own packages. To publish packages for others,
create a production key with `ignition keygen` ([signing](../signing.md)).

## Export gates

The fork exported a package only after every gate below passed. They run on the
retained production CUDA graphs inside the preparation process, after `prepare`
and before `export_package`. They now ship in `tools/`. The second-to-last
column describes the historical package. The last column is the current
package `f21b9abb`, prepared on the public inputs with 512-token prefill chunks
([gate-summary.json](receipts/g128/gate-summary.json)).

| Gate | What it checks | Pass criterion | Release package (`0e4c81f`, STATUS.md line 62) | Package `f21b9abb` (2026-10-24) |
| --- | --- | --- | --- | --- |
| Strict validation | 47 checks at batch 1 and 4 (below) | All 47 pass, byte for byte | 47/47, including every `forced_prefix_*` | 47/47 |
| Long-sequence invariance | 28 identities over 512 greedy tokens per coding prompt, 128-token prefill chunks | All 28 identical | 28/28 | 28/28 |
| Commit growth | Recurrent commit into a row admitted after logical capacity growth | All 4 checks pass | 4/4, including the negative capacity round | 4/4 |
| Consumer identity | Fused projection outputs equal their member projections | All pass | pass | pass |
| Components | Isolated MTP draft and absorb/save graphs measured | Measurement completes | complete | complete |
| Accuracy | Full-vocabulary KL(reference ‖ candidate) and top-1 on fixed short and long references | Measurement completes; there is no threshold | complete | complete; output byte-identical to the previous package's |
| Coding-token (tau) goldens | Fixed coding prompts at batch 1 and 4, draft depths 3 and 4 | Token IDs and summary identical to the golden file | identical | identical to the previous package's goldens |

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
page is restored; at prefill capacities above 128 the harness first backs the
pages the probe's longer prompt needs, see
[VALIDATE.md](VALIDATE.md#the-injected-status-probe-at-chunks-above-128)),
`sm70_decode_attention_nonfinite_probe` (NaN K/V in a visible
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
