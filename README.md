# Flare on V100 (SM70)

This lets you run the Qwen3.8-27B model on one 32 GB Tesla V100: about
83 tokens per second for one user, or about 65 per user for four users at
once, with contexts of up to 32,768 tokens. It is meant for people who have
V100 cards and want a current 27B coding model running locally. It is an
Apache-2.0 fork of [NVIDIA Flare](https://github.com/NVIDIA/flare-local-inference)
that adds a V100 path; everything else is upstream, whose README is kept as
[README-UPSTREAM.md](README-UPSTREAM.md).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/sm70/img/comparison-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/sm70/img/comparison-light.svg">
  <img alt="Same model, same card: Qwen3.8-27B on one Tesla V100 PCIe 32 GB. Decode speed per user: with one user, this port with MTP speculation 83.2 tokens per second, llama.cpp Q4_K_M with the checkpoint's MTP head 64.6, this port without speculation 30.9, llama.cpp without speculation 33.2; with four users at once, 64.9, 31.8, 28.3 and 20.5 per user. First token for a 30,720-token prompt: this port 35.5 s, llama.cpp 49.3 s. Four such prompts at once: llama.cpp first tokens at 60, 108, 157 and 200 s; this port all four at 140 to 142 s by default, or 36, 72, 108 and 145 s with --prefill-srpt. Mean KL from a full-precision run of the same checkpoint: this port 0.0011, llama.cpp Q8_0 0.0013, llama.cpp Q4_K_M 0.0088. vLLM family, greedy, no speculation: the 1Cat-vLLM fork with the same NVFP4 file 29.3 and 27.9 tokens per second and a 32.9 s first token; official vLLM 0.18.0 with a GPTQ-Int4 checkpoint 39.9 and 30.7, a 191.2 s first token and mean KL 0.0383. Official vLLM 0.32.0 and SGLang 0.5.22 cannot serve this checkpoint on a V100. Several cards, independent single-card servers with four agents each, aggregate output tokens per second end to end on short prompts: 2 cards 284.2, 3 cards 430.0, 4 cards 571.1, 5 cards 726.6, 6 cards 867.5, with 61.7 to 63.9 tokens per second per agent against 64.9 on one card; 1Cat-vLLM with tensor parallelism over four PCIe cards and four users: 203.4." src="docs/sm70/img/comparison-light.svg">
</picture>

The same checkpoint on the same card against llama.cpp and the vLLM family, with and without
speculation; official vLLM 0.32.0 and SGLang 0.5.22 cannot serve it on a V100 today. Methods,
dates and caveats are in [WHY-NOT-LOW-BIT.md](docs/sm70/WHY-NOT-LOW-BIT.md#same-model-on-llamacpp).
The last panel covers two to six cards: independent single-card servers add about 143 tokens per
second each end to end ([BENCHMARKS-SM70.md](BENCHMARKS-SM70.md#several-cards-independent-servers-2-to-6-v100s)),
set beside 1Cat-vLLM's tensor-parallel four-card run, which is a different design.

## Before you start

- One Tesla V100 32 GB card.
- An NVIDIA driver that still supports Volta (the R580 branch is the last
  one) and CUDA 12.9. CUDA 13 cannot build for the V100
  ([BUILD.md](docs/sm70/BUILD.md)).
- About 22 GB of disk for the model checkpoint and about 20 GB for the
  prepared package.
- Rust, `uv` and the build packages listed in [BUILD.md](docs/sm70/BUILD.md).
- A one-time preparation step checks the V100 output against full-precision
  (FP32) reference outputs. Making those references yourself needs a 96 GB GPU
  (an RTX PRO 6000) for one step. With only one V100, download the references
  the fork used instead (about 426 MB, from
  [Atlas3D/flare-sm70-references](https://huggingface.co/datasets/Atlas3D/flare-sm70-references)),
  as step 5 shows. The package is only written if every check passes
  ([EXPORT.md](docs/sm70/EXPORT.md#one-v100-only)).

Two values appear in the commands below. Set `THREADS` to the number of CPU
threads you want to use (leave one core free), and `V100_GPU_UUID` to your
card's ID from `nvidia-smi -L`. The `/data/...` paths are examples; change
them if you cannot write there.

## Quickstart

1. Get the code and build the server.

   ```sh
   git clone -b atlas/sm70-public https://github.com/Atlas3DSS/flare-local-inference.git && \
     cd flare-local-inference
   cargo build --release --locked --manifest-path flare/Cargo.toml
   ```

2. Set up the Python environment for preparation, point it at CUDA 12.9 and fetch CUTLASS
   (header files the V100 kernels need).

   ```sh
   uv sync --locked --extra prepare --extra test
   uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu126 \
     --reinstall-package torch --reinstall-package torchvision \
     torch==2.14.1+cu126 torchvision==0.29.1+cu126
   export CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH
   git clone --depth 1 --branch v4.8.1 https://github.com/NVIDIA/cutlass.git /data/src/cutlass
   ```

3. Download the model checkpoint (about 22 GB).

   ```sh
   .venv/bin/hf download nvidia/Qwen3.8-27B-NVFP4 \
     --revision 482ca0f3832238542f8f5295dde86b5f22711d80 \
     --local-dir /data/models/Qwen3.8-27B-NVFP4
   ```

4. Create the work folders and the test prompts.

   ```sh
   mkdir -p /data/flare-run /data/flare-refs /data/flare-keys /data/flare-cache
   .venv/bin/python tools/generate_prompts.py --output /data/flare-run/prompts.jsonl
   .venv/bin/python tools/generate_prompts.py --output /data/flare-refs/prompts.jsonl \
     --checkpoint /data/models/Qwen3.8-27B-NVFP4 --target-tokens 4096
   ```

5. Get the FP32 references. With one V100, download them; the script checks every
   file's SHA-256:

   ```sh
   bash tools/fetch_references.sh /data/flare-refs
   ```

   Or make them yourself, once, on a 96 GB GPU (`REFERENCE_GPU_UUID` is that card's ID):

   ```sh
   CUDA_VISIBLE_DEVICES="$REFERENCE_GPU_UUID" .venv/bin/python tools/reference_model.py \
     --root /data/flare-refs --checkpoint /data/models/Qwen3.8-27B-NVFP4 \
     --device cuda:0 --threads "$THREADS"
   CUDA_VISIBLE_DEVICES="$REFERENCE_GPU_UUID" .venv/bin/python tools/gpu_reference.py long \
     --root /data/flare-refs --out /data/flare-refs --calib-out /data/flare-refs \
     --checkpoint /data/models/Qwen3.8-27B-NVFP4 --prompts /data/flare-refs/prompts.jsonl \
     --threads "$THREADS" --count 6 --new-tokens 32 --positions 48
   ```

6. Prepare the V100 package. This runs every correctness check and writes the package only if
   all pass; it took about 36 minutes on one V100.

   ```sh
   CUDA_VISIBLE_DEVICES="$V100_GPU_UUID" bash tools/prepare_replay.sh .venv/bin/python \
     --root /data/flare-run --checkpoint /data/models/Qwen3.8-27B-NVFP4 \
     --prompts /data/flare-run/prompts.jsonl --compile-workers "$THREADS" \
     --cutlass-include /data/src/cutlass/include --kernel-cache /data/flare-cache \
     --signing-dir /data/flare-keys --short-reference /data/flare-refs/reference.safetensors \
     --long-reference /data/flare-refs/long-code-reference.safetensors \
     --measure /data/flare-run/tau.json --output /data/flare-run/model.safetensors \
     --report /data/flare-run/prepare.json
   ```

   If you cannot get the references at all, `--skip-reference-gates` (without the two
   `--...-reference` options) runs every other check and writes a package marked as unqualified.
   Its accuracy against the FP32 model is then unmeasured
   ([EXPORT.md](docs/sm70/EXPORT.md#one-v100-only)).

7. Start the server. Loading takes several minutes (about 8 minutes from a spinning disk in the
   fork's runs, [EXPORT.md](docs/sm70/EXPORT.md#loading-several-packages-at-once)).

   ```sh
   ./flare/target/release/flare run /data/flare-run/model.safetensors \
     --publisher-keys /data/flare-keys/development-publisher.json \
     --trust-store /data/flare-keys/trust.toml \
     --verify-weights --gpus 0 --memory-limit 31G --max-concurrency 4 --mtp-draft-tokens 3 \
     --no-mtp-adaptive-verification --prefix-cache-memory 16G --host 127.0.0.1 --port 8300
   ```

8. The first time, Flare asks you to trust the key that signed your package: type
   `local-development`. Later starts do not ask.
9. Wait until `curl http://127.0.0.1:8300/health` returns `{"status":"ok"}`.

## Try it

The server speaks the OpenAI API at `http://127.0.0.1:8300/v1`, so OpenAI-compatible
clients and coding agents can use it. A first request:

```sh
curl -sN http://127.0.0.1:8300/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "Qwen3.8-27B", "messages": [{"role": "user", "content": "Write a Python function that reverses a linked list."}], "max_tokens": 64, "stream": true, "return_token_ids": true, "return_token_timestamps": true, "stream_options": {"include_usage": true, "continuous_usage_stats": true}}'
```

## Options for several users at once

Before answering, the server reads the whole prompt (prefill; about 35 seconds for a
30,720-token prompt). Each active request's context lives in the KV pool, the part of GPU
memory that holds conversations. All three options are off by default; add them to
`flare run`.

- `--prefill-srpt` reads several long prompts one after another instead of taking turns,
  so the first users get answers much sooner; the cost is that users who already have an
  answer get it more slowly while the other prompts are read.
- `--prefill-srpt-short-tokens 2048 --prefill-srpt-decode-priority-tokens 4096` (together
  with `--prefill-srpt`) lets short prompts go in between and lets users who are getting
  an answer go first; the cost is that later users wait longer for their first word, and
  under heavy load the longest prompts can wait longer than without it
  ([details](BENCHMARKS-SM70.md#refined-prefill-ordering-opt-in-not-in-the-release-binaries)).
- `--prefix-cache-restore-wait`: when the KV pool runs out and a request has to pause, it
  waits for its saved context instead of reading its whole prompt again; that request waits,
  but at most as long as rereading would take.

## What to expect

Measured on one V100 (package `f21b9abb`, server binary `48b69ed4`;
[BENCHMARKS-SM70.md](BENCHMARKS-SM70.md)):

| Measurement | Value |
| --- | ---: |
| Prefill, 1 request, 2,048 / 30,720 tokens | 994 / 866 tok/s |
| Short prompts, decode per agent, 1 / 4 agents | 83.2 / 64.9 tok/s |
| Four cold 28.7k-token agents, first-token p50 | 132 s default, 83 s with `--prefill-srpt` |

In practice: once an answer starts it streams at about 65-83 tokens per second, but a long
prompt takes up to about half a minute before the first word, and four long prompts arriving
together take over two minutes.

## Known limits

- The KV pool holds 124,928 tokens, so four users at the full 32,768-token context do not
  fit at once.
- If the pool runs out, the server pauses a request and resumes it later; the resumed answer
  can differ from what an uninterrupted run would have produced.
- The options above trade one kind of waiting for another; that is why they are off by default.
  Details: [LIMITATIONS.md](docs/sm70/LIMITATIONS.md).

## Where to read more

- [README-SM70.md](README-SM70.md): results, trade-offs and the original quickstart
- [BENCHMARKS-SM70.md](BENCHMARKS-SM70.md): all measurements with their receipts
- [docs/sm70/EXPORT.md](docs/sm70/EXPORT.md): preparing a package
- [docs/sm70/SERVE.md](docs/sm70/SERVE.md): every server flag and the API
- [docs/sm70/VALIDATE.md](docs/sm70/VALIDATE.md): the correctness checks
- [docs/sm70/BENCHMARK.md](docs/sm70/BENCHMARK.md): how the numbers were measured
- [docs/sm70/ARCHITECTURE.md](docs/sm70/ARCHITECTURE.md): how the V100 path works, and numerics
- [docs/sm70/LIMITATIONS.md](docs/sm70/LIMITATIONS.md): what does not work or was not tested
- [docs/sm70/WHY-NOT-LOW-BIT.md](docs/sm70/WHY-NOT-LOW-BIT.md): comparison with low-bit alternatives
- [README-UPSTREAM.md](README-UPSTREAM.md): NVIDIA's original README

## Reviews

Every change in this fork was reviewed by someone other than its author; see `git log`.
