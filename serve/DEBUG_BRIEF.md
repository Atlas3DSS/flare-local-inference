# Qwen3.5 Correctness Debug Brief

Status on 2026-03-23: resolved.

## Root Cause

There were two real bugs and one major testing error.

1. The comparison methodology was invalid.
   `tests/serve/test_qwen35_correctness.py` was using a hard-coded 80-token
   multimodal-style prompt full of `248319` placeholders, while the sglang
   baseline was actually sending the raw 5-token completion prompt
   `[760, 6511, 314, 9338, 369]` for `"The capital of France is"`.
   Any conclusions drawn from those mismatched inputs were not trustworthy.

2. Full-attention KV heads were loaded incorrectly under TP.
   In `serve/model/recipe_qwen3_5.py`, when `tp_world_size > total_num_kv_heads`
   we were effectively loading all KV heads on every rank instead of matching
   sglang's replicated-KV-head layout.

3. The shared expert was silently dropped on the real NVFP4 checkpoint.
   The routed experts were loaded correctly, but the quantized shared expert
   weights (`torch.uint8`) returned `None` from `_load_shared_expert()`, so the
   shared expert path was missing from every layer in production.

## What Fixed It

- Canonicalized the prompt comparison with `tests/serve/qwen35_trace.py`.
  Both backends now run from the same exact prompt IDs and write comparable
  JSON traces.
- Fixed KV-head TP loading in `serve/model/recipe_qwen3_5.py`.
- Dequantized the NVFP4 shared expert once at load time to BF16, TP-sharded it
  like the BF16 path, and ran it through the existing dense shared-expert code.
- Disabled radix prefix-cache reuse for hybrid models in
  `serve/engine/scheduler.py` because we cache KV prefixes but not the
  corresponding linear-attention SSM prefix state. That was causing repeated
  greedy runs in one process to diverge after the first request.

## Final Validation

Using the exact raw completion prompt IDs:

```json
[760, 6511, 314, 9338, 369]
```

b12x now produces:

```json
{
  "generated_ids": [11751, 13, 248046],
  "generated_text": " Paris."
}
```

That was validated on the real checkpoint:

```text
/data/models/Qwen3.5-397B-A17B-NVFP4
```

with TP=4 on the serving stack.

## Correct Methodology

Do not compare activations unless these artifacts match first:

1. Request payload
2. Final formatted prompt text
3. Final prompt token IDs
4. Decoded prompt text from those IDs

The canonical harness already records those fields. If they differ, stop and
fix the prompt path before debugging internals.

## Current Repro Commands

Run b12x correctness:

```bash
PYTHONPATH=. ~/projects/sglang/.venv/bin/python \
  tests/serve/test_qwen35_correctness.py \
  --gpu-ids 0,1,2,3
```

Run the sglang baseline:

```bash
PYTHONPATH=. ~/projects/sglang/.venv/bin/python \
  tests/serve/run_sglang_baseline.py \
  --gpu-ids 0,1,2,3
```

Run the lower-level trace harness directly:

```bash
PYTHONPATH=. ~/projects/sglang/.venv/bin/python \
  tests/serve/qwen35_trace.py \
  --backend b12x \
  --prompt-ids '[760,6511,314,9338,369]' \
  --max-new-tokens 8 \
  --out /tmp/b12x_trace.json
```

```bash
PYTHONPATH=. ~/projects/sglang/.venv/bin/python \
  tests/serve/qwen35_trace.py \
  --backend sglang \
  --gpu-ids 0,1,2,3 \
  --prompt-ids '[760,6511,314,9338,369]' \
  --max-new-tokens 8 \
  --out /tmp/sglang_trace.json
```

## Useful Trace Files

- `/tmp/b12x_trace_after_shared_dense.json`
- `/tmp/b12x_qwen35_correctness_trace.json`
- `/tmp/sglang_qwen35_baseline_trace.json`

## Notes

- The earlier multimodal-embedding theory was a dead end caused by comparing
  different prompts.
- The env-gated debug hooks in `serve/engine/runner.py` and
  `serve/engine/serving.py` can still be used if another regression appears.
