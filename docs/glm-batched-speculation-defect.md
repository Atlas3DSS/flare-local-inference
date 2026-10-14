# GLM-5.3 Flash: batched speculative decoding corrupts answers on DGX Spark

Status: open, under investigation (Oct 14, 2026). Root cause not yet identified.

## 1. Summary

On DGX Sparks (SM121, ranks on separate hosts over RoCEnante), GLM-5.3 Flash
produces corrupted answers whenever MTP speculative decoding runs with several
requests in one batch. It reproduces at tensor parallel 2 and 4, so it is not
specific to one sharding width. With eight or more concurrent requests the
replies are mostly empty or invented. The same packages are correct with
speculation disabled, and with speculation at concurrency one or two. The
GB300 (SM103, one rank) is correct on the identical test at concurrency 16.

The defect also explains two earlier symptoms:

- **Long uncached prompts answer wrongly.** Right after its prefill, a request
  runs one speculative round at every larger batch shape the server allows
  (16, 8, 4, 2 rows, the request live in row zero), then drops to batch one.
  Those first rounds corrupt it; cached requests resume at batch one and skip
  them.
- **C16 streams fail with device error 6.** The verifier finds an invalid
  target distribution (diagnostic code 82: non-finite or empty target
  probabilities) in several rows of the same step.

## 2. Environment

| Item | Value |
| --- | --- |
| Flare | `rebase-series` a02621e |
| FlashInfer / b12x | local-inference-lab/flashinfer f6552169 (PR #12) |
| Checkpoint | local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD dec48abd |
| TP4 package | `ignition export glm-5.3-flash --ranks 4 --gpus 0 --max-concurrency 16 --prefill-chunk-size 4096 --no-context-parallel` |
| TP2 package | the same with `--ranks 2 --max-concurrency 8` |
| Hardware | DGX Spark (GB10, SM121), RoCE 200G; TP4 ranks hertz/maxwell/faraday/ampere, TP2 ranks hertz/maxwell |
| Capacity (TP4) | 6,716,416 KV tokens per rank, 176 state rows, FP32 recurrent / BF16 convolution state |
| Sampling | T=1, top-p 0.95 (model card); per-request seeds |

## 3. Results

`scripts/repro_glm_batched_speculation.py` drives every row below. "Cached"
publishes one prompt and then sends N simultaneous copies; "uncached" sends
prompts with distinct leading nonces one at a time. The 10K prompt is
`--size 8000` (about 10.5K tokens, answer `2U2GYCZF, 198`).

| Server | Test | Correct |
| --- | --- | --- |
| GB300, adaptive MTP, concurrency 16 | cached 10K x16 | 15/16, 13/16 |
| GB300, adaptive MTP | cached 96K x16 | 16/16 |
| GB300, adaptive MTP | uncached 133K–260K | 14/14 |
| Spark TP4, adaptive MTP, concurrency 16 | cached 10K x16 | 1/16 |
| Spark TP4, adaptive MTP, concurrency 16 | cached 96K x16 | 0/16 (13 hit the 16,384-token cap) |
| Spark TP4, adaptive MTP | cached 10K x2 / x4 / x8 | 2/2, 3/4, 1/8 |
| Spark TP4, MTP fixed width 2, concurrency 8 | cached 10K x8 | 3/8 |
| Spark TP4, MTP fixed width 4, concurrency 8 | cached 10K x8 | 1/8 |
| Spark TP4, speculation off (`--mtp-draft-tokens 0`) | cached 10K x16 | 10/16, coherent |
| Spark TP4, speculation off | uncached 48K–133K | 5/5 |
| Spark TP4, adaptive MTP, default concurrency | uncached 48K–133K | about 2/15 |
| Spark TP4, adaptive MTP, `--max-concurrency 1` / `2` | uncached 96K–133K | 5/5, 4/4 |
| Spark TP2, adaptive MTP, concurrency 8 | cached 10K x8 | 1/8 |
| Spark TP2, adaptive MTP | cached 10K x2 | 2/2 |

The misses with speculation off name a neighbouring record (199 for 198), the
same model-level miss the GB300 makes at a similar rate. They are a separate
question. The speculative failures are empty replies, invented codes, and
claims that the prompt contains no code.

Row count alone is not the trigger. Eight requests at width 2 (16 rows, the
same row count as plain decode at concurrency 16) still fail. The failing
configuration is several requests, each verifying several tokens, on SM121
ranks. The TP2 misses look like the TP4 ones: empty replies, a garbled code
(`2U2GYZF`), and records far from the needle.

## 4. Reproduce on DGX Sparks

1. Export a package without decode context parallelism, following
   [multi-host.md](multi-host.md). Four Sparks at TP4 are shown; two Sparks at
   TP2 (`--ranks 2`, one `join`, `--hosts` naming the one peer) reproduce the
   same way with `--max-concurrency 8` and `--concurrency 8` in step 3:

   ```sh
   .venv/bin/ignition export glm-5.3-flash --config ~/glm/config.toml --ranks 4 \
     --rendezvous 10.200.0.13:29630 --gpus 0 --max-concurrency 16 --prefill-chunk-size 4096 \
     --no-context-parallel --output ~/glm/pkg/glm53-tp4-nodcp.safetensors --verbose
   # on each peer, in rank order:
   .venv/bin/ignition join 10.200.0.13:29630 --rank 1 --ranks 4 --config ~/glm/config.toml
   ```

2. Serve it from rank zero with the package defaults (adaptive MTP,
   concurrency 16):

   ```sh
   ./flare/target/release/flare run ~/glm/pkg/glm53-tp4-nodcp.safetensors --gpus 0 \
     --hosts maxwell,faraday,ampere --rendezvous 10.200.0.13:29640 --bind 0.0.0.0:8000 \
     --publisher-keys ~/glm/development-publisher.json --dangerously-approve-key <fingerprint> \
     --dangerously-accept-publisher local-development
   ```

3. Run the cached test. A healthy server answers at least 13 of 16; the
   defect gives 0–3, mostly empty:

   ```sh
   python3 scripts/repro_glm_batched_speculation.py cached --url http://hertz:8000 --concurrency 16
   ```

4. Controls. Restart the server with each option and rerun step 3:
   - `--mtp-draft-tokens 0`: coherent answers (about 10–15 of 16).
   - `--max-concurrency 2` and `--concurrency 2` in the script: 2/2.

5. Long uncached prompts (one at a time, each about 96K tokens; fails on the
   default server, passes with either control):

   ```sh
   python3 scripts/repro_glm_batched_speculation.py cold --url http://hertz:8000 --size 70000 --count 3
   ```

Raw outputs, probes and logs from the runs above are on gracie under
`~/research/flare-spark-resume/` (`claude-*.json`, `claude-findings-20261014.md`).

## 5. Reproduce on RTX PRO 6000 (SM120)

Since the defect appears at both TP2 and TP4 on the Sparks, any tensor-parallel
width that fits should do on SM120. Flare's GLM recipe branches only on SM103
versus everything else, so an SM120
host runs the same b12x path as the Sparks: the DSA indexer, sparse MLA, KDA
recording and commit, fused NVFP4 MoE, blockscaled GEMMs, and the short
convolution. Three things differ: b12x compiles for `sm_120a`, autotuning
measures tactics on the local GPU (a GB10 has 48 SMs; an RTX PRO 6000 has
188), and tensor-parallel collectives use b12x's PCIe IPC transport instead of
RoCEnante. A reproduction on SM120 would therefore separate a defect in
Flare's SM12x speculative path from one specific to GB10 tactics or to RoCE.
We have not yet run GLM on RTX PRO 6000s ourselves.

The checkpoint holds about 166 GB of weights. Use four 96 GB cards in one
host, tensor parallel 4 (about 44 GB of weights per rank); two cards do not
leave room for the KV cache.

1. Export on the host, one rank per GPU:

   ```sh
   .venv/bin/ignition export glm-5.3-flash --gpus 0-3 --max-concurrency 16 \
     --no-context-parallel --output glm53-tp4-sm120.safetensors --verbose
   ```

   `--prefill-chunk-size 4096` matches the Spark package; the default is fine.

2. Serve:

   ```sh
   ./flare/target/release/flare run glm53-tp4-sm120.safetensors --gpus 0,1,2,3 --bind 0.0.0.0:8000
   ```

3. Run the cached test and both controls exactly as in section 4, steps 3–4,
   against `http://<host>:8000`. Report the counts, the controls, and the
   GPU model (Workstation or Max-Q).

Outcomes:

- **Fails like the Sparks, controls pass:** the defect is in Flare's SM12x
  speculative path or the b12x kernels it uses, independent of RoCE and of
  GB10 tuning. SM120 is then the faster place to bisect.
- **Passes:** the defect needs GB10-tuned tactics, RoCEnante, or the
  multi-host runtime. Compare against a Spark package built with heuristic
  defaults (section 6).

## 6. What has been ruled out

Each check below passed at the shapes the failing rounds use. Scripts are in
`~/research/flare-spark-resume/` on gracie.

| Component | Check | Result |
| --- | --- | --- |
| RoCEnante all-reduce and vocabulary all-gather | 4 Sparks, 2–96 rows (to 7.4 MB per rank), CUDA graph replays with per-replay patterns (`roce_gather_stress.py`) | no mismatched bytes |
| b12x KDA verification recording and commit | TP4 geometry (16 heads), 1–16 sequences, mixed lengths, padding, Flare's strided state indices (`kda_batch_recovery_check.py`) | exact against reference |
| b12x DSA indexer (32 heads, top-512) | fused route (≤16 rows) and tiled route (>16), graph replays with changing inputs, padding, scratch overwritten between replays (`indexer_rows_check.py`) | true top-k on every row |
| b12x sparse MLA decode | 16 heads, 2051 slots, 528-byte FP8 records, 16–96 rows with padding (`sparse_mla_rows_check.py`, SM120) | cosine ≥ 0.99998 |
| b12x fused NVFP4 MoE | padding tokens routed to expert -1 with NaN inputs (what route top-k emits for NaN logits), 16–96 tokens (`moe_padding_ids_check.py`, SM120) | live tokens bit-identical |
| b12x short convolution, speculative mode | 6144 and 24576 channels, 1–16 sequences, 7 draft columns (`short_conv_spec_check.py`, SM120) | exact |
| Auxiliary-stream overlaps | diagnostic package with the MLA query chain and MoE shared expert serialized | still 0/16 |
| KDA verification and commit implementation | diagnostic package using the portable Triton recorder and replay (SM103's) on the Sparks | still 0/16 |
| Autotuned b12x tactics | diagnostic package prepared with b12x heuristic defaults (autotuning off) | still 0/16 |
| Step inputs to peer ranks | wire format sends each upload separately; device copies are partial and ordered before launch | correct by inspection |
| Rank divergence | the step validator compares every rank's pending token each round; no error 32 | none observed |

Standalone checks used b12x's default configurations with autotuning off;
packages serve autotuned tactics measured on the GB10. vLLM's GLM-5.3 Flash
deployment on two Sparks uses the same b12x kernel families but with tuning
off, PyNCCL collectives, and at most two concurrent requests in its logs, so
it does not cover this case.

## 7. Next steps

1. A TP2 package whose decode rounds skip the MTP draft model entirely (each
   draft repeats the pending token with certainty; no draft forward and no
   draft-cache writes during decode) is being tested at fixed width 4. A pass
   places the corruption in the MTP draft path; a failure places it in target
   verification and commit.
2. Run the RTX PRO 6000 reproduction in section 5.
