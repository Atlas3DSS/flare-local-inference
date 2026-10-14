# GLM-5.3 Flash: batched speculative decoding corrupts answers on DGX Spark

Status: open. The root cause is not yet identified.

## 1. Summary

On DGX Spark (GB10, SM121), with one rank per Spark over RoCEnante, GLM-5.3
Flash produces corrupted answers when MTP speculative decoding verifies several
requests in one round. It reproduces at tensor parallel 2 and 4. With eight or
more concurrent requests most replies are empty, invent content, or name the
wrong record. The same packages answer correctly with speculation disabled
(`--mtp-draft-tokens 0`) and with speculation limited to one or two concurrent
requests. On a GB300 (SM103, one GPU) the identical test is correct at 16
concurrent requests.

Two other symptoms follow from it:

- **Long prompts without a cached prefix answer wrongly.** Right after its
  prefill, a request runs one speculative round at each larger batch capacity
  the server allows (16, 8, 4, then 2 rows, with the request in row zero and the
  rest inactive) before settling at batch one. Those first rounds corrupt it.
  A request that restores a cached prompt starts at batch one and skips them.
- **Concurrent streams occasionally fail with device error 6
  (`INVALID_SPECULATION`).** An instrumented build that reports the failing
  check shows the verifier rejecting the target distribution (non-finite or
  empty probabilities), often in several rows of the same round.

## 2. Environment

| Item | Value |
| --- | --- |
| Reproduced on | A development branch of Flare `main` carrying unmerged multi-host changes, with FlashInfer 64916e85 plus the RoCEnante variable-length all-gather from [local-inference-lab/flashinfer#12](https://github.com/local-inference-lab/flashinfer/pull/12) |
| Flare `main` | Reproduction with `main` and its pinned FlashInfer (64916e85, `pyproject.toml`) is in progress; this section will record the result. Sections 4 and 5 use `main` |
| Checkpoint | `local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD` (revision dec48abd) |
| Packages | TP2: `--ranks 2 --max-concurrency 8`; TP4: `--ranks 4 --max-concurrency 16`; both `--gpus 0 --prefill-chunk-size 4096 --no-context-parallel` with the default adaptive MTP |
| Hardware | DGX Spark, ConnectX-7 at 200G through one switch |
| Capacity per rank | TP2: 1,107,968 KV tokens, 54 state rows (1,159,680 tokens and 56 rows with speculation off). TP4: 6,716,416 KV tokens, 176 state rows. FP32 recurrent and BF16 convolution state |
| Sampling | The model card's T=1, top-p 0.95, with a seed per request |

## 3. Results

[`scripts/repro_glm_batched_speculation.py`](../scripts/repro_glm_batched_speculation.py)
produced every row. The 10K prompt is the script's default `--size 8000`
(about 10.5K tokens; answer `2U2GYCZF, 198`). "Cached" publishes the prompt
with a one-token request, then sends N copies at once with different seeds, so
every row of the speculative batch is a live request restored from the same
checkpoint. "Uncached" sends prompts that begin with distinct nonces, one at a
time.

| Server | Test | Correct |
| --- | --- | --- |
| GB300, adaptive MTP, concurrency 16 | cached 10K ×16, two runs | 15/16, 13/16 |
| GB300, adaptive MTP | cached 96K ×16 | 16/16 |
| GB300, adaptive MTP | uncached 133K–260K | 14/14 |
| Spark TP2, adaptive MTP, concurrency 8 | cached 10K ×8, two seeds | 1/8, 0/8 |
| Spark TP2, adaptive MTP | cached 10K ×2 | 2/2 |
| Spark TP2, speculation off, concurrency 8 | cached 10K ×8 | 8/8 |
| Spark TP2, MTP draft removed from decode rounds (diagnostic, see section 6), fixed width 4, concurrency 8 | cached 10K ×8, two seeds | 0/8, 0/8 |
| Spark TP2, same diagnostic | cached 10K ×2 | 2/2 |
| Spark TP4, adaptive MTP, concurrency 16 | cached 10K ×16 | 1/16 |
| Spark TP4, adaptive MTP, concurrency 16 | cached 96K ×16 | 0/16 (13 reached the 16,384-token limit) |
| Spark TP4, adaptive MTP | cached 10K ×2, ×4, ×8 | 2/2, 3/4, 1/8 |
| Spark TP4, fixed width 2 (`--mtp-draft-tokens 1 --no-mtp-adaptive-verification`), concurrency 8 | cached 10K ×8 | 3/8, all five misses the neighbouring record (no corruption) |
| Spark TP4, fixed width 4 (`--mtp-draft-tokens 3`), concurrency 8 | cached 10K ×8 | 1/8 |
| Spark TP4, speculation off | cached 10K ×16 | 10/16 |
| Spark TP4, speculation off | uncached 48K–133K | 5/5 |
| Spark TP4, adaptive MTP, default concurrency | uncached 48K–133K | about 2/15 |
| Spark TP4, adaptive MTP, `--max-concurrency 1` or `2` | uncached 96K–133K | 5/5, 4/4 |

With speculation on, the failures are empty replies that reason until the
token limit, invented or truncated codes (`2U2GYZF`, `2Q7XV81M`), claims that
the prompt holds no code, and records far from the needle. The misses with
speculation off all name the neighbouring record (199 for 198), a model-level
miss the GB300 also makes at a similar rate. Those are a separate question.

The trigger is the number of query rows in the speculative graph, batch
capacity times verification width: corruption appears above 16 rows and not
at or below it. Plain decode at 16 requests, eight requests at width 2, two at
width 6 and the draftless diagnostic at two requests and width 4 are all
correct apart from the neighbouring-record misses; eight requests at width 4
or 6, and sixteen at any width above one, corrupt. A single live request also
fails when it runs in a larger graph (the post-prefill rounds above), so the
graph's planned capacity matters, not the number of live requests. With the
MTP draft removed, every round commits exactly one token per request, as plain
decode does, and still fails, so the fault is in the target verification
graph rather than in the draft model. Kernels that change their planned
variant above 16 rows on SM12x include the b12x DSA indexer, which switches
from its fused route to its tiled route there.

## 4. Reproduce on DGX Spark

Build Flare `main` with its pinned FlashInfer (`git -C ../flashinfer checkout
64916e85`) and set up multi-host serving as in [multi-host.md](multi-host.md):
`flare agent` on every peer with the same token, and RoCE between the Sparks.
Two Sparks suffice. Host names and addresses below are examples.

1. Export a TP2 package. On rank zero:

   ```sh
   .venv/bin/ignition export glm-5.3-flash --ranks 2 --rendezvous 10.200.0.12:29630 \
     --gpus 0 --max-concurrency 8 --no-context-parallel \
     --output /data/flare-runtime/glm53-tp2.safetensors --verbose
   ```

   On the peer:

   ```sh
   .venv/bin/ignition join 10.200.0.12:29630 --rank 1 --ranks 2
   ```

   For four Sparks, use `--ranks 4 --max-concurrency 16`, run one `join` per
   peer with ranks 1–3, and pass `--concurrency 16` in step 3.

2. Serve from rank zero with the package defaults (adaptive MTP):

   ```sh
   ./flare/target/release/flare run /data/flare-runtime/glm53-tp2.safetensors \
     --hosts spark2 --gpus 0 --bind 0.0.0.0:8000
   ```

   Accept the development publisher on first use, or pass `--publisher-keys`,
   `--dangerously-approve-key` and `--dangerously-accept-publisher` for an
   unattended start.

3. From any machine that can reach the server:

   ```sh
   python3 scripts/repro_glm_batched_speculation.py cached --url http://spark1:8000 --concurrency 8
   ```

   A healthy server answers nearly all eight. The defect gives zero or one,
   mostly empty.

4. Controls. Restart the server with one option at a time and repeat step 3:
   - `--mtp-draft-tokens 0`: correct (8/8 at TP2).
   - `--max-concurrency 2`, with `--concurrency 2` in the script: 2/2.

5. Long prompts without a cached prefix, sent one at a time (about 96K tokens
   each). They fail on the default server and pass under either control:

   ```sh
   python3 scripts/repro_glm_batched_speculation.py cold --url http://spark1:8000 --size 70000 --count 3
   ```

## 5. Reproduce on RTX PRO 6000 (SM120)

Flare's GLM recipe selects SM103 kernels on SM103 and b12x kernels everywhere
else, so an SM120 host runs the same path as Spark: the b12x DSA indexer, sparse
MLA, KDA verification recording and commit, NVFP4 MoE, blockscaled GEMMs and
short convolution. Three things differ: b12x compiles for `sm_120a`; autotuning
measures tactics on the local GPU (188 SMs against GB10's 48); and
tensor-parallel collectives use b12x's PCIe transport instead of RoCEnante.
A reproduction on SM120 therefore separates a defect in Flare's SM12x
speculative path from one that needs GB10 tactics or RoCE. GLM has not yet
been run this way.

The checkpoint holds about 166 GB of weights, so use four 96 GB cards in one
host at tensor parallel 4 (about 44 GB of weights per rank). Two cards leave no
room for the KV cache. The defect appears at both TP2 and TP4 on Spark, so TP4
is a fair test.

1. Export, one rank per GPU:

   ```sh
   .venv/bin/ignition export glm-5.3-flash --gpus 0-3 --max-concurrency 16 \
     --no-context-parallel --output glm53-tp4-sm120.safetensors --verbose
   ```

2. Serve:

   ```sh
   ./flare/target/release/flare run glm53-tp4-sm120.safetensors --gpus 0,1,2,3 --bind 0.0.0.0:8000
   ```

3. Run section 4's steps 3 and 4 against this server with `--concurrency 16`
   (and 2 for the control). Report the counts and the GPU model (Workstation or
   Max-Q).

If it fails like Spark while the controls pass, the defect lies in Flare's
SM12x speculative path or the b12x kernels it uses, independent of RoCE and of
GB10 tuning, and SM120 becomes the quicker place to bisect. If it passes, the
defect needs GB10 tactics, RoCEnante, or ranks on separate hosts.

## 6. What has been ruled out

Each check passed at the shapes the failing rounds use. Kernel checks ran
standalone against b12x or PyTorch references.

| Component | Check | Result |
| --- | --- | --- |
| RoCEnante all-reduce and vocabulary all-gather | Four Sparks, 2–96 rows (up to 7.4 MB per rank), CUDA-graph replays with a new pattern each replay | No mismatched bytes |
| b12x KDA verification recording and commit | TP4 geometry (16 heads), 1–16 sequences of mixed length, inactive rows, strided state indices as Flare passes them | Exact |
| b12x DSA indexer (32 heads, top-512 pools) | Fused route (up to 16 query rows) and tiled route (more), graph replays with changing inputs, inactive rows, scratch overwritten between replays | True top-k on every row |
| b12x sparse MLA decode | 16 heads, 2051 selected slots, 528-byte FP8 records, 16–96 rows with inactive rows (SM120) | Cosine ≥ 0.99998 |
| b12x NVFP4 MoE | Inactive tokens routed to expert -1 with NaN inputs, which route top-k produces for NaN logits; 16–96 tokens (SM120) | Live tokens bit-identical |
| b12x short convolution, speculative mode | 6144 and 24576 channels, 1–16 sequences, 7 draft columns (SM120) | Exact |
| Concurrent branches | Package with the MLA query chain and the MoE shared expert (`ConcurrentSharedExpert`) run serially | Still fails (0/16) |
| KDA verification implementation | Package that records and replays KDA verification through the portable Triton path (`glm53/kda_recovery.py`) on SM121 | Still fails (0/16) |
| Autotuned b12x tactics | Package prepared with b12x heuristic defaults (autotuning off) | Still fails (0/16) |
| MTP draft model and draft-cache writes | Package whose decode rounds propose the pending token with certainty, with no draft forward or draft-cache writes (`glm53/speculative.py`), fixed width 4 | Still fails (0/8) |
| Per-step inputs to peer ranks | Each upload is sent separately and copied before the graph launches | Correct by inspection |
| Ranks disagreeing on tokens | The step validator compares every rank's pending token each round (error 32 on mismatch) | No mismatches |

The standalone kernel checks used b12x's default configurations; packages
normally serve autotuned tactics. A vLLM deployment of GLM-5.3 Flash on two
Sparks that we inspected uses the same b12x kernel families, but with b12x
tuning off, PyNCCL collectives and at most two concurrent requests in its
logs, so it does not cover this case.

## 7. Next steps

1. Confirm the reproduction on Flare `main` with FlashInfer 64916e85
   (section 2).
2. Test a package exported with `--no-compile`. Each rank then captures its
   own uncompiled forwards, which removes both suspects that depend on the
   speculative graph's shape: compiled forwards shared between ranks with
   per-process integer handles baked in, and Inductor or Triton code generated
   for those shapes on `sm_121`.
3. Run the RTX PRO 6000 reproduction in section 5.
