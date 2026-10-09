# Validation index

Sparkgap targets `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`,
revision `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`. Hardware results in this
index use NVIDIA SM120; they do not establish SM121 behavior.

[Acceptance coverage](acceptance-gaps.md) states the established behavior and
remaining gates. The [correction plan](../core-corrections-closeout.md) contains
the implementation checklist. Individual reports define their exact model,
source, workload, and measurement conditions.

## Model correctness

| Contract | Result and limitation | Reports |
|---|---|---|
| Text serving reference | All 48 layers are traced. Cross-stack normalization differences precede amplified logit differences; repeated vLLM runs also diverge because their MoE reductions are nondeterministic. | [Serving oracle](text-serving-oracle.md), [layer comparison](text-layer-comparison.md) |
| Generation agreement | The fixed 82-prompt, 64-token corpus exists. Sparkgap and repeated-reference captures are pending. | [Corpus](greedy-agreement-corpus.json), [comparison tool](../../qualify/generation_agreement.py) |
| MTP operators | Common-input checks isolate projection arithmetic and normalization behavior; complete learned-MTP serving remains unqualified. | [Operator comparison](mtp-operators.md), [module comparison](mtp-module.md) |
| Vision | Serving-path encoder differences remain open; image admission and preprocessing checks are separate. | [Serving comparison](vision-serving-reference.md), [encoder diagnostics](vision-eager-bf16.json) |
| Prefix arithmetic | Fixed QSA partitions and sequential single-key-split GDN pass the recorded exact restoration cases. Free policies need calibrated cross-capacity acceptance. | [4,096-token catalog](prefix-replay-prefill4096.md), [GDN policy](gdn-recurrence-policy.md), [unconstrained failures](prefix-replay-autotuned.md) |

Complete-logit and encoder relative L2 are diagnostics. Acceptance requires
identical-input operator checks, exact metadata/state lifecycle, generation
agreement against the reference's own variation, and task-level output checks.
Same-graph replay and same-capacity restoration remain bitwise checks.

## Execution, preparation, and serving

| Contract | Result and limitation | Reports |
|---|---|---|
| Python/native replay | Ordinary and copy catalogs preserve outputs, persistent state, and allocation counts through the tested sequences. Copy coverage includes all 14 shapes and snapshot/restore; independent W1 numerics remain open. | [Ordinary replay](graph-execution-parity-prefill4096.md), [copy replay](copy-state-replay.md) |
| Shared scratch and memory planning | Operator workspaces are caller-owned and reused across layers. Final KV sizing accounts for graph and allocator consumption. | [Shared scratch](unified-scratch.md), [hyperconnection allocation](hyperconnection-priming.md), [prefill capacity](prefill-capacity-planning.md) |
| Autotuning | Preparation declares work before tuning, refreshes selected bindings, and freezes before capture/handoff. CPU tests invoke real constructors and plans for ordinary/copy/MTP catalogs without CUDA. | [Bootstrap](autotune-bootstrap.md), [constructor tests](../../tests/preparation/test_bootstrap_construction.py) |
| Text, images, and grammar | Core HTTP cases, JSON/SSE output parsing, string stops, and image-conditioned cache identity pass under their recorded configurations. All 26 JPEG/EXIF fixtures are admitted; two constrained outputs hit their explicit budget in whitespace loops. Tool/schema composition remains open. | [JPEG/EXIF HTTP](jpeg-exif-http.md), [output protocol](output-protocol-http.md), [text HTTP](server-http-autotuned.md), [vision/grammar HTTP](server-http-vision-grammar-autotuned.md), [prefill HTTP](prefill-capacity.md) |
| Agentic workloads | Copy-adaptive serving completes 23 of 24 admissions and passes 16 of 24 manual checks. The report retains the API and output failures. | [Copy-adaptive workload](speculative-agentic-serving.md), [69-scenario comparison](tool-eval-standard69-comparison.md) |
| Copy graph execution | Forced-literal serving exercises every supported shape and accepts copied drafts. Independent one-token/state comparisons remain required. | [Copy serving](speculative-copy-serving.md) |
| Capacity | Recorded maximum-context and aggregate-image requests execute and reclaim resources. Numerical accuracy remains a separate gate. | [Context limit](server-http-context-limit.json), [aggregate images](server-http-vision-aggregate.md) |
| Resource and failure lifecycle | Recorded pressure, cancellation, startup rejection, and worker failures test bounded ownership and drain behavior. | [Pressure](/data/sparkgap-artifacts/validation-receipts/server-http-pressure.json), [startup](native-startup-validation-abi2.md), [worker failure](native-worker-failure.md) |

The production sampler has one FlashInfer path. Temperature zero produces a
point mass through it. Historical reports describing separate greedy graphs
retain their measurement scope and do not describe the production catalog.
Ordinary execution uses ABI 2; copy and learned speculation use ABI 3.

## Performance and kernel contracts

| Area | Established measurement | Remaining work |
|---|---|---|
| Serving and replay | [Ordinary serving](serving-performance.md) measures replay overhead, concurrency, prefill, and prefix reuse with its recorded policies. The [node attribution probe](graph-node-attribution.md) verifies capture-to-trace identity. | Named B1/B4 kernel timelines and speculative replay-floor comparison. |
| Sampling | [FlashInfer composition](sampling-flashinfer.md) measures filtering and sampling cost with explicit RNG and allocation checks. | Bounded top-k is deferred. |
| Chain acceptance | [Uniform-endpoint regression](flashinfer-chain-endpoint.md) verifies probability-one acceptance through the public FI sampler. | Preserve the regression during sampler changes. |
| Attention | [Fixed split contract](qsa-fixed-splits-public-api.md) and [long partitions](qsa-long-partitions.md) qualify their tested shapes. | Measure and calibrate the adaptive split policy before removing the recipe pin. |
| Speculation | All 14 legal graph shapes are prepared and served. | Fixed-width distributions, band transition, recurrent-state write cost, learned-MTP qualification, and paired adaptive throughput/latency. |

The historical ordinary measurement finds most step time inside captured
kernels. It does not attribute that time to named operators, establish learned
MTP speedup, or resolve complete-model numerical acceptance.

## Debugging data

Large payloads live under `/data/sparkgap-artifacts`; repository JSON locators
and [artifact locations](artifact-locations.json) resolve moved files. The
[FlashInfer source migration](flashinfer-source-migration.json) describes the
preserved staged-source material. Development source revisions are tracked in
Git and pinned in `pyproject.toml`.
