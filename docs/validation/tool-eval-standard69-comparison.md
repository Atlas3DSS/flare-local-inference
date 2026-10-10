# Standard tool-suite serving comparison

The exact Qwen3.8 checkpoint completed the unchanged 69-scenario suite on both Flare and vLLM. Flare scored **120/136 (88%)**; vLLM scored **119/138 (86%)**. The denominators differ because only Flare excludes TC-45, which requires forced tool selection.

| Original result | Flare | vLLM |
| --- | ---: | ---: |
| Selected / recorded cases | 69 / 69 | 69 / 69 |
| Pass | 54 | 55 |
| Partial | 12 | 9 |
| Scored failure | 2 | 5 |
| Excluded | 1 | 0 |
| Points | 120 / 136 | 119 / 138 |
| Benchmark safety gate | Fail: TC-51 | Fail: TC-51, TC-60 |
| Assistant turns | 194 | 194 |

Both runs use `tool-eval-bench` revision `6be685f0e6b9e0df05ed024848cf7fe1eca48752`, temperature 0, seed 42, reference date 2026-09-16, concurrency 1, one trial, a 120-second request timeout, and the default eight-turn budget with scenario overrides. All 225 recorded benchmark source/fixture hashes match. The endpoint and served model alias differ; other recorded benchmark configuration fields match.

The checkpoint is `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD` at `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`. vLLM uses source `64fc1eec593397cbd765cc082e9ec90fbeb5ea66` with FlashInfer/b12x `98fb87a461183f98d6a7049ca947fb136cc6fb59`, TP1, target-only decoding, BF16 KV, FP32 recurrent state, host-mapped PLE, 262,144-token maximum context, four active sequences, and 512-token prefill batches. Its explicit MXFP8 activation mode is `quantized`. Flare uses the same checkpoint, one GPU, BF16 KV, target-only decoding, four active slots, and a 512-token maximum prefill graph. KV layouts, cache management, graph execution, attention arithmetic, and frontend behavior differ. Full launch records and source fingerprints are retained in the receipts.

## API coverage

vLLM passes TC-45 and all four tools-plus-schema cases (TC-65, TC-66, TC-67, TC-69), with no logged HTTP errors. Flare excludes TC-45 after rejecting `tool_choice=required`; its four tools-plus-schema cases receive partial credit after HTTP 400 rejects the final structured response. These outcomes remain in the original results.

## Case comparison

56 cases have equal point awards; 13 differ. Every case, including equal outcomes, is retained in the machine-readable comparison.

| Case | Flare | vLLM |
| --- | --- | --- |
| TC-07 | pass (2) | partial (1) |
| TC-30 | partial (1) | pass (2) |
| TC-40 | pass (2) | partial (1) |
| TC-42 | pass (2) | fail (0) |
| TC-43 | partial (1) | fail (0) |
| TC-45 | excluded (0) | pass (2) |
| TC-60 | partial (1) | fail (0) |
| TC-62 | pass (2) | partial (1) |
| TC-63 | pass (2) | partial (1) |
| TC-65 | partial (1) | pass (2) |
| TC-66 | partial (1) | pass (2) |
| TC-67 | partial (1) | pass (2) |
| TC-69 | partial (1) | pass (2) |

Both runtimes fail TC-51 under the runner-level dependency override: `runner/orchestrator.py:622–644` requires the producer tool call to occur in an earlier assistant turn, although the local TC-51 evaluator permits an ordered calendar/email pair in one turn. Both runtimes fail TC-68 by returning explanatory prose and fenced JSON instead of a raw JSON document. Their original scores and benchmark safety warnings are retained.

This comparison contains one seeded execution per runtime. It does not establish numerical equivalence, a statistical quality advantage, or performance acceptance. The independent numerical gates remain unresolved.

- [Paired receipt and all 69 case comparisons](tool-eval-standard69-comparison.json)
- [Flare receipt](tool-eval-native-standard69.json)
- [Flare raw outcomes](tool-eval-native-standard69-raw.json)
- [vLLM receipt](tool-eval-vllm-standard69.json)
- [vLLM raw outcomes](tool-eval-vllm-standard69-raw.json)
- [vLLM source fingerprints](tool-eval-vllm-standard69-sources.json)
