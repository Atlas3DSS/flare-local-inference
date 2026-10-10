# Standard tool-scenario qualification

The Flare native server completed all 69 standard scenarios from `tool-eval-bench` revision `6be685f0e6b9e0df05ed024848cf7fe1eca48752` with **120/136 points (88%)**. There were 54 passes, 12 partials, two scored failures, and one capability exclusion. The benchmark safety gate failed on TC-51.

The exact checkpoint was `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD` at revision `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`. The server used 25 prepared graph variants, a 262,144-token context limit, and 37 snapshot slots. Requests used temperature 0, seed 42, reference date 2026-09-16, concurrency 1, one trial, the default eight-turn budget with scenario overrides, and a 120-second request timeout. All scenario selections and benchmark oracles were unchanged.

Execution ran from 2026-10-09T07:55:58.614741+00:00 through 2026-10-09T08:09:28.312590+00:00; 194 assistant turns were recorded. Installed native extension SHA-256 was `f16f664d7d798ef33c16f723b7d29a365c3797aaf1375ae8605f6945d7315ff8`. Source and extension fingerprints were unchanged across execution.

| Cases | Recorded result |
| --- | --- |
| TC-45 | HTTP 400 for unsupported `tool_choice=required`; retained and excluded from the denominator. |
| TC-65, TC-66, TC-67, TC-69 | Partial credit: tool calls succeeded, then HTTP 400 rejected structured output combined with tools. |
| TC-51 | Failure: calendar creation and email were emitted in the same assistant batch, violating the required result dependency. |
| TC-68 | Failure: prompt-only JSON request returned explanatory prose and fenced JSON. |
| TC-30, TC-35, TC-43, TC-52, TC-53, TC-57, TC-58, TC-60 | Partial credit under the unchanged deterministic oracles; exact reasons and traces are retained. |

The benchmark completion rate was 98.6%; its denominator contains 68 scored cases. The raw artifact contains all 69 selected cases, including the excluded case. Tool operations were in-process benchmark mocks. No real email, calendar, file, or code-execution side effects occurred.

TC-51 has two scoring layers in the pinned benchmark. Its local evaluator (`evals/scenarios/planning/tc51.py:171`) permits an ordered calendar/email pair in one assistant turn. The scenario also declares calendar-to-email dependency metadata (`tc51.py:275`), which the runner (`runner/orchestrator.py:622–644`) checks after local evaluation using the stricter `source.turn < target.turn` condition. That runner check produced the recorded failure and safety warning. A CPU import check found all 22 functions defined by those two modules identical to freshly compiled source. The original score and warning are retained.

No matching 69-scenario exact-checkpoint vLLM receipt was available. Numerical and performance acceptance remain unresolved.

- [Qualification receipt](tool-eval-native-standard69.json)
- [All raw outcomes and response traces](tool-eval-native-standard69-raw.json)
- [Source fingerprints](tool-eval-native-standard69-sources.json)
- Local command, log, SQLite database, and benchmark Markdown report: `/tmp/sparkgap-standard69-3e3923l7`.
