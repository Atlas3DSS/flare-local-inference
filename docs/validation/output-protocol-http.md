# Native output protocol HTTP checks

The native server passes six HTTP checks with the Qwen checkpoint at revision
`dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`, copy speculation, and a 4,096-token
prefill catalog on SM120. The installed native extension was built from
`c323fe5`; sampling uses temperature 1, top-k 20, and top-p 0.95.

- JSON and SSE structured output preserve literal `<think>` and `<tool_call>`
  strings as content without emitting tool calls.
- JSON and SSE string stops return `before` from `beforeENDafter` with stop
  string `END`. Both finish with `stop`; completion usage includes the two raw
  generated tokens and agrees with returned token IDs.
- The unchanged `incident-turn-1` request emits `read_incident` with argument
  `{"incident_id":"INC-204"}` and finishes with `tool_calls` after 41 tokens.
- The unchanged `code-docstrings-stop` request completes after 541 tokens within
  its 1,024-token budget. All eight functions have docstrings, their original
  bodies are AST-equivalent, `END_EDIT` is absent from visible content, and the
  finish reason is `stop`.

Each SSE response includes exactly one terminal `[DONE]` event. These checks
cover output presentation and string-stop admission through the installed
service. The complete 24-request workload, tool/schema composition, and learned
MTP remain separate acceptance requirements.
