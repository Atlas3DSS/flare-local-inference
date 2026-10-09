# Copy speculation through native HTTP serving

The exact Qwen checkpoint executes copy-only speculation at temperature 1,
top-k 20 and top-p 0.95 through the Rust HTTP server. No MTP weights are loaded.
All **36 assertions pass across 14 requests**, including concurrency 1/2/4,
grammar-constrained output and budgets of 1, 2, 3, 7, 8, 9 and 15 tokens.

The native metrics record **300 accepted copied draft tokens** and execution of
all 14 prepared speculative shapes: batch one at widths 1–8, batch two at
widths 1–4 and batch four at widths 1–2. Responses match the required literal
text; truncated responses match its prefix and exact requested token count.
The workload also records six prefix-cache hits and 384 reused tokens.

Preparation captures 39 graphs with 4,096-token chunked prefill, a 262,144-token
context limit and a configured 4 GiB headroom. The shared KV capacity is 415,040
tokens. After requests drain, SIGTERM exits zero. The [receipt](speculative-copy-serving.json)
links raw requests, responses, metrics, memory accounting, source fingerprints
and the preparation log under `/data/sparkgap-artifacts/mtp-implementation/`.

Forced literal output establishes transaction and grammar behavior on these
requests. Representative agentic acceptance, paired performance, an independent
all-width-one comparison, image-conditioned speculation and complete persistent
state parity remain unqualified. MTP drafting and its numerical agreement are
separate requirements.

A separate sampler regression reproduces a rare FlashInfer failure at a Philox
uniform draw of exactly one: equal proposal and target distributions reject a
certain draft and emit token zero from an empty correction distribution. Its
raw failure is retained in the receipt. The [endpoint correction](flashinfer-chain-endpoint.md)
passes the paired GPU reproducer and 72 sampling regressions. The HTTP results
above retain their original source and sampler identity.
