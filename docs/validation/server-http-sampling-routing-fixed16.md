# Sampling graph routing with fixed QSA partitions

The saved sampling-routing fixtures pass **118/120 checks** with fixed 16-way
QSA attention. Eleven model requests and 126 total HTTP requests complete with
zero case errors. The [receipt](/data/sparkgap-artifacts/validation-receipts/server-http-sampling-routing-fixed16.json) and
[archive](/data/sparkgap-artifacts/validation-receipts/server-http-sampling-routing-fixed16-artifacts.tar.gz) preserve all
requests, responses, source identities, counters and failures.

Greedy/general graph selection, seeded full-hit repeatability, mixed-temperature
batching, constrained/unconstrained shared decode, the one-in-flight grammar
barrier and final resource accounting pass. The two failures remain
`grammar-mixed.stream_complete` and `grammar-mixed.exact_literal`: the constrained
request reaches its unchanged 1,024-token budget and finishes with `length` after
3,066 bytes of the required 3,074-byte JSON string. Its content is byte-identical
to the [row-dependent QSA result](server-http-sampling-routing.md) and is an exact
prefix of the required literal. No budget, fixture or assertion changed.

Fixture SHA256 is
`b8f529f1522343d88e3b97874748f1a1c300ca08e0e37c3f30dd0b4853a56f6e`.
The embedded sampler hash references the installation snapshot, as detailed in
[sampling library provenance](sampling-binary-provenance.md); the original HTTP
receipt does not pin the loaded sampler binary.
The checkpoint, FlashInfer sampler, native ABI 2 library and 37-graph catalog
match [the default HTTP qualification](server-http-fixed16.md). The actual CLI
PID was 1078008, explicitly bound to `127.0.0.1:46279`. Its shutdown exits zero,
releases GPU storage and closes the listener. Request timings are diagnostic;
they do not establish a serving performance bound.

An attempt against an earlier `--bind 127.0.0.1:0` process was rejected before
HTTP traffic because its literal CLI bind argument differed from the resolved
endpoint. The [rejection receipt](server-http-sampling-routing-fixed16-bind-failure.json)
and log are retained. The explicit-port server passes the unchanged process
identity check.

The evidence archive SHA256 is
`8aa0c2b07bdf569a7ad311695082e034d6f8496baca8663a152e21beb699a2cf`.
Use the saved `fixtures.json`, an exclusively assigned server with an explicit
bind address, and fresh artifact/output paths to reproduce the command recorded
in the JSON receipt.
