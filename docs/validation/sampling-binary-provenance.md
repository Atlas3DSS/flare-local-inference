# FlashInfer sampling library provenance

Sampler SHA-256
`88017d41217c4428528fb6b198279603d021790152e818a0ccec6f593a1f77ed`
was measured after the Ninja build completed on 2026-10-09 at 02:23:40 PDT.
The component tests and 552-block qualification in
[sampling-flashinfer.md](sampling-flashinfer.md) followed that build.

The shared JIT cache was rebuilt at 03:26:35–03:27:12 PDT. Subsequent filesystem
inspection measured SHA-256
`abf5c8569d4a6b948120d4b2628b7c90e3725b0a86a684d2e283289bb5552451`,
with matching modification time and ELF build ID
`b3b0274c44157ccb71c5f27a3a48cf5cb54971c2`.

The three fixed-QSA HTTP receipts embed
[qsa-partitions-install.json](qsa-partitions-install.json) in `installed_patch`.
Its sampler hash identifies the referenced installation; the original HTTP
receipts contain no measurement of the library mapped by each server. Those
requests ran after the 03:27 rebuild, but the retained evidence does not establish
a server-to-binary mapping. Original request outcomes and receipts are preserved.

Six recorded sampler source hashes and the Ninja compiler/link command hashes
match across the inspected builds. Complete transitive build-input identity and
machine-code equivalence are not established. No copy of the `88017…` binary was
found in the inspected sampling caches or evidence archives.

The [audit](sampling-binary-provenance.json) and
[archive](/data/sparkgap-artifacts/validation-receipts/sampling-binary-provenance.tar.gz) preserve the `abf5…` binary, historical
build logs, source hashes and original HTTP receipts. The independently measured
`9dd90584…` build has its own
[37-test and 552-block qualification](sampling-autotune-legality.md), including
the tested binary.
