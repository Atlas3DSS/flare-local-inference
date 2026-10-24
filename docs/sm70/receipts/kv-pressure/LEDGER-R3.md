# Prefill lab round 3 ledger (Flare SM70, V100 PCIe 32 GB x 6, 2026-10-24 morning)

Targets: (1) KV-pressure preemption + ~33 s re-prefill at 4 x 30.7k-class contexts (cached prefixes or 2k outputs) on
the 976-page G128 pool; (2) the prefix-cache credit anomaly (c1 long leg 0/12 credits with --prefill-srpt vs 7/12
default). Base: qualified G128 package f21b9abb + public binary 48b69ed4 (v3 SRPT) and lab v2 binary 83b71a1c (round-2
SRPT refinement, config Y). Lab tree <release-host> <release-host>/flare/<lab-tree>-r3 (cp -a of <lab-tree>);
outputs <release-host> <release-host>/flare/prefill-lab-r3/; local scripts prefill-lab/r3/scripts/. Rules as rounds 1-2:
run_card.sh only, --memory-limit 31G, own units Nice=10, scheduler/cache-policy changes only
after Argon clears them, gates never relaxed. Measurement = stock binaries + a read-only 0.25 s /metrics poller
(poll_metrics.py) next to the release long leg (r3_long.sh = r2_chain.sh's long leg + poller).

| # | goal | change | review | card / lease | result | exact | decision | GPU-min |
|---|---|---|---|---|---|---|---|---|
| D0 | per-run counter deltas of existing long receipts (CPU) | none (run_deltas.py over requests.json metrics_before/after per group) | n/a | <build-host>/<release-host> CPU | see "Target 2" and "Target 1" sections | n/a | measurement | 0 |
| M1 | KV re-prefill repro: default policy, release long leg c4 x 3, **2,048 output tokens** | none (public 48b69ed4) | n/a | <card> (<unit>) | c4 walls 275.3 / 109.8 / 183.7 s; cold mean / max last token 229.0 / 275.3 s; 4 preemptions, 4 stalls > 30 s; prefill counter 289,445 (vs 172k of prompts actually missed). Cold rep: 3 boundary checkpoints (back 24 -> 40 -> 64 pages), 4th prompt (long-03) none; 145.09 s PREEMPT long-00 (host record resident) -> 147.34 s readmission no device / no host hit -> full re-prefill; 163.10 s 2nd PREEMPT long-03 (no checkpoint); stalls 88.5 / 91.7 s | preempted requests' greedy tokens differ from same-seed unpreempted runs a few tokens after resume (pre-existing) | mechanism (see Target 1) | 25 |
| M2 | same, config Y (lab v2 83b71a1c, short 2048 + decode budget 4096) | none | n/a | <card> (<unit>) | walls 252.9 / 207.1 / 151.8; cold mean / max 157.1 / 252.9 (no preemption in the cold rep: Y serialises prefill + decode); cached rep r1: 2 host restores at start, 265.92 s PREEMPT long-01 -> readmission no hit -> 123.4 s stall; long-00 169.1 s stall | as M1 | Y avoids the cold-rep cascade; cached reps hit the fallback | 26 |
| M3 | credit anomaly repro: v3 --prefill-srpt, 512 tokens, c4 x 3 + c1 x 12 | none | n/a | <card> (<unit>) | c4 walls 153.5 / 54.6 / 84.5; preemptions 2, each followed by a host restore (114.3 s, 168.6 s; stalls 35 s = waiting behind the SRPT leader, not re-prefill); **c1 credits 1/12** (g128 chain: 0/12) | n/a | anomaly not reproducible as a number (0 -> 1) | 27 |
| M4 | control: default, 512, c4 + c1 | none | n/a | <card> (<unit>) | walls 152.2 / 86.5 / 54.1; 3 preemptions, all readmitted by host restore or device hit (no stall > 30 s); **c1 credits 6/12** (g128 chain 7/12), all device hits | n/a | control | 26 |
| M5 | knob: v3, 512, c4 + c1, **--prefix-cache-memory 24G** | none | n/a | <card> (<unit>) | **c1 credits 12/12** (11 host restores ~2.0 GB in, each c1 request 6.6-8.3 s); host records 19-25; c4 walls 153.5 / 84.9 / 89.2 | n/a | host capacity resolves the c1 credits for the option | 22 |
| M6 | knob: default, 2k, c4, --prefix-cache-memory 24G | none | n/a | <card> (<unit>) | walls 279.6 / 148.9 / 69.4; cold mean / max 233.5 / 279.6; 4 preemptions; cold rep 143.33 s PREEMPT long-03 (message + prompt checkpoints host-stored 125.6 s) -> 148.08 s readmission no hit (record resident) -> 93.6 s stall, 2nd PREEMPT long-01 (no checkpoint) 92.5 s | n/a | **host size does not fix target 1** | 26 |
| SIM-3 | CPU event sim (<lab-tree> helios/src/scheduler/prefill_lab_r3_tests.rs, ignored tests): real scheduler + prefix cache, 976 pages, 32 rows aliased top-down (~22 pages each), growable outputs, message/prompt boundaries, runtime park/readmit protocol, write-through LRU host model (729 + 12 slabs) | none | n/a | <build-host> CPU | reproduces the structure (no 4th boundary checkpoint, no terminal snapshot at preemption, fallback re-prefills, cyclic-LRU c1 credits); W (host hits wait) never worse on c4 means (-9 .. -25 s), W+F1 removes the cold-rep MISS; c1 credits chaotic (0-9/12 for tiny changes) | n/a | design input for W / F1 | 0 |
| M7 / M8 | c1-only legs from a cold server (no c4 history): --prefill-srpt (M7) and default (M8), 512 tokens, 3 reps | none | n/a | <card> / <card> | **both 0/12 credits, identical pattern** (every request a miss: device and host each hold the 3 most recent of 4 cycling prompts) | n/a | the option has no effect at c1 | 18 + 18 |
| M11 | knob: default, 512, c4 + c1, 8 GiB memory + **--prefix-cache-disk NVMe 64G** | none | n/a | <card> (<unit>) | **c1 12/12** (10 read-through host restores, 2 device hits); disk dir 12 GB | n/a | disk tier also resolves the credits | 22 |
| ARGON-1 | review of W + F1 (<review record>, 19 KB, diff-only) | - | **CLEARED WITH REQUIRED CHANGES** (PROMPT-prefill-r3-kv.answer.md): diagnosis + page accounting confirmed (inbound short by exactly one row: 252 > 250 pages); no deadlock; defaults byte-for-byte; required (1) reset capacity_since_ns on admission (stale wait poisoned readmissions), (2) F1 newest-first without a host store + stronger test, (3) unit test of the wait/fallback gate | - | - | - | all three applied -> lab binary flare-lab-r3-w3 b9b007a9 (diff prefill-lab/r3/r3-w3-core.diff sha 175abb76); helios 812/0/21 ignored, flare 104+4+4; both tests mutation-checked | 0 |
| ARGON-2 | final delta review of w3 (<review record>, 17 KB) | - | **CLEARED** (PROMPT-prefill-r3-kv-v2.answer.md): all three required changes verified; defaults unchanged with both env knobs unset | - | - | - | w3 cleared for the lab GPU A/B (already running) | 0 |
| BASE-2 | baseline repeats for the A/B (stock binaries): M9 default 2k (<card>), M10 Y 2k (<card>), M12 Y 512 c4 + c1 (<card>), M13 default 2k (<card>, running) | none | n/a | as listed | readmission outcome after each preemption (poller: next single lookup): default 2k M1 MISS x4 (waste 117k prefill tokens), M6 MISS x3 + HOST (90k), M9 HOST x1 (34 s wait; waste 0.2k); Y 2k M2 MISS x2 (59k), M10 MISS (30k); 512: M4 HOST/HOST/GPU, M3 HOST/HOST, M11 HOST/HOST, M5 HOST/MISS/MISS (58k), M12 MISS (29k). Cold-rep mean last token default 2k 229.0 / 186.1 (M1 / M9; r2 195.6), Y 2k 157.1 / 158.7 | n/a | variance reference | 24 + 26 + 22 + (running) |
| AB1 | **W** (lab knob W): default policy, 2k outputs, c4 x 3 | lab w3 b9b007a9 | ARGON-1/2 | <card> (<unit>) | walls 198.8 / 143.5 / 134.0; cold mean / max last 183.4 / 198.8 s; all-rep mean 131.2; 4 preemptions, readmissions HOST +34 / +69 / +127 s (waits, no plain re-prefill), **re-prefill waste 1.8k tokens**; cold rep: PREEMPT 140.8 s -> restore 175.1 s (long-03 finished) -> resume 176.6 s, no second preemption | unpreempted requests token-identical to same-seed baselines; preempted ones diverge after resume as in the baselines | **W kept** | 19 |
| AB2 | **W + F1** (both knobs), default 2k | lab w3 | ARGON-1/2 | <card> (<unit>) | walls 207.6 / 190.9 / 122.3; cold 185.3 / 207.6; all mean 137.9; readmissions HOST / MISS +59 s / MISS +154 s / HOST; **waste 30.3k** | as AB1 | F1 does not separate (worse here) -> **F1 dropped** | 19 |
| AB3 | **W**, config Y, 2k | lab w3 | ARGON-1/2 | <card> (<unit>) | walls 253.9 / 159.4 / 124.1; cold 159.0 / 253.9 (no preemption in Y's cold rep); all mean **106.3**; 1 preemption -> HOST +33 s; **waste 1.2k** | as AB1 | W kept | 19 |
| AB4 | **W + F1**, Y 2k | lab w3 | ARGON-1/2 | <card> (<unit>) | walls 252.5 / 158.3 / 104.2; cold 160.6 / 252.5; all mean 103.1; 1 preemption -> HOST +29 s; waste 1.1k | as AB1 | = AB3 (F1 inert) | 16 |
| M13 | third baseline: default 2k (stock 48b69ed4) | none | n/a | <card> (<unit>) | walls 205.4 / 183.8 / 179.5; cold 185.5 / 205.4; all mean 162.0; readmissions HOST +35 / MISS / MISS / MISS; waste 90.0k | n/a | baseline | 21 |
| AB5 | **W + F1**, default policy, 512 tokens, c4 x 3 + c1 x 12 | lab w3 | ARGON-1/2 | <card> (<unit>) | c4 walls 146.8 / 118.6 / 52.4 (M4 152.2 / 86.5 / 54.1); cold mean / max 144.6 / 146.7; 1 preemption -> HOST +3 s; waste 0.3k; c1 credits 6/12 | as AB1 | no regression at 512 (preemptions were already restored by the host at this length) | 22 |
| PUB-W | public form of W only (F1 omitted: the A/B does not separate it): `flare run --prefix-cache-restore-wait` -> server config `host_restore_wait` -> PolicyConfig::host_restore_wait (read by the runtime admission; no environment variable), default off. prefill-lab/r3/r3-w-public.diff (293 lines, sha256 74305ae2), on top of the round-2 integration diff (srpt-r2-public.diff, df92551a; `patch -p1 --dry-run` clean on the 4528c220 + df92551a tree, does NOT apply to bare 4528c220). Tests: helios 812 passed / 0 failed / 19 ignored (810 + config parse + host_hit_waits_for_its_restore_within_the_recompute_bound, mutation-checked), flare 105 + 4 + 4 (+ CLI default-off test) | <build-host> <reference-host>/flare/prefill-lab-build/<build-tree>, binary flare-lab-r3wpub-pub1 3bc108ca | W logic as cleared by Argon (ARGON-2); the env -> CLI/config plumbing is new and NOT yet reviewed | <card> (<unit>) smoke: see PUB-SMOKE | - | - | handed to the coordinator | 0 |

## LIMITATIONS text (non-exact preemption), for docs/sm70
Preemption under KV pressure is not output-exact. When concurrent long requests outgrow the KV pool (for example four
~29k-token contexts with long outputs on the 976-page G128 package, where each prefix-checkpoint row also borrows about
22 KV pages), the server preempts a decoding request and readmits it once pages free up: the prompt prefix is restored
from a checkpoint (from the host prefix cache with --prefix-cache-restore-wait, otherwise often recomputed) and the
tokens the request had already generated are prefilled again. Those tokens' KV is then produced by the prefill path
rather than by the decode/verification path that generated them, so a greedy request can continue differently from an
uninterrupted run: in every preempted request we compared, output diverged 5-31 tokens after the resume point, while
unpreempted requests stayed token-identical to same-seed runs. No request is truncated and restored bytes are the
original ones; only the continuation after a preemption can differ. To avoid preemption, keep the concurrent contexts
plus their output lengths within the reported KV capacity (fewer concurrent long requests or shorter outputs).

## Target 2: prefix-cache credit anomaly (ANSWERED: policy consequence + measurement artefact, not a bug)
- The c1 long leg sends long-00..03 one at a time, 3 cycles, after the c4 leg on the same server. Each tier holds about
  three of the four ~28.7k-token prompt checkpoints: device 976 pages minus snapshot-row backing (cached_pages stays at
  669-685 = 3 x ~227 pages in every leg), host 8 GiB = 729 page slabs + 12 rows = 3.27 prompts, written through at
  publication (it holds the same most-recent set). With LRU leaf eviction in both tiers and a cyclic working set of
  4 > 3, each miss evicts the prompt that is needed next (Belady's worst case), so a leg's credits are decided by which
  three prompts, in which recency order, the c4 leg left behind; within c1 the scheduler sees one request at a time and
  makes the same choices with or without --prefill-srpt.
- Evidence: c1-only from a cold server 0/12 for both default (M8) and --prefill-srpt (M7), identical; after c4:
  default 7/12 (g128 chain) and 6/12 (repeat M4); v3 0/12 (chain) and 1/12 (repeat M3); round-2 X / X2 / Y (all with
  --prefill-srpt) 6/6/6 of 12; every c1 hit with 8 GiB is a device hit (host hits 0). Knobs: --prefix-cache-memory 24G
  -> 12/12 (M5, v3: 11 host restores of 2.0 GB, 6.6-8.3 s per request); 8 GiB + --prefix-cache-disk on NVMe (64G) ->
  12/12 (M11). Guidance: quote cold rows (as the release docs do) or run the long leg with >= 16 GiB host cache or a
  disk tier; no code change needed.

## Target 1: KV-pressure preemption + re-prefill (ANSWERED; fix W kept, F1 dropped)
- Trigger: each snapshot row borrows ~22 KV pages at the top of the 976-page pool (backing 24 / 40 / 64 pages for
  1 / 2 / 3 rows); 4 x 28.7k prompts = 900 pages, so only 3 boundary checkpoints fit and the 4th prompt captures none;
  decode growth (1 page / 128 tokens / request) uses the last 0-48 free pages after ~0.5-2k total output tokens; then
  nothing is evictable (each checkpoint is pinned by its running request) and preempt_newest runs, with no terminal
  checkpoint (no free row). Output length is the lever: at 512 tokens the cold rep preempts once near the end, at 2k
  1-2 times per rep; Y serialises the cold rep and avoids it there.
- Re-prefill path: the victim's device checkpoint must be evicted (its pages feed the decoders); its host copy exists
  in most cases, but the restore needs one snapshot row more than a plain admission (252 vs 250 free pages in the
  cold-rep case, Argon), so admit_with_host_store fell back to a plain admission: 28.7k tokens prefilled again, the
  pages re-taken, a second preemption (often of a request with no checkpoint). Host cache size does not matter (M6).
- Resume from message-boundary checkpoints: host copy yes (W), device copy no (pages needed), terminal checkpoint no
  (not captured under pressure). Knobs: --kv-cache-limit no effect here (pool memory-limited at 124,928 tokens);
  --prefix-cache-memory / --prefix-cache-disk fix target 2, not this; the chunk-512 package lost 52 pages vs chunk
  128/256, which is why 4 x 28.7k + 2k outputs no longer fit.
- Fix W (host restores wait while other requests run, bounded by the recompute time): re-prefill waste per 2k c4 leg
  1.8k (default) / 1.2k (Y) tokens vs 0.2k-117k / 30k-59k; readmissions all host restores; all-rep mean last token
  131.2 vs 130.5-167.9 s (default), 106.3 vs 121.2-123.1 s (Y); cold rep default 183.4 vs 185.5-233.5 s; 512 legs
  unchanged; unpreempted requests token-identical. Waits of 34-127 s (bounded by gained / measured prefill rate, which
  under interleaved decode is below the solo rate). F1 (restorable victim) did not separate (30k waste in its default
  arm) and is dropped.
| AB6 / EQ | equivalence of the lab binary w3, knobs off vs on (W+F1): lab_prefill_r2 2048 x 1 + 30720 x 4 (2 output tokens) + greedy code-00/01/02 + long-00 vs the release reference | lab w3, NVMe pkg f21b9abb | ARGON-2 | <card> (<unit>) | off: 944.1 tok/s c1, c4 x 30,720 window 142.53 s; **on: 944.2 tok/s, 142.60 s** (round-1 G128 default 141.4-142.5) | **greedy 4/4 identical, both** | defaults and knobs-on unchanged without KV pressure | 9 |
| PUB-SMOKE | public form on GPU: `flare run ... --prefix-cache-restore-wait` (binary flare-lab-r3wpub-pub1 3bc108ca), default policy, 2k outputs, c4 cold rep only | PUB-W | plumbing not yet reviewed | <card> (<unit>) | wall 202.4 s, cold mean / max 185.1 / 202.4; 1 preemption at 143.1 s -> wait -> host restore at 174.1 s -> resume 175.0 s; **waste 174 tokens**; status complete | - | public form behaves as the lab W | 7 |

## Round-3 decisions and hygiene (08:0x)
- Target 1: **W kept** as an opt-in (`--prefix-cache-restore-wait`, default off); public diff prefill-lab/r3/r3-w-public.diff
  (sha256 74305ae2, on top of srpt-r2-public.diff). W logic Argon-CLEARED (ARGON-1/2); the CLI/config plumbing of the
  public form still needs Argon before integration. **F1 dropped** (A/B does not separate it). Lab knobs (env) exist only
  in the lab tree <lab-tree> / binary w3.
- Target 2: **no code change**; guidance + LIMITATIONS text above.
- Hygiene: all my units finished (no <unit> units loaded), workboard shows no plr3 leases, all six V100s at 0 MiB;
  scratch disk-tier directory <scratch dir> removed. The stuck round-2 until-loop is
  the round-2 agent's.
- GPU-minutes: ~370 summed from artefact timestamps over M1-M13, AB1-AB6, PUB-SMOKE (excludes lease acquisition and
  package verification before each arm's first log, ~2-3 min per arm).
