# ADVISOR BRIEF — the decisive window-absent-preserving ISA-L clean-tail WALL oracle + reconciliation

Read-only disproof advisor. Try to BREAK the owner's two conclusions below. HEAD 7aae6c4a,
measured build /tmp/gzbuild-head (gzippy-isal native), locked guest 10.30.0.199.

## What was run (the decisive removal oracle, window-absent-PRESERVING)
The prior turn ATTRIBUTED the window-absent decodeBlock 1.6× gap to the pure-Rust clean u8
tail (resumable.rs, ~2.3× ISA-L per byte in decodeBlock-SUM). The advisor on that turn said:
NOT actionable until the already-wired GZIPPY_ISAL_ENGINE_ORACLE is run TO THE WALL
(decodeBlock-SUM is slack-masked at Fill 85%). This turn did exactly that, UNSEEDED.

Oracle = GZIPPY_ISAL_ENGINE_ORACLE=1, NO GZIPPY_SEED_WINDOWS. It replaces ONLY the
post-flip clean tail decode (finish_decode_chunk_impl, the resumable.rs term) with real
ISA-L FFI, while the u16 marker bootstrap runs as in production. Self-test PASSED:
- OFF==ON==rg sha 028bd002…cb410f (byte-exact, OFF==identity).
- isal_oracle_chunks=14 isal_oracle_fallbacks=0 (fires unseeded, no contamination).
- routing = flip_to_clean=12 finished_no_flip=4 window_seeded=2 — identical to prod unseeded,
  so the 89% window-absent marker bootstrap is preserved.

## Result 1 — THE WALL NUMBER (3 interleaved N=11 runs, sha-OK every run, gov=perf, taskset 0,2,4,6,8,10,12,14)
| | run1 | run2 | run3 |
| prod (pure-Rust clean tail)   | 0.744× | 0.754× | 0.755× rg |
| ocl  (ISA-L clean tail, unseeded) | 0.698× | 0.686× | 0.702× rg |
**The ISA-L clean-tail oracle does NOT move the wall toward a tie — ocl ≈ prod or slightly
SLOWER (~0.70× vs ~0.75×), nowhere near the 0.90× tie threshold.** Pre-registered falsifier:
"clean-engine-rate is the binder" REQUIRED ocl to move substantially toward 1.0 (≥0.85×). It
did not. ⇒ CONCLUSION 1: the clean-tail ENGINE RATE is SLACK-MASKED, NOT the wall binder.
(The decodeBlock-SUM 2.3× clean-tail figure is real but does not reach the wall.)

## Result 2 — THE RECONCILIATION (seedfull-TIE vs production-0.73×), engine held CONSTANT = ISA-L in both
Vary ONLY the window-absent structure (2 runs, sha-OK):
| | run1 | run2 |
| ocl_unseed (marker bootstrap PRESERVED) | 0.697× | 0.701× rg |
| ocl_seed   (windows SEEDED, no bootstrap) | 0.860× | 0.857× rg |
Same ISA-L clean engine in both; seeding away the marker bootstrap moves the wall
0.70×→0.86× (~19%). Verbose (engine constant):
  ocl_unseed: decodeBlock 0.869s, Real Decode 0.129s, Fill 84%, marker bootstrap body
              321.7ms/73.0M@227MB/s, flip_to_clean=12 finished_no_flip=4, spec-fail header=13.
  ocl_seed:   decodeBlock 0.749s, Real Decode 0.101s, Fill 93%, marker bootstrap body 0/0,
              flip=0 finished=0 window_seeded=17, spec-fail=0.
The decodeBlock delta (0.869−0.749 = 0.120s) ≈ the marker bootstrap body (0.322s SUM/overlap).
⇒ CONCLUSION 2: the seedfull↔production gap is the WINDOW-ABSENT u16 MARKER BOOTSTRAP
(the prefix every window-absent chunk decodes before flipping + the 4 never-flip full-u16
chunks), NOT the clean engine. ISA-L cannot decode it (it can't emit u16 markers).

## Instrument confound to scrutinize (owner-disclosed, does NOT change the deltas)
The oracle does a per-chunk 64 MiB alloc + to_vec copy that production's resumable does not.
This depresses the ABSOLUTE ocl numbers (ocl_seed reaches only 0.86× where pure seedfull
reached 1.029× last turn — the ~0.17× gap is the oracle copy). But BOTH ocl cells pay the
same copy, so the unseed↔seed DELTA (the bootstrap attribution) is clean, and the prod↔ocl
comparison (Result 1) is conservative: a copy-free ISA-L clean tail would be at-best slightly
above ocl's 0.70×, still ≈prod's 0.75× — the clean engine swap still buys ~nothing at the wall.

## Questions for the advisor (try to break these)
1. Is CONCLUSION 1 sound — does ocl≈prod genuinely falsify "clean-engine-rate is the binder",
   given the oracle copy confound depresses ocl? (Owner claims yes: the copy only makes ocl
   look WORSE, so if even ocl-with-ISA-L can't beat prod, the clean engine is not the binder.)
2. Is CONCLUSION 2's attribution to the marker bootstrap airtight, or does seeding remove
   something ELSE too (block-finder real boundaries vs partition guess; the 13→0 spec-failures)
   that the unseed↔seed delta wrongly credits to the bootstrap? (This is the Phase-0 Claim-3
   bundle concern — does it reappear here?)
3. Does this END the engine↔scheduling oscillation, or is there a third unmeasured term?
4. The fork implication: the binder is the window-absent u16 marker bootstrap RATE (227 MB/s,
   pure-Rust, ISA-L-untouchable by construction). Is no-FFI 1.0× reachable by a faster pure-Rust
   u16 marker bootstrap — given rapidgzip ties UNSEEDED with the SAME 34.5% markers (its u16
   marker decode is fast)? Or does the prior VAR_VI 0.6× plateau bound this too?
