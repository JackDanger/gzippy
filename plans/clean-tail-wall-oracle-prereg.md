# PRE-REGISTRATION — window-absent-PRESERVING ISA-L clean-tail removal oracle (the decisive wall-binder test)

Owner turn, 2026-06-07, branch reimplement-isa-l, HEAD 7aae6c4a + measured build /tmp/gzbuild-head.

## The question (the decisive fork-deciding measurement)
Does replacing ONLY the post-flip CLEAN u8 tail decode with REAL ISA-L — while PRESERVING
production's window-absent marker bootstrap (do NOT seed windows) — move the T8 WALL toward
a tie with rapidgzip?

This converts the slack-masked ΔdecodeBlock-SUM attribution (clean tail is the bigger term)
into a WALL ceiling (Measurement PROCESS Rule 3: bound a speed-up by REMOVAL, never the slope).
It is window-absent-PRESERVING by construction (charter OSCILLATION rule): GZIPPY_SEED_WINDOWS
is NOT set, so the 89% window-absent marker bootstrap runs as in production; only the
FlipToClean post-flip clean tail (resumable.rs, the attributed dominant decodeBlock term) and
the naturally-window-present chunks route through ISA-L (finish_decode_chunk_isal_oracle).

## Oracle mechanism (already wired, NOT a new build)
GZIPPY_ISAL_ENGINE_ORACLE=1 (gzip_chunk.rs:539) → finish_decode_chunk_isal_oracle (:160).
- Fires inside finish_decode_chunk_impl, which on the FlipToClean path receives the full
  32 KiB clean_window (gzip_chunk.rs:959-970). Guard at :178 (initial_window.len()==MAX_WINDOW)
  PASSES on the flip tail ⇒ ISA-L decodes the clean tail; the u16 marker bootstrap is untouched.
- finished_no_flip chunks (full u16, never flip) never call finish_decode ⇒ stay pure-Rust
  marker engine (correct — window-absent path preserved).
- Watch isal_oracle_fallbacks: must be ~0 for a clean run (contamination guard).

## Self-test (Rule 4 — run BEFORE trusting)
- OFF (no oracle, unseeded) stdout sha == rapidgzip reference sha (the canonical 028bd002… on
  the decompressed stream as measure.sh checks it).
- ON (oracle, unseeded) stdout sha == same reference sha (byte-exact; OFF==ON==identity).
- ON verbose shows isal_oracle_chunks>0 (oracle actually fired unseeded) AND
  isal_oracle_fallbacks small/0.

## Contenders (T8, interleaved, sha-verified, locked guest REDACTED_IP, taskset 0,2,4,6,8,10,12,14, gov=perf)
- rg   = rapidgzip 0.16.0 (reference)
- prod = pure-Rust production (no oracle, no seed) — the baseline 0.73× wall
- ocl  = GZIPPY_ISAL_ENGINE_ORACLE=1, NO seed — clean tail on ISA-L, marker bootstrap preserved

## PRE-REGISTERED FALSIFIER (decide BEFORE the numbers)
Let r_prod = prod/rg ratio (currently ~0.73×, i.e. prod is ~1.37× rg wall).
Let r_ocl  = ocl/rg ratio.

- **CLEAN-ENGINE-RATE IS THE WALL BINDER (the fork-forcing branch):** if r_ocl moves
  SUBSTANTIALLY toward 1.0 — i.e. ocl ties rg (≥0.90×) OR closes ≥half the prod→tie gap
  (r_ocl ≥ ~0.85×, well outside the interleaved spread) — then the clean-tail ENGINE RATE
  is the confirmed wall binder. Since the pure-Rust+ASM clean engine ceiling is PROVEN
  ~0.6× ISA-L (VAR_VI, advisor-upheld), no-FFI 1.0× is then likely UNREACHABLE for the clean
  tail ⇒ GENUINE USER FORK (goal #1 no-FFI vs the 1.0× bar).

- **CLEAN ENGINE IS SLACK-MASKED (NOT the wall binder):** if r_ocl ≈ r_prod (Δ within the
  interleaved spread, TIE vs prod) — then the clean-engine rate is slack-masked at the wall;
  it is NOT the binder. The binder is elsewhere (re-perceive). This RECONCILES with seedfull
  (which also ties only because seeding removes the window-absent structure, not the engine).

- Either way: record the WALL number first. SUM is a footnote.

## The reconciliation (seedfull-TIE vs production-0.73×) — independent of the oracle result
seedfull (all chunks window-present, SAME pure-Rust clean engine) TIES at 1.029×.
production (89% window-absent, same clean engine for the flip tail) is 0.73×.
Same clean engine, opposite wall verdicts. So the window-absent STRUCTURE costs something
BEYOND the clean-engine rate. Candidate: the u16 marker BOOTSTRAP prefix that every
window-absent chunk must decode before it can flip to clean (seedfull skips it entirely —
window present ⇒ clean from block 0). Hold the clean engine constant (the oracle does exactly
this: ISA-L clean tail in BOTH the oracle and a seeded-oracle) and compare unseeded-oracle vs
seeded-oracle: the delta is the window-absent marker-bootstrap cost at the wall, NOT the clean
engine. That names the seedfull↔production gap causally.
