# DISPROOF-ADVISOR BRIEF — Phase-0 ISA-L-in-pipeline WALL oracle

You are an INDEPENDENT, READ-ONLY disproof advisor. Adversarially try to BREAK the claims
below. Source-verify every file:line you rely on. Do NOT trust this brief's numbers blindly —
check the logic. Write your verdict to plans/asmport-phase0-advisor-verdict.md.

## CONTEXT
gzippy → rapidgzip T8 parity. Prior turns bounded the pure-Rust engine at ~0.6× ISA-L in
ISOLATION and (floor-to-floor) found decodeBlock ~1.86× rg, concluding "the engine binds."
The user then decided to transliterate igzip's AVX2 asm to pure-Rust to close the gap.
PHASE-0 (this turn) scopes that asm port BEFORE writing it, by dropping a REAL ISA-L engine
into gzippy's PRODUCTION parallel-SM pipeline and measuring the T8 WALL (Measurement PROCESS
#3: bound the speed-up by REMOVING/replacing the region, never extrapolate the isolation slope).

## THE ORACLE (measurement-only, NOT production; code in src/decompress/parallel/gzip_chunk.rs
## `finish_decode_chunk_isal_oracle` + gate at top of `finish_decode_chunk_impl`)
`GZIPPY_ISAL_ENGINE_ORACLE=1` routes the clean-tail decode through REAL ISA-L FFI
(`isal_decompress::decompress_deflate_from_bit_with_boundaries`, a patched-boundary igzip),
feeding ISA-L's bytes/boundaries/end-bit through the SAME ChunkData primitives the pure path
uses (commit + per-byte CRC + append_block_boundary_at + finalize_with_deflate). The ISA-L
input is bounded to `[..stop_hint/8 + 256KiB]` so each worker decodes only ITS chunk, not the
whole member. Everything else in the pipeline (pool, consumer, segmented ring buffers,
window-publish, CRC, finalize, scheduling) is UNCHANGED.

To make the bulk run on ISA-L, windows are SEEDED: `GZIPPY_SEED_WINDOWS_CAPTURE` (T1 records
every aligned predecessor window) then `GZIPPY_SEED_WINDOWS` replays them so all 18 chunks are
window-PRESENT and reach `finish_decode_chunk_impl` (the only path the oracle hooks). Seeding
provides the correct predecessor window; it changes NO output bytes (all modes sha 028bd002…).

PROVEN ISA-L ran: T8 verbose `isal_oracle_chunks=16 isal_oracle_fallbacks=1` (94% of clean
decode is real ISA-L; 1 chunk fell back to pure on a bounded-slice contract miss).

## MEASURED (locked guest REDACTED_IP, 16c gov=performance turbo-on, load 2.7-4.2, measure.sh
## interleaved N=11, taskset 0,2,4,6,8,10,12,14, RAW=68229982, sha-OK every run, 2 runs)
| contender | T8 wall | vs rg | verdict | run2 |
| rg (rapidgzip 0.16.0)        | 0.134s | 1.000 | —    | 0.131s |
| isal (ISA-L engine, seeded)  | 0.148s | 0.905 | TIE  | 0.892 TIE |
| pure (pure-Rust eng, seeded) | 0.134s | 1.002 | TIE  | 0.968 TIE |
| prod (pure-Rust, NO seed)    | 0.194s | 0.690 | LOSS | 0.652 LOSS |

Per-stage --verbose (T8): PROD decodeBlock SUM 1.048s, Real Decode 0.169s, Theo-Opt 0.131s,
Fill 77%, 16/18 window-absent (12 flip_to_clean + 4 finished_no_flip), bootstrap body_rate
168 MB/s, 13 header-speculation failures. PURE-SEED decodeBlock SUM 0.781s, Real Decode 0.108s,
Theo-Opt 0.098s, Fill 90.55%, 0 window-absent, 0 speculation failures, 0 bootstrap.
rapidgzip --verbose: 34.5% replaced-marker symbols (genuinely window-absent at runtime, NO
seeding), decodeBlock 0.517s.

## CLAIMS TO BREAK
1. The oracle genuinely measures an igzip-class engine in gzippy's REAL pipeline (pool/
   consumer/ring/CRC kept; only the per-chunk byte-production swapped to ISA-L), byte-exact.
2. An igzip-class engine ALONE does NOT close the T8 production gap: `pure` (pure-Rust engine,
   seeded) ALREADY ties rg, and `isal` (ISA-L engine, seeded) also ties — the engine swap is
   TIE-vs-TIE. Therefore the per-thread engine is NOT the T8 wall binder.
3. The T8 binder is the WINDOW-ABSENT marker-bootstrap/speculation path (removed by seeding):
   prod 0.69× → seeded 1.00×, with decodeBlock SUM −25%, Fill 77%→90%, 13→0 speculation
   failures. rapidgzip runs the SAME 34.5% marker workload WITHOUT seeding yet ties, so
   gzippy's marker path is the slow one — apples-to-apples, not a seeding artifact.
4. CONSEQUENCE for the asm port: the asm engine port targets a path (clean/seeded) that
   ALREADY ties; it cannot move the prod T8 wall, which is gated by the marker path. Phase-1
   should target the window-absent path, NOT the asm engine port — at least at T8.

## DISPROOF ANGLES TO CHECK (at minimum)
- Is seeding an UNFAIR oracle? (Does it skip real decode work, or only remove speculation
  overhead? Output is byte-identical — does that prove no work skipped?) Is comparing seeded-
  gzippy to un-seeded rapidgzip apples-to-apples, given rg is genuinely window-absent (34.5%
  markers)? Or does rg get its windows cheaply some other way the seed shortcuts?
- Is the 1-chunk fallback (isal_oracle_fallbacks=1) contaminating the isal number? (1/17 pure.)
- Does the bounded-input slice (stop_hint/8 + 256KiB) make ISA-L decode LESS than the pure
  path (fewer bytes ⇒ artificially fast)? Output is byte-exact — does that rule it out?
- Is the engine gap (pure decodeBlock SUM 0.781 vs rg 0.517 ≈ 1.51×) REAL but wall-NEUTRAL at
  T8 because Fill 90% leaves slack? Could it bind at T1 or T16? (The asm port's real value.)
- Could the prod LOSS be scheduling (Fill 77%) rather than the marker engine being slow per se?
  decodeBlock SUM drops 1.048→0.781 with seeding — is that the marker COMPUTE or the Fill?
