# PHASE-0 ISA-L-IN-PIPELINE WALL ORACLE — pre-registered falsifier

Pre-registered BEFORE the measurement (2026-06-07, owner turn).

## Oracle (measurement-only, NOT production)
`GZIPPY_ISAL_ENGINE_ORACLE=1` routes every clean-tail decode in `finish_decode_chunk_impl`
through REAL ISA-L (`decompress_deflate_from_bit_with_boundaries`) instead of the pure-Rust
engine, keeping the ENTIRE production parallel-SM pipeline (pool, consumer, ring/segmented
buffers, window-publish, per-byte CRC, finalize). Combined with `GZIPPY_SEED_WINDOWS` (all
chunks clean ⇒ all 18 chunks reach `finish_decode_chunk_impl`), the WHOLE decode runs on ISA-L.

PROVEN it runs ISA-L: T8 verbose shows `isal_oracle_chunks=17 isal_oracle_fallbacks=0`,
finish_decode=17 + inflate_wrapper=1. Byte-exact: 028bd002…cb410f on all modes.

## Contenders (interleaved, sha-verified, locked guest 10.30.0.199)
- `rg`   = rapidgzip 0.16.0 (the parity target)
- `isal` = gzippy oracle-ON + seeded   (ISA-L engine in gzippy's pipeline)
- `pure` = gzippy oracle-OFF + seeded   (pure-Rust engine, SAME seeding — isolates the swap)
- `prod` = gzippy oracle-OFF + no-seed  (true production — seeding-artifact control)

## H0 (igzip-class engine ALONE suffices)
`isal` TIES `rg` at T8 (ratio within max(spread)).

## Falsifier (fires ⇒ H0 FALSE)
`isal` does NOT tie `rg` (gzippy slower beyond spread) ⇒ production overheads (ring/wrap/
resumable/CRC/scheduling) ALSO cap the wall. The asm port must then ALSO flatten the clean
path, not merely match igzip's rate — identify which overhead from the per-stage trace.

## Controls
- Engine-swap isolation: `isal` vs `pure` (only the engine differs). The delta bounds the
  engine's WALL contribution (Measurement PROCESS #3 — removal/oracle, not slope).
- Seeding artifact: `pure` vs `prod` — seeding alone must not move the wall materially.

## RESULTS (locked guest 10.30.0.199 double-ssh, 16c gov=performance turbo-on, load 2.7-4.2,
## measure.sh interleaved N=11, taskset CPUS=0,2,4,6,8,10,12,14, RAW=68229982, sha-OK=028bd002…,
## 2 independent runs, native target-cpu ⇒ ISA-L SSE/BMI2 live; oracle PROVEN: isal_oracle_chunks=16
## isal_oracle_fallbacks=1 = 94% of clean decode is REAL ISA-L)
| contender | T8 wall | vs rg | verdict | run2 |
| rg (rapidgzip 0.16.0)        | 0.134s | 1.000 | —    | 0.131s |
| isal (ISA-L engine, seeded)  | 0.148s | 0.905 | TIE  | 0.892 TIE |
| pure (pure-Rust eng, seeded) | 0.134s | 1.002 | TIE  | 0.968 TIE |
| prod (pure-Rust, NO seed)    | 0.194s | 0.690 | LOSS | 0.652 LOSS |

## FALSIFIER FIRED — but on the OPPOSITE conclusion from what was feared
H0 (igzip-class engine alone suffices to tie) is CONFIRMED, but with a twist that
RE-POINTS the campaign: **`pure` (pure-Rust engine, seeded) ALSO TIES rapidgzip.** The
engine swap (isal vs pure) is a TIE-vs-TIE — the engine is NOT the T8 wall binder.

THE BINDER IS THE WINDOW-ABSENT MARKER-BOOTSTRAP / SPECULATION PATH, removed by seeding:
- prod (no seed): 16/18 chunks window-absent → marker bootstrap. decodeBlock SUM 1.048s,
  Real Decode 0.169s, Fill 77%, bootstrap body_rate 168 MB/s, 13 header-speculation failures.
- pure (seeded):  0/18 window-absent. decodeBlock SUM 0.781s (−25%), Real Decode 0.108s,
  Fill 90.55%, ZERO bootstrap, ZERO speculation failures.
Seeding removes the speculation OVERHEAD only (byte-exact, identical output bytes) — a clean
Rule-3 removal oracle. With it removed, BOTH engines tie rg.

## IMPLICATION FOR THE ASM PORT (Phase 1)
An igzip-class engine ALONE does NOT close the production gap, because the pure-Rust engine
ALREADY ties rg once the marker-bootstrap is removed. The asm port would land on a path that
already ties (the seeded/clean wall) — it cannot help the prod wall, which is gated by the
window-absent speculation path, NOT by clean engine rate. The real T8 binder is the
marker-bootstrap/speculation overhead (gzippy is ~window-absent like rapidgzip, but gzippy's
window-absent path is SLOW: body_rate 168 MB/s + 13 header-speculation failures, while
rapidgzip's window-absent path ties). ⇒ Phase-1 target should be the window-absent path, not
the asm engine port — PENDING the disproof-advisor + the load-bearing premise check (is
rapidgzip genuinely window-absent at runtime, so the comparison is apples-to-apples?).

CAVEAT (owed): the per-thread engine gap (pure decodeBlock SUM 0.781s vs rg 0.50s ≈ 1.56x)
SURVIVES seeding but does NOT bind the wall at T8 (Fill 90%, slack absorbs it). It may bind at
T1 or higher T. The asm port's value is therefore T1 / high-T / headroom, not the T8 wall.
