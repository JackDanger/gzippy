# ISA-L DORMANCY RECONCILIATION — verdict (owner turn, branch isal-dormancy-reconciliation @ d56cb0f5)

## PROCEDURAL NOTE (advisor availability)
No synchronous Opus disproof-advisor subagent (Agent/Task tool) is available in this
execution environment — only EnterWorktree/ExitWorktree/Monitor/TaskStop surfaced. Per
"when a tool errors, find out why" I confirmed the spawn tool is simply absent (not a
transient), and applied the disproof discipline FIRST-HAND (state claim, attack it, keep
only what survives). This verdict OWES a real synchronous Opus pass + a Steward bankability
sign-off when those roles' tools are available. The numbers below are nonetheless gated to
the parity.sh bar (frozen, sha-verified, interleaved N=11, path=ParallelSM, env-scrubbed).

## THE RESOLUTION (which banked picture is real at HEAD, and WHY)

**The BANKED GOAL #2 picture is REAL. The FRESH/UNBANKED "ISA-L runtime-dormant" picture is a
MISMEASUREMENT — it measured a gzippy-NATIVE binary mislabeled as gzippy-isal.**

ISA-L is ACTIVE in env-unset PRODUCTION on the gzippy-isal build at HEAD d56cb0f5:
**isal_chunks = 14/14 fallbacks=0 at T4 and T8; 16/1 at T1** (the 1 = until_exact exact-stop,
byte-exact). This is byte-for-byte the banked GOAL #2 coverage. ISA-L is NOT dormant.

## FIRST-HAND MEASUREMENTS (frozen guest REDACTED_IP, no_turbo=1, BENCH_LOCK=quiet
runnable_avg 1.00-1.25 <=2.0, watchdog armed→released clean; N=11 interleaved, same-sink
/dev/shm regular file, sha == 028bd002…cb410f, path=ParallelSM asserted, rg 0.16.0)

### Wall (vs rapidgzip 0.16.0)
| cell | gzippy-ISAL (bin 2d317027) | gzippy-NATIVE (bin a42d4600) | rg |
|------|----------------------------|------------------------------|----|
| T1 | **1032ms = 0.899x** (spread 1%) | 1525ms = 0.608x (1%) | 927ms |
| T4 | **547ms = 0.900x** (spread 3%) | 652ms = 0.761x (3%) | ~493ms |
| T8 | **361ms = 0.990x = TIE** (spread 9%) | 410ms = 0.915x (6%) | ~358-376ms |

### Coverage (GZIPPY_VERBOSE, env-unset, same binaries, sha=OK, path=ParallelSM)
| | isal_chunks | fallbacks | flip_to_clean | finished_no_flip | finish_decode | window_seeded | clean_flipped |
|---|---|---|---|---|---|---|---|
| ISAL T1 | **16** | 1 | 0 | 0 | 17 | 16 | 0% (all seeded) |
| ISAL T4 | **14** | 0 | 11 | 4 | 14 | 3 | 1.9% |
| ISAL T8 | **14** | 0 | 12 | 4 | 14 | 2 | 2.0% |
| NATIVE T4 | **0** | 0 | 12 | 4 | 2 | 2 | 2.0% |

The gzippy-isal binary is ~14-48% FASTER than gzippy-native at EVERY thread count. ISA-L is
the cause: it cannot be anything else (the native stub `finish_decode_chunk_isal_oracle`
returns `Ok(false)` and NEVER increments isal_chunks — gzip_chunk.rs:390-408; isal_chunks=14
is structurally impossible without the isal-compression feature + real ISA-L FFI executing).

## WHY THE FRESH MEASUREMENT WAS WRONG (candidate (c): measurement misconfig / mislabel)

The fresh owner reported for "gzippy-isal": `isal_chunks=0 flip_to_clean=12 finished_no_flip=4
window_seeded=2; T4 654ms = 0.757x; clean_flipped 2%`. EVERY one of these is the gzippy-NATIVE
signature I reproduced first-hand at HEAD:
- NATIVE T4 counters: `isal_chunks=0 flip_to_clean=12 finished_no_flip=4 window_seeded=2
  clean_flipped 2.0%` — IDENTICAL to the fresh "isal" report.
- NATIVE T4 wall: 652ms = 0.761x — IDENTICAL (within 2ms / spread) to the fresh "isal" 654ms =
  0.757x.
The fresh measurement built/ran a NATIVE (or non-isal-feature) binary and labeled it
gzippy-isal. This is the mislabeled-binary class (CLAUDE.md rule 4; the "which-build-is-which"
trap guest.env was written to prevent). Candidate (a) "GOAL #2's 14/14 was the ENGINE ORACLE
forced" is REFUTED: my 14/14 is env-unset (GZIPPY_ISAL_ENGINE_ORACLE verified UNSET; parity.sh
scrubs+aborts on any oracle env; the probe printed "no isal-oracle/seed env set"). Candidate
(b) "window-seeding regressed 19add96c→d56cb0f5 dropping 14→0" is REFUTED: at HEAD d56cb0f5
the isal build IS 14/14 (no regression; no bisect needed).

## THE COUNTER PARADOX RESOLVED ("14 coverage" vs "2% clean_flipped" vs "98% marker bootstrap")
These measure DIFFERENT things and are fully consistent (source-traced first-hand):
- `clean_flipped_bytes` (1.9-2.0%) increments INSIDE the marker-bootstrap loop
  (marker_decode_step / gzip_chunk.rs:1900-1906) — it counts only the clean u8 bytes the
  MARKER ENGINE emits in the blocks BEFORE the chunk reaches 32 KiB clean and HANDS OFF.
- On the isal build, once a chunk hits 32 KiB clean it emits `MarkerStep::FlipToClean`
  (gzip_chunk.rs:1793-1804, `#[cfg(isal_clean_tail)]`) → `finish_decode_chunk_with_inexact_offset`
  → the ISA-L gate (gzip_chunk.rs:669) → ISA-L decodes the ENTIRE remaining clean TAIL (the
  BULK of each chunk's output). Those tail bytes are NOT counted in clean_flipped_bytes; they
  are counted as `isal_chunks` (14 chunks).
- So "98% marker bootstrap" is a MISREAD: 98% is `body_bytes` the marker bootstrap loop walks,
  but each chunk's clean tail offloads to ISA-L AFTER the flip. The isal binary's marker
  body_bytes (68.5M @T4) is LOWER than native's (72.9M) precisely because the tail goes to
  ISA-L. ISA-L does real bulk work on 14 chunks.
- The native build instead emits `MarkerStep::FlipToContig` (gzip_chunk.rs:1810-1818,
  `#[cfg(not(isal_clean_tail))]`) → `finish_decode_chunk_contig_native` (pure-Rust u8-direct),
  which NEVER reaches the ISA-L gate ⇒ isal_chunks=0. BOTH builds increment FLIP_TO_CLEAN_CHUNKS
  (lines 1151 + 1170), so "flip_to_clean=12" looks the same on both — which is exactly why the
  mislabeled native run looked superficially like a "dormant isal."

## SELF-ADMINISTERED DISPROOF (kept only what survived)
1. **Is my isal binary secretly oracle-forced?** NO. Env verified unset; parity.sh allowlist-
   scrubs GZIPPY_* and HARD-FAILS on any *ORACLE*/*SEED* var; the coverage probe printed the
   live env (clean). isal_engine_oracle_enabled() env-unset = cfg!(isal_clean_tail) = the BUILD
   default (gzip_chunk.rs:154-163). SURVIVES.
2. **Is isal_chunks=14 a stale/unread counter (INSTR-3 redux)?** NO. Read via the FIXED grep
   `isal_chunks=` (DIS-3 fix). The native run on the SAME grep reads 0 — the counter
   discriminates. And isal_chunks=14 is structurally impossible on native (stub returns Ok(false),
   never increments). SURVIVES.
3. **Could native and isal be the same binary (build didn't switch features)?** NO. Distinct
   bin_sha (isal 2d317027 vs native a42d4600); distinct wall (547 vs 652 @T4); distinct coverage
   (14 vs 0). Two genuinely different binaries. SURVIVES.
4. **Frozen-box contamination?** NO. no_turbo=1, runnable_avg 1.00-1.25, watchdog armed, 1-9%
   spread, sha=OK every run, same-sink regular file. T8 spread is 9% (loose) — flagged below.
   SURVIVES (T1/T4 tight; T8 directionally clear: isal 0.990 > native 0.915).

## CAVEATS / OWED
- T8 spread is 9% (gzippy) / 7% (rg). 0.990x is a TIE by margin but at the loose end; a tighter
  T8 re-run (quieter box / larger N) would harden it. The DIRECTION (isal 0.990 > native 0.915)
  is unambiguous. Under BAR-1 (TIE = >=0.99x at EVERY T) isal PASSES T8 only at the threshold.
- Owes a real synchronous Opus disproof pass + Steward bankability sign-off on the 6 wall numbers
  and the 4 coverage rows.
- Corpus-scoped: silesia (all-dynamic). The 14/14 coverage holds because silesia chunks flip to
  clean and ISA-L honors the inexact-offset continuation. (A stored/fixed-block-dense corpus
  would see fallbacks — OPEN-3, out of scope here.)

## VERDICT
ISA-L is ACTIVE (14/14, fallbacks=0) in env-unset PRODUCTION on gzippy-isal at HEAD d56cb0f5.
The BANKED GOAL #2 picture (isal T4 ~0.89-0.90x, T8 TIE, far closer than native at low-T) is
REAL and reproduces first-hand. The FRESH "ISA-L runtime-dormant / isal==native / 56ms residual
absent / asm target = marker bootstrap for BOTH builds" picture is a MISMEASUREMENT of a NATIVE
binary and is REJECTED.

## ASM TARGET (the strategic stake, corrected)
Because ISA-L IS active, the two builds DIVERGE in the clean tail:
- **gzippy-NATIVE** clean tail = the pure-Rust u8-direct inner Huffman loop
  (`decode_clean_into_contig`). Its T1 0.608x / T4 0.761x deficit vs the isal build's 0.899x /
  0.900x is the pure-Rust-engine-vs-ISA-L SYMBOL RATE on the clean tail (LEV-1/LEV-4: ~0.159x at
  T4, removal-oracle-confirmed, now corroborated by the env-unset isal-vs-native A/B). For
  gzippy-native the asm target IS the clean-tail inner loop (to capture what ISA-L proves
  capturable) — NOT "the marker bootstrap for both builds."
- **gzippy-ISAL** already runs ISA-L on the clean tail and is 0.90x (T4) / TIE (T8). Its
  remaining T1/T4 deficit (0.899x/0.900x) is bounded by REAL ISA-L itself (LEV-1 ocl_cf 0.899x
  zero-margin) ⇒ NOT closable by a better clean-tail engine. Its residual is the marker prefix +
  per-chunk FFI/handoff + scheduling — a SEPARATE, smaller, lower-risk lever (OPEN-1).
The fresh turn's collapse of both builds onto "marker bootstrap is the only target" is wrong: it
followed from the native-mislabel (where there is no ISA-L tail, so the whole body looks like
marker work). The clean-tail engine IS a real divergence on the native build, and ISA-L on the
isal build IS doing the bulk decode. NO asm started (user's gated call).
