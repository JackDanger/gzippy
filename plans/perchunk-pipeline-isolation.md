# PER-CHUNK / PARALLELSM-PIPELINE ISOLATION (the final isal low-T capstone) — owner charter 2026-06-09

## WHY (the last suspect standing)
The gzippy-ISAL ~0.10 low-T gap has been removal-/source-eliminated as: engine (==rg ISA-L),
placement (dead), output (shared floor), marker bootstrap (shared), D1 over-reserve (page-faults
written-page-only — refuted), per-byte glue/LTO/avail_out (kernel is NASM; avail_out matched —
refuted). The ONLY remaining suspect (DIS-14, plans/isal-perbyte-convergence.md): gzippy routes
EVERY chunk through the full ParallelSM pipeline even at T1 — 16 separate ISA-L invocations, each
with init + set_dict(32 KB window) + boundary loop, + ring/window-map/CRC/handoff — vs rg's leaner
consumer. ARCHITECTURE-level, NOT yet removal-proved. SIZE IT cleanly before any big port.

## THE CLEAN ORACLE (T1 single-shot vs chunked pipeline)
At T1 (sequential) gzippy COULD decode the whole stream in ONE ISA-L call — no chunking, no
per-chunk set_dict, no ring/window-map. Build/measure that as a removal oracle:
- Prefer an EXISTING direct ISA-L whole-stream decode if one still exists (e.g. the old
  isal_decompress::decompress_gzip_stream / a single-shot ISA-L path) — source-check first.
- Else wire a minimal measurement-only whole-stream ISA-L decode of silesia at T1 (byte-exact:
  sha == 028bd002…; this is an oracle, not production).
- COMPARE at T1: production ParallelSM (16-chunk) vs single-shot-ISA-L vs rapidgzip 0.16.0, same
  frozen guest, interleaved N>=11, same-sink, sha-verified.
PRE-REGISTER falsifier:
- If single-shot-ISA-L ~= rg (closes most of the ~0.10): the per-chunk/ParallelSM pipeline IS the
  gap => an ARCHITECTURE-PORT lever (lean the consumer toward rg's), real, sized. Report the size.
- If single-shot-ISA-L STILL loses to rg by ~0.10: the gap is NOT the chunking overhead — it's
  deeper (the ISA-L call itself in-process, or measurement) => isal low-T is closer to a PROVED
  FLOOR. Report that (a user-finding).
Also instrument the per-chunk components (16x init+set_dict(32KB) self-time, ring/window-map/CRC
per-chunk) so the architecture overhead is decomposed, not just bounded.

## STEP 0 proof-of-binary: gzippy-isal at HEAD, isal_chunks>=14@T4/16@T1, env-unset, path=ParallelSM.

## OUTPUT
The T1 single-shot-vs-chunked-vs-rg numbers + the per-chunk component decomposition + the verdict:
is the isal low-T gap the chunked-ParallelSM ARCHITECTURE (a faithful-port lever, rg = existence
proof) or a deeper/proved floor? This COMPLETES the isal low-T attribution. (gzippy-NATIVE is
separate — its low-T is additionally the 0.667x engine floor; do NOT touch native.)

## GATES + DISCIPLINES
git WORKTREE; box is FREE — bench-lock freeze, release clean. Numbers ONLY from the bench-locked
quiet guest (procs_running gate; the prior run hit a noisy acquisition — re-acquire until
runnable_avg<=2.0); matched same-sink; interleaved N>=11; sha-verified; path=ParallelSM +
isal_chunks>=14 asserted (for the production arm). The Agent/advisor tool is UNAVAILABLE to you —
run rigorous self-disproof (pre-registered falsifier; the single-shot arm must be byte-exact) and
hand ME (supervisor) RAW numbers + provenance + isal_chunks readback for my Opus gate; do NOT
claim "advisor-vetted." Run measurements YOURSELF holding the ssh. SOURCE-VERIFY first-hand.
Serialize builds via cargo-lock.sh, df -h around builds. No multi-line python via Bash (write a
.py). Wrap hang-prone cmds in timeout. Diagnose the FIRST error before retrying. NO orphan
processes / sleep sentinels — pgrep clean on local + guest + neurotic before finishing (a prior
agent left two 100%-CPU orphans for 3.5h). Update plans/orchestrator-status.md + the
disproof-ledger. STOP at the checkpoint and report the numbers + the architecture-lever-or-floor
verdict + raw numbers for my gate.
