# FLAT-BUFFER (DE-FRAG) WALL A/B ORACLE — owner charter 2026-06-09

## WHY (the gate's owed decisive measurement)
The isal attribution is COMPLETE: the +1.54e9 instruction excess is ~71% marker inner loop, of
which ~1.70e9 is u16 OUTPUT/BACKREF FRAGMENTATION — gzippy's `SegmentedU16::push_slice` (segment-
list append) + separate u16-ring `emit_backref_ring` (modular copy) vs rapidgzip's ONE flat
aligned `m_window16` with inlined `window[pos++]` writes + single `memcpy` back-ref
(deflate.hpp:926/1319/1376), flipped in-place by `replaceMarkerBytes` (:1765). Flattening toward
rg's flat buffer is FAITHFUL (matches the governing one-MarkerRing memory) and is the LARGEST
faithful sub-lever. BUT instruction-count is a HYPOTHESIS, not a wall verdict — TIE-6 already
sized the u16-footprint axis as WALL-SLACK. So before any architectural de-frag PORT, SIZE its
WALL contribution. This decides whether the de-frag is worth a multi-session port or is wall-slack.

## THE A/B (byte-transparent, OFF==identity)
Build a measurement-only flat-buffer mode for the marker output/backref path: write the u16
marker output into a single flat pre-allocated buffer with inlined index writes + a single memcpy
back-ref (rg's m_window16 shape), SKIPPING the SegmentedU16 segment-list bookkeeping + the u16-ring
modular-copy overhead. It MUST produce byte-identical output (sha == 028bd002…; this is the
load-bearing gate — a flat buffer that changes bytes is void). OFF = production
(SegmentedU16 + ring). Measure T4 (+ T1) WALL ON vs OFF, interleaved N>=15, frozen quiet box,
isal_chunks>=14 asserted, matched same-sink. ALSO read instr-count ON vs OFF (confirm the flat
path actually cuts ~1.70e9 — the mechanism check).

PRE-REGISTER falsifier: if flat-buffer recovers a MEANINGFUL fraction of the ~55ms T4 gap (> spread)
=> the de-frag is a REAL wall lever, worth the faithful architectural port — report the size +
project isal T4 toward 0.99. If wall-slack (delta <= spread) => the 1.70e9 instruction win does
NOT translate to wall (off critical path / memory-latency-hidden, like the footprint axis) => the
de-frag is NOT worth porting; isal T4 is near-floor for the faithful (non-asm) path. NO
WORK-DISPLACEMENT (report whole-system wall + any displaced stage).

## STEP 0 proof-of-binary: gzippy-isal at HEAD, isal_chunks>=14, env-unset, path=ParallelSM.

## GATES + DISCIPLINES
git WORKTREE; box is FREE — bench-lock freeze (re-acquire until runnable_avg<=2.0), release clean.
Numbers ONLY from the bench-locked quiet guest; matched same-sink; interleaved N>=15 (report the
spread); sha-verified (the flat path MUST be byte-exact); path=ParallelSM + isal_chunks>=14
asserted. The Agent/advisor tool is UNAVAILABLE to you — run rigorous self-disproof (pre-registered
falsifier) and hand ME (supervisor) RAW wall+instr numbers + provenance + isal_chunks readback for
my Opus gate; do NOT claim "advisor-vetted." Run measurements YOURSELF holding the ssh.
SOURCE-VERIFY first-hand. Serialize builds via cargo-lock.sh, df -h around builds (earlier turns
overwrote /root/gzippy-bench/target — rebuild if needed). No multi-line python via Bash (write a
.py). Wrap hang-prone cmds in timeout. Diagnose the FIRST error before retrying. NO orphan
processes / sleep sentinels — pgrep clean on local + guest + neurotic before finishing (prior turns
leaked find/ + timeout-sleep orphans — sweep them). Update plans/orchestrator-status.md + the
disproof-ledger. STOP at the checkpoint and report: the flat-buffer T4(+T1) wall delta + instr
delta + the de-frag-pays-or-wall-slack verdict + raw numbers for my gate.
