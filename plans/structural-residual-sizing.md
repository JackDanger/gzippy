# STRUCTURAL-RESIDUAL SIZING — the low-T BAR-1 floor (owner charter, 2026-06-09)

## WHY (the capstone measurement)
All engine + placement levers are now resolved: native pure-Rust+asm clean-tail ceiling =
0.667x ISA-L (VAR_VIII, bench-bounded, integration HELD); placement DEAD (OPEN-1); FFI-handoff
negligible. Even a PERFECT engine (gzippy-isal, real ISA-L) loses T1 0.899x / T4 0.900x. So the
low-T BAR-1 gap (~0.10, BOTH builds) is the STRUCTURAL RESIDUAL = a hybrid of THREE terms:
(a) the u16 MARKER-BOOTSTRAP engine (pure-Rust decode of the window-absent prefix — fires at
T4+, NOT at T1), (b) the SERIAL-OUTPUT floor (~200 MB writev), (c) the CHUNK-0 / per-chunk
bootstrap overhead. This is the LEADING HYPOTHESIS for why BAR-1 (>=0.99x every T) is likely
unreachable — a REMOVAL ORACLE sizing the three terms converts it to a PROVED FLOOR (var8-gate
verdict claim 3). This is the one measurement that gives a definitive BAR-1 answer.

## STEP 0 (MANDATORY proof-of-binary) — build gzippy-isal at HEAD, assert isal_chunks>=14
@T4/T8 on env-unset silesia BEFORE any wall number (isal_chunks increments only in real-ISA-L
cfg gzip_chunk.rs:386; native stub :390-400 never does). Oracle/SEED env UNSET, path=ParallelSM.

## THE DECOMPOSITION (exploit the T1-has-no-marker-bootstrap simplification)
On gzippy-ISAL (engine matched to rg, so the residual is pure structural):
- T1 has NO marker bootstrap (sequential => windows present => clean ISA-L path). So T1's
  ~0.10 gap = (b) serial-output + (c) chunk-0/per-chunk ONLY. Size these at T1:
  * SERIAL-OUTPUT: the GZIPPY_SKIP_WRITEV removal oracle (skip the output syscall;
    byte-transparent OFF==identity, wrong final state but measures the output floor) — does
    removing output close the T1 gap? Compare to rg's output exposure (rg pays a serial output
    too — measure rg file-vs-null to get its floor; the gzippy-SPECIFIC excess is the lever).
  * CHUNK-0 / per-chunk: instrument the first-chunk + per-chunk FFI-setup self-time.
- T4 ADDS the marker bootstrap. So (T4 residual) - (T1 structural floor, parallelism-adjusted)
  ~= (a) the marker-bootstrap contribution. Confirm with a marker-bootstrap perturbation
  (slow-inject the u16 marker decode; freq-neutral control) to verify it's on the T4 critical
  path, and compare gzippy's marker-bootstrap rate to rg's (rg ALSO marker-decodes ~34.5% —
  the gap is gzippy's being heavier, not a missing feature).

## OUTPUT (the morning deliverable)
The {marker-bootstrap / serial-output / chunk-0} split at T1 and T4, EACH as a fraction of the
~0.10 gap, with: which (if any) is gzippy-SPECIFIC EXCESS over rg (= a faithful convergence
lever) vs a SHARED floor (rg pays it too = irreducible). VERDICT: is BAR-1 at low-T a PROVED
FLOOR (residual is shared/irreducible => native pure-Rust >=0.99-every-T UNREACHABLE, a
user-finding) OR is a term gzippy-specific-excess (=> a faithful lever exists, name the rg
vendor mechanism). NO WORK-DISPLACEMENT in any oracle.

## GATES + DISCIPLINES
git WORKTREE; the box is FREE — bench-lock freeze, release clean. Numbers ONLY from the
bench-locked quiet guest (procs_running gate); matched same-sink; interleaved N>=11 (N>=15 any
T8). The Agent/advisor tool is UNAVAILABLE to you — run rigorous self-disproof (pre-register
each falsifier; freq-neutral controls for perturbations) and hand ME (supervisor) RAW numbers +
provenance + the isal_chunks readback for my Opus gate; do NOT claim "advisor-vetted." Run
measurements YOURSELF holding the ssh. SOURCE-VERIFY first-hand. Serialize builds via
cargo-lock.sh, df -h around builds. No multi-line python via Bash. Wrap hang-prone cmds in
timeout. Diagnose the FIRST error before retrying. NO orphan processes / sleep sentinels —
pgrep clean on local + guest + neurotic. Update plans/orchestrator-status.md + the
disproof-ledger. STOP at the checkpoint and report the split + the PROVED-FLOOR-or-lever
verdict + raw numbers for my gate.
