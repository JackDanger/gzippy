# LOW-T GATE + ISAL STORED/FIXED COVERAGE — owner charter (supervisor, 2026-06-08)

## BINDING BAR CORRECTION (user, 2026-06-08) — read FIRST
A TIE with rapidgzip = **>=0.99x at EVERY thread count** (T1, T4, T8, ...). "Within
spread" / "ties at T8" is REJECTED; 0.88x is NOT a tie. Under this bar NEITHER build is
done: gzippy-isal passes only T8 (1.030x) and LOSES T1 0.904x / T4 0.885x; gzippy-native
loses every cell (T1 0.608 / T4 0.755 / T8 0.885). The LOW-T deficit is the headline.
Note the tell: gzippy-isal uses REAL ISA-L FFI and STILL loses at low-T => the low-T gap
is likely largely NON-engine (pipeline/bootstrap/scheduling), since ISA-L is already the
fastest engine. The gate below is exactly the disambiguator.

## JOB 1 (PRIORITY, the user-chosen first step) — the owed quiet-box ocl_cf-T4 GATE
Run ONE TIGHT (<=5% inter-run spread) measurement on a Plex-FROZEN quiet box:
**ocl_cf-T4 vs native-T4 vs rapidgzip-T4**, interleaved best-of-N>=9, sha-verified,
matched same-sink (both -> regular file on /dev/shm), path=ParallelSM asserted,
isal_oracle_fallbacks==0 asserted in-script. Use scripts/bench/oracle.sh +
host/bench-lock.sh (freezes Plex + noisy LXCs, verifies quiet via INSTANTANEOUS
procs_running, NOT lagging loadavg). If the box cannot be made quiet (procs_running gate
fails), report BLOCKED — do NOT bank a loaded number (rule 8 / past ocl_cf drift).

PRE-REGISTER THE FALSIFIER before measuring:
- ocl_cf-T4 >= 0.99x rg AND native-T4 < 0.99x  => low-T is ENGINE-CLOSABLE => the
  full-kernel hand-asm rewrite is justified at low-T (report it as the next gated build,
  do NOT start it).
- ocl_cf-T4 ~ native-T4 (both < 0.99x, delta < spread) => the low-T gap is NON-ENGINE =>
  close the engine chapter at low-T; the lever is scheduling/bootstrap/pipeline
  (project_confirmed_offset_prefetch_gap is the located candidate: head-of-line stalls at
  confirmed offsets). Name the binder from the per-stage trace (fulcrum_total / the
  consumer decompose), do NOT attribute from producer-side.
Report BOTH the T4 ratios AND the verdict + the named next lever.

## JOB 2 (user-directed: "close that gap") — port ISA-L read_in resync for stored/fixed
gzippy-isal degrades to native (byte-exact, zero ISA-L coverage) on stored/fixed-block-
heavy inputs because ISA-L's END_OF_BLOCK stop doesn't fire there (the advisor caveat in
the GOAL#2 entry). Port rapidgzip's non-dynamic `read_in` resync (vendor isal.hpp:392-405)
so the gzippy-isal clean tail keeps faithful ISA-L coverage across stored/fixed blocks too.
- SOURCE-VERIFY isal.hpp:392-405 first-hand + the gzippy decline site
  (finish_decode_chunk_impl, the `end_bit<=stop_hint` BFINAL-only accept added in 19add96c).
- Byte-exact: dual-sha 028bd002...cb410f BOTH features (gzippy-native UNCHANGED, x86 via
  Rosetta), full lib suite + the btype01-heavy + multi-subchunk routing traps + the
  isal_tail_parity differential gate green. Ship the silesia differential in the same commit.
- Add a stored/fixed-heavy differential fixture that EXERCISES the new resync path (prove
  ISA-L coverage > 0 on it, not just that it doesn't crash). gzippy-native must stay byte-
  identical and untouched.
- Re-measure on the locked guest that the all-dynamic parity TIE (T8 1.030x) does NOT
  regress, AND report the stored/fixed-heavy corpus wall before/after.

## SEQUENCING
JOB 1 is the gate (cheap, decides whether the big asm build is ever justified) — do it
FIRST so the gate verdict is fresh. JOB 2 is an independent byte-exact port — land it in a
worktree, advisor-vet it, do NOT let it contaminate the JOB 1 measurement binary.

## CHECKPOINT (STOP for supervisor gate)
Report: JOB 1 T4 ratios + engine-closable verdict + named next lever (or BLOCKED if no
quiet box); JOB 2 landed byte-exact + the stored/fixed coverage proof + no parity regress.
Route EACH consequential claim through an independent disproof advisor (SYNCHRONOUS,
read-only, verdict to plans/low-t-gate-advisor-verdict.md + plans/isal-resync-advisor-verdict.md).
Do NOT start the full-kernel asm rewrite (user's call, gated on JOB 1). Then STOP.

## DISCIPLINES (enforced — yields + orphans hit EVERY round)
- Always work in a git WORKTREE for JOB 2 (feedback_worktree). JOB 1 is measurement-only.
- RUN SUBAGENTS SYNCHRONOUSLY (block with timeout, collect in-turn). Do NOT background-and-
  yield for a Monitor/notification — there is NO auto-reinvoke; multiple leaders died this
  way. Run the measurement YOURSELF via Bash holding the ssh; only the advisor is delegated.
- NO detached sleep sentinel. Before finishing, pgrep MUST show none of your claude -p
  subagents and no orphaned timeout `sleep` procs — kill them explicitly.
- SOURCE-VERIFY every premise first-hand. Serialize builds via cargo-lock.sh (df -h around
  builds); don't run multi-line python via Bash (write a .py file); wrap hang-prone cmds in
  timeout; diagnose the FIRST error before retrying (feedback_when_tool_errors_find_out_why);
  numbers only from the bench-locked quiet guest. Update plans/orchestrator-status.md.
