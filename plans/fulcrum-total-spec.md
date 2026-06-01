# Fulcrum TOTAL — the complete-instrument spec (advisor-validated 2026-06-01)

Goal: running `fulcrum profile <run>` means "I know exactly what's happening" — a
single typed dataset + decision-views that return VERDICTS, not tables to argue over.
Honest promise: **descriptive-total + perturbation-gated-causal.** The irreducible
judgment that remains = *which* perturbations to run + *whether each oracle is
faithful*; surface those, don't pretend they're gone.

## Full signal enumeration (everything that could shine a light)
A. App trace (Chrome-trace via probe): per-phase span durations; instants
   (decode_mode, window_publish, decode_decision); per-chunk start_bit/end_bit/
   partition_idx/clean-vs-window-absent/had_markers. [WIRED]
B. CPU PMU/perf: cycles/instructions/IPC; TMA L1+L2; per-region time-windowed
   counters; per-span IPC + mem-stall-cycles (S3); LLC/L1/L2/L3 miss + MPKI;
   DTLB/page-walk; c2c/false-sharing; uncore IMC bandwidth (S5). [SCAFFOLD: mech,
   region_hw, mech_arch]
C. OS/kernel: page-faults; ctx-switches; migrations; syscall time; futex/lock.
D. Allocator: malloc counts/bytes; clear_page/zeroing.
E. Causal/perturbation: coz curves [scaffold]; slow-inject sweep; oracle removal.
F. Wall ground truth: interleaved best-of-N, sha, min/median/sd/ratio, freq-lock
   proof [WIRED in clean_bench, but SEPARATE from the trace dataset].
G. Cross-tool: vs / vs-sweep span subtraction [WIRED]; symmetric both-tools counters.

## Pared to BUILD (advisor-corrected — adds, drops, reorder)
PERF-FREE TIER (zero feasibility risk — build FIRST, resolves the live disagreement):
- **S1 (PROMOTED to #1, the arbiter):** per-consumer-stall start-time pair —
  start_time(decode_i) vs earliest free worker with chunk i ADMISSIBLE + admissibility
  flag (mis-schedule vs serial-frontier vs speculation-invalid). Pure trace arithmetic.
  ARBITRATES the two memory notes: stall coincides with admissible-ready-work-unused ⇒
  PLACEMENT ([[project_wall_is_consumer_critical_path]] wins, port queuePrefetched...);
  stall coincides with frontier-not-decoded ⇒ RATE ([[project_t8_saturated_pool_diag]]
  wins, ~15% bounded). (NOTE: the frontier-placement oracle already gave the causal
  PLACEMENT-refuted answer; S1 is the descriptive confirmation + per-stall breakdown.)
- **Residual tier (S2/S6/S7, perf-free — NAMES the 17-41% residual):** getrusage(RUSAGE_THREAD)
  ru_minflt/majflt (page-faults), nvcsw/nivcsw (ctx-switch), maxrss; /proc/self/task/<tid>/
  schedstat (run/runnable/blocked split — perf CAN'T easily give this); jemalloc stats or
  VmRSS deltas (allocator/clear_page). Snapshotted at region boundaries = self-keyed, no
  sampling error, NO PMU. This half resolves MOST of the disagreement with no risk.
- Worker-occupancy + out-of-order-depth + queue-depth timelines (trace-derived).

PERF TIER (spike feasibility FIRST, then build):
- **S3:** per-TID mem-stall (CYCLE_ACTIVITY.STALLS_MEM_ANY) + IPC + branch-MPKI. MUST be
  per-TID-bound (pooled-by-time is SMEARED at -pN, honest only at -p1).
- **DROP S5 (uncore IMC bandwidth):** least-feasible in LXC (system-wide, paranoid≤0,
  often unreachable from a container) AND redundant with region_hw mem-tier DRAM proxy;
  i7-13700T desktop part is nowhere near IMC saturation. Cut.
- **DROP syscall-firehose** (noise; futex is the only syscall that matters — get it from
  getrusage involuntary-ctxsw + existing WAIT spans).
- **GATE c2c behind a fingerprint** (run `fulcrum c2c` only if S1 says stall×ready-unused
  AND mem-tier shows surprising L3/remote on a small shared struct). Don't build eagerly.
- KEEP branch-MPKI (17% of gap per sm_scaling note; cheap; already wired).

## The JOIN (the contract every signal writes to)
Key = **(tid, time-window, region, partition_idx)**. CRITICAL FIX: the existing
region_hw join charges a sample to the first region by ORDER whose window contains it —
structurally SMEARED at T>1 (16 workers in the same wall window across regions). Add `tid`
to MemSample; join per-thread-then-aggregate (each thread is in exactly one span at a time).
Interval-apportionment error bound ±10-20% on boundary-straddling regions (uniform-rate
assumption); use 1ms intervals + prefer PEBS-by-timestamp (exact) over interval-apportionment.
ANTI-TAUTOLOGY RULE: never attribute by ordinal position; always by timestamp-containment
with an explicit purity/straddle metric; refuse to render a number whose purity<threshold
without a SMEARED flag. (This is what prevents the +0.0%-tautology / broken-positional-join class.)

## PERF FEASIBILITY (the killer — two-stage gate, NOT host-attached-to-guest)
Bench runs in unprivileged LXC guest 199; coz proved perf-sampling blocked there. PATH:
(1) HOST: `sysctl -w kernel.perf_event_paranoid=1` (or 0 for PEBS-mem); it's host-GLOBAL
not namespaced. (2) GUEST: RE-PROBE via mech_arch's `perf stat true` exit-0 check. If still
blocked ⇒ perf_event_open seccomp/apparmor on the container ⇒ lxc.cap.keep CAP_PERFMON or
nsenter. Guest CLOCK_MONOTONIC == host (shared kernel timekeeping) so the trace timeline
lines up either way. Guest-side perf (after paranoid lower) is SIMPLER than host-attached
(symbol --symfs pain) AND the join is by timestamp so symbols are moot. GATE HARD on the
probe: emit NO zeros if perf unavailable — degrade to the perf-free tier + say so.

## ANALYTICS API ("I know exactly what's happening")
ONE typed `ProfileBundle` per run (serialized next to the trace) + fixed DECISION-VIEWS
that each print a VERDICT line (not a table to interpret — a queryable store invites the
analyst-bias the PROCESS warns against):
- `fulcrum profile <run>` → emits the bundle.
- `fulcrum decompose` → wall = Σ(named regions) + NAMED residual (page-fault/alloc/ctxsw/futex).
- `fulcrum schedule` → S1 verdict: "X% consumer-stall = PLACEMENT vs RATE"; prints which note wins.
- `fulcrum bound` → compute-vs-memory; VERDICT only at -p1; at -pN prints "run-level only,
  per-region SMEARED, re-run -p1".
- `fulcrum delta A B` → cross-tool subtraction (from two bundles).
- `fulcrum confirm <lever>` → the perturbation gate.
ADMISSION RULE (keeps it clean, not kitchen-sink): a signal stays IFF it feeds a
decision-view's verdict OR guards another signal's purity. No orphan columns. (S5 fails;
branch-MPKI passes; getrusage-pagefault passes.)

## CAUSAL GATE (descriptive ≠ causal — bake into the type system)
Every lever claim carries `elasticity: Unmeasured` until `fulcrum confirm <region>` runs the
slow-inject sweep (GZIPPY_SLOW_BOOTSTRAP) + frequency-neutral control (sleep variant) + reads
interleaved wall response + oracle-ceiling (each oracle SELF-TESTED first — the clean-window
oracle was BROKEN here). The string "the lever is X" is UNREACHABLE without a perturbation
record in the bundle. (Bakes PROCESS rules — never conclude-from-attribution, slope≠ceiling,
always-freq-neutral — into type-state, not analyst discipline that has repeatedly failed.)

## BUILD ORDER (advisor-corrected — join first, S1 before S3, perf-spike before perf-build)
1. Unified join + ProfileBundle schema (the contract) + purity/straddle convention + no-orphan rule.
2. S1 + residual tier (S2/S6/S7) — PERF-FREE, decisive, zero risk. May settle the disagreement here.
3. Perf feasibility SPIKE (host paranoid lower → guest re-probe → cap/nsenter or declare degraded). Yes/no before building on it.
4. S3 per-TID PMU — only if (3) passed; on the tid-augmented join.
5. Perturbation orchestration + type-state elasticity gate.
6. Cross-tool symmetric (read rapidgzip's bundle) — last, smallest delta.
DROP S5. Fulcrum dev in a WORKTREE off origin/main (a copy-edit agent pushes to main).
