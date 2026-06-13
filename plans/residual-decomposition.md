# RESIDUAL-DECOMPOSITION ORACLE — owner charter (supervisor, 2026-06-08)

## WHY (the gate result that makes this THE lever)
Quiet-box T4 gate (bankable pending Steward confirm): ocl_cf (real-ISA-L engine oracle)
0.899x, native 0.740x vs rapidgzip. So at T4: engine share `ocl_cf-native = 0.159x`, and a
NON-ENGINE residual `<=0.101x / ~55ms` SURVIVES swapping in real ISA-L. Even a perfect
engine lands at 0.899x => the asm rewrite alone CANNOT reach the >=0.99x bar at T4; the
non-engine residual MUST also close. It is the lower-risk lever and helps BOTH builds. But
the 0.101x is an UPPER BOUND that bundles three buckets and cannot be turned into a lever
until decomposed.

## THE BUCKETS to separate (the 0.101x upper bound contains)
(i)  MARKER-PREFIX pure-Rust engine compute — the <=32 KiB markered prefix per chunk that
     does NOT go through ISA-L even in ocl_cf (ocl_cf only routes the clean window-continuation
     tail through ISA-L FFI). So part of the "non-engine" residual is actually still ENGINE
     (pure-Rust marker decode) hidden in ocl_cf.
(ii) FFI-HANDOFF cost — the per-chunk cost of crossing into ISA-L (reserve, bit-align, the
     decline/accept path from 19add96c).
(iii) SCHEDULING / BOOTSTRAP / WINDOW-ABSENT placement — head-of-line stalls at confirmed
     offsets != partition guess (project_confirmed_offset_prefetch_gap, ~40% of T8 wall in a
     prior reading), the window-absent marker bootstrap, pool-fill cadence.

## THE JOB — build decomposition oracles, size each bucket, name the turnable sub-lever
Build BYTE-TRANSPARENT (OFF==identity) oracle knobs, each removing/neutralizing ONE bucket,
and measure the interleaved wall response per CLAUDE.md rule 3 (REMOVE and measure a
ceiling; never extrapolate a slow-injection slope):
- Bucket (i): route the marker prefix through ISA-L too (or a "prefix-free" seed oracle) to
  size how much of the residual is still pure-Rust marker compute.
- Bucket (ii): an FFI-null / FFI-batching oracle to size the per-chunk handoff overhead.
- Bucket (iii): a perfect-placement / window-seed-all oracle (the SEEDFULL family already
  exists — reuse it) to size scheduling/bootstrap.
PRE-REGISTER the falsifier per oracle BEFORE measuring. The output is: the 0.101x split into
(i)/(ii)/(iii) with which one dominates and is FAITHFULLY closable => THAT becomes the next
build lever (its own gated turn). Do NOT build the fix this turn — size the buckets first.

## DISCIPLINES + GATES
- Work in a git WORKTREE. Byte-exact: every oracle OFF==identity, dual-sha 028bd002...cb410f
  BOTH features (native + isal via Rosetta), full lib suite green.
- Numbers ONLY from the bench-locked quiet guest (procs_running gate); if no quiet box,
  report the SPLIT as EXPLORATORY and say so (Steward will not bank it). Matched same-sink,
  interleaved best-of-N>=9, sha-verified, path=ParallelSM asserted.
- Route the decomposition through (a) the Measurement-Integrity Steward for bankability and
  (b) an independent disproof advisor for the inference (does the split survive: are the
  oracles additive or do they double-count? is each a true REMOVAL not an attribution?).
  Verdict to plans/residual-decomposition-advisor-verdict.md.
- RUN SUBAGENTS/ADVISORS SYNCHRONOUSLY (no background-and-yield). Source-verify first-hand.
  Serialize builds via cargo-lock.sh, df -h around builds. No multi-line python via Bash.
  Diagnose the FIRST error before retrying. NO orphan processes / sleep sentinels. Update
  plans/orchestrator-status.md. STOP at the checkpoint for the supervisor gate — do NOT start
  the chosen fix, and do NOT touch the asm rewrite (user's gated call).
