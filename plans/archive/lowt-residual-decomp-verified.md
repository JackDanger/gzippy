# LOW-T RESIDUAL DECOMPOSITION (verified isal binary) — owner charter (supervisor, 2026-06-08)

## WHY (verified scorecard, reconciled)
gzippy-isal (real ISA-L, env-unset production, frozen N=11): T1 0.899x / T4 0.900x / T8 0.990x.
Even REAL ISA-L tops out at ~0.90 at T1/T4 — so the ~0.10 low-T gap is NOT the clean-tail
engine (ISA-L is already running it). It is the RESIDUAL: {marker-prefix pure-Rust engine,
per-chunk FFI-handoff, scheduling/placement}. This residual is the BINDING constraint for BOTH
builds at low-T (native's low-T deficit = this residual + the clean-tail engine gap). It is
UNDECOMPOSED — the prior residual-attribution turn ran on a MISLABELED gzippy-native binary
(isal_chunks=0 = the native signature) and is VOID (see
[[project_owners_cannot_self_gate_verify_binary]], plans/isal-dormancy-advisor-verdict.md).

## STEP 0 (MANDATORY — verify the binary; the prior turn skipped this and mismeasured)
Build the gzippy-isal binary at HEAD d56cb0f5 and PROVE it is the isal build before ANY
measurement: run a GZIPPY_VERBOSE decode of silesia and assert isal_chunks >= 14 at T4/T8
(isal_chunks is incremented ONLY in the real-ISA-L cfg, gzip_chunk.rs:386; the native stub
:390-400 returns Ok(false) and never increments — so isal_chunks>0 PROVES the isal feature-set
+ real ISA-L ran). Also assert GZIPPY_ISAL_ENGINE_ORACLE UNSET and path=ParallelSM. If
isal_chunks==0, STOP — you built native; rebuild. Report the coverage as proof-of-binary.

## STEP 1 (decompose the low-T residual on the VERIFIED isal binary, T4 primary + T1)
The ~0.10 residual = isal at 0.900 (T4) below 1.0. Decompose via REMOVAL oracles (rule 3,
pre-register each falsifier), each removing ONE sub-term, measure the interleaved WHOLE-SYSTEM
wall response:
- MARKER-PREFIX engine: how much of the residual is the pre-flip u16 marker-prefix decode
  (pure-Rust, runs even on isal before FlipToClean)? (a seed-all / prefix-shrink oracle).
- FFI-HANDOFF: the per-chunk ISA-L entry/exit cost (an FFI-batch / handoff-null oracle).
- SCHEDULING/PLACEMENT: re-run the OPEN-2 clean-only / seed-all oracle ON THE VERIFIED ISAL
  BINARY (its prior baseline was the void native wall — RE-OPENED in the ledger). Does
  perfect placement move isal's low-T wall?
NO WORK-DISPLACEMENT: report the whole-system wall AND any displaced-to stage before/after.

## OUTPUT
The low-T residual split {marker-prefix / FFI / scheduling} with which DOMINATES and is
FAITHFULLY closable (matches a rg vendor mechanism, no displacement). This is the binding
low-T lever for BOTH builds; it determines whether ANY path reaches >=0.99x at T1/T4, and it is
INDEPENDENT of (and additive to) the native clean-tail asm (user-gated). Do NOT start the asm.

## GATES + DISCIPLINES
git WORKTREE; numbers ONLY from the bench-locked quiet guest (procs_running gate); matched
same-sink, interleaved N>=9 (use N>=15 for any T8 cell — the 9% T8 spread is too loose),
sha-verified, path=ParallelSM. SOURCE-VERIFY the binary feature-set (STEP 0) + counter
semantics + env first-hand. The synchronous Opus advisor + Steward may be UNAVAILABLE to you
(Agent tool absent in owner env) — if so, run rigorous self-disproof AND hand the SUPERVISOR
your raw numbers + provenance so the supervisor runs the Steward/advisor gate (do NOT claim
"advisor-vetted" if you couldn't spawn one — say "self-disproof, owes supervisor gate"). Run
measurements YOURSELF holding the ssh. Serialize builds via cargo-lock.sh, df -h around builds.
No multi-line python via Bash. Wrap hang-prone cmds in timeout. Diagnose the FIRST error before
retrying. NO orphan processes / sleep sentinels — pgrep clean on local + guest + neurotic.
Update plans/orchestrator-status.md + the disproof-ledger. STOP at the checkpoint and report
to me (supervisor): STEP-0 proof-of-binary (isal_chunks), the residual split, the dominant
faithfully-closable sub-term, and your raw numbers + provenance for my Steward/advisor gate.
