# OVERNIGHT AUTONOMY — supervisor operating mandate (2026-06-08 night)

USER (heading to bed) delegated FULL decision authority: "Maintain oversight, delegate
everything to the leader agent, and make all the decisions yourself." This includes the
previously-user-gated CLEAN-TAIL ASM decision — it is now the SUPERVISOR's call. Do NOT stall
waiting for the user; decide on gated evidence. (If context compacts, this file + the summary
carry the authority grant — it is for THIS overnight run; the user resumes in the morning.)

## STANDING DECISION RULES (apply autonomously)
1. GATE EVERY OWNER NUMBER MYSELF before trusting/acting/recording it (owners CANNOT spawn
   advisor/Steward in this env — [[project_owners_cannot_self_gate_verify_binary]]). Run the
   Opus disproof + bankability check at the supervisor level. A number contradicting a banked
   result = presumed measurement-error until reconciled (the mislabeled-binary bombshell).
2. VERIFY THE BINARY (isal_chunks>=14 = real-ISA-L fingerprint) on every isal measurement.
3. NO WORK-DISPLACEMENT; CONVERGE-don't-lever (match rg's runtime behavior). Byte-exact wins
   KEPT (rule 7a). TIE bar = >=0.99x at EVERY thread count.
4. Keep ONE leader working at a time per front; gate its checkpoint; launch the next. Keep the
   chain alive so completions re-invoke me. HYGIENE: release the box clean (no_turbo=0, thaw),
   pgrep clean (no orphans/sleep sentinels) — a prior agent leaked 2 cores for 3.5h.

## VERIFIED STATE (gated)
Scorecard (env-unset production, frozen N=11): isal 0.899/0.900/0.990 @T1/T4/T8; native
0.608/0.761/0.915. T8 near-tie; T1/T4 ~0.10 deficit even on real-ISA-L isal = structural
(window-bootstrap + serial-output + FFI-handoff + chunk-0), faithful-pipeline territory,
partly entangled with the gated asm. FFI-handoff negligible at T4. Instrument bug found+fixed
(clean-only NO-OP). asm = native clean-tail -> rg's ISA-L instructions (pure-Rust+inline-asm,
no C-FFI); necessary for native, INSUFFICIENT alone for low-T.

## OVERNIGHT PROGRAM (sequence; decide at each gate)
1. CLEANUP (running a8076f8a) -> review, then launch the SYNTHESIS/GATE-PREP partner (user's
   "help you" agent): maintains an authoritative 1-page STATE doc + warm first-pass disproof
   of owner checkpoints.
2. PLACEMENT ORACLE (running af441b6) -> GATE. If the faithful-placement slice > spread (with
   isal_chunks==14, no engine leak): port rg's runtime window-map (faithful, owner-turnable) ->
   launch it. If < spread: low-T is asm-bound -> go to (4).
3. JOB-2 COVERAGE (running a9e7e1a, box-free) -> GATE byte-exactness + coverage; if sound, KEEP
   (faithful gzippy-isal correctness win). Run the supervisor advisor gate on the diff.
4. ASM (now my call): SCOPE the full-kernel native clean-tail asm FIRST (source-map the ISA-L
   kernel vs gzippy's clean loop, byte-exact harness, feasibility) as a leader turn — the prior
   NO-GO was per-symbol transliteration, NOT full-kernel. It is the user's core "steal ISA-L's
   techniques in pure Rust" goal + the headline native lever. If scoping shows it tractable,
   begin incrementally with byte-exact + isolation-bench gates (do NOT bank without the wall).
   Hold the multi-session full commit if scoping is unfavorable; record the finding for morning.
5. PARALLEL-PATH NAMING (agent 3, user-requested) — after the live measurement owners settle
   (avoid churning the path they edit). Known targets: the "oracle" misnomers on PRODUCTION
   ISA-L (isal_engine_oracle_enabled / ISAL_ENGINE_ORACLE_CHUNKS / finish_decode_chunk_isal_oracle
   gate production on the isal build), GZIPPY_SEED_WINDOWS (a file-path, not a bool). Byte-
   transparent renames; both feature-sets compile + tests green; my gate.

## MORNING DELIVERABLE
A synthesized status: what each gate decided, the asm scoping verdict + whether I started it,
the placement-slice resolution, JOB-2 outcome, cleanup + naming results, and the open calls (if
any) left for the user. Be honest — TIEs as TIEs, blocked as blocked.
