# INTEGRATE VAR_V + MEASURE THE REAL WALL — leader charter (supervisor, 2026-06-07)

VAR_V (speculative software-pipelined loop on flat-u8) isolation-benched at 156 MB/s = 0.555×
ISA-L, byte-exact, +48% over scalar — the biggest inner-loop gain of the campaign. By the
pre-registered RATE falsifier it's a PLATEAU (0.555 < 0.85), BUT the advisor overturned
rate-ratio-as-verdict: it's producer-side ATTRIBUTION, and CLAUDE.md rule 1 says the only
verdict is the interleaved WALL. The §3 projection (decode_wall ≈ 0.410s, below both the 0.604s
bar AND the ~0.54s floor) says decode may STOP BINDING. So:

## DECISION (supervisor, advisor-endorsed): INTEGRATE + MEASURE THE WALL
Principled, not goalpost-moving: (a) rule 7a — a byte-exact gain is KEPT regardless; (b) rule 1
— only the interleaved wall settles whether decode stopped binding. The microbench cannot.

## THE JOB (you implement; delegate the gates)
1. **Integrate the VAR_V speculative pipeline into the PRODUCTION flat-u8 clean path** (Engine M
   post-flip clean tail, the window-absent path that is 89% of production). FAITHFULLY, with the
   REAL overheads the bench elided (ring/wrap, resumable contract, CRC) — NO SHORTCUT (the bench's
   0.555 is optimistic precisely because it stripped these; integrating a stripped version would
   be the shortcut the user forbids). Byte-exact: dual-sha 028bd002…cb410f BOTH features, all lib
   tests + the adversarial seam test green.
2. **Measure the real interleaved production wall** on the locked guest YOURSELF (hold the ssh;
   idle-before, restore-after, N≥9 interleaved, sha-verified) vs rapidgzip's 0.604s same-sink
   wall, at T1 AND T8. Report the wall delta and — critically — WHAT NOW BINDS: did decode stop
   binding (wall improved toward/below 0.604 / the gap collapsed)? Or did the elided overheads
   re-bind decode (wall ≈ unchanged)? Use the existing per-stage trace to name the new binder.

## THE TWO CONTINGENT NEXT LEVERS (do NOT start them this turn — report which applies)
- If decode STOPPED binding ⇒ PLACEMENT is now the live binder (legitimately re-open it; it was
  only "slack" while decode dominated).
- If decode STILL binds (overheads re-bound it) ⇒ the inline-asm/BMI2 spike to push VAR_V's rate
  0.555→0.85 (charter-authorized inner-loop open territory).
Both are charter-authorized; the FFI/bar fork re-opens ONLY if both later fail.

## CHECKPOINT (STOP)
Report: integration landed byte-exact?; the real T1/T8 production wall vs 0.604s; the new binder
(decode vs placement). Route through an independent disproof advisor (SYNCHRONOUS, read-only,
verdict to plans/varv-integration-advisor-verdict.md) attacking byte-exactness, whether the
integration kept the real overheads (no stripped shortcut), and the wall/binder claim. Then STOP
for supervisor gate.

## DISCIPLINES (enforced — yields + orphans hit EVERY round)
- RUN SUBAGENTS SYNCHRONOUSLY (block with timeout, collect in-turn). Do NOT background-and-yield
  for a Monitor/notification — NO auto-reinvoke; multiple leaders died this way. Run the
  measurement YOURSELF via Bash holding the ssh; only the advisor is a delegated SYNCHRONOUS call.
- NO detached sleep sentinel. Before finishing, pgrep MUST show none of your claude -p subagents
  and no orphaned timeout `sleep` procs — kill them explicitly (a sleep orphan leaked again).
- SOURCE-VERIFY premises first-hand. Serialize builds via cargo-lock.sh (df -h around builds);
  don't run multi-line python via Bash (write a .py file); wrap hang-prone cmds in timeout;
  diagnose the FIRST error before retrying; numbers only from the locked guest. Update
  plans/orchestrator-status.md.
