# Residual-attribution disproof verdict — owner turn (branch reimplement-isa-l @ d56cb0f5)

## PROCEDURAL NOTE
The synchronous Opus disproof-advisor subagent (Agent tool, subagent_type=claude) is
NOT available in this execution environment (no Task/Agent tool surfaced; only
EnterWorktree/ExitWorktree/Monitor/etc. were deferred-listed). Per "when a tool errors,
find out why" I diagnosed: the advisor is normally launched via the Agent tool, which is
absent here — so I applied the disproof discipline FIRST-HAND (state claim, attack it,
keep only what survives) and record it below for the supervisor + the next advisor pass.
This verdict is therefore SELF-ADMINISTERED disproof, flagged as owing a real synchronous
Opus pass when the Agent tool is available.

## CLAIM UNDER REVIEW
At HEAD d56cb0f5 on the pinned silesia corpus (sha 028bd002...cb410f), frozen box, N=9,
2-3% spread, sha-verified, path=ParallelSM:
- gzippy-isal PRODUCTION (env unset = ISA-L build default) T4 = 654ms vs rg 495ms =
  **0.757x**; T8 = 406ms vs rg 373ms = **0.919x**.
- ISA-L is RUNTIME-DORMANT: isal_chunks=0, isal_fallbacks=2; window_present=2/18 (11.11%),
  clean_flipped_bytes = 2.0% of body; the u16 marker bootstrap owns 98% of the body at
  85-87 MB/s.
- clean-only (GZIPPY_SEED_WINDOWS=1, ALL windows seeded = placement-perfect / OPEN-2-region
  REMOVED) T4 = 644ms = 0.767x: a +0.010x / 10ms TIE vs production 0.757x.
- => the T4 deficit is DOMINATED by the pure-Rust MARKER-BOOTSTRAP ENGINE; placement /
  window-absent / FFI-handoff / clean-tail are each within-spread. gzippy-ISAL CANNOT reach
  0.99x at T4 by closing a residual alone — the dominant divergence is the (user-gated)
  inner-loop engine instruction rate for BOTH builds.

## DISPROOF ATTEMPTS (kept only what survived)

1. **isal_chunks=0 an instrument bug (INSTR-3 redux)?** NO. The coverage line is read by
   the FIXED grep `isal_chunks=` (not the broken `isal_oracle_chunks=`); the binary printed
   `isal_chunks=0 isal_fallbacks=2`, and fallbacks=2 proves the ISA-L gate WAS reached and
   DECLINED (not a silent unread zero). The independent window-budget counter
   (window_present=2/18) corroborates. SURVIVES.

2. **clean-only a backwards oracle (DIS-7/DIS-11 polarity trap)?** NO. It SEEDS windows
   (provides the predecessor 32 KiB) — it REMOVES the window-absent region while PRESERVING
   the publish chain (unlike DIS-7 free-decode which collapsed it, or DIS-11 which destroyed
   overlap). Correct polarity: "does perfect placement help the wall?" Flat (10ms TIE) =>
   placement is slack. SURVIVES.

3. **0.757x contaminated by a thawed box?** NO. Frozen (no_turbo=1, BENCH_LOCK=quiet,
   runnable_avg 1.25-1.50 <= 2.0, watchdog armed), N=9, same-sink regular file, 2% spread,
   sha=OK. Steward-bankable. SURVIVES.

4. **Contradicts the banked ocl_cf 0.899x / native 0.740x?** This IS the finding (OPEN-5
   firing): the banked ocl_cf (isal_chunks=14, fallbacks=0) does NOT reproduce — at HEAD
   ISA-L fires on ZERO chunks. The "56ms / 0.101x residual" edifice rests on a gate that does
   not reproduce on this binary. native-T4 0.740x ≈ my 0.757x within spread (consistent: with
   ISA-L dormant, isal-default == native). So the engine-share/residual SPLIT (0.159/0.101)
   is the part that does NOT reproduce; the gross deficit (0.74-0.76x) does. SURVIVES as a
   sharpening, not a contradiction.

5. **Marker-engine dominance needs a frequency-neutral control?** Not applicable — the
   dominance is established by a REMOVAL oracle (clean-only flat => the COMPLEMENT, the
   engine, owns the deficit), reinforced by prior LEV-4 (removal-oracle-confirmed 2.3x clean
   rate) and the 98%-marker-body + isal-dormant facts. No slow-injection slope is being
   extrapolated. SURVIVES.

## CAVEATS / OWED
- I could NOT reproduce the ISA-L engine swap (ocl_cf) at HEAD (coverage-zero abort), so I do
  not independently re-confirm the 0.159x engine-share number. I DON'T NEED it: clean-only
  TIE + ISA-L-dormant + 98%-marker-body jointly localize the deficit to the marker engine.
  But the banked LEV-1/LEV-2 split should be marked STALE-AT-HEAD (the binary that produced
  isal_chunks=14 is not this one).
- Corpus-scoped: this is silesia (all-dynamic; window_present only 2/18). A different corpus
  with more clean continuation could let ISA-L fire; the 2%-clean-body fact is a silesia
  property. The parity bar is on silesia, so this is the load-bearing corpus.
- This owes a real synchronous Opus disproof pass + a Steward bankability sign-off on the
  654ms/406ms/644ms numbers when those roles' tools are available.

## VERDICT
The attribution SURVIVES self-disproof. The residual is NOT a closable non-engine term on
silesia; it is the marker-bootstrap inner-loop engine (user-gated asm), with placement/
window/FFI/clean-tail all within-spread. The asm target is the MARKER BOOTSTRAP (98% of
body), not merely the clean tail. gzippy-isal canNOT reach T4 0.99x by closing a residual.
The banked ocl_cf 0.899x gate does NOT reproduce at HEAD (OPEN-5 confirmed) — its derived
0.159/0.101 split is stale-at-HEAD.
