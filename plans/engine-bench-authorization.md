# ENGINE ISOLATION BENCH — authorization (supervisor, 2026-06-07)

Strategic advisor (aa214edf) verdict: PIVOT-TO-ENGINE-BENCH. The engine is the unbounded,
TIE-DETERMINING unknown (2.38× clean-rate gap, +13.7% A.2 residual that survives perfect
placement at 3.47σ); placement is bounded +13.7% with 3 rounds of diminishing returns. The
§2.3 isolation bench is the ONLY legitimate way to bound a speed-up ceiling (CLAUDE.md rule 3:
removal/isolation oracle, NOT attribution) and it answers the question no placement work can
move: can pure-Rust+inline-ASM plausibly reach igzip-class clean rate, or does it plateau at
the ~pure-decoder ceiling? If it plateaus, the 1.0× bar itself is in question — a
supervisor/USER-level finding we want BEFORE any multi-week SIMD build.

## PRIMARY: build + run the §2.3 engine isolation bench
A STANDALONE microbench (NOT the production pipeline): decode ONE known-window CLEAN silesia
chunk (the d_c case) on the guest x86_64 under the build lock, three ways, byte-exact each:
  (i) gzippy's CURRENT clean inner loop (scalar u16 baseline),
  (ii) the proposed engine technique(s) — START with E1 (u8-direct clean write, lowest-risk,
       also the 1b sub-lever); add E2 (wide SIMD back-ref copy) / E3 (packed multi-literal
       store) / E4 (wide refill) as tractable, inline-ASM where Rust codegen lags,
  (iii) ISA-L `isal_inflate` itself as the upper-bound ORACLE (call the C lib in the bench —
       a MEASUREMENT oracle only; FFI stays OFF the native decode graph).

### SELF-TEST FIRST (CLAUDE.md rule 4 — two instruments were silently broken this campaign)
Before trusting ANY (ii) number: variant (iii) ISA-L must read ~2× variant (i) scalar on
single-thread silesia — the GUEST-measured ratio (NOT the discredited 337/720 absolutes). If
it does not reproduce, the bench is broken — FIX before proceeding.

### PRE-REGISTER the plateau falsifier (BEFORE running the technique variants)
State it now: if (ii) plateaus near the pure-decoder ceiling (~2× slower than ISA-L, i.e.
≈ gzippy's current class) and the residual to igzip-class EXCEEDS the inter-run spread, the
engine front is NOT proven → report the achievable floor, DO NOT integrate / DO NOT start the
multi-week engine build, and re-confront placement-consumer-pace as the only remaining lever.
PASS = (ii) reaches a clean rate that, fed through §3, projects a T8 wall ≤ rapidgzip + spread.

## RIDERS (near-free — run alongside, do NOT silently defer)
1. **Re-read `nearest_le_start` at cap=256** from the existing step-0 stall-residency probe
   (or a quick re-run): closes the placement never-retained-vs-EVICTED sub-question. If
   nearest_le_start=-1 even at cap=256 ⇒ NEVER-RETAINED (consumer-pace), not capacity-evict.
2. **Same-sink production-output floor:** the 0.61s / 0.015s-serial floor was measured
   output→/dev/null. Measure gzippy's floor with a REAL file sink (writev ~0.245s) AND
   confirm rapidgzip's 0.54s is the SAME-SINK comparison. The §3 tie verdict is contingent on
   a same-sink floor ≤0.54s — the bench cannot reach this; capture it here.

## DROP (do not rely on)
"Placement dissolves once the engine is faster" — separability is UNPROVEN, only ~24% is
engine-coupled, the cold-gets are structural. Re-measure placement AFTER the engine moves as a
BONUS, never as a go/no-go term.

## CHECKPOINT (STOP)
Report: the engine ceiling (variant ii's achievable clean rate vs the ISA-L oracle, guest
ratio) + the plateau-falsifier verdict (proven / not-proven); the two riders' results. Route
through an independent disproof advisor (verdict to plans/engine-bench-advisor-verdict.md)
attacking the bench validity (self-test, byte-exactness, is it really isolating engine compute)
and the ceiling conclusion. Then STOP for supervisor gate. Do NOT start the multi-week
production engine integration/build — that is the NEXT gate after this bench proves the ceiling.

## DISCIPLINES (enforced)
Run subagents SYNCHRONOUSLY (no auto-reinvoke); NO detached sleep sentinel (orphaned twice);
guest runs from a Bash task that HOLDS the ssh; verify guest idle before + restore host after;
leave NO orphaned processes (advisors/sleeps have leaked every round — kill them); serialize
builds via cargo-lock.sh (df -h around builds); don't run multi-line python via Bash (write a
.py file); wrap hang-prone commands in timeout; SOURCE-VERIFY any premise first-hand; diagnose
the FIRST error before retrying; numbers only from the locked harness/guest. Update
plans/orchestrator-status.md.
