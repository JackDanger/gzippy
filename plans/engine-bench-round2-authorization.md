# ENGINE BENCH ROUND 2 — settle the plateau falsifier (supervisor, 2026-06-07)

Round-1 bench (advisor-vetted, byte-exact): scalar 118 / E1 125 (+6% — output minor) / ISA-L
388 MB/s. Ceiling = gzippy ~3.10× slower than pure ISA-L; **the gap is in the INNER HUFFMAN
LOOP**, not output. Plateau falsifier NOT-PROVEN-YET (E2–E4 unbuilt). Same-sink tie bar =
rapidgzip 0.604s (favorable, +12% room). Placement sub-question CLOSED (never-retained).

This round SETTLES the tie-reachability question: can pure-Rust + inline-ASM E2–E4 approach
igzip-class clean rate? Still a STANDALONE BENCH — NOT the multi-week production integration
(that remains separately gated AFTER this proves the ceiling).

## TASK
0. **Commit the round-1 bench harness first** (benches/engine_isolation.rs + scripts/bench/
   {run,guest}_engine_isolation.sh + the lib.rs measurement re-export + Cargo.toml) — it is
   this round's instrument; don't leave it uncommitted. Recalibrate the self-test band to
   ~[2.5×,3.6×] (round-1 "FAIL" was mis-calibrated: gzippy-vs-pure-ISA-L is a purer denominator).
1. **Prototype E2–E4 in variant (ii)** of the bench (inner Huffman loop — OPEN territory,
   inline ASM authorized where Rust codegen lags):
   - E2: wide SIMD back-ref copy (overlapping vector LZ77 copy — igzip technique; silesia is
     back-ref-heavy, highest-confidence inner-loop lever).
   - E3: packed multi-literal store (collapse the existing 2-/3-literal chain into one wide
     store; ca52389 regression is NON-BINDING — predates PRELOAD + u16-ring fold, re-measure).
   - E4: wide refill (≥56-bit buffer, refill amortized over multiple symbols when headroom).
   Each technique: byte-exact (SHA-equal vs scalar + ISA-L in the bench), measured separately
   then stacked, so we know which carries the gap.
2. **Chunk-sweep ≥3–5 representative clean silesia chunks** (round-1 was one chunk =
   suggestive, not a bound). Report per-chunk + aggregate.
3. **Settle the plateau falsifier** (compute the PASS threshold via §3 with the UPDATED
   0.604s same-sink bar): PASS = variant (ii) with E2–E4 reaches a clean MB/s that, fed
   through §3, projects a same-sink T8 wall ≤ 0.604s + spread. PLATEAU/FAIL = (ii) stays near
   the pure-decoder class (~2× slower than ISA-L) with residual > spread ⇒ engine front NOT
   provable in pure-Rust+ASM as prototyped ⇒ SUPERVISOR/USER-level finding (the 1.0× bar may
   be unreachable without FFI / needs the bar revisited). Report which, with the numbers.

## CHECKPOINT (STOP)
Report: per-technique + stacked (ii) clean rate vs the ISA-L oracle (guest ratio), chunk-sweep
spread, and the SETTLED plateau verdict (PROVEN can-approach-igzip-class / PLATEAU). Route
through an independent disproof advisor (verdict to plans/engine-bench-round2-advisor-verdict.md)
attacking byte-exactness, whether the bench still isolates engine compute with E2–E4 in, and
the falsifier math. Then STOP for supervisor gate. Do NOT start the production engine
integration — that is the NEXT gate, contingent on PROVEN here.

## DISCIPLINES (enforced — orphans leaked EVERY round; clean up)
Run subagents SYNCHRONOUSLY (no auto-reinvoke); NO detached sleep sentinel; guest runs from a
Bash task that HOLDS the ssh; verify guest idle before + restore host after; **leave NO
orphaned processes — kill your subagents' claude -p AND any timeout-wrapper sleeps before
finishing** (4 orphan sleeps + advisors had to be cleared this round). Serialize builds via
cargo-lock.sh (df -h around builds); don't run multi-line python via Bash (write a .py file);
wrap hang-prone commands in timeout; SOURCE-VERIFY premises first-hand; diagnose the FIRST
error before retrying; numbers only from the locked guest. Update plans/orchestrator-status.md.
