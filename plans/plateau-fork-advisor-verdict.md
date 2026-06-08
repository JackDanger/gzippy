Warning: no stdin data received in 3s, proceeding without it. If piping from a slow command, redirect stdin explicitly: < /dev/null to skip, or wait longer.
I've verified the brief's technical claims against the actual code. The decode path is a faithful ISA-L igzip port: single 12-bit root (`short_code_lookup[next_12_bits]`) + long-fallback, multi-symbol packing (sym_count 1–3), software-pipelined `pre` fast loop, dist table already single-root+fallback (no 2-level→1-level lever exists — it's already 1-level). The brief is accurate. Here is my disproof verdict.

---

# DISPROOF VERDICT — Plateau / Fork gate

## Q1 — Is the plateau claim valid? Mostly YES; one overstatement that doesn't change the conclusion.

All 5 STEP-2 techniques check out against source:
- **#1 L1 table** ✓ — `short_code_lookup: [u32; 4096]` = 16 KiB, the igzip DECODE_LOOKUP geometry, confirmed in `lut_huffman.rs:243`.
- **#4 FASTLOOP** ✓ — `decode_clean_into_contig`'s `'fast:` loop is the real VAR_V pipeline: `pre` decode-ahead, `sym_count==1` single-byte fast path, speculative 8-byte packed store, direct `base[*pos-d]` backref, yield-check amortized. Production-gated off `slow_spin==0` only.
- **#7 no bounds checks** ✓ — the contig path is `unsafe` with a written headroom proof.

**#5 prefetch — "structurally impossible" is OVERSTATED, but the no-headroom conclusion still holds.** A decode-2-ahead pipeline is NOT impossible: once `pre` is decoded you know `bit_count0`, so the bit position of symbol N+1 is known and you *could* peek+prefetch `short_code_lookup[next_idx]` before consuming. The reason it buys nothing is not the data dependency — it's that the table is **16 KiB and L1-resident already**, so there is no miss to prefetch away. Correct destination, wrong signpost. This doesn't open a lever.

## Q2 — Un-tried algorithmic headroom before inline-asm? NO.
- **Wider root table** — refuted by mechanism, not TIE: 12-bit/16 KiB is igzip's deliberate L1 ceiling. 14-bit = 64 KiB blows the 32 KiB L1; the fallback rate it would save is already <1% on L9-dynamic-Huffman text (codes cluster at 6–9 bits, ≥99% hit the root). Net: trades a sub-1% fallback for L1 misses on the 99% common case → worse.
- **2-level→1-level dist** — non-existent lever: `LutDistCode` is *already* single-root (`InflateHuffCodeSmall`) + long-fallback (`lut_huffman.rs:1180`), the igzip geometry.
- **decode-2-symbols-ahead** — the only un-refuted "could try." But it lives squarely in the **LLVM-codegen-vs-asm scheduling basin** that IS the 36 ms wall; more software-pipelining in Rust adds live state (2 pre-decoded symbols → register pressure under LLVM) and complicates the careful-loop/resumable boundary, with no control over the register allocation that asm gets. TIE-likely grind, not clear headroom — and if attempted it belongs *inside* the inline-asm decision, not before it.

**The 36 ms is genuinely the intrinsic hand-asm-vs-LLVM gap.** Confirmed.

## Q3 — Is this the right fork to escalate? NO — it's mis-framed, and there is a cheaper unblocked path.

This is my main disproof finding. The fork frames the decision as *engine-bound*, but the validated numbers say the engine is **only ~63% of the gap**:

| segment | size | closes with |
|---|---|---|
| native → ocl_cf (**engine**, asm-vs-LLVM) | ~36 ms | inline-asm igzip transliteration (huge/risky) |
| ocl_cf → rg (**C2 non-engine residual**) | ~21 ms | marker-region decode + bootstrap + scheduling |

The decisive fact buried in the brief: **C2's 21 ms is present even in `ocl_cf` running real ISA-L** (0.945×). So:
1. The engine is **not the sole binder** — a perfect engine still lands at 0.945×.
2. C2 is unclosed **even with FFI**, so it is *not* an engine problem at all. It is **architecture/scheduling/marker-region** — i.e. exactly the *faithful-port* territory the charter actually prioritizes (goal text: "faithfully port rapidgzip's decode pipeline").
3. C2 is **lower-risk** than inline-asm, and **benefits BOTH goals** (moves ocl_cf toward 1.0× *and* moves native).
4. There's a concrete, already-located C2 lever: the confirmed-offset prefetch gap (recalled: "high-T wall = 4 head-of-line stalls at confirmed offsets ≠ partition guess; ~40% of T8 wall; fixable, NOT architectural").

Escalating inline-asm now would commit the highest-risk lever to close the **smaller-leverage 63%** while a lower-risk, dual-goal, charter-aligned 37% sits open. That's premature.

## Q4 — Frozen-host 1.4 GHz mis-state the plateau? Minor risk, doesn't block, worth hardening.
Two compute-bound codes on the same core have a **roughly frequency-invariant ratio**, and a fixed low clock actually *exposes* an IPC gap most cleanly — so 0.945× is unlikely to be a low-freq artifact. The one real risk: ISA-L's memory-level parallelism could hide latency differently at turbo, shifting the *absolute* split between the 36 ms and 21 ms segments. Cheap to settle. Recommend one turbo-freq confirmation run to **harden** the escalation decision — but it does not block the verdict.

---

## VERDICT

**KEEP-GRINDING(C2 non-engine 21 ms residual — confirmed-offset prefetch + marker-region/bootstrap scheduling).**

The engine plateau itself is **REAL and validly bounded**: STEP-2 techniques are genuinely done/inapplicable, the 36 ms is the intrinsic asm-vs-LLVM gap, and inline-asm transliteration of igzip's hot loop is correctly identified as the *last, highest-risk* lever. **Do NOT escalate that fork yet** — it optimizes the smaller-leverage segment at the highest risk while the larger-leverage, lower-risk, charter-aligned C2 residual (which blocks *both* goals and ties rg in *neither* path until closed) is still open. Close C2 first; re-measure. Escalate the inline-asm fork **only when** the residual is provably engine-only (ocl_cf within noise of rg) — at which point the fork is a clean "asm-or-accept-0.945×" decision instead of the current mixed one. Bring the turbo-freq number to that escalation.
