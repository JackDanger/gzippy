**VERDICT: UPHELD-WITH-CAVEATS**

The PARTIAL *direction* is sound — the engine swap recovers a large, real, >>spread share (delta 0.159 ≈ 5× spread; this is decisively NOT a tie, F-NON-ENGINE is correctly rejected). But the precise 0.159/0.101 split is a contaminated decomposition and the banked "even the real ISA-L engine loses 0.101x" overclaims.

Disproof attempts:

- **"ISA-L engine ceiling" is a BLEND, not a pure-engine ceiling — CONFIRMED in source.** `gzip_chunk.rs:128-131,196-223`: the oracle routes ONLY the *clean 32 KiB-window continuation* through ISA-L FFI; the markered prefix and chunk-0 marker bootstrap stay in the pure-Rust u16 marker engine. So ocl_cf=0.899 still contains pure-Rust *engine* work. Consequence: the engine-closable share is **larger** than 0.159 and the "non-engine residual" 0.101 is an **upper bound**, not a measurement of scheduling/bootstrap. The headline claim "~55ms survives even the real ISA-L engine" is false as stated — it survives the real ISA-L engine *on the clean tail only*.

- **FFI overhead pollutes both terms.** ocl_cf's 545ms includes per-chunk ISA-L FFI boundary + clean/markered handoff cost that a native asm engine would not pay (and a native engine wouldn't get ISA-L speed either). So 107ms "engine" and 55ms "residual" are each muddied by a third bucket (FFI/handoff) that is neither engine-speed nor scheduling. The clean two-way split is an approximation.

- **asm ceiling = 0.899, zero margin — under-flagged.** A pure-Rust asm engine's *best case* is matching ISA-L (the existence-proof ceiling), i.e. 0.899 at T4 — never 0.99 alone. The verdict says this, but the consequence isn't banked: parity needs BOTH levers fully realized, and even both at 100% lands at exactly 1.0 with **no slack**, on the optimistic assumption asm==ISA-L. That's a thin plan; the non-engine residual must close *completely*, and per caveat #1 part of that residual is itself still engine (marker prefix) — which actually helps (more is asm-capturable) but means the scheduling lever alone cannot be sized from 0.101.

- **min-of-N estimator: survives.** delta 0.159 ≈ 5× spread → robust to min's downward bias and to differing gz/rg spreads. ocl_cf 0.899 sits ~2.5× spread below 0.99 → the "engine alone misses parity" leg is solid. The 0.101 residual at ~2.5-3× spread survives but is the thinnest banked quantity; treat it as directional, not precise.

- **"asm justified" is policy-gated, not perf-gated — state it.** The 0.159 engine share is capturable *today* via the ISA-L FFI already measured at 0.899. asm is required ONLY because the campaign goal forbids C-FFI on the decode graph. Sound given the constraint, but the justification should read "asm to capture in pure-Rust a share ISA-L proves is capturable," not "asm to close a gap we've shown only asm can close."

Bank: PARTIAL holds. Re-state the split as **engine share ≥0.159 (underestimate), non-engine residual ≤0.101 (upper bound, contains marker-prefix engine + FFI)**. Flag the zero-margin / both-levers-required risk. The confirmed-offset prefetch gap is a legitimate non-engine candidate but cannot be sized from 0.101 until the marker-prefix-engine and FFI buckets are separated (e.g. an oracle that also ISA-Ls the marker prefix, or an FFI-overhead null run).
