# Engine-optimization plan (the EARNED pure-Rust speedup) — PLAN ONLY, DO NOT EXECUTE YET

Status: gzippy has reached FAITHFUL pattern parity with rapidgzip
(`plans/parity-final.md`, advisor-signed-off). The remaining wall gap
(**1.83×**, T8 silesia) is the accepted pure-Rust-vs-ISA-L ENGINE-SPEED residual:
every Fulcrum-visible lever traces to decode rate, all rate terms collapse to a
uniform ~0.54× scalar (`d_c 0.52×`, `d_w 0.54×`, `L_resolve 0.54×`), and the
`fulcrum schedule` arbiter reads **100% RATE / 0.0% PLACEMENT**.

This is the LICENSE (CLAUDE.md "Permission to fully reimplement the inner inflate"
+ "build the fastest possible raw Huffman decoder") for an inner-loop speedup —
NOW that the structure is faithful. The inner Huffman loop / `LitLenTable` /
`DistTable` / `Bits` primitives are open territory; architecture stays vendor-shaped.

> **DO NOT START the speedup before the GATE below passes.** The 2.3× busy gap is
> PRODUCER-SIDE ATTRIBUTION, and CLAUDE.md rule 1 + Measurement PROCESS say
> busy-share manufactures phantom levers. The wall is bound by the in-order
> publish chain; a faster engine helps ONLY until the next term binds. Prove the
> engine moves the WALL (not just self-time) with a causal perturbation first.

---

## GATE 0 — validate the instrument (do this FIRST; advisor caveats from parity sign-off)

Two known instrument defects must be fixed before any quantitative knee is trusted:

1. **Classify `worker.decode` in the Fulcrum stage map.** The model currently
   scores gzippy `window-absent frac f = 0.0%` (artifact: `worker.decode` is
   unclassified, Span Atlas `⚠ 4833.0ms busy in spans with no config stage`), so
   it used `d_w_eff = d_c` and reported `worker-bound = 499ms`. With the real
   f≈90.2% and `d_w=122.5ms`, gzippy's worker-bound is materially higher — which
   shifts the "cutting L_resolve stops paying at the 499ms knee" conclusion. Tag
   `worker.decode` (or split it into clean/window-absent) and re-run so the
   worker-bound knee is correct. The wall stays rate-bound either way; this is
   about quoting the knee honestly.
2. **Assert `worker.isal_stream_inflate` (gzippy trace) is the pure-Rust engine,
   not residual ISA-L FFI.** It is a vendor-NAMED span (config region
   `worker.isal_clean`). Its gzippy/rapidgzip busy ratio is **2.17×** — slower
   than the 1.83× wall, which itself proves it is NOT the ISA-L C library (that
   would read ≈1.0×). Add one explicit source/trace assertion (e.g. the span is
   emitted from the pure-Rust clean-tail path, FFI off the decode graph) so a
   future reader is not misled by the ported name. Self-test: binary-vs-itself
   must read 1.0 ± spread (CLAUDE.md rule 4).

---

## GATE 1 — sub-span engine profile (refill vs Huffman vs copy_match), per-T

Extend Fulcrum (DEVELOP the instrument, never a hand-rolled script — CLAUDE.md
rule 8) to decompose the engine's own self-time into the three candidate regions,
per T (1,4,8,16), on the locked host, interleaved, sha-verified:

- **refill** — `Bits::refill` / branchy availability checks in the hot loop.
- **Huffman** — lit/len + (now) `HuffmanCodingReversedBitsCached` distance decode.
- **copy_match** — `emit_backref_ring` / ring `% RING_SIZE` per-write / window copy.

Target spans: `worker.block_body` (window-absent bootstrap),
`worker.isal_stream_inflate` (clean tail), and the canonical/specialized inner
loops in `marker_inflate.rs`. Output: a per-region self-time table gzippy vs
rapidgzip per T, with the double-count-free decomposition Fulcrum already does for
the consumer (busy+idle==span asserted).

This tells us WHICH inner region carries the ~0.54× — without it we'd guess.

---

## GATE 2 — causal perturbation: prove ENGINE speed moves the WALL

Per CLAUDE.md Measurement PROCESS (perturb don't attribute; frequency-neutral
control; removal oracle for the ceiling):

1. **Slow-injection.** `GZIPPY_SLOW_BOOTSTRAP`-style spin of the engine region by
   +50% / +100% of its own measured self-time; measure the interleaved wall
   response. Monotonic + ~proportional ⇒ engine is on the critical path.
   (The bootstrap perturbation already showed +50% spin → +6.0% wall, +100% spin
   → +20.3% wall at an earlier HEAD — re-confirm at this HEAD.)
2. **Frequency-neutral control.** Re-run with a SLEEP (yields the core) of the
   same magnitude; if the wall delta survives, criticality is real, not a turbo
   artifact. (The harness already threads `SLOW_MODE`/`SLOW_KIND` knobs —
   scripts/bench/{run_locked_fulcrum,guest_fulcrum_capture}.sh — for spin-vs-sleep.)
3. **Removal oracle for the CEILING.** Slow-down slope ≠ speed-up ceiling. To
   bound how much a faster engine can buy, REMOVE/replace the engine decode
   (oracle that emits correct bytes with ~0 decode cost) and measure the
   interleaved wall — that sets the knee where the publish-chain / worker-bound
   term re-binds. Use a REPAIRED oracle (the existing `drive_clean_window_oracle`
   bypasses the scheduler+bootstrap+publish chain and isolates nothing — do NOT
   trust it). Without this, do not quote a speed-up target.

PASS to proceed = (1) monotonic AND (2) control-agreeing AND (3) oracle shows a
material wall headroom above the engine-removed knee. If the oracle shows the wall
re-binds on the publish chain before the engine gap closes, the engine speedup is
CAPPED there — record the cap, do not over-invest.

**The CEILING (existence proof).** rapidgzip's per-chunk window-absent decode is
**d_w = 66.7ms** vs gzippy's **122.5ms** (1.84×, `plans/parity-final.md:40`) on
the SAME structure — so a faithful pure-Rust engine CAN reach ~66.7ms d_w; that is
the existence-proof target, not an aspiration. The GATE 2 removal-oracle knee says
how much of that 56ms d_w gap actually reaches the WALL before the publish chain
re-binds. Quote the speed-up target as `min(66.7ms engine ceiling, oracle knee)`.

---

## Candidate speedups (only after GATE 0–2; ranked by expected leverage)

These are the open-territory inner-loop techniques (CLAUDE.md authorizes full
reimplementation here; prior falsifications are non-binding — re-measure fresh).

> **KNOWN HAZARD — the reverted SIMD multi-literal (`ca52389`
> "feat(inflate/resumable): SIMD multi-literal fastloop").** A packed multi-literal
> SIMD write path was attempted in the resumable fastloop and REVERTED as a
> regression. CLAUDE.md marks it non-binding ("measured against the pre-PRELOAD,
> pre-BMI2 hot loop"), so re-attempt is allowed — BUT treat any multi-literal /
> packed-write candidate as the historically-regressing path: gate it on a GATE 1
> attribution that Huffman lit/len decode (not refill, not copy_match) dominates,
> and require a sha-gated locked-bench WIN, never a TIE, before keeping it. Document
> the deviation in the commit message so the falsification record stays honest.

1. **Enable BMI2 PEXT/BZHI runtime dispatch.** BMI2 is currently OFF
   (`unified.rs:121-127`, `build.rs:88`, `resumable.rs:1008`). PEXT for Huffman
   table index extraction + BZHI for masking are the canonical ISA-L/libdeflate
   levers; this is the most likely single source of the ~0.54× lit/len gap. Gate
   on GATE 1 showing Huffman is the dominant sub-region; ship behind runtime
   dispatch so portable binaries are unaffected.
2. **Wider / branch-lean refill.** If GATE 1 attributes a large share to refill,
   widen the bit buffer / hoist the availability checks out of the per-symbol path
   (FASTLOOP_OUTPUT_MARGIN headroom path that elides the resumable yield-check).
3. **copy_match / ring write.** If GATE 1 attributes to copy_match, eliminate the
   per-write `% RING_SIZE` on the address dependency chain (the `phys` u16
   pointer-bump experiment already present at `marker_inflate.rs:1537` is the seed)
   and SIMD the window copy.

Each candidate: correctness ABSOLUTE (byte-exact suites routing/correctness/
pure_rust_inflate_corpus, Rosetta-x86 silesia sha
`028bd002c89c9a909ccdbc2af0a223de285348edb014ccc8e27d297f52cb410f`), measured on
`scripts/measure.sh` / locked Fulcrum interleaved vs rapidgzip — never an internal
slice. A correct change that TIEs is KEPT and layered (CLAUDE.md rule 7); a
rejection needs a mechanism (rule 7), not a narrow miss.

---

## Done-criterion for the engine arc (future)

gzippy T8 silesia wall converges toward rapidgzip's (TIE-or-better, interleaved +
sha-verified) WITHOUT breaking the faithful structure recorded in
`plans/parity-final.md` (window-absent fraction stays ~90%, no two-phase resolve,
no re-introduction of FFI into the decode graph, no clean-decoder drift toward the
31% static fraction). Bounded by the GATE 2 removal-oracle knee.
