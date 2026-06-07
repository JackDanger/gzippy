# FULL igzip ASM-KERNEL PORT — FEASIBILITY REPORT (analysis/design only, no build)

Charter: plans/asm-kernel-feasibility-scope.md. Triggered by the PLATEAU verdict
(plans/engine-bench-round2-advisor-verdict.md, memory project_engine_plateau_pure_rust):
grafting E2–E4 onto the faithful pure-Rust **u16-ring** marker engine plateaus at +13%
(0.41× ISA-L; the tie needs 0.85×, 12σ short). The user chose to SCOPE a full igzip AVX2
decode-kernel port — transliterate igzip's asm as OUR OWN inline-Rust-asm (NOT C-FFI),
feasibility-FIRST. THIS IS ANALYSIS ONLY. No build started.

All vendor file:line below were source-verified first-hand this session (a synchronous
read-only subagent mapped the kernel; the leader spot-verified the load-bearing claims and
the stub structure). Vendor root for the kernel:
`/home/user/www/gzippy-reimplement-isal/vendor/isa-l/igzip/`. `asm` =
`igzip_decode_block_stateless.asm` (22 KB; `_01.asm`=SSE / `_04.asm`=Haswell each %include it
with `ARCH` + `USE_HSWNI` — verified, the two stubs are 61/79 bytes). `C` = `igzip_inflate.c`.
`struct` = `inflate_data_structs.asm`.

---

## Q1 — WHAT MAKES igzip FAST (beyond the grafts we tried)

The +13% graft plateau is real because the grafts (E2 AVX2 copy / E3 packed store / E4 wide
refill) were bolted onto a per-symbol LUT decode that writes into a **u16 ring**. igzip's
speed is NOT one trick — it is **four tightly-coupled tricks that only work together on a
flat u8 buffer**. The grafts reproduced two of them weakly and could reproduce none of them
at full strength because the u16 ring forbids the precondition. The four:

1. **Packed flat short-code table (one load retires up to 3 symbols).**
   lit/len short table = a flat `u32[1<<12]` indexed by the low 12 bits (`ISAL_DECODE_LONG_BITS`),
   struct:82-83, C:46-57. Each `u32` packs **up to three 8-bit literals** in bits[0:25]
   (`LARGE_SHORT_SYM_MASK`), a long-code flag at bit 25, a 2-bit symbol-count at [26:28], and
   the **total code length to consume** at [28:32] (asm:57-68; build at C:448-450, 484-487,
   594-596). The common short-code path is **a single indexed load + a few shifts/masks, no
   dependent second load** (asm:322-372: `decode_next_lit_len` pulls len from bits≥28,
   count from 26-27, tests the flag, and only on a long code does the second `movzx` into the
   long table fire, asm:359). dist uses the same scheme with a `u16[1<<10]` short table
   (struct:95-96). This is cheaper than a classic 2-level table: no root→subtable pointer
   chase for the >99% short case, AND one load can retire multiple output bytes.

2. **The speculative, software-pipelined bulk loop `loop_block` (asm:507-627) — the heart.**
   Per iteration it:
   - speculatively **stores up to 8 bytes** of packed literals UNCONDITIONALLY then advances
     `next_out` by the *actual* count (`mov [next_out], next_sym` / `add next_out, next_sym_num`,
     asm:518-519) — wrong-guess bytes are simply overwritten next iter. Branchless multi-literal output.
   - **preloads the next lit/len symbol AND the next dist symbol before it knows the current
     symbol's type** (asm:524-525, 540, 550-552) — hides table-load latency.
   - **refills the 64-bit bit register every iteration, branchlessly**, via `SHLX`/`or`
     (asm:528-530, 543-547), exploiting the 8-byte `IN_BUFFER_SLOP` so it can always read 8 bytes.
   - detects EOB with `cmp next_sym2,256` (asm:536-537,555).

3. **SIMD overlap-doubling back-ref copy (asm:558-627).** Non-overlap → `MOVDQU` in 16-byte
   strides (`large_byte_copy`, asm:603-612); overlap (dist<16) → write 16, **double
   `look_back_dist`** each step so the pattern self-replicates (`small_byte_copy`, asm:614-627).
   NOTE the surprise: it is **SSE-width `MOVDQU` (xmm), NOT AVX2 ymm**. The "AVX2"/Haswell
   build (`_04` / `USE_HSWNI`) differs mainly in the **BMI2** path (`SHLX/SHRX/BZHI` vs a
   `neg/shl/shr` fallback, `CLEAR_HIGH_BITS` asm:299-318), not vector width. The C back-ref
   copy is actually a scalar `memcpy`/`byte_copy` (C:126-132, 1697-1701) — the SIMD lives only
   in the asm.

4. **The slop-margin headroom guard (asm:48, 488-489, 509-512).** The fast loop runs ONLY
   while `next_in ≤ end_in-8` AND `next_out ≤ end_out-(16+258)`. Inside that region there are
   **NO per-symbol bounds checks** — it freely over-reads/over-writes within the slop. A
   "careful" symbol-by-symbol tail (asm:637-703, `rep movsb` copy) handles the boundary. So it
   is **not per-symbol resumable**; it bails to the careful tail before any boundary and uses
   `write_overflow_*`/`copy_overflow_*` spill slots for a symbol that straddles `avail_out`
   exhaustion (asm:711-712, C:2411-2427).

**The crux (Q1's load-bearing finding):** all four depend on the **flat u8 output buffer with
the 32 KiB window as the tail of that same buffer** (asm:518/591/605/618, C:1641/1698; window
lower bound = `start_out`, asm:589). The speculative 8-byte literal store, the `MOVDQU` byte
copy from `next_out-dist`, and the unchecked-over-write headroom ALL assume bytes are u8 and
already-final. This is precisely what our grafts could not have, because they wrote u16 into a
marker ring — which is exactly why they plateaued. The igzip kernel is not "the grafts but
more SIMD"; it is a *different output model* that makes the grafts cheap.

---

## Q2 — CAN IT BE INTEGRATED FAITHFULLY, AT WHAT COST? (the fork)

### Q2a — Is igzip-class achievable WITHIN the one-engine u16-flip arch? — **NO.**

Source-grounded mechanism, not opinion: every one of igzip's four fast tricks (Q1) is
defined over a flat u8 buffer where the copy source is always already-final bytes. The gzippy
faithful engine's `output_ring: Box<[u16; RING_SIZE]>` (marker_inflate.rs:290, a byte-for-byte
port of vendor `m_window16`, deflate.hpp:805) stores u16 and narrows u16→u8 at drain. On that
ring:
- the speculative 8-byte multi-literal store cannot be a single u8 store (each literal is a
  u16 slot) → forfeits trick #2's store;
- the `MOVDQU` byte copy cannot read a flat u8 source → forfeits trick #3's copy;
- the unchecked over-write headroom collides with ring-wrap arithmetic (the documented
  05a3835 hazard: a rounded word-copy corrupts repetitive data via circular-ring overshoot —
  the u16 path tolerates it only with a per-byte tail) → forfeits trick #4's licence.

This is the SAME wall the round-2 bench measured: E2 was "AVX2 copy on the STILL-u16 ring,"
and it (plus E3/E4) recovered only ~9% because the dominant cost — per-symbol LUT decode +
2-bytes-per-byte u16 traffic — is exactly what these techniques don't remove (engine bench
verdict, attack-2). **An igzip-class kernel REQUIRES a flat-u8 clean output path.** It cannot
live on the u16 ring. That is a code-level structural fact, not a perf preference.

### Q2b — THE FAITHFULNESS FORK (user-level — surfaced, NOT pre-decided)

There are two mutually exclusive ways to host a flat-u8 igzip-class kernel, and they sit on
OPPOSITE sides of the 2026-06-05 governing memory (project_faithful_unified_decoder_over_perf).

**FORK ARM A — "our-asm on a u8 CLEAN TAIL" (most faithful to rapidgzip's REAL structure).**
rapidgzip itself reaches igzip-class by a **two-phase handoff**: `deflate::Block` decodes the
≤32 KiB markered prefix into `m_window16`, then at the flip hands the CLEAN TAIL to a u8
ISA-L path (the `#ifdef LIBRAPIDARCHIVE_WITH_ISAL` clean-tail delegation,
`finishDecodeChunkWithInexactOffset` / the `cleanDataCount >= MAX_WINDOW_SIZE` cutover —
confirmed COMPILED-OUT in the no-ISA-L build the governing memory analyzed, but it IS
rapidgzip's real fast structure when ISA-L is present). So:
- "our igzip-asm-port on a u8 clean tail" is the **MOST FAITHFUL port of rapidgzip's actual
  igzip-class structure** — it mirrors what rapidgzip does to be fast (markered prefix on the
  u16 Block, bulk clean tail on a flat-u8 igzip kernel), and replaces ONLY the FFI'd ISA-L
  call with OUR inline-Rust-asm transliteration (so it still satisfies "no C-FFI in native").
- BUT it is a **TWO-PHASE HANDOFF** — exactly what the 2026-06-05 governing memory named
  "Divergence #2": "Reverting the clean tail to a 2nd engine for speed IS the 600-commit
  Divergence #2 — do not." The memory's faithful target is the *no-ISA-L* build = ONE
  `deflate::Block` decoding the whole chunk on `m_window16`. Arm A picks the *with-ISA-L*
  rapidgzip as the blueprint instead — a defensible reading ("port the FAST rapidgzip"), but
  it directly contradicts the memory's chosen blueprint and its explicit prohibition.

**FORK ARM B — "u8-direct one-engine ring" (keeps the one-engine invariant, our asm in it).**
Keep the ONE MarkerRing engine; change its backing to vendor's actual memory model — one
128 KB store viewed as `[u16; 65536]` pre-flip / `[u8; 131072]` post-flip, with
`setInitialWindow`'s full-65536 rotate+value-downcast at the flip (the FAITHFUL PLAN already
written in project_faithful_unified_decoder_over_perf §"REMAINING MEMORY-MODEL DEVIATION";
vendor deflate.hpp:1245,1742-1785,806). Then the post-flip clean reads write u8 DIRECTLY into
the u8 view, and OUR igzip-asm kernel runs on that flat-u8 post-flip region.
- This is faithful to the governing memory's ONE-engine, m_window16, flip-in-place mandate
  (it is literally the memory's own next step, just with the kernel made igzip-class instead
  of the current per-symbol loop).
- BUT: it is the harder integration. The flat-u8 region post-flip is a **circular ring**, not
  a linear buffer — igzip's tricks #2/#3/#4 assume a LINEAR flat buffer with over-write
  headroom past `end_out`. On a ring you must (a) special-case the wrap for the speculative
  8-byte store and the MOVDQU copy (per-byte tail at the wrap — the 05a3835 hazard again, now
  on u8 where it's worse), and (b) the "window is the tail of the same buffer" trick (#4) maps
  to "window is the ring modulo RING_SIZE" — a `% RING_SIZE` on every copy-source address,
  which is exactly the per-symbol cost igzip avoids. The E1 round-1 bench measured u8-direct
  OUTPUT at only +6% precisely because the *ring* (not the output width) is the tax. So Arm B
  may NOT reach igzip-class even with the kernel — the ring arithmetic is structurally at odds
  with the linear-buffer assumption the kernel's speed rests on.

**ADVISOR CORRECTION (folded in — verified first-hand, marker_inflate.rs:1746-1754):** my Arm B
"ring tax = `% RING_SIZE` per copy" was OVERSTATED. The live loop ALREADY amortizes the modulo
to a physical-pointer bump (`phys: u16` advanced as a pure pointer bump, no per-literal
`% RING_SIZE`; back-refs resync via `pos & (RING_SIZE-1)` after emit). So the real ring cost is
**wrap-handling per 64 KiB boundary, not per-copy modulo.** Consequently my Arm B 150–200 MB/s
band is an **UNMEASURED GUESS, not a grounded projection** — Arm B is the ONLY arm that keeps
BOTH no-FFI AND the one-engine memory, and it deserves an isolation bench rather than dismissal
by projection. The Arm B "likely narrow miss" below should be read as "unmeasured, plausibly
better than my estimate; bench it before rejecting."

**The honest framing of the fork:** the two user constraints — (1) 1.0× tie, (2) no C-FFI in
native — are in tension with a THIRD constraint, the 2026-06-05 one-engine/no-two-phase
memory. Pick at most a faithful TWO of the three:
- **A** satisfies (1)+(2)+faithful-to-FAST-rapidgzip, but breaks the one-engine memory (two-phase).
- **B** satisfies (2)+the one-engine memory, but likely MISSES (1) (ring tax defeats the kernel).
- The previously-rejected option (accept FFI) satisfies (1)+the structure but breaks (2).

### Q2c — INTEGRATION COST (both arms, beyond the kernel itself)

The kernel asm is small (Q3) — integration is the cost. Common to both arms:
- **Table build port:** `make_inflate_huff_code_lit_len/dist` (C:387-599, ~300 LOC pure
  scalar C) transliterates directly to safe Rust — but it produces a DIFFERENT table format
  (packed-u32 short + long split) than gzippy's current LUTs. New tables, new build, new tests.
- **Resumability bridge:** gzippy's chunk pipeline needs the resumable contract (callers in
  parallel/ resume mid-chunk). igzip is NOT per-symbol resumable (Q1 #4) — it bails to a
  careful tail and uses overflow spill slots. The integration must wrap the kernel's
  fast-loop + careful-tail + spill-slot replay into gzippy's resumable boundary, OR confine
  the kernel strictly to the post-flip clean tail where a chunk-at-a-time call suffices.
- **Byte-exactness gate:** all 850+ lib tests + silesia differential + the adversarial
  flip-seam test (A‖A + distance-32768 across the repositioned seam — the documented top trap).
- **Arm A extra:** the markered-prefix→clean-tail HANDOFF seam (window materialization for the
  kernel's first 32 KiB of history). This is a NEW seam — the thing the governing memory spent
  600 commits removing.
- **Arm B extra:** the ring backing rewrite (128 KB dual-view store, setInitialWindow rotate+
  downcast) + wrap special-casing inside the ported kernel. Larger blast radius; touches the
  one production engine directly.

Net: **the asm is the easy ~20%. The integration (tables, resumability, the handoff or the
ring rewrite, byte-exact seam tests) is the hard ~80%, and it is the part the round-2 advisor
already flagged as the real risk.**

---

## Q3 — WILL IT PROJECT TO THE TIE?

### Ported-kernel rate estimate
The kernel is OUR transliteration of IGZIP'S OWN CODE, so the upper bound is igzip's own
single-thread clean rate measured ON THE GUEST: the round-2 bench put pure ISA-L at **283 MB/s**
(round-1: 388 MB/s; the guest median-chunk anchor is 283). A faithful transliteration that
reproduces all four tricks would approach that — call it **~250–283 MB/s** if the flat-u8
linear model is achieved (Arm A), DISCOUNTED for our-asm-vs-vendor-codegen and the
resumability wrapper. On Arm B's ring, subtract the per-copy `% RING_SIZE` and wrap tail —
plausibly **~150–200 MB/s** (between today's 118 and the linear ceiling), i.e. likely SHORT of
igzip-class.

### Projection via §3 (tier1-design-v2 §3, same-sink 0.604s bar)
The §3 model: decode_wall = 39 chunks × per-chunk-ms / 8 × 1.36 ramp; the tie bar is
rapidgzip's MEASURED same-sink wall **0.604s**, and the PASS criterion is STRUCTURAL —
(ii)/(iii) ≥ 0.85 so the decode-bound wall drops below the shared pipeline floor and the wall
**RE-BINDS off decode** (not the loose 0.542 ≤ 0.604 numeric coincidence, which the round-2
advisor showed is a 13%-engine-bump artifact while decode still binds).

- **Arm A at ~250–283 MB/s** (igzip-class, if the linear flat-u8 model is truly reached):
  per-chunk ≈ 92.7ms × 104/283 ≈ **34 ms** → decode_wall ≈ 39 × 34 / 8 × 1.36 ≈ **0.225s**.
  Decode STOPS binding; the wall re-binds on the shared floor. (ii)/(iii) ≈ 1.0 ≥ 0.85 → **PASS
  on the decode term.** BUT the wall then sits at gzippy's NON-decode floor — and per
  project_pregate STEP-0, the consumer SERIAL floor is only ~0.015s, BUT the SAME-SINK output
  write adds ~0.245s and the consumer cold-get lag (placement, ~318ms, NOT closed —
  getIndexedChunk/offset-supply both REFUTED) is still present. So Arm A's realistic wall is
  **max(decode 0.225, placement-laden consumer floor) ≈ 0.54–0.66s = TIE-to-+10%**, the SAME
  conditional the §3 verdict reached — **and ONLY if the placement front is ALSO closed**
  (still open, prefetch-horizon question unresolved). Engine alone, even at igzip-class, does
  NOT tie without placement (the co-primary finding, project_pregate).

- **Arm B at ~150–200 MB/s** (ring-taxed): per-chunk ≈ 92.7 × 104/175 ≈ **55 ms** →
  decode_wall ≈ 39 × 55 / 8 × 1.36 ≈ **0.365s**. (ii)/(iii) ≈ 175/283 ≈ 0.62 — STILL below the
  0.85 PASS line; decode still binds. This is the §3 NARROW-MISS band (50–60 ms/chunk → total
  0.54–0.60s, possibly TIE within spread, possibly +5–10%). **Arm B is a likely NARROW MISS,
  not a clean PASS** — the ring tax keeps the kernel out of igzip-class.

**ADVISOR SHARPENING (Attack 1, UPHELD-WITH-CORRECTION — strengthens the conclusion):** the
Arm A tie is ROBUST to a large engine-rate miss. At 283/220/200 MB/s the decode_wall is
0.225/0.290/0.318s — all far below the ~0.54s floor; decode stops binding at ANY rate above
~120 MB/s. So once the flat-u8 linear model is achieved, the tie depends on the SHARED ~0.54s
FLOOR (which gzippy matches only with PLACEMENT closed), NOT on hitting igzip's exact 283.
The circular 283-anchor (it is ISA-L's own rate) therefore matters LESS than it appears — the
binding term is the floor. Correction: the "0.54–0.66 consumer floor" is decode/placement-
COUPLED (STEP-0: serial 0.015s + writev 0.245s), not an independent hard floor; the binding
term is rapidgzip's shared ~0.54s floor that gzippy reaches only via the placement co-primary.

### Tie verdict
- **The kernel CAN project to the tie ONLY on Arm A (flat-u8 linear, two-phase handoff) AND
  ONLY if placement is also closed.** Engine-at-igzip-class is necessary-but-insufficient
  (co-primary with placement, project_pregate). The asm-kernel port addresses the ENGINE
  co-primary; it does NOTHING for the placement co-primary (still open, two refuted attempts).
- **Arm B (the faithful one-engine ring) likely does NOT reach igzip-class** — the ring
  arithmetic is structurally at odds with the linear-buffer assumptions that give igzip its
  speed; projected ~0.62× ISA-L, a NARROW MISS, better than today's 0.41× but short of the
  0.85 PASS.
- The §3 non-decode-floor caveat still binds: even Arm A re-binds on a ≥0.54s consumer floor
  that the engine cannot shrink, so the tie is the OPTIMISTIC edge, not the expected outcome.

---

## RECOMMENDATION (leader — for supervisor + USER ratification; do NOT pre-decide the fork)

The feasibility answer is **CONDITIONAL-FEASIBLE, with a hard user-level fork that I cannot
resolve without the user's intent on the three-way constraint tension.**

1. **The asm itself is NOT the risk** — it is igzip's own code, ~80–100 hot instructions, ~6
   macros, no stack frame, BMI2 + SSE `MOVDQU` (not even AVX2 ymm). A Rust port is ~500–700
   LOC (inline `asm!`) or ~500–600 LOC (idiomatic `unsafe` + `std::arch` intrinsics — MORE
   maintainable, compiler can match the speculative scheduling if the packed-table format and
   single-store-multi-literal pattern are preserved). **REVISED per advisor (Attack 3
   OVERTURNED my intrinsics recommendation):** igzip ships BOTH a C decoder (C:1641, one
   literal at a time) AND hand-written asm (asm:518, speculative packed 8-byte store) on the
   SAME packed table — that is direct evidence the C compiler does NOT auto-emit igzip-class
   codegen from the C, or vendor wouldn't maintain the asm. So "prefer intrinsics for
   maintainability" is DEMOTED to a **gated codegen spike** (prototype the intrinsics loop,
   measure it against the rate; only if it hits the class do we avoid `asm!`). The only PROVEN
   path to the rate is `asm!`; treat the intrinsics route as a hopeful-but-unproven option to
   be measured, not the default.

2. **The fork is the real decision, and it is the USER's:**
   - If the user's TOP priority is the **1.0× tie**, the only projecting path is **FORK ARM A**
     (u8 clean-tail, our-asm) — which means **formally accepting the two-phase handoff as
     faithful-to-FAST-rapidgzip and OVERRIDING the 2026-06-05 one-engine memory** (re-read: it
     forbade reverting to a 2nd engine *for speed via microbench*; Arm A is reverting for a
     *source-cited structural reason* — rapidgzip's real fast path IS two-phase with ISA-L.
     That is a legitimate re-reading, but it is the user's call, not mine).
   - If the user's TOP priority is the **one-engine faithful invariant**, the path is **FORK
     ARM B** — which I project to a **NARROW MISS (~0.62×), not a tie**. The user must then
     accept "faithful but ~1.6× ISA-L" OR revisit the bar.
   - The third option (accept FFI in native's clean tail) remains on the table and is the
     LOWEST-effort tie, at the cost of the no-C-FFI constraint.

3. **Even on Arm A, do NOT authorize the engine build alone.** Engine is co-primary with
   PLACEMENT (project_pregate). Placement is still OPEN (getIndexedChunk + offset-supply both
   REFUTED; the prefetch-horizon / decode_NOT_STARTED question is the only un-run diagnostic).
   An igzip-class engine on an un-fixed placement front still lands at ~0.54–0.66s only if the
   consumer floor cooperates. **The asm-kernel port should be authorized, if at all, AS PART
   OF a plan that ALSO resolves placement — not as a standalone engine sprint.**

**My single-sentence recommendation:** scope-wise the port is feasible and the asm is the easy
part; the decision the user must make is the three-way fork (tie vs no-FFI vs one-engine —
faithful pick-two), and my technical read is that **only Arm A (two-phase u8 clean tail with
our inline-asm) projects to a tie, and only if placement is also closed** — so if the user
wants the tie without FFI, the honest path is to ratify Arm A (overriding the one-engine
memory as a re-reading of "faithful = the FAST rapidgzip") AND fund the placement co-primary;
otherwise Arm B is faithful-but-narrow-miss and the bar should be revisited.

---

## DISCIPLINE / PROVENANCE
- igzip kernel map: synchronous read-only subagent, all file:line first-hand; leader
  spot-verified the stub structure (_01/_04 %include, 61/79 bytes) and the load-bearing
  flat-u8 dependency.
- Rate/floor numbers: from the round-2 bench (project_engine_plateau_pure_rust: ISA-L 283,
  scalar 104, E234 118) + §3 model (tier1-design-v2) + STEP-0 floors (project_pregate).
- NO build run. NO production code touched. This is the checkpoint deliverable.
- Independent disproof advisor verdict: plans/asm-kernel-feasibility-advisor-verdict.md.
