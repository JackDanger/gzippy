# CLEAN-KERNEL-DESIGN — the converged clean-decode kernel (per-arch: x86 BMI2 / aarch64 NEON)

Status: **DESIGN** (no kernel implementation in this turn — "fully design, then fully implement").
Scope: **macOS-aarch64, NOT-YET-LAW cross-arch.** Every Deliverable-1 number is deterministic-
instruction (Gate-0 self-validated, byte-exact, /dev/null both arms). Cross-ISA LAW requires the
Intel **asm-OFF** pairing (designed in §5). No floor/settled/accept claim is made anywhere here.

Subject binary: gzippy-native (FFI off), `cargo build --release --no-default-features --features
gzippy-native`, path=ParallelSM, build-flavor=parallel-sm+pure (confirmed via GZIPPY_DEBUG=1).
Blueprint: **libdeflate** (`vendor/libdeflate/`), the aarch64 SOTA we are behind; igzip
(`vendor/isa-l/igzip/`) for the two-level table; the existing x86 asm kernel
(`src/decompress/parallel/asm_kernel.rs:477 run_contig`) for the x86 specialization.

---

## 0. DELIVERABLE 1 — the finer clean-core decomposition (what AIMS this design)

`localize_mac.sh` already split the program: **pipeline scaffold ~0% · table-build ~5.4% · CLEAN
DECODE CORE 94.7% (11.09–11.69 instr/B excess on silesia).** This turn decomposes that clean core
into CRC / backref-copy / Huffman+bitreader using deterministic-instr perturbations, each PROVEN
non-inert + byte-exact. Harness: `scripts/bench/standing/clean_core_decomp_mac.py` (+
`gen_decomp_corpora.py`). N=9, min-of-N, /dev/null, -p1, all arms sha==gzip -d.

| corpus | gz instr/B | ld instr/B | ratio | excess/B | gz cyc/B | ld cyc/B | FOLD_NOCRC Δinstr% |
|--------|-----------:|-----------:|------:|---------:|---------:|---------:|-------------------:|
| decomp_literal (per-SYMBOL extreme) | 35.970 | 16.470 | **2.184** | **+19.500** | 9.496 | 6.536 | +0.001 |
| decomp_backref (per-COPY-byte extreme) | 1.972 | 1.712 | **1.152** | **+0.260** | 0.970 | 0.645 | −0.007 |
| silesia (real mixed anchor) | 19.382 | 7.690 | 2.521 | +11.693 | 5.776 | 3.385 | −0.064 |

`decomp_literal` = alphabet-16 random → ~1 Huffman symbol per output byte, ~0 backref copy.
`decomp_backref` = 64-byte motif × N → ~1 symbol per 258 bytes, ~all bytes from the word-copy.

### Component verdict (STRONG-attribution HYPOTHESIS — deterministic-instr; corpus-contrast = WEAK-tier confounds)

1. **CRC ≈ 0 of the gap — already HW-converged, NOT a lever.** `GZIPPY_FOLD_NOCRC` is **INERT** on
   the native -p1 path (Δinstr = +0.001% / −0.007% / −0.064%, all at/below the 0.04% determinism
   floor; not consistently negative ⇒ the per-byte fold-sink CRC branch is not on this path). Root
   cause confirmed by source: `crc32fast` 1.5.0 uses the aarch64 **HW `__crc32d/__crc32b`** path
   (`crc32fast-1.5.0/src/specialized/aarch64.rs:52-65`, gated `cfg!(target_feature="crc")`, present
   in the Apple-Silicon baseline). **An inert knob's "share" does not exist (Gate-0c) — CRC is off
   the table.** (Design note: keep the HW-CRC compile feature on every ship target; on generic
   aarch64 this needs `+crc`/`target-cpu=native`.)

2. **Backref COPY is near-parity — a MINOR prize.** On the copy-dominated extreme gzippy is 1.972
   vs libdeflate 1.712 instr/B = **1.152×, only +0.260 instr/B excess**. The wrap-ring word-copy is
   already close to libdeflate's NEON copy. On silesia (a mix where a large fraction of bytes come
   from backrefs) copy can contribute **at most ≈0.15–0.2 instr/B** of the 11.69. ⇒ **This
   REVISES the AMD-FREE-PATH "NEON copy = biggest single-arch win" HYPOTHESIS down to a minor
   item.** Worth a NEON pass for the cyc-side (copy cyc ratio is 1.5×, larger than its instr
   ratio), but it is NOT where the instruction gap lives.

3. **The per-SYMBOL Huffman-decode + bitreader path IS THE GAP.** On the per-symbol extreme gzippy
   is 35.970 vs libdeflate 16.470 = **2.184×, +19.50 instr/B excess** — and silesia's entire 11.69
   clean-core excess tracks this, not the copy. **THE BIGGEST COMPONENT = the per-symbol Huffman
   decode loop (table lookup + packed-symbol unpack + bit refill/consume + distance decode).** This
   is the design centerpiece.

CRC and copy cannot be split finer with an INERT knob (don't manufacture a phantom). The per-symbol
excess could in principle be split table-lookup vs bitreader with a dedicated inner-loop oracle, but
that is a kernel modification (out of scope this turn) and the design below converges ALL of
{table, unpack, refill cadence, distance} together — the correct aim regardless of their internal split.

---

## 1. TARGET STRUCTURE — one kernel, per-arch specialized, mirroring libdeflate

### 1.1 What gzippy runs today (the thing to replace)

- **aarch64 (and any non-asm build):** the clean path is `MarkerDecoder::decode_clean_fast_loop`
  (`src/decompress/parallel/marker_inflate.rs:2434-2614`), feeding a wrapping **u8 ring**
  (`U8_RING_SIZE`), with a careful tail (`decode_careful_tail`, :2641) for the resumable boundary.
- **x86_64 (asm build):** `asm_kernel.rs:477 run_contig` (hand BMI2 asm) with the pure-Rust
  reference `run_contig_ref_biased::<0>` (:2167) as its differential oracle. aarch64 has **no**
  asm counterpart — it takes `decode_clean_fast_loop`. So the clean core is *shared pure-Rust in
  shape*; converging it advances both arches.

### 1.2 libdeflate's clean fastloop (the blueprint — `vendor/libdeflate/lib/decompress_template.h`)

- **Flat masked litlen table**, `LITLEN_TABLEBITS=11` (`deflate_decompress.c:372`): one masked load
  `entry = litlen_decode_table[bitbuf & litlen_tablemask]` per symbol (template :347/391/430).
- **Single fused consume**: `saved_bitbuf=bitbuf; bitbuf>>=(u8)entry; bitsleft-=entry;` (:358-360) —
  the whole entry subtracted from bitsleft (only low 8 bits matter under the refill watermark).
- **3-literal lookahead** off ONE refill (:366-435): up to 3 literals + 1 match decoded between
  refills, each preloading the NEXT entry before the consume so the dependent table load overlaps.
- **`saved_bitbuf` extract** of length/offset extra bits (:496, :547) — no re-read of the bitstream.
- **Flat masked offset table**, `OFFSET_TABLEBITS=8` (:374): one load `offset_decode_table[bitbuf &
  BITMASK(OFFSET_TABLEBITS)]` (:505), subtable only on the exceptional flag.
- **Contiguous output** `*out_next++` / wide word copy (:547-560), bounds checked ONLY in the loop
  condition (`out_fastloop_end`, FASTLOOP_MAX_BYTES_WRITTEN headroom — template :52-53, macros
  :280-295), never per symbol.
- **One refill per iteration** amortized over ≥3 symbols (`REFILL_BITS_IN_FASTLOOP`,
  `deflate_decompress.c:260`), bit budget statically proven by `CAN_CONSUME_AND_THEN_PRELOAD`.

### 1.3 The converged target

ONE clean kernel `decode_clean_contig` taking a **contiguous output region** + a `Bits` cursor,
structurally a transliteration of `decompress_template.h:344-560`:
- shared pure-Rust skeleton (flat-table decode, fused consume, 3-literal lookahead, saved_bitbuf
  extract, single amortized refill, contiguous output with margin-bounded loop condition);
- arch-specialized only at the **copy** (NEON `vld1q/vst1q` on aarch64; the existing MOVDQU path on
  x86, `asm_kernel.rs` copy arm) and, on x86, the existing BMI2 asm body kept as the x86 spec of
  the SAME contract;
- the resumable/marker surplus moved OUT of the per-symbol path to the region boundary (§4).

This is the "stateless clean-T1 kernel" (AMD-FREE-PATH Rank-2) given a concrete shape, AIMED by
Deliverable 1 at the per-symbol path (the actual gap), not at the copy.

---

## 2. PER-COMPONENT CONVERGENCE (each cites the blueprint file:line)

### (a) CRC → keep HW, do nothing in the kernel
Already HW (`__crc32d`, crc32fast aarch64.rs:52-65) and instr-inert (Deliverable 1). **No kernel
change.** Only a build-hygiene assertion: ship targets compile with `crc` (Apple baseline ✓; generic
aarch64 needs `+crc`). The bulk-update site (`chunk_decode.rs:1719`) stays; do NOT add a per-byte
CRC into the new clean loop (libdeflate also CRCs the whole output once, not per symbol).

### (b) backref COPY → NEON wide overlapping copy (minor instr, real cyc)
Mirror libdeflate's match copy (`decompress_template.h:547-560`: `src=out_next-offset; dst=out_next;`
then a `do { copy_word } while` over a length-rounded span). Replace the wrap-ring
`emit_backref_ring_u8` (`marker_inflate.rs:4637`) — whose `% U8_RING_SIZE` masking is part of the
copy cost — with a **contiguous** copy that, on aarch64, uses `core::arch::aarch64::{vld1q_u8,
vst1q_u8}` 16-byte overlapping stores (the libdeflate-shaped analog of the x86 MOVDQU copy already in
`run_contig`, `asm_kernel.rs` copy arm). Overlap (offset<16) handled exactly as libdeflate: the
length-rounded `FASTLOOP_MAX_BYTES_WRITTEN` slop lets the copy overshoot without a per-iteration
bound. Expected instr win small (+0.26 ceiling); cyc win the larger half (copy cyc ratio 1.5×).

### (c) Huffman inner loop → flat-table + 3-literal lookahead + single amortized refill (THE big one)
Today (`decode_clean_fast_loop` + `LutLitLenCode::decode`, `lut_huffman.rs:1068-1140`):
- a **two-level ISA-L table** (short_code_lookup[12] + long_code_lookup) returning a PACKED multi-
  symbol `DecodedSymbol{symbol, sym_count, bit_count}`, then a branch-heavy **inner unpack loop**
  (`while remaining>0 { s>>=8; if code<=255||remaining>1 … }`, marker_inflate.rs:2500-2516);
- **two refills per iteration** (top preload `bits.refill()` :2464 + bottom `bits.refill()` :2602,
  plus `decode()`'s own `available()<32` refill, lut_huffman.rs:1072) → poor refill amortization;
- per-symbol **ring wrap math** `dst_phys = pos % U8_RING_SIZE` + dual `out_ok && in_ok` bounds
  (:2468-2471) + `record_backreference_for_sparsity` (:2595) — resumable/marker cost in the hot path.

Converge to libdeflate's shape:
1. **Flat masked litlen table** entry layout matching `decompress_template.h` (low 8 bits = consume
   length; LITERAL flag; literal value in `entry>>16`; length base + extra-count in `entry>>8`;
   EXCEPTIONAL flag for subtable/EOB). Build via a port of `build_decode_table`
   (`deflate_decompress.c:722`, called for litlen at :1037, LITLEN_TABLEBITS=11). Drop the packed
   multi-symbol unpack loop — replace with libdeflate's **explicit 3-literal lookahead** (:366-435),
   which gets the same "multiple literals per refill" benefit without the per-symbol `s>>=8` branch
   chain.
2. **Fused consume** `saved_bitbuf=bitbuf; bitbuf>>=entry; bitsleft-=entry` (gzippy already has
   `consume_entry`, consume_first_decode.rs:304 — extend to the libdeflate full-entry-subtract form)
   and **preload the next entry before consuming** so the table load overlaps (:391/430).
3. **One refill per iteration** via `REFILL_BITS_IN_FASTLOOP` semantics (`Bits::refill` already
   branchless libdeflate-style, consume_first_decode.rs:245), with a static bit budget so 3 literals
   + length + offset preload fit (the `CAN_CONSUME_AND_THEN_PRELOAD` invariant, :159). Remove the
   redundant in-`decode()` refill on the fast path (use a `decode_prefilled`-style backstop-free
   entry, lut_huffman.rs:1093).

### (d) bit reader → keep the branchless 8-byte refill, fix the CADENCE
`Bits` (consume_first_decode.rs:197-314) is already a faithful libdeflate misaligned single-load
refill. No primitive change needed; the win is **calling it once per iteration not 2–3×** (§2c)
and keeping no live register state across the refill (§4, N32). `peek`/`consume`/`available` already
match libdeflate's `(u8)bitsleft` pattern (:311).

### (e) distance/offset → flat masked offset table (replace the cached reversed-bits decoder)
Today the clean loop decodes distance via `self.dist_hc` (`DistanceShortBitsCached`,
marker_inflate.rs:2540) + `DISTANCE_EXTRA`/`DISTANCE_BASE` array lookups + a separate refill-check
(:2557) — a dependent multi-step chain the source itself flags as costly (marker_inflate.rs:624).
Converge to libdeflate's **flat masked offset table** (`offset_decode_table[bitbuf &
BITMASK(OFFSET_TABLEBITS)]`, OFFSET_TABLEBITS=8, template :505; built by `build_decode_table`,
`deflate_decompress.c` offset call), with the base+extra packed in the entry and extracted from
`saved_bitbuf` (:547). One load, no separate EXTRA/BASE arrays, no separate refill-check.

---

## 3. ASM vs INTRINSICS — decision

**aarch64: `core::arch::aarch64` NEON INTRINSICS, not hand asm.** Justification:
- The hot path is a **latency-bound dependent scalar chain** (refill → masked table load → consume →
  preload). Hand asm cannot beat a well-shaped intrinsics loop here — there is no SIMD parallelism in
  the symbol decode; LLVM schedules the scalar chain as well as hand asm, and the x86 asm kernel's
  entire value was BMI2 (`pext/bzhi/shrx`) which aarch64 lacks an analog worth asm for.
- The only genuinely vectorizable parts — the **literal packed store** and the **match copy** — are
  exactly where NEON intrinsics (`vld1q_u8`/`vst1q_u8`) are clean, portable, and inlinable; raw asm
  would block inlining and the FASTLOOP margin reasoning.
- Maintainability + the cross-arch shared skeleton: intrinsics keep ONE Rust kernel with a small
  `#[cfg(target_arch)]` copy specialization; asm forks the whole loop and re-introduces the
  differential-harness burden the x86 side already pays.
- The authorization explicitly allows asm; we DECLINE it for aarch64 on the merits above and reserve
  it as a fallback IF a gated measurement shows intrinsics leave codegen on the table.

**x86_64:** keep the existing BMI2 asm `run_contig` (asm_kernel.rs:477) as the x86 specialization of
the SAME `decode_clean_contig` contract; it already meets/beats its ref. The shared SKELETON is the
pure-Rust kernel (which Intel runs when built **asm-off** — the cross-ISA LAW arm, §5).

---

## 4. RESUMABLE-CONTRACT handling — shed per-symbol surplus, keep T>1, avoid N32

The T>1 pipeline needs: resume mid-block at an `n_max_to_decode` boundary, and a marker (window-
absent) decode mode. Today both leak into the clean path per symbol (ring `% U8_RING_SIZE`, dual
bounds, marker bookkeeping). Design:

1. **Clean kernel writes a CONTIGUOUS region, not the wrapping ring.** Decode into the chunk's
   contiguous decoded buffer with libdeflate's `out_fastloop_end` margin (FASTLOOP_MAX_BYTES_WRITTEN
   headroom) so the per-symbol path has **no** `% U8_RING_SIZE`, no dual bounds — bounds are the loop
   condition only (template :344, :52-53). The ring/marker view is reconciled in BULK at region exit.
2. **Markers never enter the clean kernel.** Window-absent chunks already dispatch to the MARKER
   loop (`read_internal_compressed_specialized::<true>` → `decode_marker_fast_loop`,
   marker_inflate.rs:1699/1994). The clean kernel is `CONTAINS_MARKERS=false` only — so all marker
   bookkeeping (`record_backreference_for_sparsity`, distance_marker) is **out of scope** for it and
   is removed from the clean hot path (it stays in the marker kernel, unchanged).
3. **Resumable boundary handled like libdeflate's `generic_loop` fallback** (template :344
   `goto generic_loop`): the clean loop runs while margin holds; on margin/boundary it breaks to
   `decode_careful_tail` (marker_inflate.rs:2641) with the bit cursor sitting before a fresh
   un-consumed preloaded entry (today's contract, :2605-2613 — preserved verbatim). The careful tail
   owns the wrap-straddle and the `n_max_to_decode` split. No state carried across the seam.
4. **N32 hazard avoidance (no register live-range across the refill / no lengthened loop-carried
   chain):** mirror libdeflate's `saved_bitbuf` pattern EXACTLY — the only value alive across the
   refill is `entry` (the preloaded next entry) and the bit cursor; length/offset extra bits come
   from `saved_bitbuf` captured BEFORE the consume, never held across the refill. No extra accumulator
   (no per-iter sparsity counter, no marker distance) lengthens the chain. This is the precise
   structure that made the x86 N32 attempt a trap; the clean kernel adopts libdeflate's invariant
   instead of inventing a new one.

Net: the clean kernel sheds the ~per-symbol ring+marker surplus (the cross-arch "~75% residual"
shape) WITHOUT touching the marker kernel or the careful tail that enforce T>1 correctness.

---

## 5. BYTE-EXACT + GATING + CROSS-ARCH LAW plan

Every step is gated identically; a step is KEPT only on a byte-exact + instr-win pass, TIE'd (kept
if it's strictly-less-code) or REVERTED otherwise.

**Byte-exact (blocking, every step):**
- The existing reference differential: the new `decode_clean_contig` is pinned against
  `run_contig_ref`-style reference over random bitstreams × {fixed, dense, long-code} tables ×
  varying out budgets (the c2 harness pattern, asm_kernel.rs:2383) — assert exit class, full bit
  cursor, dst advance, every output byte.
- `kernel_ab` / `prop_structured` (existing) + the lib test suite (635+).
- **sha == zcat on T1/T4/T8** across silesia + monorepo + nasa + the synthetic literal/backref
  corpora (real-corpus differential ships in the SAME commit, per the standing rule).
- Marker path untouched ⇒ T>1 marker-grid sha must stay identical (the clean kernel is
  `CONTAINS_MARKERS=false` only).

**Gated instr/cyc (Deliverable-1 harness as the per-step gate):**
- `clean_core_decomp_mac.py` re-run after each step: the per-symbol `decomp_literal` instr/B is the
  primary gauge (it isolates exactly the component under work); silesia is the headline; backref +
  FOLD_NOCRC as non-regression guards. min-of-N≥9, /dev/null, -p1, drop cold run, reject E-core
  outliers (cyc>1.4×median). instr gate at 0.1%; cyc reported Δ-vs-spread (HYPOTHESIS-tier).

**Cross-arch LAW (designed in, runs when an Intel box is reachable):**
- Build Intel **asm-OFF** (`--no-default-features --features gzippy-native`, no `asm-kernel`, or
  `GZIPPY_ASM_KERNEL=0`) so Intel runs the SAME pure-Rust `decode_clean_contig` skeleton. Run the
  standing rig there. A finding is LAW only when the instr win replicates on **Intel(asm-off) +
  macOS(aarch64)** (cross-ISA, strictly stronger than Intel-vs-AMD). The x86 BMI2-asm
  specialization stays AMD-blocked and is NOT part of this LAW claim.
- The x86 asm kernel keeps its own c2/c3 differential vs its ref (unchanged).

**Per-step KEEP/TIE/REVERT:** KEEP if (byte-exact PASS) AND (decomp_literal instr/B drops > 0.1%
on macOS) AND (no silesia/backref regression). TIE→keep only if strictly-less-code. REVERT otherwise,
recording a FALSIFY entry (premise + scope + re-open trigger).

---

## 6. IMPLEMENTATION PLAN — ordered FULL sub-builds (not micro-patches)

Each step is a complete buildable kernel state with its own byte-exact gate. Expected instr/B deltas
are STRUCTURAL feedback (from Deliverable 1's shares), NOT a progress promise — the gate is the wall/
instr measurement, not the prediction.

- **Step 0 — scaffolding + the contiguous-output clean kernel skeleton.** Add `decode_clean_contig`
  (CONTAINS_MARKERS=false only) writing a contiguous region with the `out_fastloop_end` margin;
  wire it behind the existing clean dispatch with the careful tail as the boundary fallback (§4).
  No table change yet — call the current decode. Verify byte-exact + that it routes (a counter).
  Expected: ~0 instr change (pure restructure); proves the contract before the hot changes.

- **Step 1 — flat masked litlen table + fused consume + saved_bitbuf (port `build_decode_table` +
  template :344-435,496).** Replace the two-level packed decode with the flat 11-bit table and the
  explicit 3-literal lookahead. Drop the per-symbol unpack loop. *Biggest expected instr win* (the
  per-symbol gap, Deliverable-1 §3, +19.5 instr/B excess lives here). Risk: HIGH (table layout +
  byte-exactness). Gate: c2-style differential + full sha grid.

- **Step 2 — single amortized refill cadence.** Remove the redundant top/bottom + in-`decode` refills;
  one `REFILL_BITS_IN_FASTLOOP`-equivalent per iteration with the static bit budget. Risk: MED
  (bit-budget proof). Gate: same; watch `decomp_literal` instr/B.

- **Step 3 — flat masked offset table (replace dist_hc + DISTANCE_EXTRA/BASE).** Port the offset
  table (OFFSET_TABLEBITS=8); extract extra from saved_bitbuf. Risk: MED. Gate: backref + silesia sha
  + instr (this also helps the match path the backref corpus exercises).

- **Step 4 — NEON wide overlapping match copy (aarch64) / confirm x86 MOVDQU arm.** Contiguous
  `vld1q/vst1q` overlapping copy with FASTLOOP overshoot. Risk: LOW-MED (overlap correctness — copy
  the libdeflate length-rounding exactly). Expected small instr (+0.26 ceiling) but the larger cyc
  win (copy cyc ratio 1.5×). Gate: backref corpus cyc + sha.

- **Step 5 — x86 specialization reconcile + cross-ISA LAW.** Make the BMI2 asm `run_contig` the x86
  spec of the `decode_clean_contig` contract (or confirm the pure-Rust skeleton suffices on x86
  asm-off). Run Intel(asm-off)+macOS standing rig → promote the wins to LAW. Risk: MED (two arches
  agree). Gate: cross-ISA replication.

- **Step 6 — CRC build-hygiene assertion + cleanup.** Assert HW-CRC feature on ship targets; delete
  the superseded `decode_clean_fast_loop` ring path if Step 0-4 fully replace it (no dead code on
  the clean path). Gate: full suite + measure.sh vs libdeflate (and rapidgzip where available).

**Done-criterion for the kernel:** silesia clean-core instr/B excess closed toward libdeflate
(target: the +11.69 driven down to the copy-floor ~tie), byte-exact across the grid, replicated on
Intel(asm-off)+macOS = cross-ISA LAW, clean kernel the sole clean-decode path per arch.

---

## SELF-CHECK against the anti-bias preamble
- No "the lever is X" banked as prose: the component verdict is the OUTPUT of a self-validated,
  byte-exact, /dev/null, deterministic-instr Fulcrum-style harness (clean_core_decomp_mac.py), with
  the INERT CRC knob explicitly NOT counted (Gate-0c).
- The biggest-component claim (per-symbol Huffman) is measured (2.18× / +19.5 instr/B), and it
  REVISES a prior HYPOTHESIS (NEON-copy-as-biggest-win) DOWN — over-correction guarded by keeping the
  copy step in the plan at its measured (minor) size.
- Scope stamped macOS-aarch64 NOT-YET-LAW; cross-ISA LAW is DESIGNED IN (Intel asm-off), not claimed.
- Corpus-contrast is labeled WEAK-tier (confounds named); the design converges all sub-components so
  it does not over-bet on the internal table-vs-bitreader split it did not measure.
- No floor/settled/accept claim. Every step ends in a gated measurement, not a prose conclusion.
