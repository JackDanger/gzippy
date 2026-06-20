# CLEAN-KERNEL-DESIGN — the converged clean-decode kernel (per-arch: x86 BMI2 / aarch64 NEON)

Status: **DESIGN — REVISED per STEP-0.5 (premise falsified; reframed to REUSE).**
The pre-Step-0.5 premise ("BUILD a flat clean kernel from scratch") is FALSIFIED:
gzippy ALREADY has a flat libdeflate-style decoder (engine A,
`decode_huffman_libdeflate_style`, consume_first_decode.rs:632) and the Step-0.5 A/B
proves it BEATS the production two-level clean path (engine B,
`Block::decode_clean_into_contig`) 2.07–3.84× instr/B on real corpora and closes the
ENTIRE +19.5/+11.69 aarch64 clean-core excess. So this turn's work product is no
longer a from-scratch build — it is **WIRING engine A into the parallel-SM clean
dispatch + resumable seam** (§1, §6). Full Step-0.5 results + Gate-0 evidence:
`plans/STEP-0.5-RESULTS.md`.

Scope: **macOS-aarch64, NOT-YET-LAW cross-arch.** Every Deliverable-1 and Step-0.5
number is deterministic-instruction (Gate-0 self-validated, byte-exact, /dev/null /
in-RAM both arms). Cross-ISA LAW requires the Intel **asm-OFF** pairing (designed in
§5). No floor/settled/accept claim is made anywhere here.

## STEP-0.5 A/B VERDICT (the re-ranking measurement that gates this build)

Measured (macOS-aarch64, `examples/kernel_ab_aarch64.rs` + `kernel_ab_aarch64.py`,
N=7, two-point R/2R marginal instr, byte-exact A==B==oracle, FLAT_DECODE_CALLS
non-inert proof):

| corpus | A (flat) instr/B | B (two-level) instr/B | B/A | A cyc/B | B cyc/B |
|--------|-----------------:|----------------------:|----:|--------:|--------:|
| webster | 8.374 | 17.306 | **2.067×** | 3.174 | 6.220 |
| mozilla | 4.330 | 16.637 | **3.842×** | 2.018 | 4.716 |
| lit_extreme (per-symbol) | 13.910 | 32.916 | **2.366×** | 6.332 | 12.639 |
| backref_extreme (per-copy) | 0.626 | 0.814 | **1.301×** | 0.096 | 0.182 |

- **Flat WINS on every corpus**, instr and cyc. On the per-symbol extreme engine A
  (13.91) even BEATS libdeflate (16.47, Deliverable-1) and saves −19.0 instr/B vs
  engine B — ≈ the whole +19.5 clean-core excess. The gap WAS the engine, and the
  flat engine already exists.
- **Table-width vs cadence/unpack:** HYPOTHESIS (unvalidated) — the dominant lever is
  the engine STRUCTURE (two-level packed table + unpack branch-chain + double refill),
  NOT table WIDTH (flat-11 libdeflate ≈ flat-12 engine A, both ~14–16 instr/B; the
  two-level decoder is ~2× higher). An exact split needs a third engine (out of scope).
  The design converges the whole flat bundle (= engine A) rather than over-betting the split.

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

### 1.2 The flat clean kernel ALREADY EXISTS — engine A (the thing to WIRE, not build)

`decode_huffman_libdeflate_style` (consume_first_decode.rs:632) is a faithful
libdeflate transliteration that gzippy ALREADY runs in production (bgzf /
scan_inflate / multi-member). Step-0.5 proved it beats engine B 2.07–3.84× instr/B.
Its libdeflate-shaped properties (all PRESENT today, mapped to
`vendor/libdeflate/lib/decompress_template.h`):

- **Flat masked litlen table** — `LitLenTable` with `TABLE_BITS = 12` (libdeflate_entry.rs:361),
  one masked load `(*litlen_ptr.add(bitbuf & MASK)).raw()` per symbol (`lookup!`, :698-702).
  NOTE: gzippy uses **12**, NOT libdeflate's 11 — and **12 is correct for arm64 M-series**
  (the TABLE_BITS=13 attempt was falsified flat on arm64, libdeflate_entry.rs:364-371; Step-0.5
  confirms flat-12 ≈ flat-11 libdeflate, ~14–16 instr/B). KEEP 12; the "=11" target is dropped.
- **Single fused consume** — `consume!`/`huffdec!` macros: `saved=bitbuf; bitbuf>>=entry; bitsleft-=entry` (:724-744).
- **Multi-literal lookahead** — up to **8** literals off batched refills (:777-933), NOT the "3"
  the pre-Step-0.5 draft assumed (see §4 literal-count reconciliation). `saved_bitbuf` extract of
  length/dist extra bits (:976, :1015-1037) — no bitstream re-read.
- **Flat masked dist table** — `DistTable::lookup(bitbuf)` (:1012), subtable only on the exceptional flag.
- **Contiguous output** `out_ptr.add(out_pos)` / `copy_match_fast` (:1050), bounds checked only in
  the loop condition (`out_pos + FASTLOOP_MARGIN <= out_end`, FASTLOOP_MARGIN=320, :763), never per symbol.
- **One refill per iteration** amortized over the literal batch (`refill_branchless_fast!`, :661-672),
  with intermediate refills only inside the >4-literal batch.

### 1.3 The converged target = engine A wired into the clean dispatch

Do NOT write a new `decode_clean_contig`. The converged clean kernel IS engine A
(`decode_huffman_libdeflate_style`) plus its EXISTING resumable variant
(`decode_huffman_body_resumable` / `resume_decode_dynamic_resumable`, resumable.rs:946-1007,
which already handles mid-block yield + window-aware copies + EOB state transitions).
The work is:
- **WIRE** engine A as the parallel-SM CLEAN (`CONTAINS_MARKERS=false`) decode for
  FixedHuffman/DynamicHuffman chunks, replacing the engine-B two-level contig loop
  inside `decode_clean_into_contig` (marker_inflate.rs:3108+) and/or
  `chunk_decode.rs:1695` (§6 sequences this);
- arch-specialize ONLY the copy (engine A already calls NEON on aarch64 via
  `copy_match_fast`/`simd_copy`; on x86 keep the existing BMI2 asm path as the x86 spec — §3/§4(iv));
- handle the resumable `n_max_to_decode` boundary at the SEAM, not per symbol (§4) —
  engine A's fastloop already breaks to a generic loop on margin; the seam re-reads
  the bit cursor (`reclass_reread`-style) rather than carrying a per-iter anchor (N32-safe).

This is the AMD-FREE-PATH Rank-2 "stateless clean-T1 kernel" — and Step-0.5 shows it
is already built and already wins; the remaining cost is integration, not codegen.

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

### (c) Huffman inner loop → USE engine A (already flat-table + multi-literal + single refill) (THE big one)
Engine B today (the thing being REPLACED — `decode_clean_into_contig`'s inline loop +
`LutLitLenCode::decode`):
- a **two-level ISA-L table** returning a PACKED multi-symbol `DecodedSymbol{symbol, sym_count,
  bit_count}`, then a branch-heavy **inner unpack loop** (`while remaining>0 { s>>=8; … }`,
  marker_inflate.rs:2500-2516 for the ring variant; the contig variant's P3.2 chain is analogous);
- **double refill cadence** (top preload + bottom refill, plus `decode()`'s own `available()<32`
  refill) → poor amortization.
- Step-0.5 measured this at 32.9 instr/B (per-symbol) vs engine A's 13.9 — a 2.37× tax.

The convergence is NOT new code — engine A (`decode_huffman_libdeflate_style`) ALREADY has the
libdeflate shape and already won the A/B:
1. **Flat masked litlen table** — `LitLenTable` (TABLE_BITS=12) with the libdeflate entry layout
   (LITERAL flag = sign bit; literal in `entry>>16`; EXCEPTIONAL flag bit15; length in `entry>>16`,
   extra-count in `entry>>8`). Built by `LitLenTable::build` (libdeflate_entry.rs:372). No packed
   multi-symbol unpack loop — engine A uses the explicit multi-literal lookahead (up to 8) which
   gets "multiple literals per refill" WITHOUT the per-symbol `s>>=8` branch chain.
2. **Fused consume** — engine A's `consume!`/`huffdec!` (consume_first_decode.rs:724-744) already
   do `saved=bitbuf; bitbuf>>=entry; bitsleft-=entry` and preload the next entry before consuming.
3. **One refill per iteration** — engine A's `refill_branchless_fast!` (:661) with REFILL_THRESHOLD
   gating; intermediate refills only inside the >4-literal batch. No redundant per-decode refill.

The WORK in §2c is therefore: route the clean dispatch to engine A and shed the engine-B-only
ring/marker per-symbol surplus at the seam (§4). The literal-count and FASTLOOP-margin
reconciliation is in §4 (engine A does 8 literals, FASTLOOP_MARGIN=320; the seam must respect
`n_max_to_decode`).

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

**x86_64 — SCOPE THE ASM OUT of this convergence (advisor change iv).** Wiring engine A into the
x86 clean path would create a TWO-TABLE divergence: engine A uses the flat `LitLenTable`/`DistTable`,
while the x86 BMI2 asm `run_contig` (asm_kernel.rs:477) is built against the ISA-L two-level
`asm.lut_litlen`/`asm.dist`. Reconciling them means either (a) porting the asm to read the flat
table layout (a full asm rewrite + a new c2/c3 differential burden), or (b) running engine A on x86
asm-off and KEEPING the asm as a separate, faster x86 specialization (two clean kernels on x86). The
PRICE of (a) is high and its payoff is unmeasured; the PRICE of (b) is carrying two x86 tables.

DECISION for THIS convergence: **converge only the PURE-RUST skeleton (engine A) — the arm that
Intel runs asm-OFF and that aarch64 always runs.** That is the cross-ISA LAW arm (§5: Intel asm-off
+ macOS aarch64). The x86 BMI2 asm `run_contig` is LEFT AS-IS (it already meets/beats its ref and
is AMD-blocked); it is explicitly OUT of this convergence's LAW claim. Whether to later (a) rewrite
the asm against the flat table or (b) replace it with engine A on x86 is a SEPARATE, post-LAW
decision gated on an x86 A/B (engine A asm-off vs the BMI2 asm) — do NOT pre-judge it here.

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
3. **Resumable boundary via a bit-cursor RE-READ at the seam, NOT a per-iter anchor (advisor
   change ii; N32-safe).** Engine A's fastloop runs while `out_pos + FASTLOOP_MARGIN <= out_end`
   AND `in_pos < in_fastloop_end`; on either bound it falls through to its generic loop
   (consume_first_decode.rs:1059), and the generic loop / `decode_huffman_body_resumable`
   (resumable.rs:1001) owns the tail. The parallel-SM seam must, on the RARE bail (margin reached
   mid-block, or the `n_max_to_decode` cap), recover the bit cursor by **re-deriving the iteration-
   top bit position and re-reading from the data** — the `reclass_reread` pattern: `p0 = lb.pos*8 -
   lb.bitsleft` (asm_kernel.rs:2186), then `reclass_reread(lb, p0)` (asm_kernel.rs:2293) rebuilds
   `bitbuf/bitsleft/pos` from `data[p0>>3]`. This is the N32-SAFE seam: it does NOT "preserve the
   per-iter anchor verbatim" (the deleted night9 4-value snapshot — the N32 trap that lengthened the
   loop-carried chain); it carries only `p0` and re-reads. Engine A already keeps `bitbuf`/`bitsleft`
   in locals with writebacks at every exit (:1040-1042, :1055-1057), so the seam reads a consistent
   `(pos,bitbuf,bitsleft)` and `bit_position()` is well-defined there.
4. **Literal-count + FASTLOOP-margin reconciliation (advisor change ii).** The pre-Step-0.5 draft
   said "3-literal"; engine A actually does **up to 8 literals** per batch (lit1..lit8,
   consume_first_decode.rs:777-933) with intermediate `refill_branchless_fast!` before the 5th and
   8th lookups and `in_fastloop_end = len - 32` input margin. Output margin `FASTLOOP_MARGIN = 320`
   bytes (:639) ≥ 8 speculative literal bytes + one 258-byte max back-ref + copy_match_fast overshoot
   — so a single fastloop iteration writes ≤ 8 + 258 + slop < 320, contiguous, no per-symbol bound.
   **The seam must cap the fastloop so it never overruns `n_max_to_decode`:** enter the fastloop only
   while `out_pos + FASTLOOP_MARGIN <= min(out_end, n_max_to_decode)`; below that, hand off to the
   resumable/careful tail which decodes one symbol at a time up to the exact cap. This proves the
   8-literal batch + max back-ref always fits before the resumable boundary (320 > 8+258+slop) and
   the cap is honored to the byte. No accumulator (sparsity counter, marker distance) is carried
   across the refill — engine A's loop has none, preserving the N32 invariant by construction.

Net: the clean kernel sheds the ~per-symbol ring+marker surplus (the cross-arch "~75% residual"
shape) WITHOUT touching the marker kernel or the careful tail that enforce T>1 correctness.

---

## 5. BYTE-EXACT + GATING + CROSS-ARCH LAW plan

Every step is gated identically; a step is KEPT only on a byte-exact + instr-win pass, TIE'd (kept
if it's strictly-less-code) or REVERTED otherwise.

**Byte-exact (blocking, every step):**
- Engine A vs engine B reference differential: decode the SAME block with both engines (the
  `kernel_ab_aarch64` Gate-0 cross-check already does A==B==flate2) over random bitstreams ×
  {fixed, dense, long-code} tables × varying out budgets — assert exit class, full bit cursor, dst
  advance, every output byte.
- **n_max_to_decode boundary fuzz (advisor change iii):** the seam's correctness hinges on the cap
  landing INSIDE a multi-literal batch. Fuzz `n_max_to_decode` across EVERY value in
  `[block_out - (FASTLOOP_MARGIN+258), block_out]` so the cap falls at every intra-batch position
  (after literal 1..8, mid-back-ref, at EOB) and assert byte-exact + bit-cursor identity vs a
  one-symbol-at-a-time oracle at each cap. Also fuzz the SEAM RE-ENTRY: bail at cap `k`, resume, and
  assert the concatenation == full decode, for every `k` in that window. (Engine A's existing
  resumable variant `decode_huffman_body_resumable` is the resume target; the fuzz proves the
  reclass_reread seam re-reads the cursor correctly at every batch offset.)
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

## 6. IMPLEMENTATION PLAN — WIRE engine A (reframed per Step-0.5; no from-scratch build)

The flat kernel exists and won the A/B. The plan is integration, ordered so each step is a
complete buildable state with its own byte-exact gate. Expected deltas are the Step-0.5 A/B shares
(2.07–3.84× instr/B), NOT a promise — the gate is the measured instr, not the prediction.

- **Step 0 — seam scaffolding (route the clean dispatch to a wrapper, no engine change yet).**
  Add a `decode_clean_contig_flat` wrapper inside `decode_clean_into_contig` (marker_inflate.rs:3108)
  / `chunk_decode.rs:1695`, behind a counter, that for now still calls engine B. Add the
  `reclass_reread` seam (§4.3) and the `n_max_to_decode` cap-to-min(out_end, cap) entry guard (§4.4).
  Verify byte-exact + that it routes (counter > 0). Expected ~0 instr change; proves the seam contract.

- **Step 1 — swap the clean body to engine A (`decode_huffman_libdeflate_style`) + its resumable
  variant.** Route Fixed/DynamicHuffman clean chunks to engine A for the fastloop and to
  `decode_huffman_body_resumable` (resumable.rs:1001) for the bail/cap tail. Build the flat
  `LitLenTable`/`DistTable` per block via `LitLenTable::build`/`DistTable::build` (cache by
  fingerprint, as bgzf/scan already do). *Biggest expected instr win — the entire Step-0.5 2.07–3.84×
  lives here.* Risk: HIGH (seam byte-exactness at the n_max boundary). Gate: the §5
  n_max_to_decode boundary fuzz + engine-A-vs-engine-B differential + full sha grid (T1/T4/T8).

- **Step 2 — shed the engine-B-only per-symbol surplus.** With engine A in place, the contiguous
  output already removes `% U8_RING_SIZE`; confirm no marker bookkeeping
  (`record_backreference_for_sparsity`) remains on the clean (`CONTAINS_MARKERS=false`) path. Risk:
  LOW. Gate: instr (decomp_literal) + sha; marker-grid sha unchanged.

- **Step 3 — confirm the flat dist path covers the match-heavy corpus.** Engine A already uses the
  flat `DistTable` (saved_bitbuf extract); verify the backref_extreme corpus is byte-exact and
  measure its instr/cyc (Step-0.5 already shows A 0.626 vs B 0.814 instr/B). Risk: LOW. Gate:
  backref + silesia sha + instr.

- **Step 4 — NEON match copy (aarch64) confirm.** Engine A's `copy_match_fast` already routes to the
  NEON copy on aarch64; verify it is the overlapping wide-store path under FASTLOOP overshoot (no
  per-symbol bound). Expected small instr, the larger cyc win (copy cyc ratio). Risk: LOW. Gate:
  backref corpus cyc + sha.

- **Step 5 — cross-ISA LAW (Intel asm-OFF + macOS aarch64).** Build Intel asm-off so it runs the SAME
  engine A, run `kernel_ab_aarch64`-equivalent + the standing rig there; promote the win to LAW only
  when it replicates on both. The x86 BMI2 asm is OUT of this LAW arm (§3 change iv). Risk: MED.
  Gate: cross-ISA replication.

- **Step 6 — cleanup.** Delete the superseded engine-B clean contig loop + the ring
  `decode_clean_fast_loop` IF Steps 1–4 fully replace them on the clean path (no dead code; the
  MARKER path keeps its own loop). Keep the CRC HW build-hygiene assertion (§2a). Gate: full suite +
  measure.sh vs libdeflate / rapidgzip.

**Done-criterion for the kernel:** the clean-core instr/B closed to engine A's measured level
(silesia ~8.4 vs engine B ~17.3; per-symbol ~13.9 — at/below libdeflate), byte-exact across the
grid (incl. the n_max boundary fuzz), replicated on Intel(asm-off)+macOS = cross-ISA LAW, engine A
the sole pure-Rust clean-decode path per arch.

---

## SELF-CHECK against the anti-bias preamble
- No "the lever is X" banked as prose: both the Deliverable-1 component verdict AND the Step-0.5
  engine verdict are OUTPUTS of self-validated, byte-exact, /dev/null-or-in-RAM, deterministic-instr
  harnesses (clean_core_decomp_mac.py; kernel_ab_aarch64.py with FLAT_DECODE_CALLS non-inert proof,
  cursor conservation, A==B==oracle), with the INERT CRC knob explicitly NOT counted (Gate-0c).
- Step-0.5 KILLED the design's own premise (build a flat kernel) by MEASUREMENT, not reasoning: the
  flat engine already exists and wins 2.07–3.84×. The reframe to REUSE is the disciplined response to
  a falsifier, not a new confident conclusion — the build is now wiring + a boundary-fuzz gate.
- Table-width-vs-cadence is held as an EXPLICIT unvalidated HYPOTHESIS (the A/B moved both together);
  the plan converges the whole flat bundle rather than over-betting the split it did not isolate.
- Scope stamped macOS-aarch64 NOT-YET-LAW; cross-ISA LAW is DESIGNED IN (Intel asm-off), not claimed.
  The x86 BMI2 asm is explicitly SCOPED OUT of the convergence/LAW (§3 change iv) with its two-table
  price named, not pre-judged.
- No floor/settled/accept claim. Every step ends in a gated measurement, not a prose conclusion.
