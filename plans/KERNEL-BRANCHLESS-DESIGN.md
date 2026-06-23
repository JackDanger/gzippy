# KERNEL-BRANCHLESS-DESIGN — vendor-grounded codegen plan for the 3 mispredicting branches

**Tier: DESIGN / HYPOTHESIS (unvalidated).** This doc tells the implementation leader
*what to build and from which vendor mechanism*. It claims **no win**. The verdict is the
leader's fulcrum optgate (interleaved best-of-N, /dev/null both arms, sha==zcat, cyc/byte
on the gated Intel P-core box + AMD Gate-3) — NOT this prose, NOT instruction counts.

## Source of truth this builds on
- LOCATE measurement: `MEMORY/project_kernel_ipc_locate_2026_06_22` (GATED Intel i7-13700T,
  N=11, taskset P-core, NOT-YET-LAW). The +20.7%(silesia)/+30.9%(nasa) cyc/byte gap vs igzip
  is **~50-55% branch-mispredict + ~22-31% extra instr/byte + ~15% frontend (downstream of
  mispredict); backend/memory NOT a contributor.** 88% of mispredicts at 3 branches.
- HOT PATH = the BMI2 asm `run_contig` in `src/decompress/parallel/asm_kernel.rs`
  (3 variants: primary `run_contig` @ ~L481; the pattern repeats @ ~L1117 and ~L1735 —
  **every fix below must be applied to all three copies**).
- Vendor blueprints (read at HEAD of `vendor/` in the main checkout):
  - libdeflate `lib/decompress_template.h` (fastloop) + `lib/deflate_decompress.c` (macros/encoding)
  - igzip `igzip/igzip_decode_block_stateless.asm` (the AVX2 asm gz is a port of) +
    `igzip/igzip_inflate.c` (`decode_huffman_code_block_stateless_base`, the C reference)

## The 3 branches (PEBS-attributed; gz asm line = primary `run_contig`)
| # | gz asm | site | mispredict share | character |
|---|--------|------|------------------|-----------|
| B1 | `cmp {t5:e}, 256` / `jb 2b` | asm_kernel.rs:683-684 | silesia **52%** / nasa 33% | literal-vs-length/EOB late discriminator (HOT literal back-edge) |
| B2 | `cmp {bitsleft}, 48` / `jae 51f` | asm_kernel.rs:830-831 (+1421,+1874) | silesia 36% / nasa 19% | **pre-copy conditional refill** in the back-ref arm |
| B3 | `sub {t2}, 16` / `jle 74f` | asm_kernel.rs:873-874 (+~1464,+~1917) | nasa **48%** / silesia ~5% | 16-byte MOVDQU copy-loop trip count |

Note: miss-RATE is similar to igzip (~3.5%/~5%); gz just executes **+34%(sil)/+43%(nasa)
MORE branches/byte**. So payoff comes from *removing whole branches per byte*, not from
making an existing branch predict better.

---

## B2 — conditional pre-copy refill  →  ELIMINATE (igzip single-refill design)
**RANKED #1: cleanest, dual-benefit (cuts a branch AND instr/byte), zero envelope risk,
converges to igzip. Capturable mispredict share is ~100% of B2 (igzip has no such branch).**

### What gz does now (the divergence)
gz's main-body refill (`6:`, asm_kernel.rs:641-662) is already **unconditional** (igzip-style:
`mov [in+pos]` → `shlx ,bitsleft` → `or bitbuf` → `63 - bitsleft >>3 → pos` → `or bitsleft,56`).
But the back-ref arm does a **SECOND, conditional** refill before the copy:
```
asm_kernel.rs:830  "cmp {bitsleft}, 48",
asm_kernel.rs:831  "jae 51f",          // skip refill if already >=48
       832-839     <unconditional refill body>
asm_kernel.rs:840  "51:", ...
```
This branch (and its refill) **does not exist in igzip**.

### Vendor mechanism (both eliminate the branch differently)
- **igzip — ELIMINATE by budget (preferred).** `igzip_decode_block_stateless.asm` does
  exactly ONE unconditional refill per `loop_block` iteration (lines **528-547**), to ~57 bits,
  at the *top*, BEFORE the late literal/length discriminator. That single refill covers a full
  symbol: litlen(≤15) + len_extra(≤5) + dist(≤15) + dist_extra(≤13) = **48 ≤ 57**. The dist
  entry is then **speculatively preloaded from the same `read_in`** (lines 550-552:
  `mov next_bits2, 0x1FF; and next_bits2, read_in; movzx ...`) — no further refill before the
  copy. After the copy it jumps back to `loop_block` and refills once again. **igzip never
  refills twice per symbol; it has no pre-copy refill branch at all.**
- **libdeflate — make it unconditional.** `REFILL_BITS_BRANCHLESS()`
  (`deflate_decompress.c:206-212`) is a branch-free word OR (`bitbuf |= leword(in_next) <<
  (u8)bitsleft; in_next += 7 - ((bitsleft>>3)&7); bitsleft |= MAX_BITSLEFT & ~7`), using the
  "garbage in high bits of bitsleft" micro-opt so no compare/mask is needed. The fastloop calls
  it unconditionally (`REFILL_BITS_IN_FASTLOOP`, :260-272) at every preload point
  (decompress_template.h:471, 572, etc.) — no `if (bitsleft < N)` guard in the hot path.

### Mapping onto gz
**Adopt the igzip design (preferred over libdeflate's "always refill"):** the gz main-body
refill at `6:` already restores `bitsleft` to ≥56 each iteration (`or {bitsleft},56`, L662).
The pre-copy refill at L830-839 is therefore **redundant**: after consuming a litlen code
(≤15 bits, or ≤21 on the long path), `bitsleft ≥ 56 - 21 = 35` ≥ the dist budget
(dist_code ≤15 + dist_extra ≤13 = 28). So:
1. **Delete `cmp {bitsleft},48` / `jae 51f` (L830-831)** and make the refill at L832-839
   unconditional — OR, better, **delete the whole pre-copy refill** and prove the top `6:`
   refill budget covers litlen+dist+extra for the back-ref arm (the igzip proof). If the long
   litlen path (`21:`, L755-780, which does its own refill at L764-771) is the only case that
   can drop below the dist budget, keep one unconditional refill ONLY on that cold long path
   and remove it from the hot short back-ref path.
2. Same change at the two sibling variants (L1421-1422, L1874-1875).

### Correctness invariants (byte-exact)
- The asm `EXIT_*` contract reconstructs the cursor from the iteration-top `p0` anchor
  ([ctx+56]) by re-reading from data (asm_kernel.rs:973-991), and the carried packet is
  re-derived by `decode_prefilled` which is **refill-tolerant** (refills are append-only;
  X5 guarantees `bitsleft>=48` at exit). So *adding* refills (making it unconditional) cannot
  change the decoded bytes — refill only appends high bits, never consumes.
- **HAZARD (the real one):** the production Rust *reference model* (`marker_inflate.rs`,
  the `decode_clean_into_contig` fast loop the asm mirrors, and `Bits::refill`) ALSO has this
  pre-copy `< 48` conditional refill. The X1 contract requires the asm's `(bitbuf,bitsleft,pos)`
  writeback to equal the Rust loop's at the same logical point ("REFILL PLACEMENT must be
  identical", asm doc L46-58). So **the Rust ref and the asm must change together**, OR the
  exit must be proven representation-invariant (it is: the exit re-derives from `p0` + re-reads
  from data, so the cursor *representation* — how many bytes are pre-buffered — is not observed
  by the caller; only the logical bit position is). The leader must confirm the differential
  suite (635 lib tests + silesia/nasa multi-oracle) is byte-exact after the change; if the ref
  model's conditional refill is load-bearing for the X1 equality, change the ref model too
  (it is functional/append-only, so unconditional is equivalent).
- IN_MARGIN (=40) already proves every asm refill is the fast `pos+8<=len` form (asm doc
  L103-115); an extra unconditional refill stays inside that margin.

### Mechanism / why cyc drops
Removes one data-dependent branch per back-ref (the `bitsleft` threshold crosses at irregular
symbol boundaries → ~3.5-5% miss → ~15-20 cyc/miss). Because we **eliminate** (not duplicate)
the refill via the igzip budget, instr/byte goes DOWN too (RANK-2 co-benefit). cursor flag:
"making a refill unconditional buys uops when the branch usually skips" — that flag applies to
libdeflate's *add-a-refill* variant; the igzip *delete-a-refill* variant has no such trade and
is why this is ranked #1.

---

## B3 — copy-loop trip count  →  libdeflate overshoot-burst (RANK #2; nasa-dominant)
**Capturable mispredict share is large on match-heavy corpora (nasa 48%), small on silesia.**

### What gz does now
gz uses the igzip MOVDQU copy faithfully (asm_kernel.rs:862-892):
```
871 "71:"  movdqu [{t4}+{t1}], xmm0     // store 16
873        sub {t2}, 16
874        jle 74f                       // <- B3: per-16B trip-count branch
875        add {t4}, 16 ; movdqu xmm0,[{t4}] ; jmp 71b
```
This is a 1:1 port of igzip `large_byte_copy` (asm 603-612: `sub repeat_length, COPY_SIZE;
jle loop_block`). gz ALSO already has a libdeflate-shape **scalar** path for length>240
(asm_kernel.rs:906-916: 5 unconditional u64 stores, then stride loop) — that path is correct
and branch-light, but it is only taken for the rare long tail (`cmp {t2},240; ja 70f`, L862-863).

### Vendor mechanism — libdeflate overshoot burst
`decompress_template.h:590-622` (offset >= WORDBYTES path): **5 unconditional `store_word_
unaligned` (40 bytes), then `while (dst < out_next) { 5 more }`.** For matches ≤ 40 bytes (the
overwhelming majority; mean ~6) the `while` body **never executes** — zero loop-branches; the
single loop-condition predicts not-taken correctly. The loop bound is a **pointer compare
`dst < out_next`** (out_next is computed once as `dst+length`), not a `len -= 16` countdown,
so it is decoupled from the per-step decrement. Offset==1 RLE (:623-648) and 2..7 (:649-663)
are separate unconditional bursts. libdeflate sizes the overshoot into
`FASTLOOP_MAX_BYTES_WRITTEN = 2 + 258 + 5*WORDBYTES - 1` (deflate_decompress.c:280-281).

### Mapping onto gz
Replace the hot MOVDQU `71:/72:/73:` loop (asm_kernel.rs:864-892) with the libdeflate shape:
compute `end = dst + length` once; emit a fixed unconditional burst sized to the *mean+slop*
(e.g. one or two MOVDQU = 16/32 B, or the 5-word 40 B shape gz's scalar path already uses);
then `while (dst < end)` with a pointer compare. Keep the overlap (distance < 16) handling via
the existing period-doubling `small_byte_copy` (asm 614-627) **before** entering the burst, or
via libdeflate's stride-`offset` variant (:649-663). The cold `length>240` scalar path
(L893-972) can stay or be unified.

### Correctness invariants (byte-exact) — THE GATING HAZARD
- **Output-slop envelope.** gz's `FAST_OUT_SLOP = 8 + MAX_RUN_LENGTH(258) + 16 = 282`
  (marker_inflate.rs:2165), and the top guard reserves `dst + FAST_OUT_SLOP <= RING_SIZE`
  (L2238-2239). The current MOVDQU path overshoots ≤15 B past `length` ⇒ `258+15=273 ≤ 282` ✓.
  **A 5-word (40 B) libdeflate burst can overshoot up to ~39 B past the logical end ⇒
  `258+39 = 297 > 282` — OUT OF ENVELOPE.** So if the 40 B burst shape is used, **FAST_OUT_SLOP
  must grow to ≥ 258 + 40 = 298** (match libdeflate's `FASTLOOP_MAX_BYTES_WRITTEN`), and the
  reserve guard L2238 / dst_phys check L2239 / the careful-tail bound (marker_inflate.rs:3219-
  3226) re-proven. A 16 B-burst shape (`258+15=273`) stays inside the existing 282 and avoids
  the envelope change — **prefer the 16-B-burst-then-pointer-while** variant unless the optgate
  shows the 40 B burst pays for the envelope churn. (cursor confirmed 282 ≥ 265 for the 16-B
  shape ✓.)
- Overshoot bytes above the exit `dst` are X3 garbage (never read back: back-ref src < dst).
- The overlap/period-growth correctness for distance < COPY_SIZE must hold byte-for-byte
  vs `emit_backref_contig` (the scalar walk, marker_inflate.rs) — proven by the differential.

### Mechanism / why cyc drops
For short matches (the common case) the per-16-B `sub/jle` (mispredicts on the length tail,
nasa 48%) is replaced by a single well-predicted pointer compare. cursor flag: this **trades
stores for branches** (40 B written for a 6 B match ≈ 7× store traffic) — but the LOCATE
verdict says backend/memory is NOT the gap and store-bandwidth is slack, so on cyc the trade
is expected to pay on match-heavy corpora. Verify on nasa specifically.

---

## B1 — literal-vs-length/EOB dispatch  →  match igzip pack width / libdeflate sign test (RANK #3)
**Largest raw mispredict share (silesia 52%) BUT lowest removability: igzip has the SAME
branch (`cmp 256; jl loop_block`, asm 555-556), so this branch is largely irreducible — the
capturable portion is gz's *excess* branches/byte, not the branch itself.**

### What gz does now
gz packs up to 3 literals per LUT entry (igzip LARGE_SHORT_CODE shape) and discriminates on the
trailing packed symbol: `cmp {t5:e}, 256; jb 2b` (asm_kernel.rs:683-684), after a speculative
8-byte packed store + `advance by sym_count` (asm doc L117-125). This already mirrors igzip's
`mov [next_out], next_sym; add next_out, next_sym_num; ... cmp next_sym2,256; jl loop_block`
(asm 518-519, 555-556). gz's `LitLenEntry` ALSO carries a libdeflate-style bit-31 LITERAL flag
(libdeflate_entry.rs:68, 88-90, 123-126) but the asm hot path uses the `cmp 256` value test,
not the bit-31 sign test.

### Vendor mechanisms (two different philosophies)
- **igzip packed LARGE_SHORT_CODE** (`decode_next_lit_len` macro, asm 322-372): one LUT row
  returns up to 3 symbols + `cnt`; **one** data-dependent branch is amortized over 1-3 syms.
  Higher instr/sym (mask/shrx/spec-store/rollback) — an **explicit instr-for-mispredict trade**.
- **libdeflate HUFFDEC_LITERAL bit-31 sign test** (`entry & HUFFDEC_LITERAL`,
  decompress_template.h:366; flag def deflate_decompress.c:420, ENTRY macro :507): the class is
  a table bit, tested as the sign bit (cheap `js`/`test`); literals decoded up to 3-deep as a
  **taken chain** in literal runs (decompress_template.h:381-417), which predicts well because
  literals cluster. **Lower instr/literal** than igzip packing. NOTE the vendor caps at 3
  literals because "a 4th decreases performance ... by messing with the branch prediction of
  the conditional refill" (decompress_template.h:371-376) — i.e. B1 and B2 are coupled; don't
  over-unroll.

### Mapping onto gz
Two candidate directions for the leader (pre-register one, A/B it):
1. **Widen the pack** so the common literal run resolves one B1 branch per *more* literals,
   reducing branches/byte (gz's +34% excess). Verify gz's pack width actually trails igzip's
   before assuming there is headroom — they may already match at 3 (then this direction is
   near-dead and the excess branches live in B2/EOB-gating, i.e. fixing B2 *is* the B1 win).
2. **Switch the hot discriminator to the bit-31 sign test** (`test entry, entry; js` ≈ literal)
   instead of `cmp {t5},256; jb`, eliminating the explicit immediate compare and shaving an
   instr/sym (RANK-2 co-benefit). Keep the packed store. Byte-exact: the entry already encodes
   bit-31 = LITERAL (libdeflate_entry.rs:126), so the sign test is equivalent to `sym<256`.

### Correctness invariants
Byte-exact packed store + `advance by cnt`; trailing length over-advances dst by 1 and is
undone by `dec dst` before the copy (asm_kernel.rs:709) — preserve. EOB/oversize gating
(`je 81f`, `cmp 512; ja 30f`, L705-707) must stay off the hot literal back-edge.

### Mechanism / why uncertain
Largest mispredict share, but the branch itself is shared with igzip (irreducible). The
realistic cyc lever here is (a) fewer branches/byte via wider packing IF gz trails igzip, and
(b) the cheaper sign-test instr form. cursor flag: igzip-style packing **adds instr** — gate on
cyc/byte, and beware the libdeflate note that aggressive literal unroll degrades B2's refill
prediction. **This is the most likely to TIE; attempt after B2/B3 land and re-measure** (the
frontend ~15% bucket and part of B1's share may be absorbed once B2/B3 remove their mispredicts).

---

## RANKING (likely cyc/wall payoff × confidence — HYPOTHESIS; optgate decides)
1. **B2 — eliminate the pre-copy conditional refill (igzip single-refill budget).** Fully
   capturable (igzip has zero such branch), cuts a branch AND instr/byte, no envelope risk,
   pure convergence to igzip. Mispredict share 36%/19% across BOTH corpora. **Highest confidence.**
2. **B3 — libdeflate overshoot-burst copy.** Dominant on nasa (48%), libdeflate-proven, removes
   the per-16-B trip-count branch. Cost: output-slop envelope must be re-proven (use the 16-B
   burst to stay inside the existing 282, or grow FAST_OUT_SLOP→298 for the 40-B burst);
   store-bandwidth trade (slack per LOCATE). **Big payoff on match-heavy corpora.**
3. **B1 — literal-dispatch (wider pack / bit-31 sign test).** Largest raw mispredict share
   (silesia 52%) but lowest removability (branch shared with igzip). Likely partially absorbed
   once B2/B3 land. **Attempt last, re-measure, expect a TIE risk; gate hard on cyc/byte.**

### Instr-vs-cyc honesty (gate on cyc/wall, NOT instr)
- B2 (igzip delete-a-refill): **fewer** instr AND fewer branches — strict win on both axes.
- B3 (libdeflate burst): **more** store traffic (40 B for a 6 B match) but fewer branches —
  net cyc win expected only because LOCATE proved backend/memory is slack. Verify, don't assume.
- B1 (igzip packing): **more** instr/sym; libdeflate sign-test is instr-neutral-to-cheaper.
  Since the gap is 31%/22% instr-bound too, **do NOT adopt instr-heavy packing to chase a
  mispredict** — net cyc is the only verdict.

## What is explicitly NOT a target (LOCATE-refuted; do not grind)
backend dep-chain / BMI2-PEXT-the-refill / table prefetch / table-shrink — the backend/memory
topdown bucket is not the gap.

## Cross-check provenance
cursor-agent `-p -f --model auto` multi-model review (2026-06-22) corroborated all three vendor
mechanisms and the instr-vs-cyc flags; it ranked B1>B2>B3 by raw mispredict share, this doc
ranks B2>B3>B1 by *capturable* payoff × confidence (B1's share is largely irreducible /
shared with igzip). Both agree B2 is a clean cheap win and B1 risks a TIE / instr inflation.
Full cross-check archived in the session record (/tmp/cursor_full.txt at authoring time).
