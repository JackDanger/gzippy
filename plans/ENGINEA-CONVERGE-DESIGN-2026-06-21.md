# ENGINE-A → decompress_template.h CONVERGENCE DESIGN (STEP B-2) — 2026-06-21

**Scope:** macOS aarch64 (Apple Silicon, quiet). Engine A = the aarch64 PRODUCTION
clean Huffman kernel: `decode_huffman_fastloop_bounded`
(`consume_first_decode.rs:1302`, parallel-SM clean contig path) +
`decode_huffman_libdeflate_style` (`:632`, bgzf/scan/multi-member path) — structurally
identical fastloops sharing the flat `LitLenTable`/`DistTable` (`libdeflate_entry.rs`)
and the careful tail `decode_clean_careful_flat` (`:1206`). Blueprint =
`vendor/libdeflate/lib/decompress_template.h` + `deflate_decompress.c`.

**Tier:** HYPOTHESIS (objdump/source reasoning aims; deterministic mac instr/B is the
screen; the aarch64 T1 wall vs libdeflate is the verdict). NOT-YET-LAW (single arch;
AMD/Intel-asm-off owed).

## Baseline (gz engine A vs libdeflate-gunzip, mac, best-of-13, /dev/null, instr/B)

| corpus | gz i/B | ld i/B | i-ratio | excess i/B | note |
|---|---|---|---|---|---|
| decomp_literal (~1 sym/B) | 20.669 | 16.455 | 1.256 | **+4.21** | per-SYMBOL literal extreme |
| silesia | 9.827 | 7.664 | 1.282 | +2.16 | real anchor |
| monorepo | 6.811 | 5.566 | 1.224 | +1.25 | |
| nasa (backref-heavy) | 3.396 | 3.193 | 1.063 | +0.20 | near instr-parity, cyc 1.284 |
| decomp_backref (~1 sym/258B) | 1.831 | 1.702 | 1.076 | +0.13 | per-COPY extreme |

Instr spread 0.1–0.8% → instr/B is a reliable deterministic screen. The excess is
**per-SYMBOL** (collapses 258× on the copy extreme), concentrated in the literal
decode path — confirming B-1's lever.

## RANKED per-symbol divergences (engine A vs decompress_template.h)

### 1. Table lookup geometry — TABLE_BITS  [DONE: increment 1, WIN]
- **Engine A:** litlen=12, dist=9 (hardcoded for ALL arches — an x86 Raptor-Lake
  48KB-L1d tuning). **libdeflate:** `LITLEN_TABLEBITS=11`, `OFFSET_TABLEBITS=8`
  (`deflate_decompress.c:372,374`). The `LitLenTable` doc comment itself records 11
  (= 8 KB) was fastest on ARM64 for L1d locality — the bump to 12 silently regressed
  aarch64.
- **Saved/symbol:** smaller main table → better L1d residency on the per-symbol
  dependent load (`LDR Wentry,[table,idx,LSL#2]`); cost = a few more subtable hits.
- **Byte-exact confidence:** TOTAL (table geometry only; decoded output identical).
- **MEASURED:** litlen 12→11 (aarch64) = silesia 9.827→9.640 i/B (−1.9%, gap
  1.282→1.258, Δ>spread); WIN, kept. dist 9→8 = flat (TIE), kept as faithful
  convergence. (Prior banked falsification: 13 was flat, 10 regressed — reconciled:
  11 is vendor's value and the local ARM64 optimum; nobody had tested 11 directly.)

### 2. Multi-literal lookahead depth + write strategy  [NEXT — top remaining lever]
- **Engine A:** up to **8** literals per iteration, accumulated into 8 separate
  registers (`lit1..lit8`) then packed into one u64/u32 store via a ladder of 8 nested
  `if (entry as i32) < 0` branches (`consume_first_decode.rs:774-933` / `:1386-1508`).
- **libdeflate:** up to **3** (primary + 2 extra), each written **individually**
  (`*out_next++ = lit`, `decompress_template.h:389-434`). It explicitly says doing a
  3rd extra "decreases performance slightly (messes with branch prediction of the
  conditional refill later)" — i.e. vendor deliberately caps at 2 extra.
- **Excess/symbol (HYPOTHESIS):** the packed u64 build is ~15 ALU ops per 8-batch
  (8 OR + 7 shift) vs libdeflate's ~8 byte stores — on **instruction count** the
  packing is ~+1 instr/literal, and the 8-deep branch ladder adds per-symbol
  compare+branch + mispredict pressure. Plausibly ~1.5–2.5 of the +4.21 literal excess.
- **Byte-exact confidence:** MED (large surface; the packed stores + intermediate
  refill placement must stay bit-exact; resumable careful tail unaffected).
- **Risk:** could REGRESS on literal-heavy corpora (decomp_literal) where the packed
  store genuinely cuts store-port traffic. Faithful convergence = drop to libdeflate's
  3-literal individual-write structure; must be gated against decomp_literal + silesia.
  This is the heroic increment; defer to a dedicated gated pass.

### 3. Refill cadence — conditional vs amortized-unconditional  [NEXT]
- **Engine A:** conditional top-of-loop refill `if (bitsleft as u8) < REFILL_THRESHOLD`
  (=44 at TABLE_BITS=11) PLUS scattered `if (bitsleft as u8) < 32 { refill }` after
  short-literal exits (`:907,919,930,1019,...`). Each is a compare+branch/symbol.
- **libdeflate:** `REFILL_BITS_IN_FASTLOOP()` is **unconditional branchless** on 64-bit
  (`deflate_decompress.c:260-272`), issued at fixed points (~1/iteration, placed AFTER
  the next-entry preload to hide load latency). No per-symbol refill branch.
- **Excess/symbol (HYPOTHESIS):** engine A trades sometimes-skipped refills for
  per-symbol compare+branch + mispredicts; libdeflate trades for an always-cheap
  (load+shift+or+2add, no branch) refill. The refill PRIMITIVE is already identical
  (`refill_branchless_fast!` ↔ `REFILL_BITS_BRANCHLESS`, both set bitsleft∈[56,63]); the
  divergence is purely CADENCE. ~0.5–1 instr/symbol.
- **Byte-exact confidence:** MED. Entangled with #2 (the multi-literal ladder's refill
  thresholds). Best done together with #2.

### 4. saved_bitbuf / consume fusion  [≈converged, LOW]
- Both save bitbuf before consume for extra-bit extraction; engine A's `consume!`/
  `huffdec!` macros mirror libdeflate's `saved_bitbuf = bitbuf; bitbuf >>= (u8)entry;
  bitsleft -= entry`. Already faithful. No actionable divergence.

### 5. Entry preload ahead of dependent load  [≈converged, LOW]
- Engine A preloads `entry = lookup!()` before the iteration-end refill, same as
  libdeflate's `entry = table[...]; REFILL_BITS_IN_FASTLOOP()`. Already faithful.

## PLAN
1. **[DONE] increment 1** — TABLE_BITS litlen 12→11 (WIN), dist 9→8 (TIE). Banked.
2. **[NEXT] increment 3** (heroic) — converge the literal lookahead 8→3 + the refill
   cadence to libdeflate's individual-write/amortized-unconditional structure
   (divergences #2+#3 together, since entangled). Gate hard against decomp_literal
   (the path it directly changes) + silesia/monorepo/nasa + the resumable contract.
3. Cross-ISA: re-run on Intel engine-A-asm-off + AMD for LAW.
