# AMD-T4 STRUCTURAL-FLOOR DISCRIMINATOR — RESULTS (2026-06-22)

Box: solvency AMD EPYC 7282 Zen2, FROZEN gov=performance/boost=0, cores 8,10,12,14
pinned (hot llama roaming — load 11–16, see CAVEAT). silesia-T4. rg native ELF 0.16.0
`/root/gz-base/vendor/rapidgzip/.../rapidgzip` (HAS symbols + dwarf — no rebuild
needed). gz native+pure @ `fc661a58` `/dev/shm/gztgt/release/gzippy`.

## QUESTION
Is gz's window-absent **marker fast-loop @ 11.7 cyc/B** (banked: 35% of silesia bytes,
2.5× the clean asm kernel's 4.7 cyc/B) a STRUCTURAL FLOOR that rapidgzip ALSO pays
(~11–12), or does rg decode window-absent chunks materially cheaper (~9 or less) =>
gz has CLOSEABLE marker-loop codegen?

## VERDICT: **STRUCTURAL-FLOOR FALSE → CLOSEABLE marker-loop codegen** (Gate-2-tier perf, single-arch AMD)
rg has **NO 11–12 cyc/B marker decode loop**. rg decodes window-absent chunks with
the FAST ISAL huffman loop and resolves markers in a SEPARATE cheap pass.

### Numbers (perf record -F1999 dwarf; perf stat cycles, pinned, frozen)
rg total cyc (silesia-T4) = **2.421 G**.  gz total cyc = **2.675 G** (gz/rg cyc = 1.105 this run; banked ~1.087).
rg flat self-cyc profile (decode region):
  `deflate::Block<false>::read`             33.76%   <- window-absent decoder; window
       (window = `std::array<unsigned short,65536>` = u16 MARKERS;
        body = `readInternalCompressedMultiCached< … HuffmanCodingISAL>`,
        backrefs via inlined `resolveBackreference<u16[]>` 8.65%)
  `..@37.end` / `..@42.end` (ISAL inlined)  8.12% / 2.94%
  `DecodedData::applyWindow`                 7.80%   <- SEPARATE marker-resolution pass
  `loop_block` (ISAL clean huffman)          6.23%
  `peek2` (BitReader)                        2.92%
  `decode_len_dist` (ISAL len/dist)          1.94%
  `large_byte_copy` / memmove (backref copy) 2.02% / 2.46%
  decode region total ≈ 61% (matches prior locate).

### cyc/B (over silesia 211,968,000 B)
  rg decode region (≈61% × 2.421G)            = **6.97 cyc/B blended**
  rg pure symbol-decode `Block<false>::read`  = 33.76%×2.421G/212MB = **3.85 cyc/B**
  gz decode region (≈61% × 2.675G)            = **7.70 cyc/B blended**
  gz clean asm kernel (banked, 65% bytes)     = **4.7 cyc/B**
  gz marker FAST-LOOP (banked, 35% bytes)     = **11.7 cyc/B**

rg's window-absent decode is **3.85 cyc/B (pure huffman) → ~7 cyc/B blended** — i.e.
rg's hardest decode path is ~comparable to gz's CLEAN loop (4.7), NOT to gz's marker
loop (11.7). rg does **not** pay a 2.5× marker penalty.

## MECHANISM (the codegen divergence = the convergence target)
rg DECOUPLES marker handling from the hot loop: it decodes window-absent chunks with
the SAME fast ISAL huffman (`HuffmanCodingISAL` / `loop_block`) writing u16 markers
into the window array, then resolves them in a separate `applyWindow` pass (7.8%; gz's
`resolve_chunk_markers` is 5.6% — even CHEAPER, consistent with the banked apply-window
win). gz INTERLEAVES marker logic INTO a dedicated `decode_marker_fast_loop` that uses
the slower `HuffmanCodingShortBitsCached` (non-inlined `call …::decode` per prior
annotate) + LARGE_FLAG tests + marker-tag ORs per symbol → 11.7 cyc/B. CONVERGENCE:
decode window-absent chunks with the SAME fast huffman gz uses for the clean path
(asm kernel / ISAL-class), and do ALL marker work in the separate apply_window/
replace_markers pass — eliminating the 11.7 cyc/B dedicated marker decode loop.
(Prior Lever 1 — inline the ShortBitsCached call — was a band-aid, −0.5%/−1%; the real
gap is the loop using the slow huffman, not just the call boundary.)

## NET-PRIZE BOUND (honesty — do NOT over-claim)
gz's BLENDED decode (7.70) ≈ rg's (6.97); the gz/rg TOTAL T4 cyc gap is only ~3–10%
(run-dependent, load-noisy). The marker loop is 11.7 vs rg-equivalent ~7 over 74.9MB =
~352M cyc ≈ 13% of gz total IF fully captured — BUT it is partly OFFSET by gz's clean
asm loop (4.7) over-performing rg's blended, so the NET capturable wall is bounded by
the small blended/total gap, not the full 11.7→7 marker delta. The marker-loop rewrite
is the named next lever; its REALIZED prize must be measured at the wall, not assumed
from the sub-loop cyc/B.

## CAVEAT
Single-arch AMD/Zen2, load-contended (llama 11–16). rg uses ISA-L in its clean tail
(`WITH_ISAL`); gz-native is pure-Rust by design — so "use ISAL-class huffman" means
match the asm clean kernel's speed in the marker path, not necessarily link ISA-L.
cyc-ratio (1.105) is this-run; banked frozen-clean was 1.087–1.092. AMD owed a
llama-free reconfirm for LAW; Intel replication owed.
