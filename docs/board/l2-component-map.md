# The PARSE LOOP is 81.8% of our L2 instruction excess. The matchfinder gives 12.1% back.

Measured 2026-08-01, trainer (Intel i7-13700T), callgrind + `callgrind_annotate`,
dickens (12.17 MB), **explicit `-p1`**, byte-identical output on both arms
(`fulcrum why` layer [1]: position counts match at Δ0.00%). Ours: main
`120bfa9c`, built unstripped+debuginfo, output sha `d608c061737bce1c` identical
to the shipping stripped build. Theirs: `vendor/libdeflate` @ `22f6e023`,
`RelWithDebInfo -g`, output byte-identical to the homebrew binary the board
used. Totals from callgrind's own `summary:`.

    ours   842,856,205 Ir      theirs 752,825,508 Ir      ours +11.96%

| component | ours | libdeflate | delta | share of the excess |
|---|---|---|---|---|
| **parse loop** | 224,855,684 | 151,228,259 | **+73,627,425** | **+81.8%** |
| block emission | 122,778,195 | 96,346,506 | +26,431,689 | +29.4% |
| crc32 | 2,472,956 | 1,902,275 | +570,681 | +0.6% |
| **matchfinder** | 485,630,547 | 496,567,178 | **-10,936,631** | **-12.1%** |
| unaccounted | 7,118,823 | 6,781,290 | +337,533 | +0.4% |

The shares exceed 100% because the matchfinder is NEGATIVE: it offsets part of
what the other two components spend.

## How the groups were formed

| | ours | libdeflate |
|---|---|---|
| matchfinder | `matchfinder/hc.rs`, `matchfinder/common.rs`, `core::x86/sse.rs` | `hc_matchfinder.h`, `matchfinder_common.h`, `x86/matchfinder_impl.h`, `emmintrin.h`(in greedy) |
| parse loop | `parse/mod.rs:run_resumable`, `parse/greedy.rs`, `block_split.rs`, `core::ptr/{mod,const_ptr}.rs`, `core::num/uint_macros.rs`, `core::slice/iter/macros.rs` | `deflate_compress.c:deflate_compress_greedy`, `common_defs.h`(in greedy) |
| block emission | `parse/mod.rs:emit_sequences`, `bitstream.rs`, `tables.rs`, its `core::` rows | `deflate_compress.c:deflate_flush_block`, `common_defs.h`(in flush_block) |
| crc32 | `pclmulqdq` | `emmintrin.h:crc32_x86_pclmulqdq_avx` |

Only 0.85% of our arm and 0.90% of theirs falls outside the four groups, so the
partition is not hiding a residual.

## Inside our parse loop, 224,855,684 Ir

| | Ir | share of the group |
|---|---|---|
| `parse/mod.rs:run_resumable` | 71,787,338 | 31.9% |
| `core::ptr/mod.rs` | 38,672,223 | 17.2% |
| `block_split.rs` | 33,152,259 | 14.7% |
| `core::num/uint_macros.rs` | 30,996,128 | 13.8% |
| `parse/greedy.rs` | 30,840,783 | 13.7% |
| `core::ptr/const_ptr.rs` | 16,324,965 | 7.3% |
| `core::slice/iter/macros.rs` | 3,081,988 | 1.4% |

libdeflate's whole equivalent is `deflate_compress_greedy` (126,705,652) plus
`common_defs.h` (24,522,607) = 151,228,259.

**Our actual greedy decision logic — `parse/greedy.rs` — is 30.8M, 13.7% of the
group.** The other 86% is the driver (`run_resumable`), the block-split
bookkeeping, and 89.1M Ir of `core::ptr` / `core::num` / `core::slice`
scaffolding. That scaffolding alone is 39.6% of our parse loop and has no
counterpart in the vendor's 151.2M, which is hand-written C over raw pointers.

## What this reframes

`block_split.rs` was reported earlier today as the headline item. It is
14.7% of the parse loop and (against the vendor's measured 22.3M for the same
algorithm) at most 12.0% of the total excess. It is ONE ITEM INSIDE the real
finding, not the finding. The parse loop as a whole is 6.8x larger a target.

## HYPOTHESIS, explicitly not a claim

`run_resumable` is a RESUMABLE parser: it can stop mid-input and continue, which
is what the T>1 chunked path needs. libdeflate's `deflate_compress_greedy` is a
single non-resumable loop over the whole input. A resumable loop must carry and
re-check state that a straight loop keeps in registers, which is the shape of a
`core::ptr`/bounds-check-heavy profile. **This is a source-level story that
matches the profile, and it is exactly the kind of reasoning this project has
been wrong about before (hard stop #4: source-level cost is not machine-level
cost).** It predicts that a non-resumable specialisation of the T1 path would
move the `core::ptr`/`uint_macros` rows. Nothing has been ablated; that is a
measurement, not an argument, and it has not been made.

## Scope and what is NOT claimed

- dickens only, trainer/Intel only, L2 only, T1 only. Hard stop #3: nothing
  generalises to another file, level or arch until measured there.
- **Ir is not ms.** At L9 we execute 26% MORE matchfinder instructions than
  libdeflate and WIN that wall cell by 12.3%. These numbers LOCATE; they do not
  predict the wall. A change that removes instructions here must still be
  measured with `fulcrum ab paired`.
- No component here has been ABLATED. Attribution is not causation; the next
  step is `fulcrum ab ablate`.
- The three earlier corrections in `l2-instruction-attribution.md` (T16 vs T1,
  the retracted memcpy, and block_split's 36.8% -> <=12.0%) all apply; this file
  is the post-correction picture.

## What it kills, and what it opens

**KILLS:** matchfinder inner-loop levers at L2. It is 2.2% CHEAPER than the
vendor's in instructions and contributes -12.1% to the excess. Three prior
levers aimed there; a fourth would be the fourth.

**OPENS:** the parse loop, at +48.7% and 81.8% of the excess, at the exact
level and thread count where every failing wall cell lives.
