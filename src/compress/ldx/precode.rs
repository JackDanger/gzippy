//! C: `vendor/libdeflate/lib/deflate_compress.c:1484-1631` — the precode.
//!
//! Huffman codeword lengths for dynamic Huffman blocks are compressed using a
//! separate Huffman code, the "precode", which contains a symbol for each possible
//! codeword length in the larger code as well as several special symbols to represent
//! repeated codeword lengths (a form of run-length encoding). The precode is itself
//! constructed in canonical form, and its codeword lengths are represented literally
//! in 19 3-bit fields that immediately precede the compressed codeword lengths of the
//! larger code.

use super::codes::{DeflateCodes, NUM_LITLEN_PLUS_OFFSET_SYMS};
use super::huffman::deflate_make_huffman_code;
use super::tables::DEFLATE_PRECODE_LENS_PERMUTATION;
use super::{
    DEFLATE_NUM_LITLEN_SYMS, DEFLATE_NUM_OFFSET_SYMS, DEFLATE_NUM_PRECODE_SYMS,
    MAX_PRE_CODEWORD_LEN,
};

/// C: the `precode` arm of the `union o` in `struct libdeflate_compressor` (:494).
///
/// Temporary space for block flushing. In the C this shares storage with the "full"
/// length codewords via a union, because the two are never live at once. We keep them
/// as separate types: the union is a memory economy with no observable behaviour, and
/// reproducing it in Rust would need `unsafe` for zero byte-identity benefit.
/// (`PORT_STATUS.md` records this as a deliberate, behaviour-free divergence — the
/// only category of divergence this port permits.)
pub(crate) struct Precode {
    pub(crate) freqs: [u32; DEFLATE_NUM_PRECODE_SYMS],
    pub(crate) codewords: [u32; DEFLATE_NUM_PRECODE_SYMS],
    pub(crate) lens: [u8; DEFLATE_NUM_PRECODE_SYMS],
    pub(crate) items: [u32; NUM_LITLEN_PLUS_OFFSET_SYMS],
    pub(crate) num_litlen_syms: usize,
    pub(crate) num_offset_syms: usize,
    pub(crate) num_explicit_lens: usize,
    pub(crate) num_items: usize,
}

impl Precode {
    pub(crate) const fn new() -> Self {
        Self {
            freqs: [0; DEFLATE_NUM_PRECODE_SYMS],
            codewords: [0; DEFLATE_NUM_PRECODE_SYMS],
            lens: [0; DEFLATE_NUM_PRECODE_SYMS],
            items: [0; NUM_LITLEN_PLUS_OFFSET_SYMS],
            num_litlen_syms: 0,
            num_offset_syms: 0,
            num_explicit_lens: 0,
            num_items: 0,
        }
    }
}

/// C: `deflate_compute_precode_items(const u8 lens[], const unsigned num_lens,
/// u32 precode_freqs[], unsigned precode_items[])` (:1484)
///
/// Run-length encode `lens[0..num_lens]` into precode items, accumulating the precode
/// symbol frequencies. Returns the number of items written.
///
/// An item packs a precode symbol in its low 5 bits and that symbol's extra bits
/// above them: `sym | (extra_bits << 5)`. Five bits is enough because the largest
/// precode symbol is 18, and the widest extra field is symbol 18's 7 bits, so an item
/// fits in 12 bits.
///
/// # The three RLE symbols, and the order they are tried in
///
/// * **18** — a run of 11..=138 zeroes, 7 extra bits.
/// * **17** — a run of 3..=10 zeroes, 3 extra bits.
/// * **16** — repeat the PREVIOUS length 3..=6 more times, 2 extra bits.
///
/// The order is not interchangeable. Zero runs greedily take as many symbol-18s as
/// possible first, then at most one symbol-17, then fall through to literal zeroes;
/// nonzero runs emit the length literally once and only then start repeating. The
/// `>= 4` guard on the nonzero branch is why: symbol 16 repeats a length that must
/// already have been emitted, so a run of exactly 3 identical lengths is cheaper
/// written out literally (3 literals) than as literal-plus-repeat (1 literal + 1
/// repeat covering 3, which needs a run of 4 to break even).
///
/// Every one of those thresholds is a byte-identity surface: emitting the same
/// codeword lengths through a different item sequence produces a valid but different
/// stream.
pub(crate) fn deflate_compute_precode_items(
    lens: &[u8],
    num_lens: usize,
    precode_freqs: &mut [u32; DEFLATE_NUM_PRECODE_SYMS],
    precode_items: &mut [u32],
) -> usize {
    // memset(precode_freqs, 0, ...)
    *precode_freqs = [0; DEFLATE_NUM_PRECODE_SYMS];

    let mut itemptr = 0usize;
    let mut run_start = 0usize;

    // do { ... } while (run_start != num_lens);
    loop {
        // Find the next run of codeword lengths.

        // len = the length being repeated
        let len = lens[run_start];

        // Extend the run.
        let mut run_end = run_start;
        loop {
            run_end += 1;
            if run_end == num_lens || len != lens[run_end] {
                break;
            }
        }

        if len == 0 {
            // Run of zeroes.

            // Symbol 18: RLE 11 to 138 zeroes at a time.
            while (run_end - run_start) >= 11 {
                let extra_bits = core::cmp::min((run_end - run_start) - 11, 0x7F);
                precode_freqs[18] += 1;
                precode_items[itemptr] = 18 | ((extra_bits as u32) << 5);
                itemptr += 1;
                run_start += 11 + extra_bits;
            }

            // Symbol 17: RLE 3 to 10 zeroes at a time.
            if (run_end - run_start) >= 3 {
                let extra_bits = core::cmp::min((run_end - run_start) - 3, 0x7);
                precode_freqs[17] += 1;
                precode_items[itemptr] = 17 | ((extra_bits as u32) << 5);
                itemptr += 1;
                run_start += 3 + extra_bits;
            }
        } else {
            // A run of nonzero lengths.

            // Symbol 16: RLE 3 to 6 of the previous length.
            if (run_end - run_start) >= 4 {
                precode_freqs[len as usize] += 1;
                precode_items[itemptr] = len as u32;
                itemptr += 1;
                run_start += 1;
                loop {
                    let extra_bits = core::cmp::min((run_end - run_start) - 3, 0x3);
                    precode_freqs[16] += 1;
                    precode_items[itemptr] = 16 | ((extra_bits as u32) << 5);
                    itemptr += 1;
                    run_start += 3 + extra_bits;
                    if (run_end - run_start) < 3 {
                        break;
                    }
                }
            }
        }

        // Output any remaining lengths without RLE.
        while run_start != run_end {
            precode_freqs[len as usize] += 1;
            precode_items[itemptr] = len as u32;
            itemptr += 1;
            run_start += 1;
        }

        if run_start == num_lens {
            break;
        }
    }

    itemptr
}

/// C: `deflate_precompute_huffman_header(struct libdeflate_compressor *c)` (:1572)
///
/// Precompute the information needed to output dynamic Huffman codes: how many
/// litlen and offset symbols must be sent, the RLE items encoding their lengths, the
/// precode built over those items, and how many of the 19 precode lengths must be
/// sent explicitly.
///
/// # The memmove, which is behaviour and not an optimisation
///
/// If we are not using the full set of literal/length codeword lengths, the offset
/// codeword lengths are temporarily moved DOWN so that the two alphabets are
/// contiguous, and moved back afterwards. This matters because the RLE in
/// `deflate_compute_precode_items` runs across the joined array: a zero run that ends
/// the litlen lengths and begins the offset lengths is coded as ONE run. Encoding the
/// alphabets separately would be valid DEFLATE and different bytes.
///
/// See `codes::DeflateLens::as_flat_mut` for why the flat view is sound.
///
/// # The restore is NOT a full undo, and that is faithful
///
/// The second memmove copies the offset lengths back to byte 288 but does not erase
/// the copy it made at byte `num_litlen_syms`. So on return, litlen lengths
/// `[num_litlen_syms, num_litlen_syms + num_offset_syms)` hold stale offset lengths.
/// The C leaves exactly the same residue — `memmove` is not `memmove`-and-clear — and
/// it is harmless for two independently checked reasons:
///
/// 1. Those litlen symbols were trimmed precisely BECAUSE their codeword length was
///    zero, i.e. their frequency was zero, so they are never emitted in this block.
/// 2. The next block's `deflate_make_huffman_code` rewrites every one of the 288
///    litlen lengths: `sort_symbols` stores `lens[sym] = 0` for each zero-frequency
///    symbol and `gen_codewords` stores a length for each used one.
///
/// A "cleanup" that zeroed the gap would be a divergence from the C in a function
/// whose whole purpose is byte-identity. `precompute_leaves_the_same_residue_as_c`
/// pins the residue so a future tidy-up fails a test instead of a corpus.
pub(crate) fn deflate_precompute_huffman_header(precode: &mut Precode, codes: &mut DeflateCodes) {
    // Compute how many litlen and offset symbols are needed.
    precode.num_litlen_syms = DEFLATE_NUM_LITLEN_SYMS;
    while precode.num_litlen_syms > 257 {
        if codes.lens.litlen[precode.num_litlen_syms - 1] != 0 {
            break;
        }
        precode.num_litlen_syms -= 1;
    }

    precode.num_offset_syms = DEFLATE_NUM_OFFSET_SYMS;
    while precode.num_offset_syms > 1 {
        if codes.lens.offset[precode.num_offset_syms - 1] != 0 {
            break;
        }
        precode.num_offset_syms -= 1;
    }

    // If we're not using the full set of literal/length codeword lengths, then
    // temporarily move the offset codeword lengths over so that the literal/length
    // and offset codeword lengths are contiguous.
    if precode.num_litlen_syms != DEFLATE_NUM_LITLEN_SYMS {
        let flat = codes.lens.as_flat_mut();
        flat.copy_within(
            DEFLATE_NUM_LITLEN_SYMS..DEFLATE_NUM_LITLEN_SYMS + precode.num_offset_syms,
            precode.num_litlen_syms,
        );
    }

    // Compute the "items" (RLE / literal tokens and extra bits) with which the
    // codeword lengths in the larger code will be output.
    precode.num_items = deflate_compute_precode_items(
        codes.lens.as_flat(),
        precode.num_litlen_syms + precode.num_offset_syms,
        &mut precode.freqs,
        &mut precode.items,
    );

    // Build the precode.
    deflate_make_huffman_code(
        DEFLATE_NUM_PRECODE_SYMS,
        MAX_PRE_CODEWORD_LEN,
        &precode.freqs,
        &mut precode.lens,
        &mut precode.codewords,
    );

    // Count how many precode lengths we actually need to output.
    precode.num_explicit_lens = DEFLATE_NUM_PRECODE_SYMS;
    while precode.num_explicit_lens > 4 {
        if precode.lens[DEFLATE_PRECODE_LENS_PERMUTATION[precode.num_explicit_lens - 1] as usize]
            != 0
        {
            break;
        }
        precode.num_explicit_lens -= 1;
    }

    // Restore the offset codeword lengths if needed.
    if precode.num_litlen_syms != DEFLATE_NUM_LITLEN_SYMS {
        let flat = codes.lens.as_flat_mut();
        flat.copy_within(
            precode.num_litlen_syms..precode.num_litlen_syms + precode.num_offset_syms,
            DEFLATE_NUM_LITLEN_SYMS,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::tables::DEFLATE_EXTRA_PRECODE_BITS;
    use super::*;

    /// Decode an item stream back into the codeword lengths it encodes. This is the
    /// real oracle for `deflate_compute_precode_items`: a decoder written from RFC
    /// 1951 section 3.2.7, independent of the encoder's control flow.
    fn decode_items(items: &[u32], num_lens: usize) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::with_capacity(num_lens);
        for &item in items {
            let sym = item & 0x1F;
            let extra = item >> 5;
            match sym {
                16 => {
                    let prev = *out.last().expect("symbol 16 with no previous length");
                    for _ in 0..(3 + extra) {
                        out.push(prev);
                    }
                }
                17 => out.resize(out.len() + (3 + extra) as usize, 0),
                18 => out.resize(out.len() + (11 + extra) as usize, 0),
                _ => {
                    assert_eq!(extra, 0, "literal length item carries extra bits");
                    out.push(sym as u8);
                }
            }
        }
        out
    }

    /// Every item's extra-bit field must fit the width RFC 1951 allots that symbol,
    /// and every frequency must equal the number of items bearing that symbol.
    fn check_items_well_formed(items: &[u32], freqs: &[u32; DEFLATE_NUM_PRECODE_SYMS]) {
        let mut counted = [0u32; DEFLATE_NUM_PRECODE_SYMS];
        for &item in items {
            let sym = (item & 0x1F) as usize;
            assert!(
                sym < DEFLATE_NUM_PRECODE_SYMS,
                "precode symbol {sym} out of range"
            );
            let extra = item >> 5;
            let bits = DEFLATE_EXTRA_PRECODE_BITS[sym] as u32;
            assert!(
                extra < (1u32 << bits) || (bits == 0 && extra == 0),
                "sym {sym}: extra {extra} does not fit {bits} bits"
            );
            counted[sym] += 1;
        }
        assert_eq!(&counted, freqs, "frequencies disagree with the item stream");
    }

    /// Round-trip: the items must decode back to exactly the lengths they encode.
    /// Run over shapes that exercise every branch — long zero runs (symbol 18 taken
    /// repeatedly), short zero runs (17), nonzero repeats (16), and the boundary
    /// lengths where each threshold flips.
    #[test]
    fn precode_items_round_trip() {
        let mut cases: Vec<Vec<u8>> = Vec::new();

        // Degenerate and boundary zero runs, framed by a nonzero so the run has ends.
        for zeros in [0usize, 1, 2, 3, 4, 10, 11, 12, 137, 138, 139, 149, 300] {
            let mut v = vec![5u8];
            v.extend(core::iter::repeat_n(0u8, zeros));
            v.push(7);
            cases.push(v);
        }

        // Boundary nonzero runs: 3 is literal-only, 4 crosses into symbol 16, and
        // 7+ needs a second repeat item.
        for reps in [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20] {
            let mut v = vec![1u8];
            v.extend(core::iter::repeat_n(9u8, reps));
            v.push(2);
            cases.push(v);
        }

        // All zero, and all one length — the two whole-array degenerate shapes.
        cases.push(vec![0u8; 320]);
        cases.push(vec![6u8; 320]);

        // A realistic mixed shape: a deterministic LCG over the legal length range,
        // biased toward zero so runs actually occur.
        let mut state: u32 = 0x2468_ace0;
        let mut v = Vec::with_capacity(320);
        for _ in 0..320 {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let r = (state >> 16) & 0xFF;
            v.push(if r < 160 { 0 } else { ((r % 15) + 1) as u8 });
        }
        cases.push(v);

        for lens in cases {
            let mut freqs = [0u32; DEFLATE_NUM_PRECODE_SYMS];
            let mut items = vec![0u32; lens.len() + 1];
            let n = deflate_compute_precode_items(&lens, lens.len(), &mut freqs, &mut items);
            let items = &items[..n];

            check_items_well_formed(items, &freqs);
            assert_eq!(
                decode_items(items, lens.len()),
                lens,
                "round-trip failed for lens={lens:?}"
            );
        }
    }

    /// The `>= 4` threshold on nonzero runs, pinned explicitly. A run of exactly 3
    /// must be three literal items; a run of 4 must be one literal plus one repeat.
    /// Rewriting this as `>= 3` produces a valid stream and different bytes, which is
    /// exactly the failure this port exists to avoid.
    #[test]
    fn nonzero_run_threshold_is_four() {
        let mut freqs = [0u32; DEFLATE_NUM_PRECODE_SYMS];
        let mut items = [0u32; 32];

        let lens3 = [9u8, 9, 9];
        let n = deflate_compute_precode_items(&lens3, 3, &mut freqs, &mut items);
        assert_eq!(&items[..n], &[9, 9, 9], "a run of 3 must not use symbol 16");
        assert_eq!(freqs[16], 0);

        let lens4 = [9u8, 9, 9, 9];
        let n = deflate_compute_precode_items(&lens4, 4, &mut freqs, &mut items);
        assert_eq!(
            &items[..n],
            &[9, 16],
            "a run of 4 is one literal + one repeat-3"
        );
        assert_eq!(freqs[16], 1);
        assert_eq!(freqs[9], 1);
    }

    /// Zero runs prefer symbol 18 and take it greedily. 138 zeroes is exactly one
    /// symbol-18 at full extra; 139 must be 18(138) then a literal zero, NOT two
    /// items splitting the run evenly.
    #[test]
    fn zero_runs_take_symbol_18_greedily() {
        let mut freqs = [0u32; DEFLATE_NUM_PRECODE_SYMS];
        let mut items = [0u32; 64];

        let lens = [0u8; 138];
        let n = deflate_compute_precode_items(&lens, 138, &mut freqs, &mut items);
        assert_eq!(
            &items[..n],
            &[18 | (0x7F << 5)],
            "138 zeroes is one full sym-18"
        );

        let lens = [0u8; 139];
        let n = deflate_compute_precode_items(&lens, 139, &mut freqs, &mut items);
        assert_eq!(
            &items[..n],
            &[18 | (0x7F << 5), 0],
            "139 zeroes is a full sym-18 plus one literal zero"
        );

        // 149 = 138 + 11, so the second item is a minimal sym-18, not a sym-17.
        let lens = [0u8; 149];
        let n = deflate_compute_precode_items(&lens, 149, &mut freqs, &mut items);
        assert_eq!(&items[..n], &[18 | (0x7F << 5), 18]);
    }

    /// The header precompute must trim trailing unused symbols, must restore the
    /// offset lengths to their own array, and must leave the LIVE part of the litlen
    /// lengths untouched. The items must describe exactly the joined, trimmed array.
    #[test]
    fn precompute_trims_symbols_and_restores_lens() {
        let mut codes = DeflateCodes::new();
        // A code using litlen symbols 0..=260 and offset symbols 0..=3 only.
        for i in 0..=260 {
            codes.lens.litlen[i] = 8;
        }
        for i in 0..=3 {
            codes.lens.offset[i] = 5;
        }
        let before_litlen = codes.lens.litlen;
        let before_offset = codes.lens.offset;

        let mut precode = Precode::new();
        deflate_precompute_huffman_header(&mut precode, &mut codes);

        assert_eq!(precode.num_litlen_syms, 261);
        assert_eq!(precode.num_offset_syms, 4);
        assert_eq!(
            codes.lens.offset, before_offset,
            "offset lengths must be restored"
        );
        assert_eq!(
            &codes.lens.litlen[..261],
            &before_litlen[..261],
            "the live litlen lengths must be untouched"
        );

        let mut joined: Vec<u8> = codes.lens.litlen[..261].to_vec();
        joined.extend_from_slice(&codes.lens.offset[..4]);
        assert_eq!(
            decode_items(&precode.items[..precode.num_items], joined.len()),
            joined
        );
    }

    /// The C's restore is a `memmove` back, not a move-and-clear, so litlen lengths
    /// `[num_litlen_syms, num_litlen_syms + num_offset_syms)` are left holding a stale
    /// copy of the offset lengths. Pinned deliberately: it is what the C does, it is
    /// provably unobservable (see the function's doc comment), and "tidying" it would
    /// be a silent divergence in the one function that must not diverge.
    #[test]
    fn precompute_leaves_the_same_residue_as_c() {
        let mut codes = DeflateCodes::new();
        for i in 0..=260 {
            codes.lens.litlen[i] = 8;
        }
        for i in 0..=3 {
            codes.lens.offset[i] = 5;
        }

        let mut precode = Precode::new();
        deflate_precompute_huffman_header(&mut precode, &mut codes);

        assert_eq!(
            &codes.lens.litlen[261..265],
            &[5, 5, 5, 5],
            "the C leaves the moved-down offset lengths behind; so must we"
        );
        // Reason (1) it is harmless: every symbol in the residue range had length 0,
        // i.e. zero frequency, which is exactly why it was trimmed.
        for i in 261..265 {
            assert_eq!(
                DeflateCodes::new().lens.litlen[i],
                0,
                "residue range must be zero-length symbols"
            );
        }
    }

    /// Reason (2) the residue is harmless: the next block rewrites every litlen
    /// length. Drive a second code through `deflate_make_huffman_code` over the same
    /// buffer and check nothing survives.
    #[test]
    fn a_following_block_rewrites_every_litlen_length() {
        use super::super::codes::{deflate_make_huffman_codes, DeflateFreqs};

        let mut codes = DeflateCodes::new();
        for i in 0..=260 {
            codes.lens.litlen[i] = 8;
        }
        for i in 0..=3 {
            codes.lens.offset[i] = 5;
        }
        let mut precode = Precode::new();
        deflate_precompute_huffman_header(&mut precode, &mut codes);
        assert_ne!(codes.lens.litlen[261], 0, "residue is present before");

        // A completely different block: two literals and end-of-block.
        let mut freqs = DeflateFreqs::new();
        freqs.litlen[b'x' as usize] = 10;
        freqs.litlen[b'y' as usize] = 3;
        freqs.litlen[256] = 1;
        deflate_make_huffman_codes(&freqs, &mut codes);

        for i in 0..DEFLATE_NUM_LITLEN_SYMS {
            let want_used = matches!(i, 120 | 121 | 256); // 'x', 'y', end-of-block
            assert_eq!(
                codes.lens.litlen[i] != 0,
                want_used,
                "litlen {i} was not rewritten"
            );
        }
    }

    /// The trim floors are 257 litlen symbols and 1 offset symbol — RFC 1951 has no
    /// encoding for fewer. An all-zero offset alphabet (a block with no matches) must
    /// still send one offset length.
    #[test]
    fn trim_respects_the_rfc_floors() {
        let mut codes = DeflateCodes::new();
        // Only literal 'a' and end-of-block are used; no offsets at all.
        codes.lens.litlen[b'a' as usize] = 1;
        codes.lens.litlen[256] = 1;

        let mut precode = Precode::new();
        deflate_precompute_huffman_header(&mut precode, &mut codes);

        assert_eq!(precode.num_litlen_syms, 257, "floor is 257, not 256");
        assert_eq!(precode.num_offset_syms, 1, "floor is 1, not 0");
        assert!(precode.num_explicit_lens >= 4, "floor is 4 precode lengths");
        assert!(precode.num_explicit_lens <= DEFLATE_NUM_PRECODE_SYMS);
    }

    /// The precode built over the items must itself be a valid, complete Huffman code
    /// within `MAX_PRE_CODEWORD_LEN`. Checked by Kraft equality over the used symbols.
    #[test]
    fn precode_is_a_complete_code_within_the_length_limit() {
        let mut codes = DeflateCodes::new();
        let mut state: u32 = 0x1357_9bdf;
        for i in 0..288 {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let r = (state >> 16) & 0xFF;
            codes.lens.litlen[i] = if r < 100 { 0 } else { ((r % 14) + 1) as u8 };
        }
        codes.lens.litlen[256] = 7;
        for i in 0..32 {
            codes.lens.offset[i] = ((i % 15) + 1) as u8;
        }

        let mut precode = Precode::new();
        deflate_precompute_huffman_header(&mut precode, &mut codes);

        let used = precode.lens.iter().filter(|&&l| l != 0).count();
        assert!(used >= 2, "a usable precode needs at least 2 symbols");

        let mut kraft = 0u32;
        for &l in precode.lens.iter() {
            if l != 0 {
                assert!(
                    (l as usize) <= MAX_PRE_CODEWORD_LEN,
                    "precode length {l} exceeds the {MAX_PRE_CODEWORD_LEN}-bit limit"
                );
                kraft += 1 << (MAX_PRE_CODEWORD_LEN - l as usize);
            }
        }
        assert_eq!(
            kraft,
            1 << MAX_PRE_CODEWORD_LEN,
            "precode is not a complete code"
        );
    }
}
