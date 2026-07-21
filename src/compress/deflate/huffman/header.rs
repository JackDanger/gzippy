//! DEFLATE dynamic-block header (precode/RLE) construction + emission — the
//! level engine's ONE dynamic-header builder.
//!
//! Port of libdeflate `vendor/libdeflate/lib/deflate_compress.c`:
//! `deflate_compute_precode_items` / `deflate_precompute_huffman_header`
//! (~:1482-1631). Builds on [`super::fast::make_huffman_code`] for the
//! precode (19-symbol) code itself.
//!
//! Ultra (`parse::ultra`) constructs and emits its own dynamic-block header
//! via a separate code path (`parse::ultra::deflate::encode_tree_emit` +
//! `parse::ultra::deflate_size::build_rle`) that shares the same RFC-1951
//! precode/RLE wire format but a different cost-accounting shape (ultra
//! evaluates all 8 use-16/17/18 flag combinations per block and threads
//! `hlit`/`hdist`/`hclen` through its own histogram-driven `build_rle`,
//! whereas this module always uses all three RLE symbols and trims by
//! codeword length). The two are NOT merged in Stage C — see
//! `docs/compressor-architecture.md` §3 ("residual duplication").

use super::fast::make_huffman_code;
use super::HuffmanCode;
use crate::compress::deflate::bitstream::BitWriter;
use crate::compress::deflate::tables::{
    DEFLATE_NUM_PRECODE_SYMS, MAX_PRE_CODEWORD_LEN, PRECODE_EXTRA_BITS, PRECODE_LENS_PERMUTATION,
};

/// Everything needed to emit a DEFLATE dynamic-block header.
pub struct DynamicHeader {
    pub num_litlen_syms: usize,
    pub num_offset_syms: usize,
    pub num_explicit_lens: usize,
    /// RLE items: low 5 bits = precode symbol, high bits = extra-bit value.
    pub items: Vec<u32>,
    /// Precode (19 symbols).
    pub precode: HuffmanCode,
}

/// Compute the precode RLE "items" and the precode symbol frequencies for a set
/// of contiguous litlen+offset codeword lengths. Port of
/// `deflate_compute_precode_items`.
fn compute_precode_items(lens: &[u8]) -> ([u32; DEFLATE_NUM_PRECODE_SYMS], Vec<u32>) {
    let mut freqs = [0u32; DEFLATE_NUM_PRECODE_SYMS];
    let mut items: Vec<u32> = Vec::new();
    let num_lens = lens.len();

    let mut run_start = 0usize;
    loop {
        let len = lens[run_start];

        // Extend the run of equal lengths.
        let mut run_end = run_start;
        loop {
            run_end += 1;
            if run_end == num_lens || len != lens[run_end] {
                break;
            }
        }

        if len == 0 {
            // Symbol 18: 11..=138 zeroes.
            while (run_end - run_start) >= 11 {
                let extra = ((run_end - run_start) - 11).min(0x7F) as u32;
                freqs[18] += 1;
                items.push(18 | (extra << 5));
                run_start += 11 + extra as usize;
            }
            // Symbol 17: 3..=10 zeroes.
            if (run_end - run_start) >= 3 {
                let extra = ((run_end - run_start) - 3).min(0x7) as u32;
                freqs[17] += 1;
                items.push(17 | (extra << 5));
                run_start += 3 + extra as usize;
            }
        } else if (run_end - run_start) >= 4 {
            // Symbol 16: repeat previous nonzero length 3..=6 times.
            freqs[len as usize] += 1;
            items.push(len as u32);
            run_start += 1;
            loop {
                let extra = ((run_end - run_start) - 3).min(0x3) as u32;
                freqs[16] += 1;
                items.push(16 | (extra << 5));
                run_start += 3 + extra as usize;
                if (run_end - run_start) < 3 {
                    break;
                }
            }
        }

        // Any remaining lengths emitted literally.
        while run_start != run_end {
            freqs[len as usize] += 1;
            items.push(len as u32);
            run_start += 1;
        }

        if run_start == num_lens {
            break;
        }
    }

    (freqs, items)
}

/// Build the dynamic-block header from the full (untrimmed) litlen and offset
/// codeword-length arrays. Port of `deflate_precompute_huffman_header`.
pub fn build_dynamic_header(litlen_lens: &[u8], offset_lens: &[u8]) -> DynamicHeader {
    // Trim trailing zero litlen lengths (keep at least 257).
    let mut num_litlen_syms = litlen_lens.len();
    while num_litlen_syms > 257 && litlen_lens[num_litlen_syms - 1] == 0 {
        num_litlen_syms -= 1;
    }
    // Trim trailing zero offset lengths (keep at least 1).
    let mut num_offset_syms = offset_lens.len();
    while num_offset_syms > 1 && offset_lens[num_offset_syms - 1] == 0 {
        num_offset_syms -= 1;
    }

    // Contiguous litlen+offset lengths (replaces libdeflate's in-place memmove).
    let mut combined: Vec<u8> = Vec::with_capacity(num_litlen_syms + num_offset_syms);
    combined.extend_from_slice(&litlen_lens[..num_litlen_syms]);
    combined.extend_from_slice(&offset_lens[..num_offset_syms]);

    let (precode_freqs, items) = compute_precode_items(&combined);

    let precode = make_huffman_code(
        DEFLATE_NUM_PRECODE_SYMS,
        MAX_PRE_CODEWORD_LEN,
        &precode_freqs,
    );

    // Count how many precode lengths must actually be written (>= 4).
    let mut num_explicit_lens = DEFLATE_NUM_PRECODE_SYMS;
    while num_explicit_lens > 4
        && precode.lens[PRECODE_LENS_PERMUTATION[num_explicit_lens - 1] as usize] == 0
    {
        num_explicit_lens -= 1;
    }

    DynamicHeader {
        num_litlen_syms,
        num_offset_syms,
        num_explicit_lens,
        items,
        precode,
    }
}

impl DynamicHeader {
    /// Exact bit cost of the header body (everything after BFINAL+BTYPE):
    /// HLIT/HDIST/HCLEN + precode lengths + the RLE-encoded litlen/offset
    /// lengths. Used by the stored-vs-dynamic decision.
    pub fn header_bits(&self) -> u64 {
        let mut bits = 5 + 5 + 4 + 3 * self.num_explicit_lens as u64;
        for &item in &self.items {
            let sym = (item & 0x1F) as usize;
            bits += self.precode.lens[sym] as u64 + PRECODE_EXTRA_BITS[sym] as u64;
        }
        bits
    }

    /// Emit the header body. The caller must already have written the 3-bit
    /// BFINAL + BTYPE(dynamic) prefix.
    pub fn emit(&self, bw: &mut BitWriter) {
        bw.add_bits((self.num_litlen_syms - 257) as u64, 5);
        bw.add_bits((self.num_offset_syms - 1) as u64, 5);
        bw.add_bits((self.num_explicit_lens - 4) as u64, 4);

        // Precode codeword lengths, in permutation order.
        for &sym in PRECODE_LENS_PERMUTATION.iter().take(self.num_explicit_lens) {
            bw.add_bits(self.precode.lens[sym as usize] as u64, 3);
        }

        // RLE-encoded litlen + offset codeword lengths.
        for &item in &self.items {
            let sym = (item & 0x1F) as usize;
            bw.add_bits(
                self.precode.codewords[sym] as u64,
                self.precode.lens[sym] as u32,
            );
            bw.add_bits((item >> 5) as u64, PRECODE_EXTRA_BITS[sym] as u32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precode_items_roundtrip_lengths() {
        // A lengths array with runs of zeros and repeats; reconstruct it by
        // interpreting the items and confirm we get the original back.
        let lens: Vec<u8> = vec![
            5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 7, 7, 7, 7, 7, 7,
        ];
        let (_freqs, items) = compute_precode_items(&lens);
        let mut out: Vec<u8> = Vec::new();
        let mut prev = 0u8;
        for &item in &items {
            let sym = (item & 0x1F) as u8;
            let extra = item >> 5;
            match sym {
                16 => {
                    let n = 3 + extra;
                    out.resize(out.len() + n as usize, prev);
                }
                17 => {
                    let n = 3 + extra;
                    out.resize(out.len() + n as usize, 0);
                }
                18 => {
                    let n = 11 + extra;
                    out.resize(out.len() + n as usize, 0);
                }
                l => {
                    out.push(l);
                    prev = l;
                }
            }
        }
        assert_eq!(out, lens);
    }
}
