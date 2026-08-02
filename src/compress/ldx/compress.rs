//! C: `vendor/libdeflate/lib/deflate_compress.c:2393` and `:4050` —
//! `deflate_compress_none` (the level 0 "compressor") and
//! `libdeflate_deflate_compress` (the public entry point).
//!
//! # `max_passthrough_size` is why short inputs never reach a parser
//!
//! `libdeflate_deflate_compress` routes any input of `max_passthrough_size` bytes or
//! fewer straight to `deflate_compress_none`, and the level map sets that to
//! `55 - 4 * level` (so 51 at level 1, 19 at level 9) — or `SIZE_MAX` at level 0.
//!
//! This is not a fast path that happens to also be correct: `deflate_flush_block`
//! ASSERTs `block_length >= MIN_BLOCK_LENGTH || (is_final_block && block_length > 0)`,
//! so a zero-length input reaching it is undefined behaviour in the C (it computes
//! `DIV_ROUND_UP(0, 65535) - 1`, which underflows). The passthrough is what guarantees
//! that never happens. A port that skips it appears to work on every non-empty input
//! and then produces garbage — or, in Rust, panics — on the empty one.
//!
//! I found this by writing a test that compressed empty input through
//! `deflate_compress_fastest` directly. It panicked in the cost model. The test was
//! wrong, not the port: the C never makes that call.

use super::bitstream::DeflateOutputBitstream;
use super::compress_fastest::{deflate_compress_fastest, FastestState};
use super::flush::Compressor;
use super::DEFLATE_BLOCKTYPE_UNCOMPRESSED;

/// C: `deflate_compress_none(const u8 *in, size_t in_nbytes, u8 *out,
/// size_t out_nbytes_avail)` (:2393)
///
/// This is the level 0 "compressor". It always outputs uncompressed blocks. Returns
/// the number of bytes written, or 0 if the output buffer is too small.
pub(crate) fn deflate_compress_none(r#in: &[u8], in_nbytes: usize, out: &mut [u8]) -> usize {
    let mut in_next: usize = 0;
    let in_end: usize = in_nbytes;
    let mut out_next: usize = 0;
    let out_end: usize = out.len();

    // If the input is zero-length, we still must output a block in order for the
    // output to be a valid DEFLATE stream. Handle this case specially to avoid
    // potentially passing NULL to memcpy() below.
    if in_nbytes == 0 {
        if out_end < 5 {
            return 0;
        }
        // BFINAL and BTYPE
        out[0] = 1 | ((DEFLATE_BLOCKTYPE_UNCOMPRESSED as u8) << 1);
        // LEN and NLEN
        out[1..5].copy_from_slice(&0xFFFF_0000u32.to_le_bytes());
        return 5;
    }

    loop {
        let mut bfinal: u8 = 0;
        let mut len: usize = u16::MAX as usize;

        if in_end - in_next <= u16::MAX as usize {
            bfinal = 1;
            len = in_end - in_next;
        }
        if out_end - out_next < 5 + len {
            return 0;
        }
        // Output BFINAL and BTYPE. The stream is already byte-aligned here, so this
        // step always requires outputting exactly 1 byte.
        out[out_next] = bfinal | ((DEFLATE_BLOCKTYPE_UNCOMPRESSED as u8) << 1);
        out_next += 1;

        // Output LEN and NLEN, then the data itself.
        out[out_next..out_next + 2].copy_from_slice(&(len as u16).to_le_bytes());
        out_next += 2;
        out[out_next..out_next + 2].copy_from_slice(&(!(len as u16)).to_le_bytes());
        out_next += 2;
        out[out_next..out_next + len].copy_from_slice(&r#in[in_next..in_next + len]);
        out_next += len;
        in_next += len;

        if in_next == in_end {
            break;
        }
    }

    out_next
}

/// The ported subset of `struct libdeflate_compressor`'s configuration. C: `:466` and
/// the level map at `:3874`.
///
/// **Only levels 0 and 1 are wired up so far** — those are the two whose compressors
/// are ported. `impl_for` returns `None` for anything else rather than silently
/// falling back, so a caller cannot get a wrong answer from an unported level.
pub(crate) struct LdxCompressor {
    pub(crate) compression_level: u32,
    /// C: `c->max_passthrough_size` (:468)
    pub(crate) max_passthrough_size: usize,
    /// C: `c->nice_match_length`
    pub(crate) nice_match_length: u32,
    pub(crate) c: Compressor,
    pub(crate) p_f: FastestState,
}

impl LdxCompressor {
    /// C: `libdeflate_alloc_compressor_ex` (:3874), the levels ported so far.
    pub(crate) fn new(compression_level: u32) -> Option<Self> {
        // C: `c->max_passthrough_size = 55 - (compression_level * 4);` (:3919)
        let mut max_passthrough_size = 55usize.wrapping_sub(compression_level as usize * 4);
        let nice_match_length;

        match compression_level {
            0 => {
                // C: `c->impl = NULL; c->max_passthrough_size = SIZE_MAX;` (:3922)
                max_passthrough_size = usize::MAX;
                nice_match_length = 0;
            }
            1 => {
                // C: `c->impl = deflate_compress_fastest;` (:3926)
                // `max_search_depth` is unused at this level.
                nice_match_length = 32;
            }
            _ => return None,
        }

        Some(Self {
            compression_level,
            max_passthrough_size,
            nice_match_length,
            c: Compressor::new(),
            p_f: FastestState::new(),
        })
    }

    /// C: `libdeflate_deflate_compress(...)` (:4050)
    ///
    /// Returns the compressed size in bytes, or 0 if the output buffer is too small.
    pub(crate) fn compress(&mut self, r#in: &[u8], in_nbytes: usize, out: &mut [u8]) -> usize {
        // For extremely short inputs, or for compression level 0, just output
        // uncompressed blocks.
        if in_nbytes <= self.max_passthrough_size {
            return deflate_compress_none(r#in, in_nbytes, out);
        }

        // Initialize the output bitstream structure.
        let mut os = DeflateOutputBitstream::new(out);

        // Call the actual compression function.
        debug_assert_eq!(self.compression_level, 1, "only level 1 has a ported impl");
        deflate_compress_fastest(
            &mut self.c,
            &mut self.p_f,
            r#in,
            in_nbytes,
            &mut os,
            self.nice_match_length,
        );

        // Return 0 if the output buffer is too small.
        if os.overflow {
            return 0;
        }

        // Write the final byte if needed. This can't overflow the output buffer
        // because deflate_flush_block() would have set the overflow flag if there
        // wasn't enough space remaining for the full final block.
        debug_assert!(os.bitcount <= 7);
        let mut n = os.next;
        if os.bitcount != 0 {
            debug_assert!(n < os.buf.len());
            os.buf[n] = os.bitbuf as u8;
            n += 1;
        }

        // Return the compressed size in bytes.
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn inflate(bytes: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        flate2::read::DeflateDecoder::new(bytes)
            .read_to_end(&mut out)
            .expect("emitted stream must inflate");
        out
    }

    fn compress_at(level: u32, data: &[u8]) -> Vec<u8> {
        let mut c = LdxCompressor::new(level).expect("level must be ported");
        let mut out = vec![0u8; data.len() * 2 + 4096];
        let n = c.compress(data, data.len(), &mut out);
        assert!(n > 0, "level {level}: output buffer reported too small");
        out.truncate(n);
        out
    }

    /// Level 0 must be pure stored blocks, and must round-trip at every size —
    /// including zero, which is the case the passthrough exists for.
    #[test]
    fn level_0_stores_and_round_trips() {
        for n in [0usize, 1, 2, 100, 65534, 65535, 65536, 131_070, 131_071] {
            let data: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
            let bytes = compress_at(0, &data);
            assert_eq!(inflate(&bytes), data, "level 0, {n} bytes");
            if n > 0 {
                assert_eq!((bytes[0] >> 1) & 3, 0, "level 0 must emit stored blocks");
            }
            // 5 bytes of framing per 65535-byte block, and at least one block.
            let blocks = n.div_ceil(65535).max(1);
            assert_eq!(bytes.len(), n + 5 * blocks, "level 0, {n} bytes: framing");
        }
    }

    /// **The bug this module exists for.** Level 1 must route inputs of 51 bytes or
    /// fewer to `deflate_compress_none` — `55 - 4 * 1`. Above that it must use the
    /// fastest compressor. Both must round-trip.
    #[test]
    fn level_1_passthrough_boundary_is_51_bytes() {
        let mut c = LdxCompressor::new(1).unwrap();
        assert_eq!(c.max_passthrough_size, 51);

        for n in 0..=64usize {
            let data: Vec<u8> = (0..n).map(|i| b"abcabcabd"[i % 9]).collect();
            let mut out = vec![0u8; 4096];
            let written = c.compress(&data, n, &mut out);
            assert!(written > 0, "{n} bytes: reported too small");
            out.truncate(written);
            assert_eq!(inflate(&out), data, "{n} bytes did not round-trip");

            if n <= 51 {
                assert_eq!(
                    (out[0] >> 1) & 3,
                    0,
                    "{n} bytes must pass through to a stored block"
                );
            }
            // A fresh compressor per length: the passthrough path leaves the parser
            // state untouched, so reusing it across lengths would hide state bugs.
            c = LdxCompressor::new(1).unwrap();
        }
    }

    /// Level 1 end to end, at sizes above the passthrough, through an independent
    /// decoder. This is the real gate for the whole port so far: matchfinder,
    /// sequence store, Huffman construction, precode and emitter, all at once.
    #[test]
    fn level_1_round_trips_end_to_end() {
        let mut cases: Vec<Vec<u8>> = Vec::new();

        // Every length across the passthrough boundary and well past it.
        for n in 52..400usize {
            cases.push((0..n).map(|i| b"the quick brown fox "[i % 20]).collect());
        }

        let unit = b"the rain in spain falls mainly on the plain. ";
        let mut rep = Vec::new();
        while rep.len() < 250_000 {
            rep.extend_from_slice(unit);
        }
        cases.push(rep);

        let mut state: u32 = 0x0BAD_F00D;
        cases.push(
            (0..90_000)
                .map(|_| {
                    state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    (state >> 16) as u8
                })
                .collect(),
        );

        for data in &cases {
            let bytes = compress_at(1, data);
            assert_eq!(
                &inflate(&bytes),
                data,
                "level 1 round-trip failed for {} bytes",
                data.len()
            );
        }
    }

    /// An output buffer that is too small must report 0, not panic and not write
    /// past the end. The C returns 0 for this and callers rely on it.
    #[test]
    fn a_short_output_buffer_reports_zero() {
        let data: Vec<u8> = (0..50_000).map(|i| (i % 256) as u8).collect();

        let mut c = LdxCompressor::new(1).unwrap();
        let mut out = vec![0u8; 16];
        assert_eq!(c.compress(&data, data.len(), &mut out), 0);

        // Level 0 too, including the zero-length special case.
        let mut c0 = LdxCompressor::new(0).unwrap();
        let mut tiny = vec![0u8; 4];
        assert_eq!(c0.compress(b"", 0, &mut tiny), 0, "needs 5 bytes for empty");
        let mut just_enough = vec![0u8; 5];
        assert_eq!(c0.compress(b"", 0, &mut just_enough), 5);
    }

    /// Unported levels must refuse rather than silently answer with the wrong
    /// algorithm.
    #[test]
    fn unported_levels_refuse() {
        for level in 2..=12u32 {
            assert!(
                LdxCompressor::new(level).is_none(),
                "level {level} is not ported yet and must not pretend otherwise"
            );
        }
    }
}
