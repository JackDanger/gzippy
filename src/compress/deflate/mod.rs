//! Pure-Rust DEFLATE encoder — Increment 2 (hash-chain matchfinder + parsers).
//!
//! This module is the entry point for a from-scratch, pure-Rust DEFLATE/gzip
//! compressor whose structure transliterates libdeflate
//! (`vendor/libdeflate/lib/deflate_compress.c`). Increment 1 landed the proven
//! substrate — constant [`tables`], the word-oriented [`bitstream`], the
//! length-limited canonical [`huffman`] builder + dynamic-block header, the
//! [`block_split`] statistic, and the shared [`matchfinder`] primitives.
//!
//! Increment 2 adds REAL compression: the hash-chains matchfinder
//! ([`matchfinder::hc`], a port of `hc_matchfinder.h`), the level→params table
//! ([`level`]), and the greedy / lazy / lazy2 [`parse`]rs (levels 2-9). Each
//! block chooses the cheapest of a stored, static-Huffman, or dynamic-Huffman
//! encoding of the parsed literal/back-reference token stream. Matches share a
//! 32 KiB window across block boundaries, exactly as DEFLATE allows.
//!
//! Correctness is pinned by `src/tests/deflate_encoder_matches.rs`: byte-exact
//! roundtrip through flate2, libdeflate (FFI), and system `gzip -d` for every
//! implemented level, plus a proptest generator. Wired into production T1
//! routing only under the `pure-rust-encoder` cargo feature (compile-time; the
//! default build is unchanged).
//!
//! Some substrate primitives are used only by later increments (near-optimal
//! parsing), so `dead_code` is allowed module-wide.
#![allow(dead_code)]

pub mod bitstream;
pub mod block_split;
pub mod costs;
pub mod huffman;
pub mod level;
pub mod matchfinder;
pub mod parse;
pub mod tables;

use bitstream::BitWriter;
use level::Strategy;
use tables::DEFLATE_BLOCKTYPE_UNCOMPRESSED;

/// Largest payload of a single stored (BTYPE=00) sub-block.
const MAX_STORED_SUBBLOCK: usize = 65535;

/// Compress `data` into a raw DEFLATE stream (no gzip/zlib framing) at `level`.
pub fn compress_oneshot(data: &[u8], level: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 2 + 64);
    compress_block(data, &[], level, &mut out);
    out
}

/// Compress `data` into a raw DEFLATE stream, appending to `out`.
///
/// `dict` is an optional preset-dictionary window: its bytes are seeded into the
/// matchfinder so back-references in the coded output may point into it, but the
/// dictionary itself is not emitted. The decoder must have the identical window
/// preloaded. Pass `&[]` for no dictionary (the gzip/single-member case).
pub fn compress_block(data: &[u8], dict: &[u8], level: u32, out: &mut Vec<u8>) {
    // A standalone single final block: BFINAL is set on the last internal
    // block and no sync-flush marker is appended. `bw.finish()` byte-aligns
    // the tail. This is the T1 / single-member framing.
    compress_block_streaming(data, dict, level, true, out);
}

/// Compress `data` into a raw DEFLATE stream for use as ONE CHUNK of a larger
/// concatenated single-member stream, appending to `out`.
///
/// Identical to [`compress_block`] except for the stream-position semantics
/// controlled by `is_last`:
///
/// * `is_last == true` — this chunk closes the stream. The last internal block
///   carries `BFINAL=1` and NOTHING is appended after it; `bw.finish()`
///   byte-aligns the tail. With an empty `dict` this is byte-identical to
///   [`compress_block`] (the single-member case).
/// * `is_last == false` — this chunk is followed by more chunks. Every internal
///   block (including the last) stays `BFINAL=0`, and a byte-aligned empty
///   stored block — the standard `Z_SYNC_FLUSH` marker
///   `[BFINAL=0][BTYPE=00][align][LEN=0000][NLEN=FFFF]` — is appended so the
///   chunk ends on a clean, byte-aligned block boundary.
///
/// The sync-flush suffix is the load-bearing correctness detail: independently
/// compressed chunks concatenate into ONE valid single-member DEFLATE stream
/// only when every non-final chunk ends byte-aligned, so a decoder reads chunk
/// N's tail and then chunk N+1's header with no stray bits between them. Each
/// chunk's back-references may point into `dict` (the preceding window, seeded
/// into the matchfinder but not emitted), which the decoder already holds as
/// the tail of the output decoded so far.
pub fn compress_block_streaming(
    data: &[u8],
    dict: &[u8],
    level: u32,
    is_last: bool,
    out: &mut Vec<u8>,
) {
    let mut bw = BitWriter::with_capacity(data.len() / 2 + 64);

    if data.is_empty() {
        emit_stored_block(&mut bw, &[], is_last);
    } else {
        let params = level::params(level);
        if params.strategy == Strategy::Stored {
            // Level 0: uncompressed blocks over the whole input.
            emit_stored_block(&mut bw, data, is_last);
        } else {
            // Build a padded working buffer [dict | data | pad] so the
            // matchfinder's speculative word loads always stay in bounds, and
            // parse over the data region with the dictionary seeded ahead of it.
            let dict_len = dict.len();
            let in_end = dict_len + data.len();
            let mut buf = Vec::with_capacity(in_end + parse::BUF_PAD);
            buf.extend_from_slice(dict);
            buf.extend_from_slice(data);
            buf.resize(in_end + parse::BUF_PAD, 0);

            parse::compress(&buf, dict_len, in_end, &params, is_last, &mut bw);
        }
    }

    // Non-final chunk: close on a clean byte boundary with a sync-flush marker
    // so the next chunk's stream concatenates without stray bits.
    if !is_last {
        emit_stored_block(&mut bw, &[], false);
    }

    out.extend_from_slice(&bw.finish());
}

/// Compress `data` into a gzip-framed stream (gzip header + DEFLATE + CRC32 +
/// ISIZE). This is the variant the roundtrip oracles consume.
pub fn compress_gzip(data: &[u8], level: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 2 + 32);
    // Minimal gzip header: magic, CM=8 (deflate), FLG=0, MTIME=0, XFL=0,
    // OS=255 (unknown).
    out.extend_from_slice(&[0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff]);

    compress_block(data, &[], level, &mut out);

    let crc = crc32fast::hash(data);
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out
}

/// Emit one or more stored (uncompressed, BTYPE=00) blocks covering `data`.
///
/// Port of the uncompressed-block emission in `deflate_flush_block` (~:1826).
/// A stored sub-block carries at most 65535 bytes, so long inputs use several;
/// `is_final` marks the last sub-block BFINAL.
fn emit_stored_block(bw: &mut BitWriter, data: &[u8], is_final: bool) {
    if data.is_empty() {
        write_stored_subblock(bw, &[], is_final);
        return;
    }
    let mut off = 0usize;
    while off < data.len() {
        let end = (off + MAX_STORED_SUBBLOCK).min(data.len());
        let last = end == data.len();
        write_stored_subblock(bw, &data[off..end], is_final && last);
        off = end;
    }
}

fn write_stored_subblock(bw: &mut BitWriter, sub: &[u8], bfinal: bool) {
    debug_assert!(sub.len() <= MAX_STORED_SUBBLOCK);
    bw.add_bits(bfinal as u64, 1);
    bw.add_bits(DEFLATE_BLOCKTYPE_UNCOMPRESSED as u64, 2);
    bw.align_to_byte();
    let len = sub.len() as u16;
    bw.write_u16_le(len);
    bw.write_u16_le(!len);
    bw.write_aligned_bytes(sub);
}

#[cfg(test)]
mod streaming_tests {
    use super::*;
    use std::io::{Read, Write};
    use std::process::{Command, Stdio};

    /// Deterministic mixed text+binary corpus of `len` bytes.
    fn mixed_corpus(len: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(len);
        let phrases: [&[u8]; 4] = [
            b"the quick brown fox jumps over the lazy dog; ",
            b"DEFLATE back-references span chunk boundaries. ",
            b"lorem ipsum dolor sit amet consectetur adipiscing; ",
            b"0123456789abcdef repeated structure repeated structure ",
        ];
        let mut i = 0usize;
        while v.len() < len {
            v.extend_from_slice(phrases[i % phrases.len()]);
            // Sprinkle pseudo-random binary bytes so blocks aren't trivially RLE.
            let x = (i.wrapping_mul(2654435761)) as u32;
            v.extend_from_slice(&x.to_le_bytes());
            i += 1;
        }
        v.truncate(len);
        v
    }

    /// Wrap a raw DEFLATE stream in minimal gzip framing over `original`.
    fn wrap_gzip(deflate: &[u8], original: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(deflate.len() + 18);
        out.extend_from_slice(&[0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff]);
        out.extend_from_slice(deflate);
        out.extend_from_slice(&crc32fast::hash(original).to_le_bytes());
        out.extend_from_slice(&(original.len() as u32).to_le_bytes());
        out
    }

    fn decode_flate2(gz: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(gz)
            .read_to_end(&mut out)
            .expect("flate2 failed to decode concatenated stream");
        out
    }

    /// `gzip -dc` decode; `None` only when no `gzip` binary is on PATH.
    fn decode_system_gzip(gz: &[u8]) -> Option<Vec<u8>> {
        let mut child = Command::new("gzip")
            .arg("-dc")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let mut stdin = child.stdin.take().unwrap();
        let buf = gz.to_vec();
        let writer = std::thread::spawn(move || {
            let _ = stdin.write_all(&buf);
        });
        let mut out = Vec::new();
        child.stdout.take().unwrap().read_to_end(&mut out).unwrap();
        writer.join().unwrap();
        assert!(child.wait().unwrap().success(), "gzip -dc exited non-zero");
        Some(out)
    }

    /// Three independently-compressed chunks (chunk k seeds the previous chunk's
    /// 32 KiB tail as its dictionary; only the last is `is_last`) concatenate
    /// into ONE valid single-member gzip stream that decodes byte-exact.
    #[test]
    fn three_chunks_concatenate_and_roundtrip() {
        let input = mixed_corpus(300_003);
        let n = input.len();
        let bounds = [0usize, n / 3, 2 * n / 3, n];

        for level in [1u32, 6, 9, 12] {
            let mut deflate = Vec::new();
            for c in 0..3 {
                let start = bounds[c];
                let end = bounds[c + 1];
                let dict_start = start.saturating_sub(32 * 1024);
                let dict = &input[dict_start..start];
                let is_last = c == 2;
                compress_block_streaming(&input[start..end], dict, level, is_last, &mut deflate);
            }
            let gz = wrap_gzip(&deflate, &input);

            assert_eq!(
                decode_flate2(&gz),
                input,
                "flate2 roundtrip mismatch at L{level}"
            );
            if let Some(sys) = decode_system_gzip(&gz) {
                assert_eq!(sys, input, "gzip -dc roundtrip mismatch at L{level}");
            }
        }
    }

    /// `compress_block_streaming(data, &[], level, true, ..)` must be
    /// byte-identical to the single-block [`compress_block`] (no sync marker,
    /// BFINAL set) — the regression guard the brief requires.
    #[test]
    fn is_last_no_dict_equals_compress_block() {
        let cases: [Vec<u8>; 3] = [Vec::new(), b"tiny".to_vec(), mixed_corpus(200_000)];
        for data in &cases {
            for level in [0u32, 1, 2, 6, 9, 12] {
                let mut streaming = Vec::new();
                compress_block_streaming(data, &[], level, true, &mut streaming);
                let mut block = Vec::new();
                compress_block(data, &[], level, &mut block);
                assert_eq!(
                    streaming,
                    block,
                    "streaming(is_last=true) diverged from compress_block at L{level}, len={}",
                    data.len()
                );
            }
        }
    }
}

#[cfg(test)]
mod dict_tests {
    use super::*;

    /// An empty preset dictionary must yield byte-identical output to the
    /// no-dictionary path (regression guard on the seeding wiring).
    #[test]
    fn empty_dict_equals_no_dict() {
        let data: Vec<u8> = b"the pure-rust deflate encoder must roundtrip. ".repeat(400);
        for level in [2u32, 6, 9] {
            let mut with_empty = Vec::new();
            compress_block(&data, &[], level, &mut with_empty);
            let no_dict = compress_oneshot(&data, level);
            assert_eq!(with_empty, no_dict, "empty dict diverged at L{level}");
        }
    }

    /// A dictionary whose bytes appear in the data must let the parser reference
    /// it, producing a strictly smaller stream than compressing without it.
    /// This exercises the `skip_bytes` dictionary-seeding path (matches point
    /// back into `buf[..data_start]`).
    #[test]
    fn matching_dict_shrinks_output() {
        // Data begins with content that only exists in the dictionary, so the
        // opening bytes can only be coded as matches into the seeded window.
        let dict: Vec<u8> =
            b"PRESET-DICTIONARY-CONTENT-abcdefghijklmnopqrstuvwxyz-0123456789-".repeat(30);
        let data: Vec<u8> = {
            let mut d = dict.clone(); // fully present in the dictionary window
            d.extend_from_slice(b" and then some novel trailing text to code as literals.");
            d
        };
        for level in [4u32, 6, 9] {
            let with_dict = {
                let mut v = Vec::new();
                compress_block(&data, &dict, level, &mut v);
                v.len()
            };
            let without = compress_oneshot(&data, level).len();
            assert!(
                with_dict < without,
                "L{level}: dict-seeded {with_dict} not smaller than no-dict {without}",
            );
        }
    }
}
