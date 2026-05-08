//! Zlib container: 2-byte CMF/FLG header + deflate payload + 4-byte
//! big-endian Adler-32. Port of vendor/zopfli/src/zopfli/zlib_container.c.
//!
//! gzippy itself does not use the zlib format, but the C zopfli's public
//! API exposes it; porting it here keeps `ZopfliFormat` complete.

#![allow(dead_code)]

use super::deflate::deflate;
use super::ZopfliOptions;

/// Vanilla Adler-32. Mirrors the C inner loop exactly, including the
/// 5550-element sum-overflow chunking (the largest count of bytes that
/// can be added without overflowing 32-bit `s2 = s2 + s1`).
fn adler32(data: &[u8]) -> u32 {
    const SUMS_OVERFLOW: usize = 5550;
    let mut s1: u32 = 1;
    let mut s2: u32 = 0;
    let mut rest = data;
    while !rest.is_empty() {
        let amount = rest.len().min(SUMS_OVERFLOW);
        for &b in &rest[..amount] {
            s1 += b as u32;
            s2 += s1;
        }
        s1 %= 65521;
        s2 %= 65521;
        rest = &rest[amount..];
    }
    (s2 << 16) | s1
}

/// Zlib-wraps `in_` and appends the result to `out`. Header is the 2-byte
/// CMF/FLG with CM=8, CINFO=7, FLEVEL=3, FDICT=0, FCHECK adjusted so
/// `(cmf*256 + flg) % 31 == 0`. Trailer is the Adler-32 in big-endian.
pub fn zlib_compress(options: &ZopfliOptions, in_: &[u8], out: &mut Vec<u8>) {
    let checksum = adler32(in_);

    let cmf: u32 = 120; // CM = 8, CINFO = 7 (32 KB window)
    let flevel: u32 = 3;
    let fdict: u32 = 0;
    let mut cmfflg: u32 = 256 * cmf + fdict * 32 + flevel * 64;
    let fcheck: u32 = 31 - cmfflg % 31;
    cmfflg += fcheck;

    out.push((cmfflg / 256) as u8);
    out.push((cmfflg % 256) as u8);

    deflate(options, 2, true, in_, 0, out);

    out.push(((checksum >> 24) & 0xff) as u8);
    out.push(((checksum >> 16) & 0xff) as u8);
    out.push(((checksum >> 8) & 0xff) as u8);
    out.push((checksum & 0xff) as u8);

    if options.verbose != 0 {
        let removed = if !in_.is_empty() {
            100.0 * (in_.len() as f64 - out.len() as f64) / in_.len() as f64
        } else {
            0.0
        };
        eprintln!(
            "Original Size: {}, Zlib: {}, Compression: {}% Removed",
            in_.len(),
            out.len(),
            removed
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adler32_matches_known_values() {
        // Reference values from RFC 1950 / common implementations.
        assert_eq!(adler32(b""), 0x0000_0001);
        assert_eq!(adler32(b"a"), 0x0062_0062);
        assert_eq!(adler32(b"abc"), 0x024d_0127);
        // Hand-computed: s1 = 1+sum(bytes) = 920 = 0x398;
        // s2 = sum of running s1 values = 4582 = 0x11E6.
        assert_eq!(adler32(b"Wikipedia"), 0x11e6_0398);
    }

    #[test]
    fn header_modulo_31_is_zero() {
        let opts = ZopfliOptions::default();
        let mut out = Vec::new();
        zlib_compress(&opts, b"hello", &mut out);
        let cmfflg = ((out[0] as u32) << 8) | (out[1] as u32);
        assert_eq!(cmfflg % 31, 0);
        // Length: 2 header + ≥1 deflate + 4 adler32.
        assert!(out.len() >= 7);
    }
}
