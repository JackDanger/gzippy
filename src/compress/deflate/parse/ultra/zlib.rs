//! Zlib container: 2-byte CMF/FLG header + deflate payload + 4-byte
//! big-endian Adler-32. Port of Google Zopfli zlib_container.c.
//!
//! gzippy's production `-z` routes use `crate::compress::zlib_container`
//! for the header/checksum primitives; this module contributes the zopfli
//! DEFLATE engine behind C zopfli's public zlib-container API shape.

use super::deflate::deflate;
use super::ZopfliOptions;
use crate::compress::zlib_container::adler32;

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

    deflate(options, 2, true, in_, out);

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
