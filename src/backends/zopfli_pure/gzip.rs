//! Gzip container: 10-byte header + deflate payload + 4-byte CRC32 + 4-byte
//! ISIZE. Port of vendor/zopfli/src/zopfli/gzip_container.c (RFC 1952).

#![allow(dead_code)]

use super::deflate::deflate;
use super::ZopfliOptions;

/// Gzip-wraps `in_` and appends the result to `out`. The C version writes
/// header bytes that match exactly:
///
/// - ID1=0x1f, ID2=0x8b, CM=8 (deflate), FLG=0
/// - MTIME = 4 zero bytes
/// - XFL=2 (best compression), OS=3 (Unix)
///
/// followed by `deflate(btype=2, final=true)` of the payload, then a
/// little-endian CRC32 and ISIZE-mod-2^32.
pub fn gzip_compress(options: &ZopfliOptions, in_: &[u8], out: &mut Vec<u8>) {
    let crcvalue = crc32fast::hash(in_);

    // Header: 10 fixed bytes per RFC 1952.
    out.push(31); // ID1
    out.push(139); // ID2
    out.push(8); // CM = deflate
    out.push(0); // FLG
    out.push(0); // MTIME[0]
    out.push(0); // MTIME[1]
    out.push(0); // MTIME[2]
    out.push(0); // MTIME[3]
    out.push(2); // XFL = best compression
    out.push(3); // OS = Unix

    deflate(options, 2, true, in_, 0, out);

    // CRC32 (little-endian).
    out.push((crcvalue & 0xff) as u8);
    out.push(((crcvalue >> 8) & 0xff) as u8);
    out.push(((crcvalue >> 16) & 0xff) as u8);
    out.push(((crcvalue >> 24) & 0xff) as u8);

    // ISIZE (little-endian, mod 2^32 — C uses size_t but writes only 4 bytes).
    let isize_lo = (in_.len() & 0xffff_ffff) as u32;
    out.push((isize_lo & 0xff) as u8);
    out.push(((isize_lo >> 8) & 0xff) as u8);
    out.push(((isize_lo >> 16) & 0xff) as u8);
    out.push(((isize_lo >> 24) & 0xff) as u8);

    if options.verbose != 0 {
        let removed = if !in_.is_empty() {
            100.0 * (in_.len() as f64 - out.len() as f64) / in_.len() as f64
        } else {
            0.0
        };
        eprintln!(
            "Original Size: {}, Gzip: {}, Compression: {}% Removed",
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
    fn header_is_well_formed_and_trailer_records_isize() {
        let opts = ZopfliOptions::default();
        let mut out = Vec::new();
        let data = b"hello, world!\n";
        gzip_compress(&opts, data, &mut out);

        // Header.
        assert_eq!(out[0], 0x1f);
        assert_eq!(out[1], 0x8b);
        assert_eq!(out[2], 8);
        assert_eq!(out[3], 0);
        assert_eq!(&out[4..8], &[0u8, 0, 0, 0]); // MTIME
        assert_eq!(out[8], 2); // XFL
        assert_eq!(out[9], 3); // OS = Unix

        // ISIZE in last 4 bytes (little-endian).
        let n = out.len();
        let isize_le = u32::from_le_bytes(out[n - 4..n].try_into().unwrap());
        assert_eq!(isize_le as usize, data.len());

        // CRC32 in the 4 bytes before that.
        let crc_le = u32::from_le_bytes(out[n - 8..n - 4].try_into().unwrap());
        assert_eq!(crc_le, crc32fast::hash(data));
    }
}
