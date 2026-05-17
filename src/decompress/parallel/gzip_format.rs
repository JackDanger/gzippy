//! Literal port of `rapidgzip::gzip` header + footer parsing
//! (vendor/rapidgzip/.../gzip/gzip.hpp:77-309).
//!
//! Provides byte-slice oriented `read_header` and `read_footer` that
//! return the parsed metadata + the byte position just past the header
//! / footer. Used by the parallel single-member driver and by future
//! multi-stream support to bracket each gzip stream within a multi-
//! member input.

#![allow(dead_code)]

use std::io;

pub const MAGIC_ID1: u8 = 0x1F;
pub const MAGIC_ID2: u8 = 0x8B;
pub const MAGIC_COMPRESSION: u8 = 0x08;

/// Maximum allowed length of a single FNAME / FCOMMENT field. Matches
/// rapidgzip's `MAX_ALLOWED_FIELD_SIZE` = 1 MiB (gzip.hpp:90).
pub const MAX_ALLOWED_FIELD_SIZE: usize = 1024 * 1024;

/// Port of `rapidgzip::gzip::Header` (gzip.hpp:133-148).
#[derive(Debug, Clone, Default)]
pub struct Header {
    pub modification_time: u32,
    pub operating_system: u8,
    pub extra_flags: u8,
    pub is_likely_ascii: bool,
    pub extra: Option<Vec<u8>>,
    pub file_name: Option<String>,
    pub comment: Option<String>,
    pub crc16: Option<u16>,
}

/// Port of `rapidgzip::gzip::Footer` (gzip.hpp:151-155). Already
/// re-exported as the inner gzip_footer of our `chunk_data::Footer`,
/// but kept here under the rapidgzip-faithful namespace for symmetry.
#[derive(Debug, Clone, Copy, Default)]
pub struct Footer {
    pub crc32: u32,
    pub uncompressed_size: u32,
}

/// Literal port of `rapidgzip::gzip::readHeader` (gzip.hpp:158-234) for
/// byte-slice input. Returns the parsed header and the byte offset just
/// past the header in `data`.
pub fn read_header(data: &[u8]) -> io::Result<(Header, usize)> {
    if data.len() < 10 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "data too short for gzip header (need 10 bytes minimum)",
        ));
    }
    if data[0] != MAGIC_ID1 || data[1] != MAGIC_ID2 || data[2] != MAGIC_COMPRESSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid gzip magic / unsupported compression method",
        ));
    }
    let flags = data[3];
    let modification_time = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let extra_flags = data[8];
    let operating_system = data[9];
    let is_likely_ascii = (flags & 0x01) != 0;

    let mut header = Header {
        modification_time,
        operating_system,
        extra_flags,
        is_likely_ascii,
        ..Default::default()
    };

    let mut offset: usize = 10;

    // FEXTRA (flag bit 2)
    if (flags & (1 << 2)) != 0 {
        if offset + 2 > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "incomplete FEXTRA length",
            ));
        }
        let xlen = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;
        if offset + xlen > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "incomplete FEXTRA data",
            ));
        }
        if xlen > MAX_ALLOWED_FIELD_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "FEXTRA exceeds MAX_ALLOWED_FIELD_SIZE",
            ));
        }
        header.extra = Some(data[offset..offset + xlen].to_vec());
        offset += xlen;
    }

    // FNAME (flag bit 3) — zero-terminated ISO-8859-1 string
    if (flags & (1 << 3)) != 0 {
        let (s, new_offset) = read_zero_terminated_string(data, offset)?;
        header.file_name = Some(s);
        offset = new_offset;
    }

    // FCOMMENT (flag bit 4)
    if (flags & (1 << 4)) != 0 {
        let (s, new_offset) = read_zero_terminated_string(data, offset)?;
        header.comment = Some(s);
        offset = new_offset;
    }

    // FHCRC (flag bit 1)
    if (flags & (1 << 1)) != 0 {
        if offset + 2 > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "incomplete FHCRC",
            ));
        }
        header.crc16 = Some(u16::from_le_bytes([data[offset], data[offset + 1]]));
        offset += 2;
    }

    Ok((header, offset))
}

/// Literal port of `rapidgzip::gzip::readFooter` (gzip.hpp:295-306).
/// `at` is the byte offset to start reading from (must be byte-aligned;
/// the gzip stream always ends on a byte boundary post-padding). Returns
/// the parsed footer.
pub fn read_footer(data: &[u8], at: usize) -> io::Result<Footer> {
    if at + 8 > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "incomplete gzip footer (need 8 bytes)",
        ));
    }
    let crc32 = u32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]]);
    let uncompressed_size =
        u32::from_le_bytes([data[at + 4], data[at + 5], data[at + 6], data[at + 7]]);
    Ok(Footer {
        crc32,
        uncompressed_size,
    })
}

fn read_zero_terminated_string(data: &[u8], at: usize) -> io::Result<(String, usize)> {
    let mut bytes = Vec::new();
    let mut i = at;
    while i < data.len() {
        if bytes.len() >= MAX_ALLOWED_FIELD_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "zero-terminated string exceeds MAX_ALLOWED_FIELD_SIZE",
            ));
        }
        let b = data[i];
        i += 1;
        if b == 0 {
            // Rapidgzip stores the bytes as ISO-8859-1; decode as such.
            let s: String = bytes.iter().map(|&c| c as char).collect();
            return Ok((s, i));
        }
        bytes.push(b);
    }
    Err(io::Error::new(
        io::ErrorKind::UnexpectedEof,
        "reached end of data while reading zero-terminated string",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_gzip(flags: u8, extras: &[(u8, &[u8])]) -> Vec<u8> {
        let mut v = vec![MAGIC_ID1, MAGIC_ID2, MAGIC_COMPRESSION, flags];
        v.extend_from_slice(&u32::to_le_bytes(0x12345678)); // mtime
        v.push(2); // XFL
        v.push(3); // OS = Unix
        for (kind, bytes) in extras {
            match kind {
                b'X' => {
                    // FEXTRA: u16 length + bytes
                    v.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
                    v.extend_from_slice(bytes);
                }
                b'N' | b'C' => {
                    // FNAME or FCOMMENT: null-terminated
                    v.extend_from_slice(bytes);
                    v.push(0);
                }
                b'H' => {
                    v.extend_from_slice(bytes);
                }
                _ => unreachable!(),
            }
        }
        v
    }

    #[test]
    fn parses_minimal_header() {
        let data = build_gzip(0, &[]);
        let (h, off) = read_header(&data).unwrap();
        assert_eq!(off, 10);
        assert_eq!(h.modification_time, 0x12345678);
        assert_eq!(h.extra_flags, 2);
        assert_eq!(h.operating_system, 3);
        assert!(h.file_name.is_none());
    }

    #[test]
    fn parses_header_with_fname() {
        let data = build_gzip(1 << 3, &[(b'N', b"hello.txt")]);
        let (h, off) = read_header(&data).unwrap();
        assert_eq!(off, 10 + b"hello.txt".len() + 1);
        assert_eq!(h.file_name.unwrap(), "hello.txt");
    }

    #[test]
    fn parses_header_with_extra_and_fname_and_fcomment_and_fhcrc() {
        let data = build_gzip(
            (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4),
            &[
                (b'X', b"extradata"),
                (b'N', b"name"),
                (b'C', b"comment"),
                (b'H', &[0xAB, 0xCD]),
            ],
        );
        let (h, off) = read_header(&data).unwrap();
        assert_eq!(h.extra.as_deref(), Some(&b"extradata"[..]));
        assert_eq!(h.file_name.as_deref(), Some("name"));
        assert_eq!(h.comment.as_deref(), Some("comment"));
        assert_eq!(h.crc16, Some(0xCDAB));
        assert_eq!(off, data.len());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut data = build_gzip(0, &[]);
        data[0] = 0xFF;
        assert!(read_header(&data).is_err());
    }

    #[test]
    fn read_footer_parses_crc_and_isize() {
        let mut data = vec![0u8; 10];
        data.extend_from_slice(&u32::to_le_bytes(0xDEADBEEF));
        data.extend_from_slice(&u32::to_le_bytes(1024));
        let f = read_footer(&data, 10).unwrap();
        assert_eq!(f.crc32, 0xDEADBEEF);
        assert_eq!(f.uncompressed_size, 1024);
    }
}
