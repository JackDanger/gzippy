//! Pure gzip header parsing — no I/O, no routing decisions.
//!
//! These functions inspect raw gzip bytes: magic bytes, flags, FEXTRA subfields,
//! ISIZE trailer, and gzippy's own "GZ" parallel-format marker.
//!
//! Used by: `decompression` (classify_gzip), `decompress_io` (file handling),
//!          `bgzf` (format detection in tests).

use memchr::memmem;

/// True if the data carries gzippy's "GZ" FEXTRA subfield (parallel multi-block format).
#[inline]
pub(crate) fn has_bgzf_markers(data: &[u8]) -> bool {
    // Minimum header with FEXTRA: 10 base + 2 XLEN + 4 subfield header
    if data.len() < 16 {
        return false;
    }
    if data[3] & 0x04 == 0 {
        return false;
    }
    let xlen = u16::from_le_bytes([data[10], data[11]]) as usize;
    if xlen < 6 || data.len() < 12 + xlen {
        return false;
    }
    let extra_field = &data[12..12 + xlen];
    let mut pos = 0;
    while pos + 4 <= extra_field.len() {
        let subfield_id = &extra_field[pos..pos + 2];
        let subfield_len =
            u16::from_le_bytes([extra_field[pos + 2], extra_field[pos + 3]]) as usize;
        if subfield_id == crate::parallel_compress::GZ_SUBFIELD_ID.as_slice() {
            return true;
        }
        pos += 4 + subfield_len;
    }
    false
}

/// True if `data` contains more than one gzip member (pigz / concatenated gzip style).
///
/// Uses conservative heuristics to reject magic-byte false positives embedded
/// in compressed data: checks preceding ISIZE plausibility, reserved flag bits,
/// mtime range, XFL, and OS byte.
pub(crate) fn is_likely_multi_member(data: &[u8]) -> bool {
    if data.len() < 36 {
        return false;
    }
    let header_size = parse_gzip_header_size(data).unwrap_or(10);
    const GZIP_MAGIC: &[u8] = &[0x1f, 0x8b, 0x08];
    let finder = memmem::Finder::new(GZIP_MAGIC);
    let mut pos = header_size + 1;
    while let Some(offset) = finder.find(&data[pos..]) {
        let header_pos = pos + offset;
        if header_pos + 10 > data.len() {
            break;
        }
        if header_pos < 18 {
            pos = header_pos + 1;
            continue;
        }
        let preceding_isize = u32::from_le_bytes([
            data[header_pos - 4],
            data[header_pos - 3],
            data[header_pos - 2],
            data[header_pos - 1],
        ]);
        if preceding_isize == 0 || preceding_isize > 1_073_741_824 {
            pos = header_pos + 1;
            continue;
        }
        let flags = data[header_pos + 3];
        if flags & 0xE0 != 0 {
            pos = header_pos + 1;
            continue;
        }
        let mtime = u32::from_le_bytes([
            data[header_pos + 4],
            data[header_pos + 5],
            data[header_pos + 6],
            data[header_pos + 7],
        ]);
        if mtime != 0 && mtime > 4_102_444_800 {
            pos = header_pos + 1;
            continue;
        }
        let xfl = data[header_pos + 8];
        if xfl != 0 && xfl != 2 && xfl != 4 {
            pos = header_pos + 1;
            continue;
        }
        let os = data[header_pos + 9];
        if os > 13 && os != 255 {
            pos = header_pos + 1;
            continue;
        }
        return true;
    }
    false
}

/// Parse gzip header size (variable due to FEXTRA, FNAME, FCOMMENT, FHCRC).
pub(crate) fn parse_gzip_header_size(data: &[u8]) -> Option<usize> {
    if data.len() < 10 {
        return None;
    }
    if data[0] != 0x1f || data[1] != 0x8b || data[2] != 0x08 {
        return None;
    }
    let flags = data[3];
    let mut pos = 10;
    if flags & 0x04 != 0 {
        if pos + 2 > data.len() {
            return None;
        }
        let xlen = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2 + xlen;
    }
    if flags & 0x08 != 0 {
        while pos < data.len() && data[pos] != 0 {
            pos += 1;
        }
        pos += 1;
    }
    if flags & 0x10 != 0 {
        while pos < data.len() && data[pos] != 0 {
            pos += 1;
        }
        pos += 1;
    }
    if flags & 0x02 != 0 {
        pos += 2;
    }
    Some(pos)
}

/// Read the ISIZE field from the gzip trailer (last 4 bytes), mod 2^32 per RFC 1952.
#[inline]
pub(crate) fn read_gzip_isize(data: &[u8]) -> Option<u32> {
    if data.len() < 18 {
        return None;
    }
    let b = &data[data.len() - 4..];
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// Extract MTIME from a gzip header (bytes 4–7).
pub(crate) fn extract_gzip_mtime(data: &[u8]) -> Option<u32> {
    if data.len() < 10 || data[0] != 0x1f || data[1] != 0x8b {
        return None;
    }
    Some(u32::from_le_bytes([data[4], data[5], data[6], data[7]]))
}

/// Extract the original filename (FNAME field) from a gzip header.
pub(crate) fn extract_gzip_fname(data: &[u8]) -> Option<String> {
    if data.len() < 10 || data[0] != 0x1f || data[1] != 0x8b || data[2] != 0x08 {
        return None;
    }
    let flags = data[3];
    if flags & 0x08 == 0 {
        return None;
    }
    let mut pos = 10;
    if flags & 0x04 != 0 {
        if pos + 2 > data.len() {
            return None;
        }
        let xlen = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2 + xlen;
    }
    let start = pos;
    while pos < data.len() && data[pos] != 0 {
        pos += 1;
    }
    if pos >= data.len() {
        return None;
    }
    String::from_utf8(data[start..pos].to_vec()).ok()
}
