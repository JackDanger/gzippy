//! Literal port of `rapidgzip::SimpleRunLengthEncoding`
//! (vendor/rapidgzip/librapidarchive/src/core/SimpleRunLengthEncoding.hpp).
//!
//! A tiny RLE-over-varint encoder/decoder used by rapidgzip to compress
//! the per-chunk window data (recurring runs of bytes are common in
//! the 32 KiB sliding-window slices it stores in its index). The
//! format is Protobuf-style varints — 7 bits of payload per byte,
//! high bit set on continuation — so the encoding is byte-aligned and
//! cross-platform.
//!
//! Stream grammar (per the vendor doc-comment, SimpleRunLengthEncoding.hpp:86-110):
//! ```text
//! varint(0) varint(count) <literal bytes...>
//! varint(1) varint(count - 1)  // "repeat previous symbol count times"
//! ... repeated ...
//! ```
//! `varint(N)` > 1 is reserved for future "backward reference"
//! semantics and unsupported by both the vendor and this port.

#![allow(dead_code)]

/// Mirror of `rapidgzip::SimpleRunLengthEncoding::writeVarInt`
/// (SimpleRunLengthEncoding.hpp:26-33). Appends an unsigned varint
/// (Protobuf little-endian-LSB-first) to `target`.
///
/// Faithfully preserves the vendor's "value of `0` writes one zero
/// byte" do-while semantics — important for the decoder, which
/// expects every field to occupy at least one byte.
pub fn write_var_int(target: &mut Vec<u8>, mut value: u64) {
    loop {
        let next = value >> 7;
        let byte = if next > 0 {
            ((value & 0b0111_1111) as u8) | 0b1000_0000
        } else {
            (value & 0b0111_1111) as u8
        };
        target.push(byte);
        value = next;
        if value == 0 {
            return;
        }
    }
}

/// Mirror of `rapidgzip::SimpleRunLengthEncoding::readVarInt`
/// (SimpleRunLengthEncoding.hpp:42-61).
///
/// Returns `Some((value, n_bytes_read))` on success.  Returns `None`
/// if the input ends mid-varint or the high bit of byte 9 is set with
/// a payload > 1 (vendor: "the last varint byte contains the 64-th
/// bit and nothing more") — matching the C++ "return { 0, 0 }"
/// sentinel by way of Rust's `Option`.
pub fn read_var_int(source: &[u8], offset: usize) -> Option<(u64, u8)> {
    let mut value: u64 = 0;
    let mut n_bytes_read: u8 = 0;
    let mut i = offset;
    while i < source.len() {
        let byte = source[i];
        // Vendor: "The last varint byte contains the 64-th bit and
        // nothing more!"  Once we've read 9 bytes, the 10th may only
        // carry a 0 or 1.
        if n_bytes_read == 9 && byte > 1 {
            return None;
        }
        value += ((byte & 0b0111_1111) as u64) << (7 * n_bytes_read as u64);
        n_bytes_read += 1;
        if (byte & 0b1000_0000) == 0 {
            return Some((value, n_bytes_read));
        }
        i += 1;
    }
    // Ran out of bytes mid-varint.
    None
}

/// Mirror of `rapidgzip::SimpleRunLengthEncoding::findRun`
/// (SimpleRunLengthEncoding.hpp:68-83).
///
/// Returns `(offset, length)` where `offset` is the index of the
/// first byte of a run of `>= min_length` repeated bytes, and
/// `length` is the run length (0 if no such run exists).  When no
/// run is found, returns `(data.len(), 0)`.
pub fn find_run(data: &[u8], mut offset: usize, min_length: usize) -> (usize, usize) {
    while offset < data.len() {
        let mut length = 1;
        while offset + length < data.len() && data[offset + length] == data[offset] {
            length += 1;
        }
        if length >= min_length {
            return (offset, length);
        }
        offset += 1;
    }
    (offset, 0)
}

/// Errors surfaced by [`simple_run_length_decode`]. Mirrors the three
/// `std::domain_error` / `std::logic_error` throw sites in
/// `simpleRunLengthDecode` (SimpleRunLengthEncoding.hpp:187, 188, 199,
/// 209, 239, 244).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimpleRlError {
    /// "Partial varint read for operation type!" (line 187)
    PartialVarintOperation,
    /// "Backreference points past the file start!" (line 193).
    BackreferencePastStart,
    /// "Partial varint read for literal count/match length!" (line 199).
    PartialVarintLength,
    /// "Literal count points past the end!" (line 208).
    LiteralPastEnd,
    /// "Unsupported backward reference!" (line 239).
    UnsupportedBackreference,
    /// "Decompressed size does not match container!" (line 243).
    SizeMismatch,
}

/// Literal port of `rapidgzip::SimpleRunLengthEncoding::simpleRunLengthEncode`
/// (SimpleRunLengthEncoding.hpp:113-155).
///
/// Encodes `data` using literal-run + repeat-last-symbol opcodes.
/// `min_run` defaults to 6 in the vendor (line 130) — runs shorter
/// than that do not pay back the two extra varints needed to switch
/// out of literal mode.
pub fn simple_run_length_encode(data: &[u8]) -> Vec<u8> {
    let mut encoded = Vec::new();
    if data.is_empty() {
        return encoded;
    }

    let mut i: usize = 0;
    while i < data.len() {
        let (run_offset, run_length) = find_run(data, i, 6);

        // Literal opcode = 0, followed by literal count + bytes.
        write_var_int(&mut encoded, 0);
        let literal_count = (data.len() - i).min(run_offset + 1 - i);
        write_var_int(&mut encoded, literal_count as u64);
        encoded.extend_from_slice(&data[i..i + literal_count]);
        i += literal_count;

        if i >= data.len() {
            break;
        }
        if run_length <= 1 {
            continue;
        }

        // Repeat-last-symbol opcode = 1, followed by (run_length - 1).
        write_var_int(&mut encoded, 1);
        write_var_int(&mut encoded, (run_length - 1) as u64);
        i += run_length - 1;
    }

    encoded
}

/// Literal port of `rapidgzip::SimpleRunLengthEncoding::simpleRunLengthDecode`
/// (SimpleRunLengthEncoding.hpp:167-249).
///
/// Decodes into a buffer pre-sized to `decompressed_size`. Returns
/// the buffer on success or one of the [`SimpleRlError`] variants on
/// any malformed input or size mismatch. The vendor template version
/// supports any output container; this port targets `Vec<u8>` only
/// — the only instantiation actually used in the codebase.
pub fn simple_run_length_decode(
    data: &[u8],
    decompressed_size: usize,
) -> Result<Vec<u8>, SimpleRlError> {
    let mut output = vec![0u8; decompressed_size];
    let mut decoded_size: usize = 0;
    let mut i: usize = 0;

    while i < data.len() {
        let (backward_reference, n_bytes_read) =
            read_var_int(data, i).ok_or(SimpleRlError::PartialVarintOperation)?;
        i += n_bytes_read as usize;

        // Vendor: "Backreference points past the file start!".
        if backward_reference > decoded_size as u64 {
            return Err(SimpleRlError::BackreferencePastStart);
        }

        let (length, n_bytes_read2) =
            read_var_int(data, i).ok_or(SimpleRlError::PartialVarintLength)?;
        i += n_bytes_read2 as usize;
        let length = length as usize;

        match backward_reference {
            0 => {
                if i + length > data.len() {
                    return Err(SimpleRlError::LiteralPastEnd);
                }
                // Vendor caps the write to `output.size()` because the
                // template version may be called with an undersized
                // output container — preserve that bound.
                let mut j = 0;
                while j < length && decoded_size + j < output.len() {
                    output[decoded_size + j] = data[i + j];
                    j += 1;
                }
                i += length;
                decoded_size += length;
            }
            1 => {
                // Repeat the previous symbol. If the output is already
                // full, the vendor still increments `decoded_size` so
                // the post-loop size check sees the right count.
                let symbol = if decoded_size > 0 && decoded_size - 1 < output.len() {
                    output[decoded_size - 1]
                } else {
                    0u8
                };
                if symbol != 0 {
                    let mut j = 0;
                    while j < length && decoded_size + j < output.len() {
                        output[decoded_size + j] = symbol;
                        j += 1;
                    }
                }
                decoded_size += length;
            }
            _ => return Err(SimpleRlError::UnsupportedBackreference),
        }
    }

    if decoded_size != output.len() {
        return Err(SimpleRlError::SizeMismatch);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_zero_writes_single_byte() {
        // Vendor preserves this — `value = 0` enters the do-while
        // and writes one zero byte.
        let mut buf = Vec::new();
        write_var_int(&mut buf, 0);
        assert_eq!(buf, vec![0x00]);
    }

    #[test]
    fn varint_round_trip_small() {
        for v in [0u64, 1, 7, 127, 128, 255, 256, 16_384, 1_000_000] {
            let mut buf = Vec::new();
            write_var_int(&mut buf, v);
            let (decoded, n) = read_var_int(&buf, 0).unwrap();
            assert_eq!(decoded, v, "round-trip for {v}");
            assert_eq!(n as usize, buf.len(), "byte count for {v}");
        }
    }

    #[test]
    fn varint_round_trip_max_u64() {
        // 10-byte encoding of u64::MAX.
        let mut buf = Vec::new();
        write_var_int(&mut buf, u64::MAX);
        assert_eq!(buf.len(), 10);
        let (decoded, n) = read_var_int(&buf, 0).unwrap();
        assert_eq!(decoded, u64::MAX);
        assert_eq!(n, 10);
    }

    #[test]
    fn varint_truncated_returns_none() {
        // [0x80, 0x80] — continuation set, but no terminator.
        let truncated = [0x80u8, 0x80];
        assert_eq!(read_var_int(&truncated, 0), None);
    }

    #[test]
    fn varint_overlong_invalid() {
        // 10 bytes of 0xFF would have the 10th byte's payload > 1.
        let bad = [0xFF; 10];
        assert_eq!(read_var_int(&bad, 0), None);
    }

    #[test]
    fn find_run_returns_first_long_run() {
        let data = [1u8, 2, 3, 4, 5, 5, 5, 5, 5, 5, 6, 7];
        let (off, len) = find_run(&data, 0, 6);
        assert_eq!(off, 4);
        assert_eq!(len, 6);
    }

    #[test]
    fn find_run_skips_short_runs() {
        // Two short runs (3 each) before a 6-run.
        let data = [9u8, 9, 9, 8, 8, 8, 1, 2, 3, 7, 7, 7, 7, 7, 7];
        let (off, len) = find_run(&data, 0, 6);
        assert_eq!(off, 9);
        assert_eq!(len, 6);
    }

    #[test]
    fn find_run_no_run_returns_end() {
        let data = [1u8, 2, 3, 4];
        let (off, len) = find_run(&data, 0, 6);
        assert_eq!(off, data.len());
        assert_eq!(len, 0);
    }

    #[test]
    fn round_trip_empty() {
        let encoded = simple_run_length_encode(&[]);
        assert!(encoded.is_empty());
        let decoded = simple_run_length_decode(&encoded, 0).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn round_trip_no_run() {
        let data: Vec<u8> = (0u8..50).collect();
        let encoded = simple_run_length_encode(&data);
        let decoded = simple_run_length_decode(&encoded, data.len()).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn round_trip_with_runs() {
        let mut data = Vec::new();
        data.extend(b"hello ");
        data.extend(std::iter::repeat_n(b'A', 100));
        data.extend(b" world ");
        data.extend(std::iter::repeat_n(b'Z', 40));
        data.extend(b" end");

        let encoded = simple_run_length_encode(&data);
        // Encoding should be much smaller than the input.
        assert!(
            encoded.len() < data.len(),
            "expected RLE to compress; got {} -> {}",
            data.len(),
            encoded.len()
        );
        let decoded = simple_run_length_decode(&encoded, data.len()).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn round_trip_all_zeros() {
        let data = vec![0u8; 200];
        let encoded = simple_run_length_encode(&data);
        let decoded = simple_run_length_decode(&encoded, data.len()).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn decode_rejects_truncated() {
        // Operation byte missing.
        let r = simple_run_length_decode(&[0x80], 5);
        assert_eq!(r, Err(SimpleRlError::PartialVarintOperation));
    }

    #[test]
    fn decode_rejects_size_mismatch() {
        // Encode 10 bytes, ask decoder for 5.
        let data = vec![1u8; 10];
        let encoded = simple_run_length_encode(&data);
        let r = simple_run_length_decode(&encoded, 5);
        assert!(matches!(
            r,
            Err(SimpleRlError::SizeMismatch) | Err(SimpleRlError::LiteralPastEnd)
        ));
    }
}
