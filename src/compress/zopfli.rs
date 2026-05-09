//! Zopfli-based gzip encoder
//!
//! **Single-member output, always.** The previous multi-member parallel
//! path emitted one gzip member per CPU thread, each with its own
//! Huffman tree; against the C zopfli binary that cost +2.0–2.3% on the
//! corpus audit (plan.md Phase 11). Under the "ratio is sacred" rule
//! that's a P0, so the L11 path now always produces a single member
//! whose deflate payload is bit-identical to C zopfli's output.
//!
//! Multi-thread scaling on `--ultra` is reduced to *intra-block*
//! parallelism (Step 29's `std::thread::scope` inside `deflate_part`,
//! gated on `ZopfliOptions::thread_budget = 0`). Recovery of input-
//! level parallelism without ratio loss is plan.md Phase 15
//! (single-member parallel zopfli).

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

use crate::backends::zopfli_compress::{compress_deflate, ZopfliTuning};
use crate::compress::parallel::GzipHeaderInfo;

/// Zopfli-based gzip encoder. Always emits a single gzip member;
/// `thread_count` is advisory only (informational, not used to slice
/// input — see the module doc above).
pub struct ZopfliGzEncoder {
    tuning: ZopfliTuning,
    header_info: GzipHeaderInfo,
}

impl ZopfliGzEncoder {
    pub fn new(tuning: ZopfliTuning) -> Self {
        Self {
            tuning,
            header_info: GzipHeaderInfo::default(),
        }
    }

    pub fn set_header_info(&mut self, info: GzipHeaderInfo) {
        self.header_info = info;
    }

    /// Compress from a reader with zopfli
    pub fn compress<R: Read, W: Write>(&self, mut reader: R, writer: W) -> io::Result<u64> {
        let mut data = Vec::new();
        let bytes_read = reader.read_to_end(&mut data)? as u64;
        self.compress_buffer(&data, writer)?;
        Ok(bytes_read)
    }

    /// Compress from a file with zopfli (uses mmap for large files)
    #[allow(dead_code)]
    pub fn compress_file<P: AsRef<Path>, W: Write>(&self, path: P, writer: W) -> io::Result<u64> {
        let file = File::open(path)?;
        let file_size = file.metadata()?.len();

        // For reasonably-sized files, read into memory
        // For very large files, still read (zopfli is so slow that I/O is not the bottleneck)
        let mut data = Vec::with_capacity(file_size as usize);
        std::io::BufReader::new(file).read_to_end(&mut data)?;
        self.compress_buffer(&data, writer)?;
        Ok(data.len() as u64)
    }

    /// Compress already-loaded data. Always emits a single gzip member
    /// (see module doc for the ratio-correctness rationale).
    fn compress_buffer<W: Write>(&self, data: &[u8], writer: W) -> io::Result<()> {
        if data.is_empty() {
            return self.write_empty_gzip(writer);
        }
        self.compress_single(data, writer)
    }

    /// Compress to a single-member gzip with full metadata preservation.
    /// `thread_budget = 0` lets `deflate_part` use intra-block parallelism
    /// freely — no outer pool exists to contend with on this path.
    fn compress_single<W: Write>(&self, data: &[u8], mut writer: W) -> io::Result<()> {
        let mut tuning = self.tuning.clone();
        tuning.thread_budget = 0;
        let deflate_data = compress_deflate(data, &tuning);

        self.write_gzip_header(&mut writer)?;
        writer.write_all(&deflate_data)?;

        let crc = crc32fast::hash(data);
        let size = data.len() as u32;
        writer.write_all(&crc.to_le_bytes())?;
        writer.write_all(&size.to_le_bytes())?;

        Ok(())
    }

    /// Write a gzip header with metadata
    fn write_gzip_header<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // ID1, ID2: magic bytes
        writer.write_all(&[0x1f, 0x8b])?;
        // CM: compression method (8 = deflate)
        writer.write_all(&[8])?;

        // FLG: flags
        let mut flg = 0u8;
        if self.header_info.filename.is_some() {
            flg |= 0x08; // FNAME
        }
        if self.header_info.comment.is_some() {
            flg |= 0x10; // FCOMMENT
        }
        writer.write_all(&[flg])?;

        // MTIME: modification time (4 bytes LE)
        writer.write_all(&self.header_info.mtime.to_le_bytes())?;

        // XFL: extra flags (2 = maximum compression)
        writer.write_all(&[2])?;

        // OS: operating system (255 = unknown)
        writer.write_all(&[255])?;

        // Optional FNAME field
        if let Some(ref name) = self.header_info.filename {
            writer.write_all(name.as_bytes())?;
            writer.write_all(&[0])?; // null terminator
        }

        // Optional FCOMMENT field
        if let Some(ref comment) = self.header_info.comment {
            writer.write_all(comment.as_bytes())?;
            writer.write_all(&[0])?; // null terminator
        }

        Ok(())
    }

    /// Write an empty gzip file
    fn write_empty_gzip<W: Write>(&self, mut writer: W) -> io::Result<()> {
        // Magic bytes
        writer.write_all(&[0x1f, 0x8b])?;
        // CM
        writer.write_all(&[8])?;
        // FLG
        writer.write_all(&[0])?;
        // MTIME
        writer.write_all(&[0, 0, 0, 0])?;
        // XFL
        writer.write_all(&[0])?;
        // OS
        writer.write_all(&[255])?;
        // CRC32 of empty data: 0x00000000
        writer.write_all(&[0, 0, 0, 0])?;
        // ISIZE: 0
        writer.write_all(&[0, 0, 0, 0])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    /// Phase 11.1.A invariant: at L11 the output bytes must not depend
    /// on the user's `-pN` choice. The encoder no longer takes a
    /// thread_count, so the only way this could regress is by
    /// re-introducing an input-level slicing path; the test pins that
    /// closed.
    #[test]
    fn ultra_output_is_single_member_and_deterministic() {
        let data = std::fs::read("test_data/alice.txt").expect("test_data/alice.txt missing");
        let tuning = ZopfliTuning::default();

        let encode = || {
            let encoder = ZopfliGzEncoder::new(tuning.clone());
            let mut out = Vec::new();
            encoder
                .compress(std::io::Cursor::new(&data), &mut out)
                .unwrap();
            out
        };

        // Same input, same tuning → byte-identical gzip output.
        let a = encode();
        let b = encode();
        assert_eq!(a, b, "encoder is not deterministic");
        assert_eq!(&a[0..2], &[0x1f, 0x8b], "missing gzip magic");

        // Single-member: a single-member decoder must consume the entire
        // stream and produce the original input. Multi-member output
        // would either leave trailing bytes (subsequent members) or fail
        // the round-trip, depending on the decoder; flate2's
        // `GzDecoder` is single-member and lets us assert both.
        let mut decoded = Vec::new();
        flate2::read::GzDecoder::new(a.as_slice())
            .read_to_end(&mut decoded)
            .expect("single-member gunzip");
        assert_eq!(
            decoded, data,
            "single-member decode must yield the original input \
             — multi-member output would short-decode (Phase 11.1.A)"
        );
    }
}
