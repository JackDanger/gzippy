//! Zopfli-based gzip encoder
//!
//! Supports both single-threaded and parallel compression:
//! - T1: reads all data, uses DEFLATE format + manual gzip header for metadata preservation
//! - T>1: splits into blocks, compresses each in parallel with gzip format

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

use crate::backends::zopfli_compress::{compress_deflate, compress_gzip, ZopfliTuning};
use crate::compress::parallel::GzipHeaderInfo;
use crate::infra::scheduler::compress_parallel_independent;

/// Zopfli-based parallel gzip encoder
pub struct ZopfliGzEncoder {
    thread_count: usize,
    block_size: usize,
    tuning: ZopfliTuning,
    header_info: GzipHeaderInfo,
}

impl ZopfliGzEncoder {
    pub fn new(thread_count: usize, block_size: usize, tuning: ZopfliTuning) -> Self {
        Self {
            thread_count,
            block_size,
            tuning,
            header_info: GzipHeaderInfo::default(),
        }
    }

    pub fn set_header_info(&mut self, info: GzipHeaderInfo) {
        self.header_info = info;
    }

    /// Compress from a reader with zopfli
    pub fn compress<R: Read, W: Write + Send>(&self, mut reader: R, writer: W) -> io::Result<u64> {
        let mut data = Vec::new();
        let bytes_read = reader.read_to_end(&mut data)? as u64;
        self.compress_buffer(&data, writer)?;
        Ok(bytes_read)
    }

    /// Compress from a file with zopfli (uses mmap for large files)
    #[allow(dead_code)]
    pub fn compress_file<P: AsRef<Path>, W: Write + Send>(
        &self,
        path: P,
        writer: W,
    ) -> io::Result<u64> {
        let file = File::open(path)?;
        let file_size = file.metadata()?.len();

        // For reasonably-sized files, read into memory
        // For very large files, still read (zopfli is so slow that I/O is not the bottleneck)
        let mut data = Vec::with_capacity(file_size as usize);
        std::io::BufReader::new(file).read_to_end(&mut data)?;
        self.compress_buffer(&data, writer)?;
        Ok(data.len() as u64)
    }

    /// Compress already-loaded data
    fn compress_buffer<W: Write + Send>(&self, data: &[u8], writer: W) -> io::Result<()> {
        if data.is_empty() {
            // Write empty gzip file
            return self.write_empty_gzip(writer);
        }

        if self.thread_count == 1 || data.len() <= self.block_size {
            // Single-threaded: use DEFLATE format + manual gzip header for full metadata
            self.compress_single(data, writer)
        } else {
            // Multi-threaded: split into blocks, compress each with gzip format
            self.compress_parallel(data, writer)
        }
    }

    /// Single-threaded compression with full metadata preservation
    fn compress_single<W: Write>(&self, data: &[u8], mut writer: W) -> io::Result<()> {
        // Compress to raw DEFLATE format. `thread_budget = 0` lets the
        // pure-Rust port use intra-block parallelism freely — there's no
        // outer pool to contend with on this path.
        let mut tuning = self.tuning.clone();
        tuning.thread_budget = 0;
        let deflate_data = compress_deflate(data, &tuning);

        // Write gzip header manually
        self.write_gzip_header(&mut writer)?;

        // Write DEFLATE data
        writer.write_all(&deflate_data)?;

        // Write CRC32 + ISIZE trailer
        let crc = crc32fast::hash(data);
        let size = data.len() as u32;
        writer.write_all(&crc.to_le_bytes())?;
        writer.write_all(&size.to_le_bytes())?;

        Ok(())
    }

    /// Multi-threaded compression using parallel scheduler
    fn compress_parallel<W: Write + Send>(&self, data: &[u8], writer: W) -> io::Result<()> {
        // The scheduler already runs one zopfli compression per CPU; force
        // the inner port to serial so we don't spawn N×N threads on an
        // N-CPU box (the regression that killed the Step 28a experiment;
        // see plan.md).
        let mut tuning = self.tuning.clone();
        tuning.thread_budget = 1;
        let _writer = compress_parallel_independent(
            data,
            self.block_size,
            self.thread_count,
            writer,
            move |block, output| {
                // Compress each block as complete gzip member
                *output = compress_gzip(block, &tuning);
            },
        )?;

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
