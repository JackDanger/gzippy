//! Pipelined compression with dictionary sharing for maximum compression
//!
//! At high compression levels (L7-L9), users expect maximum compression ratio.
//! This module implements pigz-style pipelined compression where each block
//! uses the previous block's data as a dictionary.
//!
//! Trade-off:
//! - Better compression (matches pigz output size)
//! - Sequential decompression only (like pigz)
//!
//! This is used when compression_level >= 7 and threads > 1.

use crate::parallel_compress::adjust_compression_level;
use flate2::write::GzEncoder;
use flate2::{Compress, Compression, FlushCompress, Status};
use std::io::{self, Read, Write};
use std::path::Path;

/// Block size for pipelined compression (same as parallel for consistency)
const BLOCK_SIZE: usize = 64 * 1024;

/// Dictionary size (DEFLATE maximum is 32KB)
const DICT_SIZE: usize = 32 * 1024;

/// Pipelined gzip compression with dictionary sharing
///
/// This produces a single gzip member with dictionary sharing between
/// internal blocks, achieving compression ratios comparable to pigz.
///
/// The output is gzip-compatible but requires sequential decompression.
pub struct PipelinedGzEncoder {
    compression_level: u32,
    _num_threads: usize, // Reserved for future parallel pipelining
}

impl PipelinedGzEncoder {
    pub fn new(compression_level: u32, num_threads: usize) -> Self {
        Self {
            compression_level,
            _num_threads: num_threads,
        }
    }

    /// Compress data with dictionary sharing
    pub fn compress<R: Read, W: Write>(&self, mut reader: R, writer: W) -> io::Result<u64> {
        // Read all input data
        let mut input_data = Vec::new();
        let bytes_read = reader.read_to_end(&mut input_data)? as u64;

        if input_data.is_empty() {
            // Write empty gzip file
            let encoder = GzEncoder::new(writer, Compression::new(self.compression_level));
            encoder.finish()?;
            return Ok(0);
        }

        self.compress_with_dict(&input_data, writer)?;
        Ok(bytes_read)
    }

    /// Compress file using memory-mapped I/O with dictionary sharing
    pub fn compress_file<P: AsRef<Path>, W: Write>(&self, path: P, writer: W) -> io::Result<u64> {
        use memmap2::Mmap;
        use std::fs::File;

        let file = File::open(path.as_ref())?;
        let file_len = file.metadata()?.len() as usize;

        if file_len == 0 {
            let encoder = GzEncoder::new(writer, Compression::new(self.compression_level));
            encoder.finish()?;
            return Ok(0);
        }

        // Memory-map the file for zero-copy access
        let mmap = unsafe { Mmap::map(&file)? };

        self.compress_with_dict(&mmap, writer)?;
        Ok(file_len as u64)
    }

    /// Core compression with dictionary sharing
    ///
    /// This produces a SINGLE gzip stream where each block uses the
    /// previous block's data as dictionary for better compression.
    fn compress_with_dict<W: Write>(&self, data: &[u8], mut writer: W) -> io::Result<()> {
        use crc32fast::Hasher;

        let level = adjust_compression_level(self.compression_level);

        // Write gzip header (standard format)
        let header = [
            0x1f, 0x8b, // Magic
            0x08, // Compression method (deflate)
            0x00, // Flags (none)
            0, 0, 0, 0, // MTIME (zero)
            0x00, // XFL
            0xff, // OS (unknown)
        ];
        writer.write_all(&header)?;

        // Create compressor
        let mut compress = Compress::new(Compression::new(level), false);
        let mut output_buf = vec![0u8; BLOCK_SIZE + 1024];
        let mut crc_hasher = Hasher::new();

        // Process blocks with dictionary sharing
        let blocks: Vec<&[u8]> = data.chunks(BLOCK_SIZE).collect();

        for (i, block) in blocks.iter().enumerate() {
            // Update CRC
            crc_hasher.update(block);

            // Set dictionary from previous block (last 32KB)
            if i > 0 {
                let prev_block = blocks[i - 1];
                let dict = if prev_block.len() > DICT_SIZE {
                    &prev_block[prev_block.len() - DICT_SIZE..]
                } else {
                    prev_block
                };
                // set_dictionary returns Ok(()) or error
                compress.set_dictionary(dict).map_err(|e| {
                    io::Error::new(io::ErrorKind::Other, format!("set_dictionary failed: {}", e))
                })?;
            }

            // Determine flush type
            let flush = if i == blocks.len() - 1 {
                FlushCompress::Finish
            } else {
                // Z_SYNC_FLUSH: ends block, allows dictionary change
                FlushCompress::Sync
            };

            // Compress this block
            let mut input_consumed = 0;
            let mut block_data = *block;

            loop {
                let before_in = compress.total_in();
                let before_out = compress.total_out();

                let status = compress.compress(block_data, &mut output_buf, flush)?;

                let consumed = (compress.total_in() - before_in) as usize;
                let produced = (compress.total_out() - before_out) as usize;

                // Write produced output
                if produced > 0 {
                    writer.write_all(&output_buf[..produced])?;
                }

                input_consumed += consumed;
                block_data = &block_data[consumed..];

                match status {
                    Status::Ok => {
                        if block_data.is_empty() && flush != FlushCompress::Finish {
                            break;
                        }
                    }
                    Status::BufError => {
                        // Need more output space, continue
                        if produced == 0 && consumed == 0 {
                            break; // Stuck, shouldn't happen
                        }
                    }
                    Status::StreamEnd => break,
                }
            }
        }

        // Write gzip trailer: CRC32 + ISIZE
        let crc = crc_hasher.finalize();
        let isize = (data.len() as u32).to_le_bytes();
        writer.write_all(&crc.to_le_bytes())?;
        writer.write_all(&isize)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::read::GzDecoder;
    use std::io::Read;

    #[test]
    fn test_pipelined_compress() {
        let data = b"Hello, world! ".repeat(10000);
        let mut output = Vec::new();

        let encoder = PipelinedGzEncoder::new(9, 4);
        encoder
            .compress(std::io::Cursor::new(&data), &mut output)
            .unwrap();

        // Verify we can decompress
        let mut decoder = GzDecoder::new(&output[..]);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed).unwrap();

        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_pipelined_vs_parallel_size() {
        use crate::parallel_compress::ParallelGzEncoder;

        let data = b"The quick brown fox jumps over the lazy dog. ".repeat(5000);

        // Pipelined (with dictionary)
        let mut pipelined_output = Vec::new();
        let pipelined = PipelinedGzEncoder::new(9, 4);
        pipelined
            .compress(std::io::Cursor::new(&data), &mut pipelined_output)
            .unwrap();

        // Parallel (independent blocks)
        let mut parallel_output = Vec::new();
        let parallel = ParallelGzEncoder::new(9, 4);
        parallel
            .compress(std::io::Cursor::new(&data), &mut parallel_output)
            .unwrap();

        // Pipelined should be smaller (dictionary sharing)
        println!(
            "Pipelined: {} bytes, Parallel: {} bytes",
            pipelined_output.len(),
            parallel_output.len()
        );
        assert!(
            pipelined_output.len() <= parallel_output.len(),
            "Pipelined should produce smaller output"
        );
    }
}
