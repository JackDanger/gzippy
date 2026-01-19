//! Block Index Caching for Repeated Decompression
//!
//! This module implements rapidgzip's index caching feature, which allows:
//! 1. Saving block boundaries and windows to disk
//! 2. Loading pre-computed indexes for instant parallel decompression
//! 3. Random access within gzip files
//!
//! ## File Format
//!
//! The index file stores:
//! - Magic number and version
//! - Number of blocks
//! - For each block:
//!   - Compressed bit offset
//!   - Uncompressed byte offset
//!   - 32KB window (compressed with zlib)
//!
//! ## Usage
//!
//! ```ignore
//! // First decompression: build and save index
//! let index = GzipIndex::build(&compressed_data)?;
//! index.save("file.gz.gzidx")?;
//!
//! // Subsequent decompressions: load index for instant parallelism
//! let index = GzipIndex::load("file.gz.gzidx")?;
//! decompress_with_index(&compressed_data, &index, writer)?;
//! ```

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(unused_assignments)]
#![allow(clippy::manual_is_multiple_of)]

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::{Arc, RwLock};

use crate::marker_decode::WINDOW_SIZE;

/// Magic number for gzippy index files
const INDEX_MAGIC: [u8; 4] = *b"GZIX";

/// Index file version
const INDEX_VERSION: u32 = 1;

/// A cached block boundary
#[derive(Clone, Debug)]
pub struct BlockEntry {
    /// Bit offset in compressed stream
    pub compressed_bit_offset: u64,
    /// Byte offset in uncompressed stream
    pub uncompressed_offset: u64,
    /// Compressed window (zlib-compressed 32KB)
    pub window: Vec<u8>,
}

/// Gzip index for parallel decompression
#[derive(Clone, Debug, Default)]
pub struct GzipIndex {
    /// Block entries sorted by compressed offset
    pub blocks: Vec<BlockEntry>,
    /// Total uncompressed size (if known)
    pub uncompressed_size: Option<u64>,
    /// Original file path
    pub source_path: Option<String>,
}

impl GzipIndex {
    /// Create a new empty index
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a block to the index
    pub fn add_block(&mut self, compressed_bit: u64, uncompressed_byte: u64, window: &[u8]) {
        // Compress the window
        let compressed_window = compress_window(window);

        self.blocks.push(BlockEntry {
            compressed_bit_offset: compressed_bit,
            uncompressed_offset: uncompressed_byte,
            window: compressed_window,
        });
    }

    /// Get the block containing a given uncompressed offset
    pub fn block_for_offset(&self, uncompressed_offset: u64) -> Option<&BlockEntry> {
        // Binary search for the block
        let idx = self
            .blocks
            .binary_search_by(|b| b.uncompressed_offset.cmp(&uncompressed_offset))
            .unwrap_or_else(|i| i.saturating_sub(1));

        self.blocks.get(idx)
    }

    /// Get all blocks in a range
    pub fn blocks_in_range(&self, start: u64, end: u64) -> &[BlockEntry] {
        let start_idx = self
            .blocks
            .binary_search_by(|b| b.uncompressed_offset.cmp(&start))
            .unwrap_or_else(|i| i.saturating_sub(1));

        let end_idx = self
            .blocks
            .binary_search_by(|b| b.uncompressed_offset.cmp(&end))
            .unwrap_or_else(|i| i);

        &self.blocks[start_idx..end_idx.min(self.blocks.len())]
    }

    /// Save index to a file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write header
        writer.write_all(&INDEX_MAGIC)?;
        writer.write_all(&INDEX_VERSION.to_le_bytes())?;
        writer.write_all(&(self.blocks.len() as u64).to_le_bytes())?;
        writer.write_all(&self.uncompressed_size.unwrap_or(0).to_le_bytes())?;

        // Write blocks
        for block in &self.blocks {
            writer.write_all(&block.compressed_bit_offset.to_le_bytes())?;
            writer.write_all(&block.uncompressed_offset.to_le_bytes())?;
            writer.write_all(&(block.window.len() as u32).to_le_bytes())?;
            writer.write_all(&block.window)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Load index from a file
    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read and verify header
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != INDEX_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid index magic",
            ));
        }

        let mut version_buf = [0u8; 4];
        reader.read_exact(&mut version_buf)?;
        let version = u32::from_le_bytes(version_buf);
        if version != INDEX_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported index version: {}", version),
            ));
        }

        let mut num_blocks_buf = [0u8; 8];
        reader.read_exact(&mut num_blocks_buf)?;
        let num_blocks = u64::from_le_bytes(num_blocks_buf) as usize;

        let mut uncompressed_size_buf = [0u8; 8];
        reader.read_exact(&mut uncompressed_size_buf)?;
        let uncompressed_size = u64::from_le_bytes(uncompressed_size_buf);

        // Read blocks
        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            let mut offset_buf = [0u8; 8];
            reader.read_exact(&mut offset_buf)?;
            let compressed_bit_offset = u64::from_le_bytes(offset_buf);

            reader.read_exact(&mut offset_buf)?;
            let uncompressed_offset = u64::from_le_bytes(offset_buf);

            let mut window_len_buf = [0u8; 4];
            reader.read_exact(&mut window_len_buf)?;
            let window_len = u32::from_le_bytes(window_len_buf) as usize;

            let mut window = vec![0u8; window_len];
            reader.read_exact(&mut window)?;

            blocks.push(BlockEntry {
                compressed_bit_offset,
                uncompressed_offset,
                window,
            });
        }

        Ok(Self {
            blocks,
            uncompressed_size: if uncompressed_size > 0 {
                Some(uncompressed_size)
            } else {
                None
            },
            source_path: None,
        })
    }

    /// Get decompressed window for a block
    pub fn get_window(&self, block_idx: usize) -> Option<Vec<u8>> {
        self.blocks
            .get(block_idx)
            .map(|b| decompress_window(&b.window))
    }
}

/// Compress a window using flate2
fn compress_window(data: &[u8]) -> Vec<u8> {
    use flate2::write::ZlibEncoder;
    use flate2::Compression;

    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
    encoder.write_all(data).unwrap_or(());
    encoder.finish().unwrap_or_default()
}

/// Decompress a window using flate2
fn decompress_window(data: &[u8]) -> Vec<u8> {
    use flate2::read::ZlibDecoder;

    let mut decoder = ZlibDecoder::new(data);
    let mut output = Vec::with_capacity(WINDOW_SIZE);
    decoder.read_to_end(&mut output).unwrap_or(0);
    output
}

/// Global index cache (thread-safe)
use std::sync::OnceLock;
static INDEX_CACHE: OnceLock<RwLock<HashMap<String, Arc<GzipIndex>>>> = OnceLock::new();

fn get_cache() -> &'static RwLock<HashMap<String, Arc<GzipIndex>>> {
    INDEX_CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Get a cached index for a file
pub fn get_cached_index(path: &str) -> Option<Arc<GzipIndex>> {
    get_cache().read().ok()?.get(path).cloned()
}

/// Cache an index for a file
pub fn cache_index(path: String, index: GzipIndex) {
    if let Ok(mut cache) = get_cache().write() {
        cache.insert(path, Arc::new(index));
    }
}

/// Try to load an index from the standard location (.gzidx extension)
pub fn try_load_index(gzip_path: &str) -> Option<Arc<GzipIndex>> {
    // Check memory cache first
    if let Some(index) = get_cached_index(gzip_path) {
        return Some(index);
    }

    // Try loading from disk
    let index_path = format!("{}.gzidx", gzip_path);
    if let Ok(index) = GzipIndex::load(&index_path) {
        let arc_index = Arc::new(index);
        if let Ok(mut cache) = get_cache().write() {
            cache.insert(gzip_path.to_string(), arc_index.clone());
        }
        return Some(arc_index);
    }

    None
}

/// Build an index by decompressing a file
pub fn build_index(data: &[u8], chunk_size: usize) -> io::Result<GzipIndex> {
    use crate::marker_decode::{skip_gzip_header, MarkerDecoder};

    let header_size = skip_gzip_header(data)?;
    let deflate_data = &data[header_size..data.len().saturating_sub(8)];

    let mut index = GzipIndex::new();
    let mut uncompressed_offset = 0u64;
    let mut bit_offset = 0usize;
    let mut window = vec![0u8; WINDOW_SIZE];

    // First block (empty window)
    index.add_block(0, 0, &[]);

    // Decode and record block boundaries
    let mut decoder = MarkerDecoder::with_window(deflate_data, 0, &window);
    match decoder.decode_all() {
        Ok(()) => {
            uncompressed_offset = decoder.output().len() as u64;
            window = decoder.final_window();
        }
        Err(e) => return Err(e),
    }

    index.uncompressed_size = Some(uncompressed_offset);

    // Add final block with window
    if uncompressed_offset > chunk_size as u64 {
        index.add_block(decoder.bit_position() as u64, uncompressed_offset, &window);
    }

    Ok(index)
}

/// Decompress using a pre-built index
pub fn decompress_with_index<W: io::Write + Send>(
    data: &[u8],
    index: &GzipIndex,
    writer: &mut W,
    num_threads: usize,
) -> io::Result<u64> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    if index.blocks.is_empty() {
        return crate::marker_decode::decompress_sequential(data, writer);
    }

    let header_size = crate::marker_decode::skip_gzip_header(data)?;
    let deflate_data = &data[header_size..data.len().saturating_sub(8)];

    // Parallel decode using index
    let outputs: Vec<std::sync::Mutex<Vec<u8>>> = index
        .blocks
        .iter()
        .map(|_| std::sync::Mutex::new(Vec::new()))
        .collect();

    let next_block = AtomicUsize::new(0);

    std::thread::scope(|scope| {
        for _ in 0..num_threads.min(index.blocks.len()) {
            let outputs_ref = &outputs;
            let blocks_ref = &index.blocks;
            let next_ref = &next_block;

            scope.spawn(move || {
                loop {
                    let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                    if idx >= blocks_ref.len() {
                        break;
                    }

                    let block = &blocks_ref[idx];
                    let window = decompress_window(&block.window);

                    let bit_offset = block.compressed_bit_offset as usize;
                    let byte_offset = bit_offset / 8;

                    if byte_offset >= deflate_data.len() {
                        continue;
                    }

                    // Determine how much to decode
                    let end_offset = if idx + 1 < blocks_ref.len() {
                        blocks_ref[idx + 1].uncompressed_offset - block.uncompressed_offset
                    } else {
                        // Last block: decode to end
                        u64::MAX
                    };

                    // Use ISA-L if available
                    #[cfg(feature = "isal")]
                    if bit_offset % 8 == 0 && !window.is_empty() {
                        if let Ok(output) = crate::marker_decode::decode_with_isal(
                            deflate_data,
                            bit_offset,
                            &window,
                            end_offset as usize,
                        ) {
                            *outputs_ref[idx].lock().unwrap() = output;
                            continue;
                        }
                    }

                    // Fallback: use marker decoder
                    let chunk_data = &deflate_data[byte_offset..];
                    let mut decoder =
                        crate::marker_decode::MarkerDecoder::with_window(chunk_data, 0, &window);
                    if decoder.decode_all().is_ok() {
                        *outputs_ref[idx].lock().unwrap() =
                            crate::marker_decode::to_u8(decoder.output());
                    }
                }
            });
        }
    });

    // Write outputs in order
    let mut total = 0u64;
    for output_mutex in &outputs {
        let output = output_mutex.lock().unwrap();
        writer.write_all(&output)?;
        total += output.len() as u64;
    }

    writer.flush()?;
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_save_load() {
        let mut index = GzipIndex::new();
        index.add_block(0, 0, &[]);
        index.add_block(1024 * 8, 10000, &[1, 2, 3, 4]);
        index.uncompressed_size = Some(20000);

        let temp_path = "/tmp/test_index.gzidx";
        index.save(temp_path).unwrap();

        let loaded = GzipIndex::load(temp_path).unwrap();
        assert_eq!(loaded.blocks.len(), 2);
        assert_eq!(loaded.uncompressed_size, Some(20000));
    }

    #[test]
    fn test_window_compression() {
        let original: Vec<u8> = (0..WINDOW_SIZE).map(|i| (i % 256) as u8).collect();
        let compressed = compress_window(&original);
        let decompressed = decompress_window(&compressed);
        assert_eq!(original, decompressed);
    }
}
