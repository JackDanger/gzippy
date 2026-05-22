#![allow(dead_code)] // vendor-faithful rapidgzip port; many items are pending consumer-port

//! Literal port of `rapidgzip::BlockMap`
//! (vendor/rapidgzip/.../core/BlockMap.hpp).
//!
//! Maintains a (compressed-bit-offset, decoded-byte-offset) index for
//! every chunk-end the parallel decoder produces. Required for
//! random-access seek: given a decoded byte offset, locate the chunk
//! that contains it (by binary-searching the decoded offsets).
//!
//! Expects `push` to be called with monotonically-increasing
//! `encoded_block_offset` arguments (rapidgzip's contract).

use std::sync::Mutex;

/// Per-block lookup result returned by `find_data_offset` /
/// `get_encoded_offset`. Mirror of `BlockMap::BlockInfo`
/// (BlockMap.hpp:30-59).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BlockInfo {
    pub block_index: usize,
    pub encoded_offset_in_bits: usize,
    pub encoded_size_in_bits: usize,
    pub decoded_offset_in_bytes: usize,
    pub decoded_size_in_bytes: usize,
}

impl BlockInfo {
    /// True iff `data_offset` lies inside this block's decoded range.
    pub fn contains(&self, data_offset: usize) -> bool {
        self.decoded_offset_in_bytes <= data_offset
            && data_offset < self.decoded_offset_in_bytes + self.decoded_size_in_bytes
    }
}

/// Internal storage: pairs of (encoded_offset_bits, decoded_offset_bytes).
type BlockOffsets = Vec<(usize, usize)>;

/// Thread-safe block-offset index. Mirror of rapidgzip's `BlockMap`
/// (BlockMap.hpp:27-296).
pub struct BlockMap {
    inner: Mutex<Inner>,
}

struct Inner {
    block_to_data_offsets: BlockOffsets,
    eos_blocks: Vec<usize>,
    finalized: bool,
    last_block_encoded_size: usize,
    last_block_decoded_size: usize,
}

impl Default for BlockMap {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockMap {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                block_to_data_offsets: Vec::new(),
                eos_blocks: Vec::new(),
                finalized: false,
                last_block_encoded_size: 0,
                last_block_decoded_size: 0,
            }),
        }
    }

    /// Push a new chunk's boundaries. Returns the decoded byte offset
    /// at which this block's data begins. Panics on out-of-order
    /// `encoded_block_offset` (mirrors rapidgzip's
    /// `std::invalid_argument` throw).
    pub fn push(
        &self,
        encoded_block_offset: usize,
        encoded_size: usize,
        decoded_size: usize,
    ) -> usize {
        let mut g = self.inner.lock().unwrap();
        if g.finalized {
            panic!("BlockMap: may not insert into finalized block map");
        }
        let decoded_offset: Option<usize> = if g.block_to_data_offsets.is_empty() {
            Some(0)
        } else if encoded_block_offset > g.block_to_data_offsets.last().unwrap().0 {
            Some(g.block_to_data_offsets.last().unwrap().1 + g.last_block_decoded_size)
        } else {
            None
        };
        if let Some(off) = decoded_offset {
            g.block_to_data_offsets.push((encoded_block_offset, off));
            if decoded_size == 0 {
                g.eos_blocks.push(encoded_block_offset);
            }
            g.last_block_decoded_size = decoded_size;
            g.last_block_encoded_size = encoded_size;
            return off;
        }
        // Duplicate or older offset — verify consistency, otherwise panic.
        let pos = g
            .block_to_data_offsets
            .binary_search_by_key(&encoded_block_offset, |&(e, _)| e);
        match pos {
            Ok(i) => {
                let match_decoded = g.block_to_data_offsets[i].1;
                if i + 1 >= g.block_to_data_offsets.len() {
                    // Same encoded start as the open tail block — update
                    // sizes instead of panicking. Happens when a spacing
                    // guess overshoots and the consumer resumes from
                    // `furthest_decoded_bit` (see chunk_fetcher).
                    g.last_block_encoded_size = encoded_size;
                    g.last_block_decoded_size = decoded_size;
                    return match_decoded;
                }
                let next_decoded = g.block_to_data_offsets[i + 1].1;
                let implied = next_decoded - match_decoded;
                if implied != decoded_size {
                    panic!("BlockMap: duplicate offset with inconsistent size");
                }
                match_decoded
            }
            Err(_) => panic!("BlockMap: inserted offsets must be strictly increasing"),
        }
    }

    /// Returns the block containing `data_offset`, or the last block if
    /// `data_offset` is past the end. Mirror of `findDataOffset`
    /// (BlockMap.hpp:126-145).
    pub fn find_data_offset(&self, data_offset: usize) -> Option<BlockInfo> {
        let g = self.inner.lock().unwrap();
        // Find the highest entry whose decoded_offset_in_bytes <= data_offset.
        let pos = g
            .block_to_data_offsets
            .binary_search_by(|&(_, d)| d.cmp(&data_offset));
        let idx = match pos {
            Ok(i) => i,
            Err(0) => return None,
            Err(i) => i - 1,
        };
        Some(get_locked(&g, idx))
    }

    /// Look up a chunk by its encoded bit offset. Mirror of
    /// `getEncodedOffset` (BlockMap.hpp:147-162).
    pub fn get_encoded_offset(&self, encoded_offset_in_bits: usize) -> Option<BlockInfo> {
        let g = self.inner.lock().unwrap();
        let pos = g
            .block_to_data_offsets
            .binary_search_by_key(&encoded_offset_in_bits, |&(e, _)| e);
        match pos {
            Ok(i) => Some(get_locked(&g, i)),
            Err(_) => None,
        }
    }

    pub fn data_block_count(&self) -> usize {
        let g = self.inner.lock().unwrap();
        g.block_to_data_offsets.len() - g.eos_blocks.len()
    }

    pub fn finalize(&self) {
        let mut g = self.inner.lock().unwrap();
        if g.finalized {
            return;
        }
        let last_enc = g.last_block_encoded_size;
        let last_dec = g.last_block_decoded_size;
        if g.block_to_data_offsets.is_empty() {
            g.block_to_data_offsets.push((last_enc, last_dec));
        } else if last_enc != 0 || last_dec != 0 {
            let (le, ld) = *g.block_to_data_offsets.last().unwrap();
            g.block_to_data_offsets.push((le + last_enc, ld + last_dec));
        }
        g.last_block_encoded_size = 0;
        g.last_block_decoded_size = 0;
        g.finalized = true;
    }

    pub fn finalized(&self) -> bool {
        self.inner.lock().unwrap().finalized
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().unwrap().block_to_data_offsets.is_empty()
    }

    pub fn back(&self) -> (usize, usize) {
        let g = self.inner.lock().unwrap();
        *g.block_to_data_offsets
            .last()
            .expect("BlockMap: empty back()")
    }

    pub fn block_offsets(&self) -> Vec<(usize, usize)> {
        let g = self.inner.lock().unwrap();
        g.block_to_data_offsets.clone()
    }
}

/// Insert every subchunk in `chunk` into `block_map` in stream order.
/// Mirror of the per-subchunk push loop inside rapidgzip's
/// `GzipChunkFetcher::appendSubchunksToIndexes`
/// (GzipChunkFetcher.hpp:371-375).
///
/// The chunk must already be finalized (so each subchunk's
/// `encoded_size_bits` is computed). Pushes happen in `subchunks`
/// order to satisfy BlockMap's monotonic-increasing-offset contract.
pub fn append_subchunks_to_block_map(
    block_map: &BlockMap,
    chunk: &crate::decompress::parallel::chunk_data::ChunkData,
) {
    for sc in &chunk.subchunks {
        block_map.push(
            sc.encoded_offset_bits,
            sc.encoded_size_bits,
            sc.decoded_size,
        );
    }
}

fn get_locked(g: &Inner, idx: usize) -> BlockInfo {
    let (enc, dec) = g.block_to_data_offsets[idx];
    let mut info = BlockInfo {
        block_index: idx,
        encoded_offset_in_bits: enc,
        decoded_offset_in_bytes: dec,
        encoded_size_in_bits: 0,
        decoded_size_in_bytes: 0,
    };
    if idx + 1 < g.block_to_data_offsets.len() {
        let (next_enc, next_dec) = g.block_to_data_offsets[idx + 1];
        info.encoded_size_in_bits = next_enc - enc;
        info.decoded_size_in_bytes = next_dec - dec;
    } else {
        info.encoded_size_in_bits = g.last_block_encoded_size;
        info.decoded_size_in_bytes = g.last_block_decoded_size;
    }
    info
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_first_block_returns_zero_offset() {
        let m = BlockMap::new();
        assert_eq!(m.push(0, 800, 1000), 0);
    }

    #[test]
    fn push_accumulates_decoded_offsets() {
        let m = BlockMap::new();
        assert_eq!(m.push(0, 800, 1000), 0);
        assert_eq!(m.push(800, 600, 500), 1000);
        assert_eq!(m.push(1400, 200, 200), 1500);
    }

    #[test]
    fn find_data_offset_returns_correct_block() {
        let m = BlockMap::new();
        m.push(0, 800, 1000);
        m.push(800, 600, 500);
        m.push(1400, 200, 200);
        // Offset 500 → first block.
        let info = m.find_data_offset(500).unwrap();
        assert_eq!(info.block_index, 0);
        assert!(info.contains(500));
        // Offset 1200 → second block (decoded [1000, 1500)).
        let info = m.find_data_offset(1200).unwrap();
        assert_eq!(info.block_index, 1);
        assert!(info.contains(1200));
    }

    #[test]
    fn get_encoded_offset_returns_block_info() {
        let m = BlockMap::new();
        m.push(0, 800, 1000);
        m.push(800, 600, 500);
        let info = m.get_encoded_offset(800).unwrap();
        assert_eq!(info.block_index, 1);
        assert_eq!(info.decoded_offset_in_bytes, 1000);
    }

    #[test]
    fn append_subchunks_to_block_map_pushes_in_order() {
        use crate::decompress::parallel::chunk_data::{ChunkConfiguration, ChunkData};
        let cfg = ChunkConfiguration {
            split_chunk_size: 100,
            max_decoded_chunk_size: 10_000,
            crc32_enabled: true,
        };
        let mut chunk = ChunkData::new(0, cfg);
        chunk.append_clean(&[0u8; 50]);
        chunk.append_block_boundary(400);
        chunk.append_clean(&[0u8; 50]);
        chunk.finalize(800);
        let m = BlockMap::new();
        append_subchunks_to_block_map(&m, &chunk);
        // Two subchunks pushed. data_block_count = 2 minus any EOS (none here).
        assert_eq!(m.data_block_count(), 2);
        // First subchunk decoded_offset = 0; second's offset = 50.
        let first = m.get_encoded_offset(0).unwrap();
        assert_eq!(first.decoded_offset_in_bytes, 0);
        let second = m.get_encoded_offset(400).unwrap();
        assert_eq!(second.decoded_offset_in_bytes, 50);
    }

    #[test]
    fn finalize_caps_last_block() {
        let m = BlockMap::new();
        m.push(0, 800, 1000);
        m.push(800, 600, 500);
        m.finalize();
        assert!(m.finalized());
        // Last entry's decoded_offset reflects the cumulative end.
        let last = m.back();
        assert_eq!(last.1, 1500); // 1000 + 500
    }
}
