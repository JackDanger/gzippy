//! Shared corpus loader for `inflate_block.rs` (Criterion) and
//! `inflate_block_iai.rs` (iai-callgrind, Phase 1.5).
//!
//! Reads the GZBLK01 binary format produced by `tools/extract_blocks/`.
//! Format documented at `tools/extract_blocks/src/main.rs` preamble.

use std::fs;
use std::path::{Path, PathBuf};

pub struct CorpusBlock {
    pub file_name: String,
    pub block_idx: u32,
    pub bit_offset_in_compressed: u64,
    pub compressed_bit_len: u64,
    pub compressed: Vec<u8>,
    pub predecessor: Vec<u8>,
    pub decoded_expected: Vec<u8>,
    pub literal_cl: Vec<u8>,
    pub distance_cl: Vec<u8>,
    pub max_litlen_code_len: u8,
    pub max_dist_code_len: u8,
}

impl CorpusBlock {
    pub fn from_bytes(file_name: &str, data: &[u8]) -> Result<Self, String> {
        if data.len() < 64 {
            return Err(format!("{}: too short for header", file_name));
        }
        if &data[..8] != b"GZBLK01\0" {
            return Err(format!("{}: bad magic", file_name));
        }
        let read_u32 =
            |off: usize| -> u32 { u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) };
        let read_u64 =
            |off: usize| -> u64 { u64::from_le_bytes(data[off..off + 8].try_into().unwrap()) };
        let read_u16 =
            |off: usize| -> u16 { u16::from_le_bytes(data[off..off + 2].try_into().unwrap()) };
        let block_idx = read_u32(8);
        let _btype = read_u32(12);
        let bit_offset_in_compressed = read_u64(16);
        let compressed_bit_len = read_u64(24);
        let compressed_byte_len = read_u64(32) as usize;
        let decoded_len = read_u32(40) as usize;
        let predecessor_len = read_u32(44) as usize;
        let literal_code_count = read_u16(48) as usize;
        let distance_code_count = read_u16(50) as usize;
        let max_litlen_code_len = data[52];
        let max_dist_code_len = data[53];
        // bytes 54-55: marker_count_during_decode (skipped)
        // bytes 56-59: crc32_of_decoded (skipped — extractor verified)
        // bytes 60-63: reserved (skipped)

        let mut cursor = 64usize;
        let predecessor = data[cursor..cursor + predecessor_len].to_vec();
        cursor += predecessor_len;
        let compressed = data[cursor..cursor + compressed_byte_len].to_vec();
        cursor += compressed_byte_len;
        let decoded_expected = data[cursor..cursor + decoded_len].to_vec();
        cursor += decoded_len;
        let literal_cl = data[cursor..cursor + literal_code_count].to_vec();
        cursor += literal_code_count;
        let distance_cl = data[cursor..cursor + distance_code_count].to_vec();

        Ok(CorpusBlock {
            file_name: file_name.to_string(),
            block_idx,
            bit_offset_in_compressed,
            compressed_bit_len,
            compressed,
            predecessor,
            decoded_expected,
            literal_cl,
            distance_cl,
            max_litlen_code_len,
            max_dist_code_len,
        })
    }
}

pub fn load_corpus(dir: impl AsRef<Path>) -> Vec<CorpusBlock> {
    let dir = dir.as_ref();
    let mut blocks: Vec<CorpusBlock> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir({}): {e}", dir.display()))
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "bin").unwrap_or(false))
        .map(|e| {
            let path = e.path();
            let file_name = path.file_name().unwrap().to_string_lossy().to_string();
            let data = fs::read(&path).unwrap_or_else(|e| panic!("read({}): {e}", path.display()));
            CorpusBlock::from_bytes(&file_name, &data).unwrap_or_else(|e| panic!("{e}"))
        })
        .collect();
    blocks.sort_by(|a, b| a.file_name.cmp(&b.file_name));
    blocks
}

#[allow(dead_code)]
pub fn corpus_dir() -> PathBuf {
    // Walk up from CARGO_MANIFEST_DIR / OUT_DIR style search.
    // Cargo runs benches from the workspace root, so `corpus/silesia_blocks` is
    // a relative path that works at runtime.
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus/silesia_blocks")
}
