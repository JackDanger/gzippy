//! `--rsyncable` (audit item #3): the content-defined-chunk split + the
//! pure-Rust per-member compressor. Extracted from `compress::parallel` so the
//! one-concept-per-file rule holds; io.rs's two call sites were the only
//! consumers.

use std::io::{self, Write};

use super::gzip_header::GzipHeaderInfo;

/// Split data into rsyncable blocks using a rolling hash.
/// Block boundaries are determined by content, so small input changes
/// only affect nearby blocks — ideal for rsync workflows.
///
/// Uses a simple Adler-style rolling hash with a window of 8KB.
/// When the hash's low bits match a trigger mask, a block boundary is created.
/// Target block size is ~128KB (mask = 0x1FFFF = 128K-1).
pub fn split_rsyncable(data: &[u8]) -> Vec<&[u8]> {
    const WINDOW: usize = 8192;
    const MASK: u32 = 0x1FFFF; // ~128KB average block size
    const MIN_BLOCK: usize = 32 * 1024; // 32KB minimum
    const MAX_BLOCK: usize = 512 * 1024; // 512KB maximum

    if data.len() <= MIN_BLOCK {
        return vec![data];
    }

    let mut blocks = Vec::new();
    let mut block_start = 0;
    let mut hash: u32 = 0;

    for i in 0..data.len() {
        // Add new byte to hash
        hash = hash.wrapping_add(data[i] as u32);

        // Remove byte leaving the window
        if i >= WINDOW {
            hash = hash.wrapping_sub(data[i - WINDOW] as u32);
        }

        let block_len = i - block_start + 1;

        // Check for boundary: hash hits trigger AND block is big enough
        if block_len >= MIN_BLOCK && (hash & MASK == MASK || block_len >= MAX_BLOCK) {
            blocks.push(&data[block_start..block_start + block_len]);
            block_start += block_len;
        }
    }

    // Last block
    if block_start < data.len() {
        blocks.push(&data[block_start..]);
    }

    blocks
}

/// Compress data with rsyncable block boundaries.
/// Each content-determined block becomes an independent standard gzip member.
///
/// Increment 7: each block is compressed with the pure-Rust DEFLATE engine
/// (`deflate::encode_gzip_bytes_to_vec` — one self-contained gzip member per block), so
/// `--rsyncable` carries ZERO C-FFI compressor. `header_info` no longer flows
/// into per-block headers (each pure member uses the minimal gzip header); the
/// content-defined boundaries from `split_rsyncable` are what rsync relies on.
pub fn compress_rsyncable<W: Write + Send>(
    data: &[u8],
    compression_level: u32,
    num_threads: usize,
    _header_info: &GzipHeaderInfo,
    mut writer: W,
) -> io::Result<u64> {
    use crate::compress::deflate;

    // Gate-4 (CLAUDE.md measurement PROTOCOL): the sole call site of
    // `compress_rsyncable`'s two production entry points in `io.rs` — print
    // here so it can't diverge from which branch (stdout vs file writer)
    // reached it.
    crate::compress::route::emit(
        crate::compress::route::RSYNCABLE,
        compression_level,
        num_threads,
    );

    let blocks = split_rsyncable(data);

    if blocks.is_empty() {
        return Ok(0);
    }

    // For single block or single thread, compress sequentially
    if blocks.len() == 1 || num_threads <= 1 {
        let mut total = 0u64;
        for block in &blocks {
            let output = deflate::encode_gzip_bytes_to_vec(block, compression_level);
            writer.write_all(&output)?;
            total += block.len() as u64;
        }
        return Ok(total);
    }

    // Parallel: compress blocks using thread pool
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    let num_blocks = blocks.len();
    let next_block = AtomicUsize::new(0);

    // Pre-allocate output slots
    let outputs: Vec<std::sync::Mutex<Vec<u8>>> = (0..num_blocks)
        .map(|_| std::sync::Mutex::new(Vec::new()))
        .collect();

    thread::scope(|scope| {
        for _ in 0..num_threads.min(num_blocks) {
            scope.spawn(|| loop {
                let idx = next_block.fetch_add(1, Ordering::Relaxed);
                if idx >= num_blocks {
                    break;
                }
                let mut output = outputs[idx].lock().unwrap();
                *output = deflate::encode_gzip_bytes_to_vec(blocks[idx], compression_level);
            });
        }
    });

    // Write outputs in order
    let mut total = 0u64;
    for (i, slot) in outputs.iter().enumerate() {
        let output = slot.lock().unwrap();
        writer.write_all(&output)?;
        total += blocks[i].len() as u64;
    }

    Ok(total)
}
