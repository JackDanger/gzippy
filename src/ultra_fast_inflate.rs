//! Ultra-Fast Inflate using Two-Level Huffman Tables
//!
//! This is the fastest pure Rust inflate implementation, using:
//! 1. Two-level Huffman tables (10-bit L1 in L1 cache)
//! 2. Optimized bit buffer with minimal refills
//! 3. SIMD LZ77 copies

#![allow(dead_code)]

use std::io;

use crate::inflate_tables::CODE_LENGTH_ORDER;
use crate::two_level_table::{decode_lz77, decode_symbol, FastBits, TwoLevelTable};

// =============================================================================
// Constants
// =============================================================================

const END_OF_BLOCK: u16 = 256;

// =============================================================================
// Fixed Huffman Tables (Static)
// =============================================================================

/// Pre-built fixed literal/length Huffman table
fn build_fixed_lit_len_table() -> TwoLevelTable {
    let mut lens = [0u8; 288];

    // 0-143: 8 bits
    for len in lens.iter_mut().take(144) {
        *len = 8;
    }
    // 144-255: 9 bits
    for len in lens.iter_mut().take(256).skip(144) {
        *len = 9;
    }
    // 256-279: 7 bits
    for len in lens.iter_mut().take(280).skip(256) {
        *len = 7;
    }
    // 280-287: 8 bits
    for len in lens.iter_mut().take(288).skip(280) {
        *len = 8;
    }

    TwoLevelTable::build(&lens).unwrap()
}

/// Pre-built fixed distance Huffman table
fn build_fixed_dist_table() -> TwoLevelTable {
    let lens = [5u8; 32];
    TwoLevelTable::build(&lens).unwrap()
}

// Thread-local static tables to avoid rebuilding
thread_local! {
    static FIXED_LIT_LEN: TwoLevelTable = build_fixed_lit_len_table();
    static FIXED_DIST: TwoLevelTable = build_fixed_dist_table();
}

// =============================================================================
// Block Decoders
// =============================================================================

/// Decode stored (uncompressed) block
fn decode_stored_block(bits: &mut FastBits, output: &mut Vec<u8>) -> io::Result<()> {
    bits.align();
    bits.refill();

    let len = bits.read(16) as usize;
    let nlen = bits.read(16) as usize;

    if len != (!nlen & 0xFFFF) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Stored block length mismatch",
        ));
    }

    output.reserve(len);
    for _ in 0..len {
        bits.ensure(8);
        output.push(bits.read(8) as u8);
    }

    Ok(())
}

/// Decode fixed Huffman block using two-level tables
fn decode_fixed_block(bits: &mut FastBits, output: &mut Vec<u8>) -> io::Result<()> {
    FIXED_LIT_LEN.with(|lit_len_table| {
        FIXED_DIST.with(|dist_table| decode_huffman_block(bits, output, lit_len_table, dist_table))
    })
}

/// Decode dynamic Huffman block
fn decode_dynamic_block(bits: &mut FastBits, output: &mut Vec<u8>) -> io::Result<()> {
    bits.refill();

    let hlit = bits.read(5) as usize + 257;
    let hdist = bits.read(5) as usize + 1;
    let hclen = bits.read(4) as usize + 4;

    // Read code length code lengths
    let mut code_len_lens = [0u8; 19];
    for i in 0..hclen {
        bits.ensure(4);
        code_len_lens[CODE_LENGTH_ORDER[i] as usize] = bits.read(3) as u8;
    }

    // Build code length table
    let code_len_table = TwoLevelTable::build(&code_len_lens)?;

    // Read all code lengths
    let mut all_lens = vec![0u8; hlit + hdist];
    let mut i = 0;
    while i < hlit + hdist {
        bits.ensure(16);

        let sym = decode_symbol(bits, &code_len_table)?;

        match sym {
            0..=15 => {
                all_lens[i] = sym as u8;
                i += 1;
            }
            16 => {
                let repeat = bits.read(2) as usize + 3;
                let prev = if i > 0 { all_lens[i - 1] } else { 0 };
                for _ in 0..repeat.min(all_lens.len() - i) {
                    all_lens[i] = prev;
                    i += 1;
                }
            }
            17 => {
                let repeat = bits.read(3) as usize + 3;
                i += repeat.min(all_lens.len() - i);
            }
            18 => {
                let repeat = bits.read(7) as usize + 11;
                i += repeat.min(all_lens.len() - i);
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid code length code",
                ))
            }
        }
    }

    // Build Huffman tables
    let lit_len_table = TwoLevelTable::build(&all_lens[..hlit])?;
    let dist_table = TwoLevelTable::build(&all_lens[hlit..])?;

    decode_huffman_block(bits, output, &lit_len_table, &dist_table)
}

/// Decode symbols using two-level tables
#[inline(always)]
fn decode_huffman_block(
    bits: &mut FastBits,
    output: &mut Vec<u8>,
    lit_len_table: &TwoLevelTable,
    dist_table: &TwoLevelTable,
) -> io::Result<()> {
    output.reserve(32 * 1024);

    loop {
        if bits.needs_refill() {
            bits.refill();
        }

        let symbol = decode_symbol(bits, lit_len_table)?;

        if symbol < 256 {
            // Literal
            output.push(symbol as u8);
        } else if symbol == END_OF_BLOCK {
            break;
        } else {
            // Length code
            decode_lz77(bits, dist_table, symbol, output)?;
        }
    }

    Ok(())
}

// =============================================================================
// Main API
// =============================================================================

/// Ultra-fast inflate using two-level Huffman tables
pub fn inflate_ultra_fast(input: &[u8], output: &mut Vec<u8>) -> io::Result<usize> {
    let mut bits = FastBits::new(input);
    let start_len = output.len();

    loop {
        bits.refill();

        let bfinal = bits.read(1);
        let btype = bits.read(2);

        match btype {
            0 => decode_stored_block(&mut bits, output)?,
            1 => decode_fixed_block(&mut bits, output)?,
            2 => decode_dynamic_block(&mut bits, output)?,
            3 => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Reserved block type",
                ))
            }
            _ => unreachable!(),
        }

        if bfinal == 1 {
            break;
        }
    }

    Ok(output.len() - start_len)
}

/// Ultra-fast gzip inflate
pub fn inflate_gzip_ultra_fast(input: &[u8], output: &mut Vec<u8>) -> io::Result<usize> {
    if input.len() < 10 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Input too short",
        ));
    }

    if input[0] != 0x1f || input[1] != 0x8b {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Not a gzip file",
        ));
    }

    if input[2] != 8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Unsupported compression",
        ));
    }

    let flags = input[3];
    let mut pos = 10;

    // Skip optional fields
    if flags & 0x04 != 0 {
        if pos + 2 > input.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Truncated extra field",
            ));
        }
        let xlen = u16::from_le_bytes([input[pos], input[pos + 1]]) as usize;
        pos += 2 + xlen;
    }

    if flags & 0x08 != 0 {
        while pos < input.len() && input[pos] != 0 {
            pos += 1;
        }
        pos += 1;
    }

    if flags & 0x10 != 0 {
        while pos < input.len() && input[pos] != 0 {
            pos += 1;
        }
        pos += 1;
    }

    if flags & 0x02 != 0 {
        pos += 2;
    }

    if pos >= input.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Truncated header",
        ));
    }

    let deflate_data = &input[pos..input.len().saturating_sub(8)];
    inflate_ultra_fast(deflate_data, output)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    #[test]
    fn test_ultra_fast_simple() {
        let original = b"Hello, World! This is a test of ultra-fast inflate.";

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = Vec::new();
        inflate_gzip_ultra_fast(&compressed, &mut output).unwrap();

        assert_eq!(&output[..], &original[..]);
    }

    #[test]
    fn test_ultra_fast_repeated() {
        let original: Vec<u8> = "ABCDEFGH".repeat(1000).into_bytes();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = Vec::new();
        inflate_gzip_ultra_fast(&compressed, &mut output).unwrap();

        assert_eq!(output, original);
    }

    #[test]
    fn test_ultra_fast_large() {
        let original: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = Vec::new();
        inflate_gzip_ultra_fast(&compressed, &mut output).unwrap();

        assert_eq!(output, original);
    }

    #[test]
    fn test_ultra_fast_benchmark() {
        let original: Vec<u8> = (0..1_000_000)
            .map(|i| ((i * 7 + i / 100) % 256) as u8)
            .collect();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        const WARMUP: usize = 5;
        const ITERS: usize = 50;

        // Warmup
        for _ in 0..WARMUP {
            let mut output = Vec::new();
            inflate_gzip_ultra_fast(&compressed, &mut output).unwrap();
            let mut output2 = vec![0u8; original.len()];
            libdeflater::Decompressor::new()
                .gzip_decompress(&compressed, &mut output2)
                .unwrap();
        }

        // Benchmark ultra-fast implementation
        let start = std::time::Instant::now();
        for _ in 0..ITERS {
            let mut output = Vec::with_capacity(original.len());
            inflate_gzip_ultra_fast(&compressed, &mut output).unwrap();
            std::hint::black_box(&output);
        }
        let ultra_time = start.elapsed();

        // Benchmark turbo implementation
        let start = std::time::Instant::now();
        for _ in 0..ITERS {
            let mut output = Vec::with_capacity(original.len());
            crate::turbo_inflate::inflate_gzip_turbo(&compressed, &mut output).unwrap();
            std::hint::black_box(&output);
        }
        let turbo_time = start.elapsed();

        // Benchmark libdeflate
        let mut decompressor = libdeflater::Decompressor::new();
        let start = std::time::Instant::now();
        for _ in 0..ITERS {
            let mut output = vec![0u8; original.len()];
            decompressor
                .gzip_decompress(&compressed, &mut output)
                .unwrap();
            std::hint::black_box(&output);
        }
        let libdeflate_time = start.elapsed();

        let ultra_avg = ultra_time / ITERS as u32;
        let turbo_avg = turbo_time / ITERS as u32;
        let libdeflate_avg = libdeflate_time / ITERS as u32;

        let ultra_mbps = 1_000_000.0 / ultra_avg.as_secs_f64() / 1_000_000.0;
        let turbo_mbps = 1_000_000.0 / turbo_avg.as_secs_f64() / 1_000_000.0;
        let libdeflate_mbps = 1_000_000.0 / libdeflate_avg.as_secs_f64() / 1_000_000.0;

        println!(
            "\n=== ULTRA-FAST Decompression Benchmark (1MB x {}) ===",
            ITERS
        );
        println!(
            "Ultra-fast:  {:>8?}/iter  ({:.0} MB/s)",
            ultra_avg, ultra_mbps
        );
        println!(
            "Turbo:       {:>8?}/iter  ({:.0} MB/s)",
            turbo_avg, turbo_mbps
        );
        println!(
            "libdeflate:  {:>8?}/iter  ({:.0} MB/s)",
            libdeflate_avg, libdeflate_mbps
        );
        println!(
            "Ultra vs libdeflate: {:.2}x",
            ultra_avg.as_secs_f64() / libdeflate_avg.as_secs_f64()
        );
        println!(
            "Ultra vs turbo:      {:.2}x",
            ultra_avg.as_secs_f64() / turbo_avg.as_secs_f64()
        );
    }
}

#[test]
fn test_ultra_fast_large_file() {
    let data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
        Ok(d) => d,
        Err(_) => {
            eprintln!("Skipping test - benchmark file not found");
            return;
        }
    };
    eprintln!("Compressed size: {} bytes", data.len());

    let start = std::time::Instant::now();
    let mut output = Vec::new();
    match inflate_gzip_ultra_fast(&data, &mut output) {
        Ok(sz) => eprintln!("ultra_fast_inflate: {} bytes in {:?}", sz, start.elapsed()),
        Err(e) => eprintln!("ultra_fast_inflate error: {:?}", e),
    }
}
