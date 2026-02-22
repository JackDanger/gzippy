//! Trace which decompression path a .gz file would take through gzippy.
//!
//! Reads the gzip header and compressed data to classify the file:
//! - BGZF (gzippy-produced, has embedded size markers)
//! - Multi-member (pigz-style, multiple gzip members)
//! - Single-member (standard gzip)
//!
//! Then predicts the code path and expected performance characteristics.

use std::fs;
use std::path::Path;

pub fn trace(path: &str) -> Result<(), String> {
    let p = Path::new(path);
    if !p.exists() {
        return Err(format!("File not found: {path}"));
    }

    let data = fs::read(p).map_err(|e| format!("Failed to read {path}: {e}"))?;
    if data.len() < 18 {
        return Err("File too small to be a valid gzip file".to_string());
    }

    // Validate gzip magic
    if data[0] != 0x1f || data[1] != 0x8b {
        return Err("Not a gzip file (bad magic bytes)".to_string());
    }

    let file_size = data.len();
    let method = data[2]; // Should be 8 (deflate)
    let flags = data[3];
    let has_extra = flags & 0x04 != 0;
    let _has_name = flags & 0x08 != 0;
    let _has_comment = flags & 0x10 != 0;

    // Check ISIZE (last 4 bytes of gzip)
    let isize = u32::from_le_bytes([
        data[data.len() - 4],
        data[data.len() - 3],
        data[data.len() - 2],
        data[data.len() - 1],
    ]);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  DECOMPRESSION PATH TRACE                                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("  File:         {path}");
    println!("  Size:         {:.2} MB ({file_size} bytes)", file_size as f64 / 1_048_576.0);
    println!("  Method:       {} ({})", method, if method == 8 { "deflate" } else { "UNKNOWN" });
    println!("  ISIZE:        {isize} ({:.2} MB)", isize as f64 / 1_048_576.0);
    println!("  Ratio:        {:.1}x", isize as f64 / file_size as f64);

    // Check for BGZF
    let is_bgzf = has_extra && data.len() > 16 && {
        let xlen = u16::from_le_bytes([data[10], data[11]]) as usize;
        xlen >= 6 && data.len() > 12 + xlen
            && data[12] == b'B' && data[13] == b'C'
            && data[14] == 2 && data[15] == 0
    };

    // Count gzip members
    let member_count = count_members(&data);

    // Classify
    let (path_name, description, expected_speed) = if is_bgzf {
        (
            "BGZF Parallel",
            "bgzf::decompress_bgzf_parallel — Pre-parsed block sizes, parallel inflate",
            "3000-4000+ MB/s (8 threads)",
        )
    } else if member_count > 1 {
        (
            "Multi-Member Parallel",
            "bgzf::decompress_multi_member_parallel — Member boundaries, parallel inflate",
            "2000-3000 MB/s (8 threads)",
        )
    } else if file_size > 10 * 1024 * 1024 {
        (
            "Single-Member (large) → Speculative Parallel",
            "speculative_parallel::decompress_speculative → fallback to libdeflate FFI",
            "1000-1400 MB/s (sequential) or 2000+ MB/s (parallel, WIP)",
        )
    } else {
        (
            "Single-Member (small) → Sequential libdeflate",
            "decompress_single_member_libdeflate — Sequential libdeflate FFI",
            "1000-1400 MB/s (sequential)",
        )
    };

    println!();
    println!("  Classification:");
    println!("    BGZF:       {}", if is_bgzf { "YES" } else { "no" });
    println!("    Members:    {member_count}");
    println!("    Path:       {path_name}");
    println!("    Function:   {description}");
    println!("    Expected:   {expected_speed}");

    // Member details
    if member_count > 1 && member_count <= 100 {
        println!("\n  Member details:");
        let mut pos = 0;
        let mut idx = 0;
        while pos < data.len() - 18 {
            if data[pos] == 0x1f && data[pos + 1] == 0x8b && data[pos + 2] == 0x08 {
                let member_isize = if let Some(end) = find_member_end(&data, pos) {
                    u32::from_le_bytes([
                        data[end - 4], data[end - 3], data[end - 2], data[end - 1],
                    ])
                } else {
                    0
                };
                if idx < 10 || idx == member_count - 1 {
                    println!(
                        "    #{:<4} offset={:<10} isize={:<10} ({:.1} KB)",
                        idx, pos, member_isize, member_isize as f64 / 1024.0,
                    );
                } else if idx == 10 {
                    println!("    ... ({} more members) ...", member_count - 11);
                }
                if let Some(end) = find_member_end(&data, pos) {
                    pos = end;
                } else {
                    break;
                }
                idx += 1;
            } else {
                pos += 1;
            }
        }
    }

    // Thread analysis
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!("\n  Threading:");
    println!("    Available cores: {num_cpus}");
    if is_bgzf || member_count > 1 {
        println!("    Will use:        {num_cpus} threads (parallel)");
    } else if file_size > 10 * 1024 * 1024 {
        println!("    Will use:        {num_cpus} threads (speculative parallel, or 1 fallback)");
    } else {
        println!("    Will use:        1 thread (sequential, file too small)");
    }

    Ok(())
}

fn count_members(data: &[u8]) -> usize {
    let mut count = 0;
    let mut pos = 0;
    while pos < data.len().saturating_sub(18) {
        if data[pos] == 0x1f && data[pos + 1] == 0x8b && data[pos + 2] == 0x08 {
            count += 1;
            // Skip past this member to avoid counting bytes within the compressed data
            if let Some(end) = find_member_end(data, pos) {
                pos = end;
            } else {
                break;
            }
        } else {
            pos += 1;
        }
    }
    count
}

fn find_member_end(data: &[u8], start: usize) -> Option<usize> {
    if start + 10 > data.len() {
        return None;
    }

    let flags = data[start + 3];
    let mut pos = start + 10;

    // FEXTRA
    if flags & 0x04 != 0 {
        if pos + 2 > data.len() {
            return None;
        }
        let xlen = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2 + xlen;
    }
    // FNAME
    if flags & 0x08 != 0 {
        while pos < data.len() && data[pos] != 0 {
            pos += 1;
        }
        pos += 1;
    }
    // FCOMMENT
    if flags & 0x10 != 0 {
        while pos < data.len() && data[pos] != 0 {
            pos += 1;
        }
        pos += 1;
    }
    // FHCRC
    if flags & 0x02 != 0 {
        pos += 2;
    }

    // Now at compressed data. Scan forward for the next member or end.
    // This is imprecise for single members but good enough for multi-member counting.
    // For BGZF we already know the block size from BSIZE.
    let search_from = pos;
    for i in search_from..data.len().saturating_sub(2) {
        if data[i] == 0x1f && data[i + 1] == 0x8b && i + 2 < data.len() && data[i + 2] == 0x08
            && i > start + 18 {
            return Some(i);
        }
    }

    Some(data.len())
}
