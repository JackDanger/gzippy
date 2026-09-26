//! BGZF (Block GZIP Format) Parallel Decompression
//!
//! BGZF files have independent blocks with embedded size markers, allowing
//! perfect parallelism with zero lock contention.
//!
//! ## Strategy
//!
//! 1. Parse BGZF headers to find all block boundaries and output sizes (ISIZE)
//! 2. Pre-allocate entire output buffer based on sum of ISIZE values
//! 3. Decompress blocks in parallel, writing directly to pre-calculated offsets
//! 4. Single write of complete output
//!
//! ## Performance Target: 4000+ MB/s with 14 threads
//!
//! With single-threaded inflate at 10700 MB/s and no lock contention,
//! theoretical max is ~150,000 MB/s. Memory bandwidth limits us to ~4000-5000 MB/s.

#![allow(clippy::needless_range_loop)]

use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};

#[allow(unused_imports)]
use crate::decompress::two_level_table::{FastBits, TurboBits, TwoLevelTable};

/// BGZF block information
#[derive(Debug, Clone)]
pub(crate) struct BgzfBlock {
    /// Byte offset of block start in compressed data
    start: usize,
    /// Total block length (including header and trailer)
    length: usize,
    /// Uncompressed size (from ISIZE trailer)
    isize: u32,
    /// Output offset (calculated during planning)
    output_offset: usize,
    /// Byte offset of raw deflate data within the block (past gzip header)
    deflate_start: usize,
}

/// Parse all BGZF blocks from compressed data
fn parse_bgzf_blocks(data: &[u8]) -> io::Result<Vec<BgzfBlock>> {
    let mut blocks = Vec::new();
    let mut offset = 0;
    let mut output_offset = 0;

    while offset + 18 < data.len() {
        // Check gzip magic
        if data[offset] != 0x1f || data[offset + 1] != 0x8b {
            break;
        }

        // Must have FEXTRA flag
        if data[offset + 3] & 0x04 == 0 {
            break;
        }

        // Get XLEN
        if offset + 12 > data.len() {
            break;
        }
        let xlen = u16::from_le_bytes([data[offset + 10], data[offset + 11]]) as usize;
        if offset + 12 + xlen > data.len() {
            break;
        }

        // Find GZ subfield with block size
        let extra_start = offset + 12;
        let extra_field = &data[extra_start..extra_start + xlen];
        let mut block_size = None;
        let mut pos = 0;

        while pos + 4 <= extra_field.len() {
            let subfield_id = &extra_field[pos..pos + 2];
            let subfield_len =
                u16::from_le_bytes([extra_field[pos + 2], extra_field[pos + 3]]) as usize;

            if subfield_id == b"GZ" {
                if subfield_len == 4 && pos + 8 <= extra_field.len() {
                    // New 4-byte format (supports blocks > 64KB)
                    let size = u32::from_le_bytes([
                        extra_field[pos + 4],
                        extra_field[pos + 5],
                        extra_field[pos + 6],
                        extra_field[pos + 7],
                    ]) as usize;
                    if size > 0 {
                        block_size = Some(size);
                    }
                    break;
                } else if subfield_len == 2 && pos + 6 <= extra_field.len() {
                    // Legacy 2-byte format (BSIZE-1)
                    let size_minus_1 =
                        u16::from_le_bytes([extra_field[pos + 4], extra_field[pos + 5]]) as usize;
                    block_size = Some(size_minus_1 + 1);
                    break;
                }
            }

            pos += 4 + subfield_len;
        }

        let length = match block_size {
            Some(l) if l > 0 && offset + l <= data.len() => l,
            _ => break,
        };

        // Read ISIZE from trailer (last 4 bytes of block)
        let isize = if length >= 8 {
            let trailer_start = offset + length - 4;
            u32::from_le_bytes([
                data[trailer_start],
                data[trailer_start + 1],
                data[trailer_start + 2],
                data[trailer_start + 3],
            ])
        } else {
            0
        };

        // Deflate data starts after the full gzip header (including optional fields)
        let mut deflate_start = offset + 12 + xlen;
        let flags = data[offset + 3];
        // FNAME: null-terminated filename
        if flags & 0x08 != 0 {
            while deflate_start < offset + length && data[deflate_start] != 0 {
                deflate_start += 1;
            }
            deflate_start += 1; // skip null terminator
        }
        // FCOMMENT: null-terminated comment
        if flags & 0x10 != 0 {
            while deflate_start < offset + length && data[deflate_start] != 0 {
                deflate_start += 1;
            }
            deflate_start += 1;
        }
        // FHCRC: 2-byte header CRC
        if flags & 0x02 != 0 {
            deflate_start += 2;
        }

        blocks.push(BgzfBlock {
            start: offset,
            length,
            isize,
            output_offset,
            deflate_start,
        });

        output_offset += isize as usize;
        offset += length;
    }

    if blocks.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "No BGZF blocks found",
        ));
    }

    Ok(blocks)
}

/// Inflate directly into a pre-allocated output slice
///
/// Decompress raw deflate data using the pure-Rust inflate engine.
///
/// This is the key function for zero-copy parallel decompression. The pure-Rust
/// decoder (`inflate_consume_first`) is stateless — there is no per-call
/// decompressor to allocate/free — and carries NO C-FFI, so the BGZF /
/// multi-member decode graph is FFI-free.
fn inflate_into(deflate_data: &[u8], output: &mut [u8]) -> io::Result<usize> {
    crate::decompress::inflate::consume_first_decode::inflate_consume_first(deflate_data, output)
}

/// Public version of inflate_into for use by other modules.
///
/// Pure-Rust inflate — no C-FFI in the decode graph.
pub fn inflate_into_pub(deflate_data: &[u8], output: &mut [u8]) -> io::Result<usize> {
    inflate_into(deflate_data, output)
}

/// Parallel BGZF decompression returning output as a Vec.
///
/// This is the zero-copy path: the output Vec is filled in-place by
/// parallel threads, then returned directly to the caller without any
/// intermediate copies.
pub fn decompress_bgzf_parallel_to_vec(data: &[u8], num_threads: usize) -> io::Result<Vec<u8>> {
    let blocks = parse_bgzf_blocks(data)?;

    if blocks.is_empty() {
        return Ok(Vec::new());
    }

    let total_output: usize = blocks.iter().map(|b| b.isize as usize).sum();
    let output = vec![0u8; total_output];

    let num_blocks = blocks.len();
    let next_block = AtomicUsize::new(0);
    let had_error = std::sync::atomic::AtomicBool::new(false);

    use std::cell::UnsafeCell;
    struct OutputBuffer(UnsafeCell<Vec<u8>>);
    unsafe impl Sync for OutputBuffer {}

    let output_cell = OutputBuffer(UnsafeCell::new(output));

    std::thread::scope(|scope| {
        for _ in 0..num_threads.min(num_blocks) {
            let blocks_ref = &blocks;
            let next_ref = &next_block;
            let output_ref = &output_cell;
            let error_ref = &had_error;

            scope.spawn(move || {
                loop {
                    let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                    if idx >= num_blocks {
                        break;
                    }

                    let block = &blocks_ref[idx];
                    let out_size = block.isize as usize;
                    if out_size == 0 {
                        continue;
                    }

                    // Raw deflate: skip gzip header, stop before 8-byte trailer
                    let deflate_end = block.start + block.length - 8;
                    let deflate_data = &data[block.deflate_start..deflate_end];

                    // SAFETY: Each block writes to a disjoint region
                    let output_ptr = unsafe { (*output_ref.0.get()).as_mut_ptr() };
                    let out_start = block.output_offset;
                    let out_slice = unsafe {
                        std::slice::from_raw_parts_mut(output_ptr.add(out_start), out_size)
                    };

                    match inflate_into(deflate_data, out_slice) {
                        Ok(actual_out) if actual_out == out_size => {}
                        _ => error_ref.store(true, Ordering::Relaxed),
                    }
                }
            });
        }
    });

    if had_error.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "CRC32 or size mismatch in BGZF block",
        ));
    }

    Ok(output_cell.0.into_inner())
}

/// Parallel BGZF decompression writing to a generic writer.
///
/// For single-thread (num_threads=1), uses a streaming path that decompresses
/// block-by-block into a reusable buffer.
///
/// For multi-thread, uses a pipelined architecture:
///   - N decoder threads pull blocks via atomic counter, decompress into
///     pooled buffers, and send (block_index, buffer) through a channel
///   - Main thread receives completed blocks, writes them in order,
///     and returns buffers to the pool
///
/// This avoids allocating the full output (~211MB for silesia) which caused
/// ~53K page faults and only 2x scaling with 4 threads. The pipeline uses
/// a small buffer pool (~1MB) and writes blocks as they complete.
pub fn decompress_bgzf_parallel<W: Write>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> io::Result<u64> {
    if num_threads <= 1 {
        return decompress_bgzf_streaming(data, writer);
    }
    decompress_bgzf_pipelined(data, writer, num_threads)
}

/// Pipelined parallel BGZF: decoder threads + ordered writer.
///
/// Buffer pool avoids per-block allocation. Completed blocks are written
/// in order as they arrive, overlapping I/O with decompression.
fn decompress_bgzf_pipelined<W: Write>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> io::Result<u64> {
    let blocks = parse_bgzf_blocks(data)?;
    if blocks.is_empty() {
        return Ok(0);
    }

    let num_blocks = blocks.len();
    let max_block_output = blocks.iter().map(|b| b.isize as usize).max().unwrap_or(0);

    // Completed blocks channel: (block_index, decompressed_data)
    // Bounded to 2*threads so decoders don't race too far ahead of the writer.
    let channel_cap = num_threads * 2 + 2;
    let (done_tx, done_rx) = std::sync::mpsc::sync_channel::<(usize, Vec<u8>)>(channel_cap);

    let next_block = AtomicUsize::new(0);
    let had_error = std::sync::atomic::AtomicBool::new(false);
    let mut total = 0u64;

    std::thread::scope(|scope| {
        // Spawn N decoder threads, each with its own reusable buffer
        for _ in 0..num_threads.min(num_blocks) {
            let done_tx = done_tx.clone();
            let blocks_ref = &blocks;
            let next_ref = &next_block;
            let error_ref = &had_error;

            scope.spawn(move || {
                let mut buf = vec![0u8; max_block_output];

                loop {
                    let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                    if idx >= num_blocks {
                        break;
                    }

                    let block = &blocks_ref[idx];
                    let out_size = block.isize as usize;
                    if out_size == 0 {
                        let _ = done_tx.send((idx, Vec::new()));
                        continue;
                    }

                    if buf.len() < out_size {
                        buf.resize(out_size, 0);
                    }

                    let deflate_end = block.start + block.length - 8;
                    let deflate_data = &data[block.deflate_start..deflate_end];

                    let actual_out = match inflate_into(deflate_data, &mut buf[..out_size]) {
                        Ok(n) if n == out_size => n,
                        Ok(n) => {
                            error_ref.store(true, Ordering::Relaxed);
                            n
                        }
                        Err(_) => {
                            error_ref.store(true, Ordering::Relaxed);
                            0
                        }
                    };

                    // Transfer buffer ownership through the channel; swap in a
                    // fresh capacity-only Vec so the next block has a buffer to
                    // fill without any copy of the decompressed bytes.
                    buf.truncate(actual_out);
                    let send_buf =
                        std::mem::replace(&mut buf, Vec::with_capacity(max_block_output));
                    let _ = done_tx.send((idx, send_buf));
                }
            });
        }
        drop(done_tx); // close channel when all decoders finish

        // Writer: receive completed blocks, write in order.
        // Blocks may arrive out of order; hold them in a BTreeMap until
        // the next sequential block is available, then flush.
        let mut next_to_write = 0usize;
        let mut pending = std::collections::BTreeMap::<usize, Vec<u8>>::new();
        let mut write_error: Option<io::Error> = None;

        for (idx, data_vec) in &done_rx {
            pending.insert(idx, data_vec);

            while let Some(block_data) = pending.remove(&next_to_write) {
                if write_error.is_none() && !block_data.is_empty() {
                    if let Err(e) = writer.write_all(&block_data) {
                        write_error = Some(e);
                    }
                    total += block_data.len() as u64;
                }
                next_to_write += 1;
            }
        }
    });

    if had_error.load(Ordering::Relaxed) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "CRC32 or size mismatch in BGZF block",
        ));
    }

    Ok(total)
}

/// Streaming BGZF decompression: decompress one block at a time into a
/// reusable buffer, write immediately. No full-output-size allocation.
///
/// Uses raw deflate decompress with a reused decompressor (skipping both
/// gzip header re-parsing and decompressor alloc/free per block).
fn decompress_bgzf_streaming<W: Write>(data: &[u8], writer: &mut W) -> io::Result<u64> {
    let blocks = parse_bgzf_blocks(data)?;
    if blocks.is_empty() {
        return Ok(0);
    }

    let max_block_output = blocks.iter().map(|b| b.isize as usize).max().unwrap_or(0);
    let mut buf = vec![0u8; max_block_output];
    let mut total = 0u64;

    for block in &blocks {
        let out_size = block.isize as usize;
        if out_size == 0 {
            continue;
        }

        if out_size > buf.len() {
            buf.resize(out_size, 0);
        }

        // Raw deflate data: between header and 8-byte trailer (CRC32 + ISIZE)
        let deflate_end = block.start + block.length - 8;
        let deflate_data = &data[block.deflate_start..deflate_end];

        let actual_out = inflate_into(deflate_data, &mut buf[..out_size]).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "deflate decompression failed in BGZF block",
            )
        })?;

        writer.write_all(&buf[..actual_out])?;
        total += actual_out as u64;
    }
    Ok(total)
}

// ============================================================================
// Multi-Member Parallel Decompression (for pigz-style files)
// ============================================================================

/// Parse a gzip header starting at `data[offset..]`, returning the byte offset
/// of the raw deflate data (past the header). Returns None if the header is
/// malformed or extends past the given `end` bound.
fn parse_gzip_header(data: &[u8], offset: usize, end: usize) -> Option<usize> {
    if end - offset < 10 {
        return None;
    }
    let mut ds = offset + 10;
    let flg = data[offset + 3];
    if flg & 0x04 != 0 {
        if ds + 2 > end {
            return None;
        }
        let xlen = u16::from_le_bytes([data[ds], data[ds + 1]]) as usize;
        ds += 2 + xlen;
    }
    if flg & 0x08 != 0 {
        while ds < end && data[ds] != 0 {
            ds += 1;
        }
        ds += 1;
    }
    if flg & 0x10 != 0 {
        while ds < end && data[ds] != 0 {
            ds += 1;
        }
        ds += 1;
    }
    if flg & 0x02 != 0 {
        ds += 2;
    }
    if ds >= end {
        None
    } else {
        Some(ds)
    }
}

/// Fast O(N) member boundary scan for multi-member gzip files (pigz-style).
///
/// Scans for gzip magic bytes (0x1f 0x8b 0x08) with header validation to find
/// member boundaries without any decompression. Reads ISIZE from each member's
/// trailer for output pre-allocation. This replaces the old `scan_member_boundaries_exact`
/// which fully decompressed every member (doing 2x total work).
///
/// Returns None if the data is not multi-member or boundaries look suspicious.
pub(crate) fn scan_member_boundaries_fast(data: &[u8]) -> Option<Vec<BgzfBlock>> {
    if data.len() < 18 || data[0] != 0x1f || data[1] != 0x8b || data[2] != 0x08 {
        return None;
    }

    let header_size = crate::decompress::format::parse_gzip_header_size(data).unwrap_or(10);
    let mut starts = vec![0usize];

    // SIMD magic-byte search (memchr::memmem) instead of a byte-by-byte linear
    // scan. This is a T-invariant SERIAL pass over the WHOLE compressed file that
    // ran BEFORE any parallel dispatch, so on a large compressible-dominant
    // multi-member stream it was ~50% of the T16 wall (Amdahl). memmem vectorizes
    // the 3-byte magic search (~10× the byte-loop throughput) for an identical
    // `starts` list — every candidate is re-validated by the same ISIZE + flag
    // predicate below, so the output is byte-for-byte unchanged.
    const GZIP_MAGIC: &[u8] = &[0x1f, 0x8b, 0x08];
    let finder = memchr::memmem::Finder::new(GZIP_MAGIC);
    let mut search_from = header_size + 1;
    while search_from + 10 < data.len() {
        let Some(rel) = finder.find(&data[search_from..]) else {
            break;
        };
        let pos = search_from + rel;
        // Replicate the byte-loop guard exactly: positions whose full 10-byte
        // header window would run past EOF were never inspected. Matches are
        // returned left-to-right, so once one crosses that bound every later one
        // does too — stop.
        if pos + 10 >= data.len() {
            break;
        }
        if data[pos + 3] & 0xE0 == 0
            // Validate preceding ISIZE field (same heuristic as is_likely_multi_member).
            // Filters false positives from stored-block streams where raw bytes appear
            // verbatim and can accidentally match the gzip magic sequence.
            && pos >= 4
            && {
                let isize = u32::from_le_bytes([
                    data[pos - 4], data[pos - 3], data[pos - 2], data[pos - 1],
                ]);
                isize > 0 && isize <= 1_073_741_824
            }
        {
            starts.push(pos);
        }
        search_from = pos + 1;
    }

    if starts.len() < 2 {
        return None;
    }

    let mut members = Vec::with_capacity(starts.len());
    let mut output_offset = 0usize;

    for i in 0..starts.len() {
        let start = starts[i];
        let end = if i + 1 < starts.len() {
            starts[i + 1]
        } else {
            data.len()
        };
        let length = end - start;

        if length < 18 {
            return None;
        }

        let isize_val =
            u32::from_le_bytes([data[end - 4], data[end - 3], data[end - 2], data[end - 1]]);

        let deflate_start = parse_gzip_header(data, start, end)?;

        members.push(BgzfBlock {
            start,
            length,
            isize: isize_val,
            output_offset,
            deflate_start,
        });

        output_offset += isize_val as usize;
    }

    // Sanity: total output shouldn't be wildly disproportionate to input
    if output_offset > data.len().saturating_mul(100) {
        return None;
    }

    Some(members)
}

/// Routing predicate (design §1.2 / [R1-#8]): decide whether the plain
/// multi-member **member-per-worker** fast path
/// ([`decompress_multi_member_parallel`]) will keep all `t_eff` workers busy —
/// i.e. whether the member size distribution is balanced enough that assigning
/// one whole member per worker approaches the ideal makespan `Σcost / t_eff`.
///
/// This replaces two coupled scalar thresholds (`count ≥ K·T && max_share ≤
/// MARGIN/T`) with a **single-member dominance** test over per-member DECODE
/// COST: the member-per-worker split can rebalance any distribution EXCEPT one
/// where a member's cost exceeds roughly one worker's fair share of the total —
/// that member pins a worker for the whole decode and cannot be sped up by more
/// workers, whereas the chunked path splits *within* it. On a `false` verdict
/// the classifier routes the whole file to the chunked path
/// ([`crate::decompress::DecodePath::MultiMemberChunked`]).
///
/// Why the dominance test rather than a literal greedy-LPT makespan-vs-ideal
/// comparison (design §1.2's first form): indivisible equal members carry an
/// inherent integer-granularity imbalance (40 members over 16 workers has a
/// forced 3-vs-2 split = 20% over the *continuous* ideal) that a naive
/// `makespan ≤ (1+EPS)·(Σcost/T)` test mis-flags as "unbalanced" — it would
/// reject the design's own canonical `mm_many` fast-path example. Granularity
/// imbalance is NOT a splittable-dominance problem; only a member exceeding a
/// worker's share is. The dominance test captures exactly the discriminating
/// signal and is granularity-robust; it also subsumes the adversarial
/// "two-huge-members" case (each huge member alone exceeds a worker share). A
/// residual mis-pick (e.g. 3 medium members on 2 workers, where within-member
/// chunking could shave the forced 3-vs-2 tail) is only a perf choice between
/// two *correct* paths [R1-#8] and is bounded by `EPS`.
///
/// Cost model (`cost_i`): the member's compressed length, blended up by the
/// output-size proxy `isize_i / global_ratio` so a stored-dense member (whose
/// decode cost tracks its OUTPUT, not its tiny compressed input) is not
/// undercounted. `global_ratio = Σ isize / Σ compressed`. All inputs are
/// content-derived from the header scan — no host benchmark scar.
///
/// This is a **perf routing predicate only**: after per-member CRC32/ISIZE is
/// equalized across paths, a wrong verdict mis-picks between two *correct*
/// paths, never between correctness levels. `EPS` is a plain named constant (no
/// production env knob) locked by the box-side OQ-2 gate.
///
/// STAGE-2d: WIRED into `classify_gzip` — a plain multi-member T>1 stream routes
/// to [`crate::decompress::DecodePath::MultiMemberGrid`] when this returns
/// `false` (dominant/uneven ⇒ the whole-file chunk grid spreads the dominant
/// member across all workers) and to `MultiMemberPar` (member-per-worker) when
/// it returns `true` (numerous + balanced).
pub(crate) fn fast_path_ok(members: &[BgzfBlock], t_eff: usize) -> bool {
    /// Slack on a member's cost over one worker's fair share (`Σcost / t_eff`)
    /// that still counts as "not dominant". Absorbs the integer granularity of
    /// indivisible members; locked by the OQ-2 schedule-predictor gate (§7).
    const EPS: f64 = 0.25;

    let t_eff = t_eff.max(1);
    let n = members.len();
    // Fewer members than workers ⇒ at least one worker idles ⇒ the fast path
    // cannot saturate the pool; the chunked path splits within members.
    if n < t_eff {
        return false;
    }

    // global_ratio = Σ isize / Σ compressed (guard against a zero denom).
    let total_compressed: u128 = members.iter().map(|m| m.length as u128).sum();
    let total_isize: u128 = members.iter().map(|m| m.isize as u128).sum();
    if total_compressed == 0 {
        return false;
    }
    // Fixed-point ratio ×256 to avoid float in the per-member blend.
    let ratio_q8: u128 = (total_isize.saturating_mul(256) / total_compressed).max(1);

    let costs = members.iter().map(|m| {
        let by_input = m.length as u128;
        // isize / global_ratio  ==  isize * 256 / ratio_q8
        let by_output = (m.isize as u128).saturating_mul(256) / ratio_q8;
        by_input.max(by_output)
    });

    let mut total_cost: u128 = 0;
    let mut max_cost: u128 = 0;
    for c in costs {
        total_cost += c;
        if c > max_cost {
            max_cost = c;
        }
    }
    if total_cost == 0 {
        return false;
    }

    // Dominance test: the largest member's cost must not exceed one worker's
    // fair share (`total_cost / t_eff`) by more than `EPS`. A dominant member
    // pins a worker and can only be sped up by the within-member chunked path.
    let ideal = total_cost as f64 / t_eff as f64;
    (max_cost as f64) <= (1.0 + EPS) * ideal
}

/// GZ coverage walk: hop member-to-member via the "GZ" FEXTRA subfield's
/// whole-member size and report whether EVERY member carries the subfield AND
/// the walk ends exactly at the end of `data`. Only meaningful once
/// [`crate::decompress::format::has_bgzf_markers`] has fired on member 1.
///
/// A pure gzippy-parallel file walks cleanly to `data.len()` → the GZ fast path
/// ([`decompress_bgzf_parallel`]) is safe. A mixed concatenation (a plain member
/// lacking the subfield, or a walk that over/undershoots the file end) returns
/// `false` so the classifier routes the whole file to the cross-member chunked
/// path deterministically at classify time — never an in-body fallback. [R2-#3]
pub(crate) fn gz_coverage_is_pure(data: &[u8]) -> bool {
    let mut offset = 0usize;
    let mut members = 0usize;
    // Bound the walk so a hostile size field cannot loop forever; a real
    // gzippy-parallel file has one member per block (≤ millions, but each step
    // advances by ≥ 1 header so the file length already bounds it).
    while offset + 12 <= data.len() {
        if data[offset] != 0x1f || data[offset + 1] != 0x8b || data[offset + 2] != 0x08 {
            return false;
        }
        // FEXTRA must be present for a GZ member.
        if data[offset + 3] & 0x04 == 0 {
            return false;
        }
        let member_len = match gz_member_len(data, offset) {
            Some(l) if l >= 18 => l,
            _ => return false,
        };
        offset = match offset.checked_add(member_len) {
            Some(o) if o <= data.len() => o,
            _ => return false,
        };
        members += 1;
    }
    members >= 1 && offset == data.len()
}

/// Parse the "GZ" FEXTRA subfield's whole-member compressed length at `start`.
/// Mirrors the size decode in the BGZF block scan (bgzf.rs:315-351): 4-byte
/// form = whole-member size, legacy 2-byte form = BSIZE-1. Returns `None` when
/// the member lacks the subfield or the header is truncated.
fn gz_member_len(data: &[u8], start: usize) -> Option<usize> {
    if start + 12 > data.len() {
        return None;
    }
    let xlen = u16::from_le_bytes([data[start + 10], data[start + 11]]) as usize;
    if start + 12 + xlen > data.len() {
        return None;
    }
    let extra = &data[start + 12..start + 12 + xlen];
    let mut pos = 0;
    while pos + 4 <= extra.len() {
        let id = &extra[pos..pos + 2];
        let sublen = u16::from_le_bytes([extra[pos + 2], extra[pos + 3]]) as usize;
        if id == b"GZ" {
            if sublen == 4 && pos + 8 <= extra.len() {
                let size = u32::from_le_bytes([
                    extra[pos + 4],
                    extra[pos + 5],
                    extra[pos + 6],
                    extra[pos + 7],
                ]) as usize;
                return if size > 0 { Some(size) } else { None };
            } else if sublen == 2 && pos + 6 <= extra.len() {
                let size_minus_1 = u16::from_le_bytes([extra[pos + 4], extra[pos + 5]]) as usize;
                return Some(size_minus_1 + 1);
            }
            return None;
        }
        pos += 4 + sublen;
    }
    None
}

/// Zero-copy parallel decompression for multi-member gzip files.
///
/// Uses the same approach as BGZF parallel: pre-allocate output, write directly
/// to disjoint slices. Member boundaries are found by `scan_member_boundaries_fast`
/// (header-only scan), and each member's deflate body is decoded with the
/// pure-Rust `inflate_into` (no C-FFI).
///
/// This avoids the old approach's issues:
/// - No intermediate Vec copies (~1GB saved for 503MB output)
/// - No per-chunk buffer allocation
/// - Work-stealing across all members for optimal load balancing
pub fn decompress_multi_member_parallel_to_vec(
    data: &[u8],
    num_threads: usize,
) -> io::Result<Vec<u8>> {
    let members = scan_member_boundaries_fast(data).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "Not a multi-member gzip file or boundary scan failed",
        )
    })?;

    let total_output: usize = members.iter().map(|m| m.isize as usize).sum();
    let output = vec![0u8; total_output];

    let num_members = members.len();
    let next_member = AtomicUsize::new(0);
    let had_error = std::sync::atomic::AtomicBool::new(false);

    use std::cell::UnsafeCell;
    struct OutputBuffer(UnsafeCell<Vec<u8>>);
    unsafe impl Sync for OutputBuffer {}

    let output_cell = OutputBuffer(UnsafeCell::new(output));

    std::thread::scope(|scope| {
        for _ in 0..num_threads.min(num_members) {
            let members_ref = &members;
            let next_ref = &next_member;
            let output_ref = &output_cell;
            let error_ref = &had_error;

            scope.spawn(move || {
                // scan_member_boundaries_fast already parsed each gzip header and
                // stored deflate_start, so we decode the raw deflate body directly
                // via the pure-Rust inflate engine (no C-FFI in the decode graph).
                loop {
                    let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                    if idx >= num_members {
                        break;
                    }

                    let member = &members_ref[idx];
                    // deflate_start is absolute offset in data; trailer is 8 bytes (CRC32+ISIZE).
                    // Defensive bounds (fuzz-found panic, bgzf.rs OOB): a malformed member from
                    // scan_member_boundaries_fast on crafted input can carry length < 8,
                    // deflate_start > deflate_end, or deflate_end past the buffer — slicing
                    // `data[deflate_start..deflate_end]` then panics ("slice index starts at N
                    // but ends at M"). For any VALID member deflate_start <= start+length-8 <=
                    // data.len(), so these guards are byte-transparent; a bad member flags the
                    // error and is skipped (the had_error check below turns it into a terminal
                    // Err, matching the inflate-failure arm).
                    let deflate_end = match member.start.checked_add(member.length) {
                        Some(end)
                            if member.length >= 8
                                && end - 8 >= member.deflate_start
                                && end - 8 <= data.len() =>
                        {
                            end - 8
                        }
                        _ => {
                            error_ref.store(true, Ordering::Relaxed);
                            continue;
                        }
                    };
                    let deflate_data = &data[member.deflate_start..deflate_end];

                    // SAFETY: Each member writes to a disjoint region. Defensive bounds: a
                    // malformed member's output_offset/isize could point past the output
                    // buffer; `from_raw_parts_mut` past the allocation is UB. Valid members
                    // satisfy output_offset + isize <= output.len() (the buffer is sized to the
                    // sum of member ISIZEs), so this guard is byte-transparent.
                    let out_total = unsafe { (*output_ref.0.get()).len() };
                    let out_start = member.output_offset;
                    let out_size = member.isize as usize;
                    if out_start
                        .checked_add(out_size)
                        .is_none_or(|e| e > out_total)
                    {
                        error_ref.store(true, Ordering::Relaxed);
                        continue;
                    }
                    let output_ptr = unsafe { (*output_ref.0.get()).as_mut_ptr() };
                    let out_slice = unsafe {
                        std::slice::from_raw_parts_mut(output_ptr.add(out_start), out_size)
                    };

                    match inflate_into(deflate_data, out_slice) {
                        Ok(actual_out) if actual_out == out_size => {}
                        _ => error_ref.store(true, Ordering::Relaxed),
                    }
                }
            });
        }
    });

    if had_error.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Decompression error in multi-member parallel",
        ));
    }

    Ok(output_cell.0.into_inner())
}

/// Parallel decompression for multi-member gzip files (pigz-style output).
///
/// Delegates to `decompress_multi_member_parallel_to_vec` for zero-copy parallel,
/// then writes the result. Falls back to sequential for single-member files.
pub fn decompress_multi_member_parallel<W: Write>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> io::Result<u64> {
    if data.len() < 18 || data[0] != 0x1f || data[1] != 0x8b {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Not a gzip file",
        ));
    }

    let output = decompress_multi_member_parallel_to_vec(data, num_threads)?;
    let len = output.len() as u64;
    writer.write_all(&output)?;
    Ok(len)
}

// ============================================================================
// Single-Member Parallel Decompression (rapidgzip strategy)
// ============================================================================
//
// For single-member gzip files, we use a two-phase approach:
// 1. Sequential first pass: decode and record block boundaries + windows
// 2. Parallel second pass: re-decode each segment using windows as dictionaries
//
// This provides speedup when the file is large enough to amortize the overhead.

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
