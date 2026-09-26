use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};

use super::{inflate_into, BgzfBlock};

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
