//! Non-speculative parallel decode for STORED-block-dominated single-member
//! gzip streams (incompressible / `gzip -1` on random data, BTYPE=00).
//!
//! Motivation (FULCRUM-measured, 2026-05-29): on incompressible input the
//! speculative parallel-SM pipeline's spacing-aligned block-finder never lands
//! on a real boundary (228 header + 69 body speculation failures observed on
//! `random100.gz`), so the `parallel_sm_unprofitable` ratio gate routes such
//! input to single-thread libdeflate — which does not scale with threads.
//!
//! The fix is grounded in DEFLATE's framing: a stored block (RFC 1951 §3.2.4)
//! carries an EXPLICIT byte-aligned `LEN`/`NLEN` followed by `LEN` raw literal
//! bytes, and (because it ends byte-aligned) the next block header is itself
//! byte-aligned. So a stored region splits for decode WITHOUT speculation —
//! walk the chain reading explicit lengths, partition the block list, and copy
//! the literal bytes in parallel. No marker bootstrap, no Huffman boundary
//! hunting, no re-decode storms.
//!
//! Two stream shapes:
//!   * Pure stored (e.g. `gzip -1`/`-9`/zlib-L0 on random data): the WHOLE
//!     output is parallel-copied. Measured on neurotic (frozen, interleaved
//!     A/B, byte-exact): pure-100 MB p8 +47% vs single-thread libdeflate, at
//!     PARITY with rapidgzip.
//!   * Stored prefix + Huffman tail (the real `random100.gz`: ~65% stored then
//!     a dynamic-Huffman tail): the prefix is parallel-copied; the tail — which
//!     has no explicit length — is decoded sequentially by the ISA-L bulk
//!     decoder (`isal_lut_bulk`). The sequential tail is an Amdahl ceiling:
//!     measured random100 p8 +12% vs libdeflate, ~0.77× rapidgzip (the
//!     un-parallelised Huffman tail is the remaining gap; parallelising it
//!     needs the window-map machinery the speculative pipeline already has).
//!
//! Safety contract (correctness is sacred — see CLAUDE.md Rule 4 / Rule 5):
//!   * The stored-chain walk is byte-exact: every stored block's extent comes
//!     from its explicit `LEN`, never a guess. The Huffman tail is decoded by
//!     the proven `isal_lut_bulk` per-block decoder into the SAME output buffer
//!     so its back-references resolve directly against the materialised prefix
//!     (no separate 32 KiB window). On targets without that bulk decoder
//!     (non-x86 / no isal-pure-rust) a Huffman tail makes us return
//!     [`StoredSplitError::NotStoredDominated`] WITHOUT touching the writer, so
//!     the dispatcher falls through to the safe one-shot path — same bytes.
//!   * CRC32 + ISIZE are verified against the gzip trailer before any byte is
//!     written; a mismatch is a terminal `Err` (no partial output, no fallback).
//!     This is STRICTER than the streaming parallel-SM path (which writes as it
//!     goes); stored decode buffers the whole output (sized exactly from ISIZE)
//!     so verification precedes the single write.

use std::io::{self, Write};

use crate::decompress::parallel::crc32::{combine_crc32, crc32};
use crate::decompress::parallel::gzip_format;

/// Env-gated phase timing for the stored decode path. When
/// `GZIPPY_STORED_PHASE_TIMING=1` is set, each phase's wall time is printed to
/// stderr. Measurement-only; zero cost when the env is unset (the closure body
/// is skipped and `Instant::now` is never called).
#[inline]
fn phase_timing_enabled() -> bool {
    std::env::var_os("GZIPPY_STORED_PHASE_TIMING").is_some()
}

#[inline]
fn time_phase<T>(name: &str, f: impl FnOnce() -> T) -> T {
    if phase_timing_enabled() {
        let t0 = std::time::Instant::now();
        let r = f();
        eprintln!("[stored-phase] {name}: {:?}", t0.elapsed());
        r
    } else {
        f()
    }
}

/// A decoded stored-block descriptor: where its raw literal bytes live in the
/// compressed input and where they land in the decompressed output.
#[derive(Clone, Copy)]
struct StoredRun {
    /// Byte offset of the first literal byte in the gzip-compressed input.
    src_off: usize,
    /// Byte offset where these literals land in the decompressed output.
    out_off: usize,
    /// Number of literal bytes (== the block's LEN field). May be 0 (an
    /// empty stored block, commonly the BFINAL terminator zlib/pigz emit).
    len: usize,
}

#[derive(Debug)]
pub enum StoredSplitError {
    /// The stream is not 100% stored blocks (a fixed/dynamic Huffman block was
    /// reached). The caller must decode this via the normal safe path — this is
    /// NOT an error condition for the input, only a "wrong specialised decoder"
    /// signal. No bytes were written.
    NotStoredDominated,
    /// The gzip header/trailer could not be parsed (truncated / malformed).
    InvalidFormat,
    /// A stored block's `LEN`/`NLEN` is inconsistent, or a block runs past the
    /// end of the input. Terminal corruption.
    Corrupt(&'static str),
    /// Decoded output size disagrees with the gzip ISIZE trailer. Terminal.
    SizeMismatch { expected: usize, actual: usize },
    /// Decoded CRC32 disagrees with the gzip trailer. Terminal corruption.
    CrcMismatch { expected: u32, actual: u32 },
    /// I/O error writing the verified output.
    Io(io::Error),
}

impl std::fmt::Display for StoredSplitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoredSplitError::NotStoredDominated => write!(f, "not a pure-stored stream"),
            StoredSplitError::InvalidFormat => write!(f, "invalid gzip header/trailer"),
            StoredSplitError::Corrupt(s) => write!(f, "corrupt stored stream: {s}"),
            StoredSplitError::SizeMismatch { expected, actual } => {
                write!(
                    f,
                    "stored output size mismatch: expected {expected}, got {actual}"
                )
            }
            StoredSplitError::CrcMismatch { expected, actual } => write!(
                f,
                "stored CRC32 mismatch: expected {expected:08x}, got {actual:08x}"
            ),
            StoredSplitError::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for StoredSplitError {}

impl From<io::Error> for StoredSplitError {
    fn from(e: io::Error) -> Self {
        StoredSplitError::Io(e)
    }
}

/// Cheap, allocation-free classifier predicate: does the first deflate block of
/// this gzip stream look like a STORED block (BTYPE=00)?
///
/// Used by the router to send stored-dominated input here instead of to the
/// single-thread one-shot. This is a HEURISTIC routing hint only — correctness
/// does not depend on it. If it mis-fires (routes a non-pure-stored stream
/// here), [`decompress_stored_parallel`] walks the chain, returns
/// `NotStoredDominated`, and the dispatcher uses the safe path. So the predicate
/// only needs to be *cheap* and *usually right* for incompressible input, never
/// exact.
///
/// Returns `false` (decline the stored path) on any parse failure.
pub fn first_block_is_stored(gzip_data: &[u8]) -> bool {
    let header_size = match gzip_format::read_header(gzip_data) {
        Ok((_h, off)) => off,
        Err(_) => return false,
    };
    // Need the trailer too (8 bytes) plus at least the first deflate byte.
    if gzip_data.len() < header_size + 8 + 1 {
        return false;
    }
    // The deflate stream starts byte-aligned at `header_size`. The first
    // block's BFINAL (bit 0) + BTYPE (bits 1-2) are the low 3 bits of that
    // byte (LSB-first bit order, RFC 1951 §3.1.1).
    let first = gzip_data[header_size];
    let btype = (first >> 1) & 0b11;
    btype == 0
}

/// How the maximal stored prefix ended.
enum WalkEnd {
    /// A BFINAL stored block was reached: the whole stream is stored. `p` is the
    /// byte offset (into the deflate slice) where the gzip trailer begins.
    Final { deflate_end: usize },
    /// A non-stored (Huffman) block was reached at byte offset `tail_byte`
    /// (into the deflate slice), which is byte-aligned because it follows a
    /// stored block. `prefix_out` is the total output bytes produced by the
    /// stored prefix so far (== where the tail's output starts). Decoding the
    /// tail requires a real DEFLATE decoder (it has no explicit length).
    HuffmanTail { tail_byte: usize, prefix_out: usize },
}

/// Walk the deflate block chain accepting stored blocks (BTYPE=00) by their
/// explicit `LEN`. Returns the stored runs found plus how the walk ended
/// ([`WalkEnd`]).
///
/// Byte-exactness: in a stored region every block header is byte-aligned (a
/// stored block ends on a byte boundary, so its successor's 3-bit header
/// occupies the low bits of a fresh byte). We exploit that to compute each
/// stored block's extent from its `LEN` with no bit-level decoding, and — when
/// a Huffman block appears — to report its byte-aligned start so a real decoder
/// can take over from exactly there.
fn walk_stored_chain(
    deflate: &[u8],
    base_off: usize,
) -> Result<(Vec<StoredRun>, WalkEnd), StoredSplitError> {
    let mut runs: Vec<StoredRun> = Vec::new();
    let mut p: usize = 0; // byte cursor into `deflate`
    let mut out_off: usize = 0;
    let n = deflate.len();

    loop {
        if p >= n {
            // Ran off the end without a BFINAL block — malformed for our
            // purposes. Treat as corruption (the safe path will surface the
            // real error if it disagrees).
            return Err(StoredSplitError::Corrupt(
                "deflate stream ended without BFINAL",
            ));
        }
        let header_byte = deflate[p];
        let bfinal = header_byte & 1;
        let btype = (header_byte >> 1) & 0b11;
        if btype != 0 {
            // Fixed (01) / dynamic (10) Huffman, or reserved (11). The stored
            // prefix ends here; the tail starts at this byte-aligned offset.
            return Ok((
                runs,
                WalkEnd::HuffmanTail {
                    tail_byte: p,
                    prefix_out: out_off,
                },
            ));
        }

        // Skip the partial byte holding the 3 header bits → next byte boundary.
        let len_off = p + 1;
        if len_off + 4 > n {
            return Err(StoredSplitError::Corrupt("truncated LEN/NLEN"));
        }
        let len = u16::from_le_bytes([deflate[len_off], deflate[len_off + 1]]) as usize;
        let nlen = u16::from_le_bytes([deflate[len_off + 2], deflate[len_off + 3]]);
        if (len as u16) != !nlen {
            return Err(StoredSplitError::Corrupt("LEN != ~NLEN"));
        }
        let data_start = len_off + 4;
        let data_end = data_start + len;
        if data_end > n {
            return Err(StoredSplitError::Corrupt(
                "stored block runs past end of input",
            ));
        }
        if len > 0 {
            runs.push(StoredRun {
                src_off: base_off + data_start,
                out_off,
                len,
            });
            out_off += len;
        }
        p = data_end;
        if bfinal == 1 {
            // Successor of a stored block is byte-aligned; the trailer starts at
            // `base_off + p`. Return p relative to the deflate slice.
            return Ok((runs, WalkEnd::Final { deflate_end: p }));
        }
    }
}

/// Decode a stored-dominated single-member gzip buffer in parallel.
///
/// Returns `Ok(total_bytes_written)` on success. Returns
/// [`StoredSplitError::NotStoredDominated`] (without writing) if the stream is
/// not 100% stored — the caller MUST then decode via the safe one-shot path.
/// Any other `Err` is terminal corruption (no fallback; matches the
/// parallel-SM no-fallback contract).
pub fn decompress_stored_parallel<W: Write>(
    gzip_data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> Result<u64, StoredSplitError> {
    let (_hdr, header_size) =
        gzip_format::read_header(gzip_data).map_err(|_| StoredSplitError::InvalidFormat)?;
    let trailer_size = 8;
    if gzip_data.len() < header_size + trailer_size {
        return Err(StoredSplitError::InvalidFormat);
    }
    let footer = gzip_format::read_footer(gzip_data, gzip_data.len() - trailer_size)
        .map_err(|_| StoredSplitError::InvalidFormat)?;
    let expected_crc = footer.crc32;
    let expected_size = footer.uncompressed_size as usize;

    let deflate = &gzip_data[header_size..gzip_data.len() - trailer_size];
    let (runs, walk_end) = walk_stored_chain(deflate, header_size)?;

    match walk_end {
        WalkEnd::Final { deflate_end } => {
            // Pure-stored stream: parallel-copy the entire output.
            //
            // Sanity: after the final block the trailer must begin exactly at
            // the deflate slice end. A short trailing gap would indicate either
            // trailing bytes we don't understand or a multi-member stream —
            // defer to the safe path rather than risk a silent truncation.
            if deflate_end != deflate.len() {
                return Err(StoredSplitError::NotStoredDominated);
            }
            let total: usize = runs.iter().map(|r| r.len).sum();
            if total != expected_size {
                return Err(StoredSplitError::SizeMismatch {
                    expected: expected_size,
                    actual: total,
                });
            }
            let mut output = time_phase("alloc_zero", || vec![0u8; total]);
            let crc = time_phase("fill_and_crc", || {
                fill_and_crc(&mut output, deflate, header_size, &runs, num_threads)
            });
            time_phase("verify_write", || {
                verify_and_write(writer, &output, crc, expected_crc, expected_size)
            })
        }
        WalkEnd::HuffmanTail {
            tail_byte,
            prefix_out,
        } => decode_with_huffman_tail(
            writer,
            deflate,
            header_size,
            &runs,
            tail_byte,
            prefix_out,
            expected_crc,
            expected_size,
            num_threads,
        ),
    }
}

/// Verify the decoded buffer against the gzip trailer, then write it. CRC + size
/// are checked BEFORE any byte reaches the writer (no partial output on
/// corruption).
fn verify_and_write<W: Write>(
    writer: &mut W,
    output: &[u8],
    crc: u32,
    expected_crc: u32,
    expected_size: usize,
) -> Result<u64, StoredSplitError> {
    if output.len() != expected_size {
        return Err(StoredSplitError::SizeMismatch {
            expected: expected_size,
            actual: output.len(),
        });
    }
    if crc != expected_crc {
        return Err(StoredSplitError::CrcMismatch {
            expected: expected_crc,
            actual: crc,
        });
    }
    writer.write_all(output)?;
    writer.flush()?;
    Ok(output.len() as u64)
}

/// Mixed stream: a stored PREFIX followed by a Huffman tail. The prefix's
/// literals are copied in parallel (bandwidth-bound); the tail — which has no
/// explicit length — is decoded sequentially with the proven ISA-L bulk block
/// decoder into the same output buffer, so its back-references resolve directly
/// against the already-materialised prefix (no separate 32 KiB window needed).
///
/// On non-x86 (or non-ISA-L/pure-rust) builds the bulk decoder is unavailable,
/// so we decline (`NotStoredDominated`) and let the safe one-shot path decode
/// the whole stream — same byte-exact result, just not parallel.
#[allow(clippy::too_many_arguments)]
fn decode_with_huffman_tail<W: Write>(
    writer: &mut W,
    deflate: &[u8],
    base_off: usize,
    prefix_runs: &[StoredRun],
    tail_byte: usize,
    prefix_out: usize,
    expected_crc: u32,
    expected_size: usize,
    num_threads: usize,
) -> Result<u64, StoredSplitError> {
    // The tail's output cannot be laid out without decoding, but we know the
    // total from ISIZE, so the tail must produce exactly `expected_size -
    // prefix_out` bytes. Guard the obviously-impossible case up front.
    if prefix_out > expected_size {
        return Err(StoredSplitError::SizeMismatch {
            expected: expected_size,
            actual: prefix_out,
        });
    }

    // The ISA-L bulk per-block decoder (`isal_lut_bulk`) is available exactly
    // when the `parallel_sm` cfg is set (x86_64 + isal/pure-rust, OR aarch64 +
    // pure-rust). Where it is not, decline so the safe one-shot path decodes the
    // whole stream — same bytes, just not parallel.
    #[cfg(parallel_sm)]
    {
        let mut output = time_phase("alloc_zero", || vec![0u8; expected_size]);
        // 1) Parallel-copy the stored prefix into output[0..prefix_out].
        time_phase("prefix_copy", || {
            let (prefix_buf, _tail_buf) = output.split_at_mut(prefix_out);
            // Reuse the parallel filler (it computes a CRC we ignore here; the
            // whole-buffer CRC is taken once at the end so prefix + tail fold
            // together without a separate combine).
            let _ = fill_and_crc(prefix_buf, deflate, base_off, prefix_runs, num_threads);
        });
        // 2) Sequentially decode the Huffman tail into output[prefix_out..].
        //    Back-refs reach into output[..out_pos] (the materialised prefix),
        //    so predecessor_window is unused.
        time_phase("huffman_tail", || {
            decode_tail_blocks(&deflate[tail_byte..], &mut output, prefix_out)
        })?;

        let crc = time_phase("crc32", || crc32(&output));
        time_phase("verify_write", || {
            verify_and_write(writer, &output, crc, expected_crc, expected_size)
        })
    }
    #[cfg(not(parallel_sm))]
    {
        let _ = (
            writer,
            deflate,
            base_off,
            prefix_runs,
            tail_byte,
            expected_crc,
            num_threads,
        );
        Err(StoredSplitError::NotStoredDominated)
    }
}

/// Decode the Huffman tail (a byte-aligned suffix of the deflate stream) into
/// `output[start..]` using the ISA-L bulk per-block decoder, looping until the
/// BFINAL block. `output[..start]` already holds the decoded prefix; all
/// back-references resolve there (`predecessor_window` is empty).
#[cfg(parallel_sm)]
fn decode_tail_blocks(
    tail: &[u8],
    output: &mut [u8],
    start: usize,
) -> Result<(), StoredSplitError> {
    use crate::decompress::inflate::consume_first_decode::Bits;
    use crate::decompress::parallel::isal_lut_bulk::{decode_block, DecoderScratch};

    let mut bits = Bits::new(tail);
    let mut out_pos = start;
    let mut scratch = DecoderScratch::new();
    loop {
        let result = decode_block(&mut bits, output, &mut out_pos, &[], &mut scratch)
            .map_err(|_| StoredSplitError::Corrupt("huffman tail decode failed"))?;
        if result.is_final_block {
            break;
        }
        if out_pos >= output.len() {
            // No room left but not final — size disagreement; surface it.
            return Err(StoredSplitError::Corrupt("huffman tail overran output"));
        }
    }
    if out_pos != output.len() {
        return Err(StoredSplitError::SizeMismatch {
            expected: output.len(),
            actual: out_pos,
        });
    }
    Ok(())
}

/// Copy every run's literals into `output` (disjoint output ranges → no
/// synchronisation) and compute the whole-stream CRC32 by combining per-
/// partition CRCs in output order. Partitions are contiguous runs of blocks so
/// their output ranges are contiguous and their CRCs fold left-to-right.
fn fill_and_crc(
    output: &mut [u8],
    deflate: &[u8],
    base_off: usize,
    runs: &[StoredRun],
    num_threads: usize,
) -> u32 {
    let total = output.len();
    if runs.is_empty() || total == 0 {
        // CRC32 of the empty stream is 0 (gzip stores crc32(b"") == 0).
        return 0;
    }

    let threads = num_threads.max(1).min(num_cpus::get_physical().max(1));
    // Below this many threads (or for tiny output) the parallel split's
    // per-partition CRC-combine overhead is not worth it — do it inline.
    // `GZIPPY_STORED_INLINE_COPY=1` forces the inline path at any thread count
    // (A/B knob: stored decode is partly bandwidth-bound, so whether the
    // partitioned copy+CRC pays depends on the box's memory subsystem).
    let force_inline = std::env::var_os("GZIPPY_STORED_INLINE_COPY").is_some();
    if force_inline || threads <= 1 || total < 1 << 20 {
        copy_runs(output, deflate, base_off, runs);
        return crc32(output);
    }

    // Partition the run list into `threads` contiguous groups, balanced by
    // output bytes (not run count) so a few huge blocks don't skew load.
    let parts = partition_runs(runs, total, threads);

    // Per-partition (crc, out_len) results, indexed by partition for ordered
    // combine. Each partition writes a disjoint slice of `output`.
    let mut results: Vec<(u32, usize)> = vec![(0u32, 0usize); parts.len()];

    // Split `output` into the per-partition disjoint slices up front so each
    // worker gets an exclusive &mut to its range (no aliasing, no unsafe).
    let mut out_slices: Vec<&mut [u8]> = Vec::with_capacity(parts.len());
    {
        let mut rest = &mut output[..];
        for part in &parts {
            let part_out = part_out_bytes(runs, part);
            let (head, tail) = rest.split_at_mut(part_out);
            out_slices.push(head);
            rest = tail;
        }
        // `rest` should be empty (partitions cover all output).
        debug_assert!(
            rest.is_empty(),
            "partition output slices must tile the buffer"
        );
    }

    std::thread::scope(|scope| {
        for ((part, out_slice), result) in parts.iter().zip(out_slices).zip(results.iter_mut()) {
            let runs_part = &runs[part.clone()];
            scope.spawn(move || {
                // Each run's out_off is absolute; translate to slice-local by
                // subtracting the partition's first run's out_off.
                let local_base = runs_part.first().map(|r| r.out_off).unwrap_or(0);
                for r in runs_part {
                    let dst = r.out_off - local_base;
                    out_slice[dst..dst + r.len].copy_from_slice(
                        &deflate[r.src_off - base_off..r.src_off - base_off + r.len],
                    );
                }
                *result = (crc32(out_slice), out_slice.len());
            });
        }
    });

    // Fold partition CRCs left-to-right in output order.
    let mut acc_crc = results[0].0;
    for (crc, len) in results.iter().skip(1) {
        acc_crc = combine_crc32(acc_crc, *crc, *len as u64);
    }
    acc_crc
}

/// Inline (single-threaded) copy of all runs.
fn copy_runs(output: &mut [u8], deflate: &[u8], base_off: usize, runs: &[StoredRun]) {
    for r in runs {
        let src = r.src_off - base_off;
        output[r.out_off..r.out_off + r.len].copy_from_slice(&deflate[src..src + r.len]);
    }
}

/// Sum of output bytes covered by a contiguous partition (run range).
fn part_out_bytes(runs: &[StoredRun], part: &std::ops::Range<usize>) -> usize {
    runs[part.clone()].iter().map(|r| r.len).sum()
}

/// Partition `runs` into ≤ `threads` contiguous index ranges, each holding
/// roughly `total / threads` output bytes. Contiguity guarantees each
/// partition's output range is contiguous (so CRCs combine in order) and
/// disjoint (so workers never alias).
fn partition_runs(runs: &[StoredRun], total: usize, threads: usize) -> Vec<std::ops::Range<usize>> {
    let target = total.div_ceil(threads).max(1);
    let mut parts: Vec<std::ops::Range<usize>> = Vec::with_capacity(threads);
    let mut start = 0usize;
    let mut acc = 0usize;
    for (i, r) in runs.iter().enumerate() {
        acc += r.len;
        // Close the partition once it reaches the target, but never create more
        // than `threads` partitions (the last one absorbs the remainder).
        if acc >= target && parts.len() < threads - 1 {
            parts.push(start..i + 1);
            start = i + 1;
            acc = 0;
        }
    }
    if start < runs.len() {
        parts.push(start..runs.len());
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a gzip stream of `payload` forced into STORED blocks of size
    /// `block` by re-framing flate2's stored output is fiddly; instead we
    /// hand-build the gzip envelope around stored deflate blocks directly.
    fn gzip_stored(payload: &[u8], block: usize) -> Vec<u8> {
        let mut deflate = Vec::new();
        if payload.is_empty() {
            // single empty BFINAL stored block
            deflate.push(0x01); // bfinal=1, btype=00
            deflate.extend_from_slice(&0u16.to_le_bytes());
            deflate.extend_from_slice(&(!0u16).to_le_bytes());
        } else {
            let mut off = 0;
            while off < payload.len() {
                let end = (off + block).min(payload.len());
                let chunk = &payload[off..end];
                let last = end == payload.len();
                deflate.push(if last { 0x01 } else { 0x00 }); // bfinal, btype=00
                let len = chunk.len() as u16;
                deflate.extend_from_slice(&len.to_le_bytes());
                deflate.extend_from_slice(&(!len).to_le_bytes());
                deflate.extend_from_slice(chunk);
                off = end;
            }
        }
        // gzip envelope: 10-byte header + deflate + crc32 + isize
        let mut gz = vec![0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff];
        gz.extend_from_slice(&deflate);
        let crc = crc32(payload);
        gz.extend_from_slice(&crc.to_le_bytes());
        gz.extend_from_slice(&((payload.len() as u32).to_le_bytes()));
        gz
    }

    fn roundtrip(payload: &[u8], block: usize, threads: usize) {
        let gz = gzip_stored(payload, block);
        assert!(first_block_is_stored(&gz) || payload.is_empty());
        let mut out = Vec::new();
        let n = decompress_stored_parallel(&gz, &mut out, threads).expect("decode");
        assert_eq!(n as usize, payload.len());
        assert_eq!(
            out, payload,
            "byte-exact mismatch (block={block}, threads={threads})"
        );
    }

    #[test]
    fn empty_payload() {
        roundtrip(b"", 64, 4);
    }

    #[test]
    fn single_small_block() {
        roundtrip(b"hello, stored world!", 1024, 4);
    }

    #[test]
    fn many_blocks_single_thread() {
        let payload: Vec<u8> = (0..200_000).map(|i| (i * 31 + 7) as u8).collect();
        roundtrip(&payload, 4096, 1);
    }

    #[test]
    fn many_blocks_multi_thread() {
        let payload: Vec<u8> = (0..5_000_000)
            .map(|i| ((i ^ (i >> 3)) * 17) as u8)
            .collect();
        roundtrip(&payload, 65535, 8);
    }

    #[test]
    fn max_size_blocks_straddle_partitions() {
        // 65535 is the max stored LEN; exercises block boundaries that don't
        // align to partition boundaries.
        let payload: Vec<u8> = (0..3_000_000).map(|i| (i % 253) as u8).collect();
        for t in [1usize, 2, 3, 4, 7, 8] {
            roundtrip(&payload, 65535, t);
        }
    }

    #[test]
    fn empty_trailing_block_after_data() {
        // data blocks then a final empty BFINAL stored block (zlib/pigz style).
        let payload: Vec<u8> = (0..100_000).map(|i| (i * 13) as u8).collect();
        let mut deflate = Vec::new();
        let mut off = 0;
        let block = 16384;
        while off < payload.len() {
            let end = (off + block).min(payload.len());
            let chunk = &payload[off..end];
            deflate.push(0x00); // non-final, btype=00
            let len = chunk.len() as u16;
            deflate.extend_from_slice(&len.to_le_bytes());
            deflate.extend_from_slice(&(!len).to_le_bytes());
            deflate.extend_from_slice(chunk);
            off = end;
        }
        // final empty BFINAL block
        deflate.push(0x01);
        deflate.extend_from_slice(&0u16.to_le_bytes());
        deflate.extend_from_slice(&(!0u16).to_le_bytes());

        let mut gz = vec![0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff];
        gz.extend_from_slice(&deflate);
        gz.extend_from_slice(&crc32(&payload).to_le_bytes());
        gz.extend_from_slice(&(payload.len() as u32).to_le_bytes());

        let mut out = Vec::new();
        let n = decompress_stored_parallel(&gz, &mut out, 4).expect("decode");
        assert_eq!(n as usize, payload.len());
        assert_eq!(out, payload);
    }

    #[test]
    fn huffman_first_block() {
        use std::io::Write as _;
        // A real flate2 deflate stream (dynamic Huffman) has NO stored prefix.
        // Production never routes such a stream here (first_block_is_stored is
        // false), but a direct call must still be correct:
        //   * on x86 the empty-prefix + Huffman-tail path decodes it byte-exact,
        //   * on other platforms (no bulk decoder) it declines without writing.
        let payload: Vec<u8> = (0..50_000).map(|i| (i % 7) as u8).collect();
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(6));
        enc.write_all(&payload).unwrap();
        let gz = enc.finish().unwrap();
        // first block must be Huffman (not stored) for this fixture.
        assert!(!first_block_is_stored(&gz));

        let mut out = Vec::new();
        let r = decompress_stored_parallel(&gz, &mut out, 4);
        #[cfg(parallel_sm)]
        {
            assert_eq!(r.map(|n| n as usize).unwrap(), payload.len());
            assert_eq!(out, payload, "empty-prefix Huffman-tail must decode");
        }
        #[cfg(not(parallel_sm))]
        {
            match r {
                Err(StoredSplitError::NotStoredDominated) => {}
                other => panic!("expected NotStoredDominated, got {other:?}"),
            }
            assert!(out.is_empty(), "must not write on NotStoredDominated");
        }
    }

    /// The random100.gz shape: a long STORED prefix followed by a Huffman tail
    /// (one valid single-member deflate stream). Where the bulk decoder exists
    /// (`parallel_sm`) the prefix is copied in parallel and the tail decoded by
    /// the ISA-L bulk decoder; output must be byte-exact. Elsewhere it declines
    /// to the safe path.
    #[test]
    fn stored_prefix_then_huffman_tail() {
        use std::io::{Read as _, Write as _};
        // Incompressible prefix (stored blocks) + compressible suffix (Huffman).
        let mut payload = vec![0u8; 1_500_000];
        let mut state = 0xfeed_face_dead_beefu64;
        for b in &mut payload {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (state >> 33) as u8;
        }
        payload.resize(payload.len() + 1_000_000, 0u8); // compressible tail
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(6));
        enc.write_all(&payload).unwrap();
        let gz = enc.finish().unwrap();

        // Confirm the fixture really is stored-prefix + huffman-tail.
        assert!(first_block_is_stored(&gz), "fixture must start stored");

        // Oracle.
        let mut oracle = Vec::new();
        flate2::read::GzDecoder::new(&gz[..])
            .read_to_end(&mut oracle)
            .unwrap();
        assert_eq!(oracle, payload, "oracle sanity");

        let mut out = Vec::new();
        let r = decompress_stored_parallel(&gz, &mut out, 8);
        #[cfg(parallel_sm)]
        {
            assert_eq!(r.map(|n| n as usize).unwrap(), payload.len());
            assert_eq!(out, payload, "stored-prefix+huffman-tail must decode");
            assert_eq!(out, oracle);
        }
        #[cfg(not(parallel_sm))]
        {
            match r {
                Err(StoredSplitError::NotStoredDominated) => {}
                other => panic!("expected NotStoredDominated, got {other:?}"),
            }
        }
    }

    #[test]
    fn corrupt_nlen_is_terminal() {
        let payload = vec![7u8; 1000];
        let mut gz = gzip_stored(&payload, 1024);
        // Corrupt the NLEN of the (only) stored block: header(10) + bfinal(1)
        // + LEN(2) → NLEN at offset 13.
        gz[13] ^= 0xFF;
        let mut out = Vec::new();
        match decompress_stored_parallel(&gz, &mut out, 4) {
            Err(StoredSplitError::Corrupt(_)) => {}
            other => panic!("expected Corrupt, got {other:?}"),
        }
    }

    #[test]
    fn crc_mismatch_is_terminal() {
        let payload = vec![42u8; 5000];
        let mut gz = gzip_stored(&payload, 1024);
        // Corrupt the trailing CRC32 (last 8 bytes are crc(4) + isize(4)).
        let crc_pos = gz.len() - 8;
        gz[crc_pos] ^= 0xFF;
        let mut out = Vec::new();
        match decompress_stored_parallel(&gz, &mut out, 4) {
            Err(StoredSplitError::CrcMismatch { .. }) => {}
            other => panic!("expected CrcMismatch, got {other:?}"),
        }
        // No partial output on a verification failure.
        assert!(out.is_empty());
    }

    #[test]
    fn matches_flate2_on_stored_random() {
        // Cross-check against an independent oracle on incompressible data.
        let mut payload = vec![0u8; 2_000_000];
        let mut state = 0x1234_5678_9abc_def0u64;
        for b in &mut payload {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (state >> 33) as u8;
        }
        let gz = gzip_stored(&payload, 65535);
        // oracle: flate2 read decoder
        let mut oracle = Vec::new();
        {
            use std::io::Read;
            let mut d = flate2::read::GzDecoder::new(&gz[..]);
            d.read_to_end(&mut oracle).unwrap();
        }
        assert_eq!(oracle, payload, "oracle sanity");

        let mut out = Vec::new();
        decompress_stored_parallel(&gz, &mut out, 8).expect("decode");
        assert_eq!(out, payload);
        assert_eq!(out, oracle);
    }
}
