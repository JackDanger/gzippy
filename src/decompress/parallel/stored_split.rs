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
//!     decoder (`lut_bulk_inflate`). The sequential tail is an Amdahl ceiling:
//!     measured random100 p8 +12% vs libdeflate, ~0.77× rapidgzip (the
//!     un-parallelised Huffman tail is the remaining gap; parallelising it
//!     needs the window-map machinery the speculative pipeline already has).
//!
//! Safety contract (correctness is sacred — see CLAUDE.md Rule 4 / Rule 5):
//!   * The stored-chain walk is byte-exact: every stored block's extent comes
//!     from its explicit `LEN`, never a guess. The Huffman tail is decoded by
//!     the proven `lut_bulk_inflate` per-block decoder into the SAME output buffer
//!     so its back-references resolve directly against the materialised prefix
//!     (no separate 32 KiB window). On targets without that bulk decoder
//!     (non-x86 / no isal-pure-rust) a Huffman tail makes us return
//!     [`StoredSplitError::NotStoredDominated`] WITHOUT touching the writer, so
//!     the dispatcher falls through to the safe one-shot path — same bytes.
//!   * CRC32 + ISIZE are verified against the gzip trailer before any byte is
//!     written; a mismatch is a terminal `Err` (no partial output, no fallback).
//!     This is STRICTER than the streaming parallel-SM path (which writes as it
//!     goes). For a PURE-stored stream the output bytes ARE the verbatim input
//!     run slices, so verification reads the input runs (computing CRC) and the
//!     trailer is checked BEFORE the runs are streamed directly from the input
//!     to the writer — NO monolithic output buffer is allocated (rapidgzip-style
//!     chunked streaming; the old `vec![0u8; total]` fault-storm is gone). A
//!     stored-prefix + Huffman tail still buffers (the tail has no explicit
//!     length and its back-refs resolve against the materialised prefix).

use std::io::{self, Write};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::decompress::parallel::crc32::{combine_crc32, crc32};
use crate::decompress::parallel::gzip_format;
use crc32fast::Hasher;

/// Counter: number of times StoredParallel was demoted to ParallelSM because
/// the Huffman tail accounts for >= 50% of total output (prefix_out < 50% of
/// expected_size). Dumped by `GZIPPY_DEBUG=1`.
pub static STORED_DEMOTE_TO_PARALLEL_SM: AtomicU64 = AtomicU64::new(0);

/// Counter: number of pure-stored streams decoded via the chunked-streaming
/// path that copies the verbatim input run slices straight to the writer with
/// NO monolithic output buffer. Non-zero proves the streaming path ran (Gate-0
/// non-inert witness). Dumped by `GZIPPY_DEBUG=1`.
pub static STORED_STREAM_RUNS: AtomicU64 = AtomicU64::new(0);

/// Counter: number of pure-stored streams whose runs were emitted via the
/// `writev` iovec-GATHER fast path (all run slices gathered into batched
/// `writev` syscalls) instead of one `write_all` per run. Non-zero proves the
/// coalesced-write path fired (Gate-0 non-inert witness for the syscall-count
/// collapse). Dumped by `GZIPPY_DEBUG=1`.
pub static STORED_WRITEV_BATCHES: AtomicU64 = AtomicU64::new(0);

/// Counter: number of Huffman ISLANDS decoded IN PLACE by the SEGMENTED stored
/// path (`decode_segmented`) — the scattered dynamic/fixed blocks that punctuate
/// an otherwise-stored stream (storedheavy: 33 islands across 3008 stored runs).
/// Non-zero proves the segmented walk ran and stayed on the stored LEN-chain
/// path instead of demoting the whole stream to ParallelSM (Gate-0 non-inert
/// witness for the anti-demote lever). Dumped by `GZIPPY_DEBUG=1`.
pub static STORED_SEGMENTED_ISLANDS: AtomicU64 = AtomicU64::new(0);

/// Counter: number of SEGMENTED streams whose ordered output segments (stored
/// runs + decoded islands) were emitted via the `writev` iovec-GATHER fast path
/// (all segments gathered into batched `writev` syscalls) instead of one
/// `write_all` per segment. Non-zero proves the segmented coalesced-write path
/// fired (Gate-0 non-inert witness). Dumped by `GZIPPY_DEBUG=1`.
pub static STORED_SEGMENTED_WRITEV_BATCHES: AtomicU64 = AtomicU64::new(0);

/// Threshold: if the stored prefix accounts for < this fraction of total
/// output (numerator/denominator), demote to ParallelSM so the Huffman tail
/// is decoded in parallel. Currently 50% (1/2).
const DEMOTE_THRESHOLD_NUM: usize = 1;
const DEMOTE_THRESHOLD_DEN: usize = 2;

/// Phase wrapper for the stored decode path. Formerly hosted an env-gated
/// per-phase wall-time dump (removed); reduced to a
/// transparent pass-through so the phase call sites keep their structure while
/// carrying zero measurement cost.
#[inline]
fn time_phase<T>(_name: &str, f: impl FnOnce() -> T) -> T {
    f()
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
#[allow(dead_code)] // called only from classify's parallel_sm branch
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
    out_fd: Option<i32>,
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

            // Pure-stored stream: the output bytes ARE the verbatim input run
            // slices — `StoredRun { src_off, len }` indexes straight into the
            // compressed input, so `output[out_off..][..len] ==
            // deflate[src_off-base..][..len]` byte-for-byte, no transform. So we
            // STREAM the runs directly from the input to the writer with NO
            // intermediate buffer, eliminating the old `vec![0u8; total]`
            // 100 MB zero-init first-touch page-fault storm AND the full second
            // copy pass. Faithful to rapidgzip's chunk-by-chunk reused-buffer
            // streaming (`DecodedData.hpp` + the `GzipChunkFetcher` writeAll
            // loop): for stored blocks the "chunk" is the input run slice.
            //
            // Verify-before-write is PRESERVED exactly: the input is fully
            // buffered, so we CRC the runs (output order == run order, since
            // `out_off` ascends) and compare to the trailer BEFORE the first
            // byte reaches the sink. On mismatch a terminal Err with no partial
            // output — identical contract to the old monolithic path.
            let crc = time_phase("crc_runs", || {
                crc_runs(deflate, header_size, &runs, total, num_threads)
            });
            if crc != expected_crc {
                return Err(StoredSplitError::CrcMismatch {
                    expected: expected_crc,
                    actual: crc,
                });
            }
            STORED_STREAM_RUNS.fetch_add(1, Ordering::Relaxed);
            time_phase("stream_write", || {
                write_runs(writer, deflate, header_size, &runs, out_fd)
            })?;
            Ok(total as u64)
        }
        WalkEnd::HuffmanTail {
            tail_byte,
            prefix_out,
        } => {
            // Multi-member defense-in-depth (P0 fix c4918aa2): a stored-dominant
            // FIRST member whose prefix ALREADY exceeds the whole-file (== last
            // member's) ISIZE means a multi-member stream slipped past the
            // router's dominant-first detector. A single-member decoder cannot
            // span members, so DECLINE to the safe multi-member-capable path
            // rather than emit a terminal SizeMismatch on EMPTY output. The
            // router now catches this shape up front at every T; this is belt-
            // and-braces (keeps `test_dominant_first_stored_multi_member_p1_byte_exact`
            // green if the scan window ever slips).
            if prefix_out > expected_size {
                return Err(StoredSplitError::NotStoredDominated);
            }
            // ROUTE 1 — preserved single-tail / empty-prefix behavior (UNCHANGED).
            // If there is NO stored prefix (prefix_out == 0, a direct-call
            // Huffman-first stream), OR the stored prefix before the first
            // Huffman block is ALREADY >= 50% of output, the overlap-friendly
            // single-tail decoder (`decode_with_huffman_tail`, which copies the
            // big prefix in parallel while it decodes the remaining suffix —
            // interspersed stored blocks included — sequentially) is the
            // efficient, byte-exact decoder. Its behavior is preserved EXACTLY.
            //   random100.gz: ~60% prefix ≥ 50% → this route (unchanged).
            if prefix_out == 0
                || prefix_out * DEMOTE_THRESHOLD_DEN >= expected_size * DEMOTE_THRESHOLD_NUM
            {
                return decode_with_huffman_tail(
                    writer,
                    deflate,
                    header_size,
                    &runs,
                    tail_byte,
                    prefix_out,
                    expected_crc,
                    expected_size,
                    num_threads,
                );
            }
            // ROUTE 2 — SEGMENTED walk (the anti-demote lever). The OLD code
            // DEMOTED the ENTIRE stream to the non-scaling ParallelSM machinery
            // the moment an EARLY Huffman island appeared with a <50% prefix
            // (storedheavy: 8.2% stored prefix < 50% → the whole 98.9%-stored
            // 100 MB went to ParallelSM). NEW: walk the WHOLE member — keep the
            // stored runs on the byte-exact LEN-chain parallel-copy path and
            // decode the (rare, scattered) Huffman islands IN PLACE against a
            // true rolling output window. Demote to ParallelSM ONLY if Huffman
            // ACTUALLY dominates (>50% of total output), decided by the walk.
            decode_segmented(
                writer,
                deflate,
                header_size,
                &runs,
                tail_byte,
                prefix_out,
                expected_crc,
                expected_size,
                num_threads,
                out_fd,
            )
        }
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
    // prefix_out` bytes. If the stored prefix ALONE already exceeds
    // `expected_size`, this cannot be the single stored-dominated member we
    // decode: `expected_size` is the WHOLE-FILE trailer's ISIZE, so a larger
    // prefix means we were handed a MULTI-MEMBER stream whose first (dominant)
    // member exceeds the small last member's ISIZE — the router's
    // `is_likely_multi_member` 16 MiB scan window slipped a big first member
    // through as "single-member". DECLINE to the safe multi-member-capable path
    // instead of a terminal error (this path used to emit a spurious
    // `stored output size mismatch: expected <last-ISIZE>, got <member1-size>`
    // and EMPTY output on files `gzip -dc` decodes fine). The router now catches
    // this shape up front (`classify_gzip` dominant-first detection at every T),
    // so this is defense-in-depth for any residual mis-route.
    if prefix_out > expected_size {
        return Err(StoredSplitError::NotStoredDominated);
    }

    // The ISA-L bulk per-block decoder (`lut_bulk_inflate`) is available exactly
    // when the `parallel_sm` cfg is set (x86_64 + isal/pure-rust, OR aarch64 +
    // pure-rust). Where it is not, decline so the safe one-shot path decodes the
    // whole stream — same bytes, just not parallel.
    #[cfg(parallel_sm)]
    {
        let mut output = time_phase("alloc_zero", || vec![0u8; expected_size]);

        // The Huffman tail's back-references reach at most MAX_WINDOW_SIZE
        // (32 KiB) bytes before its first output byte. So the tail decode only
        // depends on the LAST 32 KiB of the stored prefix — not the whole
        // prefix. That lets us OVERLAP the (single-threaded) tail decode with
        // the (parallel) bulk copy of the rest of the prefix: build a 32 KiB
        // predecessor window directly from the runs, then run the tail decode
        // and the full-prefix copy concurrently into disjoint output regions.
        //
        // The overlap path requires `prefix_out >= MAX_WINDOW_SIZE` so the
        // predecessor window is exactly 32 KiB — then for every legal tail
        // back-reference (`distance <= 32 KiB`, validated by `decode_block`)
        // `copy_match`'s window arithmetic is in-bounds. With a shorter prefix
        // the standalone-buffer window could be smaller than a (corrupt)
        // distance; the contiguous sequential path has no such edge, so we use
        // it. Stored-dominated production input always has a multi-MiB prefix,
        // so this guard never excludes the real workload.
        let overlap = prefix_out >= MAX_WINDOW_SIZE;

        let (prefix_crc, tail_crc) = if overlap {
            // Gather the predecessor window (last min(prefix_out, 32 KiB) bytes
            // of the decoded prefix) from the stored runs, independent of the
            // full-prefix copy that runs concurrently below.
            let pred = time_phase("pred_window", || {
                build_predecessor_window(deflate, base_off, prefix_runs, prefix_out)
            });

            let (prefix_buf, tail_buf) = output.split_at_mut(prefix_out);
            let tail_in = &deflate[tail_byte..];

            // Run both halves concurrently: Unit Y parallel-copies the whole
            // prefix (and returns its CRC); Unit X decodes the tail into the
            // disjoint tail buffer, resolving early back-refs against `pred`.
            let mut tail_result: Result<(usize, u32), StoredSplitError> = Ok((0, crc32(&[])));
            let prefix_crc = time_phase("overlap_copy+tail", || {
                let mut pcrc = 0u32;
                std::thread::scope(|scope| {
                    let tr = &mut tail_result;
                    scope.spawn(move || {
                        *tr = decode_tail_into(tail_in, tail_buf, &pred);
                    });
                    // The tail decode occupies one core for the whole overlap,
                    // so the parallel prefix copy gets num_threads-1 to avoid
                    // oversubscribing (copy threads + tail thread <= cores). The
                    // main thread drives the copy's own thread::scope.
                    let copy_threads = num_threads.saturating_sub(1).max(1);
                    pcrc = fill_and_crc(prefix_buf, deflate, base_off, prefix_runs, copy_threads);
                });
                pcrc
            });
            let (tail_len, tcrc) = tail_result?;
            // Guard: the tail must exactly fill the tail buffer (size agreement).
            if tail_len != tail_buf_len(expected_size, prefix_out) {
                return Err(StoredSplitError::SizeMismatch {
                    expected: expected_size,
                    actual: prefix_out + tail_len,
                });
            }
            (prefix_crc, tcrc)
        } else {
            // Sequential path: copy the whole prefix, then decode the tail into
            // output[prefix_out..] (its back-refs resolve in the now-contiguous
            // output). Used when overlap is disabled or there is no prefix.
            let prefix_crc = time_phase("prefix_copy", || {
                let (prefix_buf, _tail_buf) = output.split_at_mut(prefix_out);
                fill_and_crc(prefix_buf, deflate, base_off, prefix_runs, num_threads)
            });
            time_phase("huffman_tail", || {
                decode_tail_blocks(&deflate[tail_byte..], &mut output, prefix_out)
            })?;
            let tail = &output[prefix_out..];
            let tcrc = if tail.is_empty() { 0 } else { crc32(tail) };
            (prefix_crc, tcrc)
        };

        // Fold prefix_crc ⊕ tail_crc in output order. `combine_crc32` is the
        // standard CRC32 concatenation (tested against crc32fast's combine) so
        // prefix(parallel) + tail folds to the exact whole-buffer CRC.
        let tail_len = expected_size - prefix_out;
        let crc = if tail_len == 0 {
            prefix_crc
        } else {
            combine_crc32(prefix_crc, tail_crc, tail_len as u64)
        };
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

/// SEGMENTED (streaming) decode of a stored-DOMINANT member whose stored runs
/// are punctuated by RARE, scattered Huffman islands (storedheavy: 33 dynamic
/// islands across 3008 stored runs, 98.9% of output stored). Walks the WHOLE
/// member: stored blocks are STREAMED straight from the input mmap to the writer
/// (the byte-exact LEN-chain path — NO 100 MB intermediate buffer, exactly like
/// the pure-stored [`WalkEnd::Final`] path); the Huffman islands are decoded into
/// small per-island buffers (rare & small — ~1 MB total for storedheavy).
///
/// # Byte-exactness of the islands (the correctness trap)
/// A Huffman island's LZ77 back-references (distance ≤ 32 KiB) reach into the
/// PRECEDING output — which may be a PRIOR ISLAND'S DECODED bytes (not verbatim
/// input) when two islands fall within 32 KiB, OR a preceding STORED run. Before
/// decoding each island we reconstruct the TRUE rolling ≤32 KiB output window
/// from the ordered segments so far — stored bytes read from the input, prior-
/// island bytes read from their decoded buffers — and feed it to `decode_block`
/// as its `predecessor_window`. So an island always resolves against the actual
/// EMITTED output, making the adversarial two-islands-within-32-KiB case correct
/// by construction. CRC32 + ISIZE are verified over the FULL output BEFORE any
/// byte reaches the writer (no partial output on corruption).
///
/// # Demotion (preserved semantics)
/// Demote to ParallelSM (return [`StoredSplitError::NotStoredDominated`] WITHOUT
/// touching the writer) ONLY if Huffman ACTUALLY dominates — as soon as the
/// running Huffman output exceeds `expected_size / 2` the stored fraction can no
/// longer reach 50%, so we bail (counted in `STORED_DEMOTE_TO_PARALLEL_SM`).
///
/// On non-`parallel_sm` builds the bulk decoder is unavailable → decline.
#[cfg(parallel_sm)]
#[allow(clippy::too_many_arguments)]
fn decode_segmented<W: Write>(
    writer: &mut W,
    deflate: &[u8],
    base_off: usize,
    prefix_runs: &[StoredRun],
    tail_byte: usize,
    prefix_out: usize,
    expected_crc: u32,
    expected_size: usize,
    num_threads: usize,
    out_fd: Option<i32>,
) -> Result<u64, StoredSplitError> {
    use crate::decompress::parallel::lut_bulk_inflate::DecoderScratch;

    let n = deflate.len();
    // Ordered output segments (stored run OR decoded island), in output order.
    let mut segs: Vec<Seg> = Vec::with_capacity(prefix_runs.len() + 128);
    for r in prefix_runs {
        segs.push(Seg::Stored {
            src: r.src_off - base_off,
            len: r.len,
        });
    }
    // Per-island decoded bytes (referenced by Seg::Island.idx). ~1 MB total.
    let mut islands_buf: Vec<Vec<u8>> = Vec::new();

    let mut out_pos = prefix_out;
    let mut huffman_out: usize = 0;
    let mut islands: u64 = 0;
    let demote_limit = expected_size / 2;

    let mut scratch = DecoderScratch::new();
    let mut pred = vec![0u8; MAX_WINDOW_SIZE]; // reused predecessor scratch
    let mut p = tail_byte;

    'walk: loop {
        if p >= n {
            return Err(StoredSplitError::Corrupt(
                "deflate stream ended without BFINAL",
            ));
        }
        let header_byte = deflate[p];
        let btype = (header_byte >> 1) & 0b11;
        if btype == 0 {
            // Byte-aligned stored block: record a streamed segment.
            let bfinal = header_byte & 1;
            let (len, data_start, data_end) = read_stored_lens(deflate, p + 1, n)?;
            if len > 0 {
                segs.push(Seg::Stored {
                    src: data_start,
                    len,
                });
                out_pos += len;
                if out_pos > expected_size {
                    return Err(StoredSplitError::NotStoredDominated);
                }
            }
            p = data_end;
            if bfinal == 1 {
                break 'walk;
            }
            continue 'walk;
        }

        // ── Huffman island (byte-aligned start at `p`) ──────────────────────
        // Reconstruct the TRUE rolling ≤32 KiB output window from the ordered
        // segments emitted so far (stored from input, prior islands from bufs).
        let pred_len = build_pred_streaming(&mut pred, &segs, &islands_buf, deflate, out_pos);

        // Decode the maximal run of consecutive Huffman blocks into an adaptive
        // buffer, growing + retrying on overflow (islands are small; grows rare).
        let mut cap = 128 * 1024;
        let huffman_base = huffman_out;
        let (ibytes, exit) = loop {
            match try_decode_island(
                deflate,
                p * 8,
                n,
                &pred[..pred_len],
                &mut scratch,
                cap,
                demote_limit,
                huffman_base,
            ) {
                Ok(v) => break v,
                Err(IslandErr::Overflow) => {
                    cap = cap.saturating_mul(4);
                    continue;
                }
                Err(IslandErr::Demote) => {
                    STORED_DEMOTE_TO_PARALLEL_SM.fetch_add(1, Ordering::Relaxed);
                    if crate::utils::debug_enabled() {
                        eprintln!(
                            "[gzippy] StoredParallel(segmented) demote → ParallelSM: \
                             huffman_out>{demote_limit} (Huffman dominates)"
                        );
                    }
                    return Err(StoredSplitError::NotStoredDominated);
                }
                Err(IslandErr::Corrupt(m)) => return Err(StoredSplitError::Corrupt(m)),
            }
        };

        let ilen = ibytes.len();
        out_pos += ilen;
        huffman_out += ilen;
        if out_pos > expected_size {
            return Err(StoredSplitError::NotStoredDominated);
        }
        if ilen > 0 {
            let idx = islands_buf.len();
            islands_buf.push(ibytes);
            segs.push(Seg::Island { idx, len: ilen });
            islands += 1;
        }

        match exit {
            IslandExit::FinalHuffman => break 'walk,
            IslandExit::StoredNext { len_off, bfinal } => {
                let (len, data_start, data_end) = read_stored_lens(deflate, len_off, n)?;
                if len > 0 {
                    segs.push(Seg::Stored {
                        src: data_start,
                        len,
                    });
                    out_pos += len;
                    if out_pos > expected_size {
                        return Err(StoredSplitError::NotStoredDominated);
                    }
                }
                if bfinal == 1 {
                    break 'walk;
                }
                p = data_end;
            }
        }
    }

    if out_pos != expected_size {
        return Err(StoredSplitError::SizeMismatch {
            expected: expected_size,
            actual: out_pos,
        });
    }

    // Verify CRC (parallel over ordered segments) BEFORE writing.
    let crc = segmented_crc(&segs, &islands_buf, deflate, num_threads);
    if crc != expected_crc {
        return Err(StoredSplitError::CrcMismatch {
            expected: expected_crc,
            actual: crc,
        });
    }

    STORED_SEGMENTED_ISLANDS.fetch_add(islands, Ordering::Relaxed);
    if crate::utils::debug_enabled() {
        eprintln!(
            "[gzippy] StoredParallel(segmented) ok: {out_pos} bytes; segs={} islands={islands} \
             huffman_out={huffman_out} (stored fraction {:.2}%)",
            segs.len(),
            (expected_size - huffman_out) as f64 / expected_size.max(1) as f64 * 100.0,
        );
    }

    // Stream to the writer in output order (stored straight from the input mmap,
    // islands from their decoded bufs) — NO 100 MB intermediate copy.
    //
    // When the sink's raw fd is available (`out_fd` — production file / stdout
    // sinks), GATHER every ordered segment into batched `writev` syscalls
    // (`writev_all_to_fd`, vendor `writeAllToFdVector`, the same fast path the
    // pure-stored `write_runs` uses) rather than one `write_all` per segment.
    // On storedheavy (~3008 stored runs + islands) that collapses ~3000 serial
    // per-segment `write()` calls — each of which memcpy'd the ~100 MB output
    // through the BufWriter on the critical path (the SERIAL ORDERED-WRITE floor
    // that FALSIFIED the earlier segmented build) — to ⌈segs / IOV_MAX⌉ `writev`
    // calls straight from the input mmap / decoded island bufs (zero userspace
    // copy). BYTE-IDENTICAL: the concatenation of the gathered iovecs, in segment
    // order, equals the old per-segment write order (each iovec borrows the exact
    // slice `write_all` streamed). Verify-before-write is preserved (CRC32 + ISIZE
    // checked above); the pre-`writev` `flush` keeps ordering even if the
    // BufWriter ever held buffered bytes. `islands_buf` outlives the `writev`
    // (dropped only at function return), so every Island iovec pointer stays
    // valid. Falls back to the portable per-segment `write_all` loop when no fd is
    // present (tests / in-memory Vec / non-unix).
    #[cfg(unix)]
    if let Some(fd) = out_fd {
        writer.flush()?;
        let mut iovs: Vec<libc::iovec> = Vec::with_capacity(segs.len());
        for s in &segs {
            let slice: &[u8] = match *s {
                Seg::Stored { src, len } => &deflate[src..src + len],
                Seg::Island { idx, len } => &islands_buf[idx][..len],
            };
            if slice.is_empty() {
                continue;
            }
            iovs.push(libc::iovec {
                iov_base: slice.as_ptr() as *mut libc::c_void,
                iov_len: slice.len(),
            });
        }
        if !iovs.is_empty() {
            crate::decompress::parallel::fd_vectored_write::writev_all_to_fd(fd, &mut iovs)?;
        }
        STORED_SEGMENTED_WRITEV_BATCHES.fetch_add(1, Ordering::Relaxed);
        return Ok(out_pos as u64);
    }
    #[cfg(not(unix))]
    let _ = out_fd;

    for s in &segs {
        match *s {
            Seg::Stored { src, len } => writer.write_all(&deflate[src..src + len])?,
            Seg::Island { idx, len } => writer.write_all(&islands_buf[idx][..len])?,
        }
    }
    writer.flush()?;
    Ok(out_pos as u64)
}

/// Non-`parallel_sm` builds lack the bulk block decoder — decline so the safe
/// one-shot path decodes the whole stream (same byte-exact result, not parallel).
#[cfg(not(parallel_sm))]
#[allow(clippy::too_many_arguments)]
fn decode_segmented<W: Write>(
    _writer: &mut W,
    _deflate: &[u8],
    _base_off: usize,
    _prefix_runs: &[StoredRun],
    _tail_byte: usize,
    _prefix_out: usize,
    _expected_crc: u32,
    _expected_size: usize,
    _num_threads: usize,
    _out_fd: Option<i32>,
) -> Result<u64, StoredSplitError> {
    Err(StoredSplitError::NotStoredDominated)
}

/// An ordered output segment for the segmented streaming path.
#[cfg(parallel_sm)]
#[derive(Clone, Copy)]
enum Seg {
    /// Stored run: `deflate[src..src + len]` streamed verbatim.
    Stored { src: usize, len: usize },
    /// Huffman island: `islands_buf[idx][..len]` decoded output.
    Island { idx: usize, len: usize },
}

#[cfg(parallel_sm)]
impl Seg {
    #[inline]
    fn out_len(&self) -> usize {
        match self {
            Seg::Stored { len, .. } => *len,
            Seg::Island { len, .. } => *len,
        }
    }
    #[inline]
    fn bytes<'a>(&self, islands_buf: &'a [Vec<u8>], deflate: &'a [u8]) -> &'a [u8] {
        match *self {
            Seg::Stored { src, len } => &deflate[src..src + len],
            Seg::Island { idx, len } => &islands_buf[idx][..len],
        }
    }
}

/// How a Huffman island ended.
#[cfg(parallel_sm)]
enum IslandExit {
    /// The island's last Huffman block was BFINAL — the member ends here.
    FinalHuffman,
    /// A stored block (byte-aligned LEN at `len_off`, `bfinal` from its header)
    /// follows the island — the byte-aligned stored walk resumes after it.
    StoredNext { len_off: usize, bfinal: u8 },
}

#[cfg(parallel_sm)]
enum IslandErr {
    /// The provisional island buffer was too small — the caller grows + retries.
    Overflow,
    /// Running Huffman output exceeded `demote_limit` — Huffman dominates.
    Demote,
    /// Terminal decode corruption.
    Corrupt(&'static str),
}

/// Decode the maximal run of consecutive Huffman blocks (one island) starting at
/// bit offset `start_bit`, into a fresh `cap`-byte buffer, resolving pre-island
/// back-references against `pred` (the true rolling output window). Stops at the
/// first STORED block (byte-aligned) or a BFINAL Huffman block. Returns the
/// decoded island bytes plus how it ended. `Err(Overflow)` if the buffer filled
/// (caller grows); `Err(Demote)` if Huffman output crosses `demote_limit`.
#[cfg(parallel_sm)]
#[allow(clippy::too_many_arguments)]
fn try_decode_island(
    deflate: &[u8],
    start_bit: usize,
    n: usize,
    pred: &[u8],
    scratch: &mut crate::decompress::parallel::lut_bulk_inflate::DecoderScratch,
    cap: usize,
    demote_limit: usize,
    huffman_base: usize,
) -> Result<(Vec<u8>, IslandExit), IslandErr> {
    use crate::decompress::inflate::consume_first_decode::Bits;
    use crate::decompress::parallel::lut_bulk_inflate::{decode_block, BulkDecodeError};

    let mut buf = vec![0u8; cap];
    let mut local: usize = 0;
    let mut bits = Bits::at_bit_offset(deflate, start_bit);
    loop {
        bits.refill();
        let bh = (bits.peek() & 0b111) as u8;
        let bfinal = bh & 1;
        let bt = (bh >> 1) & 0b11;
        if bt == 0 {
            // A stored block terminates the island: consume its 3-bit header and
            // align to the byte boundary where LEN/NLEN begin.
            bits.consume(3);
            bits.align_to_byte();
            let bp = bits.bit_position();
            debug_assert!(bp % 8 == 0, "post-island stored block must byte-align");
            let len_off = bp / 8;
            if len_off > n {
                return Err(IslandErr::Corrupt("island stored terminator past end"));
            }
            buf.truncate(local);
            return Ok((buf, IslandExit::StoredNext { len_off, bfinal }));
        }
        // Decode one Huffman block into buf[local..] (out_pos LOCAL; pre-island
        // back-refs resolve against `pred`, within-island against buf[..local]).
        match decode_block(&mut bits, &mut buf, &mut local, pred, scratch) {
            Ok(res) => {
                if huffman_base + local > demote_limit {
                    return Err(IslandErr::Demote);
                }
                if res.is_final_block {
                    buf.truncate(local);
                    return Ok((buf, IslandExit::FinalHuffman));
                }
            }
            Err(BulkDecodeError::OutputOverflow) => return Err(IslandErr::Overflow),
            Err(_) => return Err(IslandErr::Corrupt("huffman island decode failed")),
        }
    }
}

/// Parse a byte-aligned stored block's `LEN`/`NLEN` at byte offset `len_off`
/// (RFC 1951 §3.2.4). Returns `(len, data_start, data_end)` where the `len` raw
/// literal bytes occupy `deflate[data_start..data_end]`. Byte-exact: `LEN` is
/// explicit and validated against `~NLEN`.
#[cfg(parallel_sm)]
#[inline]
fn read_stored_lens(
    deflate: &[u8],
    len_off: usize,
    n: usize,
) -> Result<(usize, usize, usize), StoredSplitError> {
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
    Ok((len, data_start, data_end))
}

/// Reconstruct the ≤`MAX_WINDOW_SIZE` bytes of TRUE emitted output immediately
/// preceding `out_pos` into `pred`, sourcing each byte from the segment covering
/// it (stored bytes from the input; prior-island bytes from their decoded bufs).
/// Returns the window length. `pred`'s last byte is `output[out_pos - 1]`,
/// matching `decode_block`'s `predecessor_window` contract.
#[cfg(parallel_sm)]
fn build_pred_streaming(
    pred: &mut [u8],
    segs: &[Seg],
    islands_buf: &[Vec<u8>],
    deflate: &[u8],
    out_pos: usize,
) -> usize {
    let w = out_pos.min(MAX_WINDOW_SIZE);
    if w == 0 {
        return 0;
    }
    let window_start = out_pos - w;
    // Walk segments backward accumulating their output extents; copy the portion
    // of each segment that intersects [window_start, out_pos) into pred.
    let mut seg_end = out_pos; // exclusive output offset of the current segment's end
    for s in segs.iter().rev() {
        let slen = s.out_len();
        let seg_start = seg_end - slen;
        if seg_end <= window_start {
            break;
        }
        let lo = seg_start.max(window_start);
        let hi = seg_end.min(out_pos);
        if lo < hi {
            let src_bytes = s.bytes(islands_buf, deflate);
            let src_lo = lo - seg_start;
            let dst_lo = lo - window_start;
            pred[dst_lo..dst_lo + (hi - lo)]
                .copy_from_slice(&src_bytes[src_lo..src_lo + (hi - lo)]);
        }
        seg_end = seg_start;
        if seg_end <= window_start {
            break;
        }
    }
    w
}

/// Whole-output CRC32 over the ORDERED segments (stored from input, islands from
/// their bufs). Partitions the segment list into contiguous groups balanced by
/// output bytes; each worker hashes its segments' bytes in order; the per-group
/// CRCs fold left-to-right (`combine_crc32`) to exactly `crc32(whole output)`.
#[cfg(parallel_sm)]
fn segmented_crc(segs: &[Seg], islands_buf: &[Vec<u8>], deflate: &[u8], num_threads: usize) -> u32 {
    let total: usize = segs.iter().map(|s| s.out_len()).sum();
    if total == 0 {
        return 0;
    }
    let threads = num_threads.max(1).min(num_cpus::get_physical().max(1));
    if threads <= 1 || total < (1 << 20) || segs.len() < threads {
        let mut hasher = Hasher::new();
        for s in segs {
            hasher.update(s.bytes(islands_buf, deflate));
        }
        return hasher.finalize();
    }

    // Partition segment indices into contiguous groups balanced by output bytes.
    let target = total.div_ceil(threads).max(1);
    let mut parts: Vec<std::ops::Range<usize>> = Vec::with_capacity(threads);
    let mut start = 0usize;
    let mut acc = 0usize;
    for (i, s) in segs.iter().enumerate() {
        acc += s.out_len();
        if acc >= target && parts.len() < threads - 1 {
            parts.push(start..i + 1);
            start = i + 1;
            acc = 0;
        }
    }
    if start < segs.len() {
        parts.push(start..segs.len());
    }

    let mut results: Vec<(u32, usize)> = vec![(0u32, 0usize); parts.len()];
    std::thread::scope(|scope| {
        for (part, result) in parts.iter().zip(results.iter_mut()) {
            let segs_part = &segs[part.clone()];
            scope.spawn(move || {
                let mut hasher = Hasher::new();
                let mut len = 0usize;
                for s in segs_part {
                    let b = s.bytes(islands_buf, deflate);
                    hasher.update(b);
                    len += b.len();
                }
                *result = (hasher.finalize(), len);
            });
        }
    });

    let mut acc_crc = results[0].0;
    for (crc, len) in results.iter().skip(1) {
        acc_crc = combine_crc32(acc_crc, *crc, *len as u64);
    }
    acc_crc
}

/// Maximum DEFLATE back-reference distance (RFC 1951 §3.2.5): a tail block can
/// reach at most this far before its first output byte.
#[cfg(parallel_sm)]
const MAX_WINDOW_SIZE: usize = 32 * 1024;

/// Length of the Huffman-tail output region (everything after the prefix).
#[cfg(parallel_sm)]
#[inline]
fn tail_buf_len(expected_size: usize, prefix_out: usize) -> usize {
    expected_size - prefix_out
}

/// Build the predecessor window for the Huffman tail: the last
/// `min(prefix_out, MAX_WINDOW_SIZE)` bytes of the decoded stored prefix,
/// gathered directly from the stored runs (so it does not depend on the
/// concurrent full-prefix copy). The returned buffer's LAST byte is
/// `decoded_output[prefix_out - 1]`, matching `copy_match`'s contract that
/// `predecessor_window` holds the bytes immediately preceding `output[0]`.
#[cfg(parallel_sm)]
fn build_predecessor_window(
    deflate: &[u8],
    base_off: usize,
    runs: &[StoredRun],
    prefix_out: usize,
) -> Vec<u8> {
    let w = prefix_out.min(MAX_WINDOW_SIZE);
    let mut pred = vec![0u8; w];
    if w == 0 {
        return pred;
    }
    let window_start = prefix_out - w; // output offset of pred[0]
                                       // Copy the portion of each run that intersects [window_start, prefix_out).
    for r in runs {
        let r_start = r.out_off;
        let r_end = r.out_off + r.len;
        if r_end <= window_start {
            continue;
        }
        // Overlap of [r_start, r_end) with [window_start, prefix_out).
        let lo = r_start.max(window_start);
        let hi = r_end.min(prefix_out);
        if lo >= hi {
            continue;
        }
        let dst = lo - window_start;
        let src = (r.src_off - base_off) + (lo - r_start);
        pred[dst..dst + (hi - lo)].copy_from_slice(&deflate[src..src + (hi - lo)]);
    }
    pred
}

/// Decode the Huffman tail into a STANDALONE `tail_buf` (out_pos starts at 0),
/// resolving back-references that reach before the tail against `pred` (the last
/// 32 KiB of the prefix). Returns `(bytes_written, crc32_of_those_bytes)`.
///
/// This is the overlap-friendly variant of [`decode_tail_blocks`]: because the
/// tail writes its own disjoint buffer and reaches the prefix only through the
/// immutable `pred` window, it can run concurrently with the full-prefix copy.
#[cfg(parallel_sm)]
fn decode_tail_into(
    tail: &[u8],
    tail_buf: &mut [u8],
    pred: &[u8],
) -> Result<(usize, u32), StoredSplitError> {
    use crate::decompress::inflate::consume_first_decode::Bits;
    use crate::decompress::parallel::lut_bulk_inflate::{decode_block, DecoderScratch};

    let mut bits = Bits::new(tail);
    let mut out_pos = 0usize;
    let mut scratch = DecoderScratch::new();
    loop {
        let result = decode_block(&mut bits, tail_buf, &mut out_pos, pred, &mut scratch)
            .map_err(|_| StoredSplitError::Corrupt("huffman tail decode failed"))?;
        if result.is_final_block {
            break;
        }
        if out_pos >= tail_buf.len() {
            return Err(StoredSplitError::Corrupt("huffman tail overran output"));
        }
    }
    let crc = crc32(&tail_buf[..out_pos]);
    Ok((out_pos, crc))
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
    use crate::decompress::parallel::lut_bulk_inflate::{decode_block, DecoderScratch};

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
    if threads <= 1 || total < 1 << 20 {
        // Fused copy+CRC: hash each run's bytes while they are still hot in
        // cache from the copy, instead of a SECOND full pass over `output`.
        // (The old split-copy-then-`crc32(output)` A/B arm, `GZIPPY_STORED_SPLIT_CRC=1`,
        // was removed 2026-07-07, batch 4f — same CRC semantics, this is just
        // fewer passes over `output`.)
        return copy_runs_fused_crc(output, deflate, base_off, runs);
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
                let out_len = out_slice.len();
                // Fused copy+CRC: hash each run's bytes while they are still
                // hot in cache from the copy (one pass over `output`, not
                // two). Runs within a partition are contiguous and ordered,
                // so an incremental Hasher over them yields the exact same
                // CRC32 as one `crc32(out_slice)` over the whole partition.
                let mut hasher = Hasher::new();
                for r in runs_part {
                    let dst = r.out_off - local_base;
                    let s = r.src_off - base_off;
                    out_slice[dst..dst + r.len].copy_from_slice(&deflate[s..s + r.len]);
                    hasher.update(&out_slice[dst..dst + r.len]);
                }
                *result = (hasher.finalize(), out_len);
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

/// Compute the whole-output CRC32 of a pure-stored stream DIRECTLY from the
/// input run slices — NO intermediate output buffer. The output bytes equal the
/// concatenation of the runs in output order (`out_off` ascends == run order),
/// so this yields the exact same CRC32 as `crc32(assembled_output)`. Mirrors
/// `fill_and_crc`'s parallel partition + `combine_crc32` fold, minus the copy.
///
/// For `T<=1` or small output it hashes inline;
/// otherwise it partitions runs into contiguous output-byte-balanced groups,
/// hashes each group's input slices on its own thread, and folds the
/// per-partition CRCs left-to-right (same fold as `fill_and_crc`).
fn crc_runs(
    deflate: &[u8],
    base_off: usize,
    runs: &[StoredRun],
    total: usize,
    num_threads: usize,
) -> u32 {
    if runs.is_empty() || total == 0 {
        // CRC32 of the empty stream is 0 (gzip stores crc32(b"") == 0).
        return 0;
    }

    let threads = num_threads.max(1).min(num_cpus::get_physical().max(1));
    if threads <= 1 || total < 1 << 20 {
        return crc_runs_inline(deflate, base_off, runs);
    }

    let parts = partition_runs(runs, total, threads);
    let mut results: Vec<(u32, usize)> = vec![(0u32, 0usize); parts.len()];

    std::thread::scope(|scope| {
        for (part, result) in parts.iter().zip(results.iter_mut()) {
            let runs_part = &runs[part.clone()];
            scope.spawn(move || {
                // Incremental hash over this partition's input slices, in order.
                let mut hasher = Hasher::new();
                let mut len = 0usize;
                for r in runs_part {
                    let s = r.src_off - base_off;
                    hasher.update(&deflate[s..s + r.len]);
                    len += r.len;
                }
                *result = (hasher.finalize(), len);
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

/// Inline (single-threaded) CRC32 over all run slices, read straight from the
/// input. Equals `crc32` over the concatenation of the runs in output order.
fn crc_runs_inline(deflate: &[u8], base_off: usize, runs: &[StoredRun]) -> u32 {
    let mut hasher = Hasher::new();
    for r in runs {
        let s = r.src_off - base_off;
        hasher.update(&deflate[s..s + r.len]);
    }
    hasher.finalize()
}

/// Stream every run's literal bytes DIRECTLY from the input to the writer with
/// no intermediate buffer. Faithful to rapidgzip's chunk-by-chunk `writeAll`
/// loop (`GzipChunkFetcher`). The caller has already verified CRC32 + size, so
/// the first byte written here is the first byte to reach the sink — the
/// verify-before-write / no-partial-output-on-corruption contract is preserved.
///
/// When the sink's raw fd is available (`out_fd` — production file / stdout
/// sinks pass `writer.get_ref().as_raw_fd()`), the run slices are GATHERED into
/// batched `writev` syscalls (`writev_all_to_fd`, the reaper's iovec pattern,
/// vendor `writeAllToFdVector`) rather than one `write_all` per run. On a
/// ~100 MB incompressible stream that collapses ~900 serial per-run `write()`
/// calls to ⌈runs / IOV_MAX⌉ `writev` calls. It is BYTE-IDENTICAL: the
/// concatenation of the gathered iovecs, in run order, equals the old per-run
/// write order (each iovec borrows `deflate[src_off-base..][..len]`, the exact
/// slice `write_all` streamed). The stored path is the sole producer of output
/// bytes and has written nothing through `writer` yet; the pre-`writev`
/// `flush` preserves ordering defensively even if the `BufWriter` (which wraps
/// the SAME fd) ever held buffered bytes. Without a fd (tests / in-memory Vec /
/// non-unix) it falls back to the portable per-run `write_all` loop.
fn write_runs<W: Write>(
    writer: &mut W,
    deflate: &[u8],
    base_off: usize,
    runs: &[StoredRun],
    out_fd: Option<i32>,
) -> Result<(), StoredSplitError> {
    #[cfg(unix)]
    if let Some(fd) = out_fd {
        writer.flush()?;
        let mut iovs: Vec<libc::iovec> = Vec::with_capacity(runs.len());
        for r in runs {
            if r.len == 0 {
                continue;
            }
            let s = r.src_off - base_off;
            iovs.push(libc::iovec {
                iov_base: deflate[s..s + r.len].as_ptr() as *mut libc::c_void,
                iov_len: r.len,
            });
        }
        if !iovs.is_empty() {
            crate::decompress::parallel::fd_vectored_write::writev_all_to_fd(fd, &mut iovs)?;
        }
        STORED_WRITEV_BATCHES.fetch_add(1, Ordering::Relaxed);
        return Ok(());
    }
    #[cfg(not(unix))]
    let _ = out_fd;

    for r in runs {
        let s = r.src_off - base_off;
        writer.write_all(&deflate[s..s + r.len])?;
    }
    writer.flush()?;
    Ok(())
}

/// Inline (single-threaded) copy of all runs that ALSO computes the whole-output
/// CRC32 in the same pass: each run's bytes are hashed immediately after the
/// copy, while still hot in cache, instead of a second full pass over `output`.
/// Runs are contiguous and ordered, so the incremental hash equals
/// `crc32(output)` byte-for-byte.
fn copy_runs_fused_crc(
    output: &mut [u8],
    deflate: &[u8],
    base_off: usize,
    runs: &[StoredRun],
) -> u32 {
    let mut hasher = Hasher::new();
    for r in runs {
        let src = r.src_off - base_off;
        output[r.out_off..r.out_off + r.len].copy_from_slice(&deflate[src..src + r.len]);
        hasher.update(&output[r.out_off..r.out_off + r.len]);
    }
    hasher.finalize()
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

    /// Build a gzip with `stored_data` as STORED deflate blocks (not BFINAL) followed
    /// by `huffman_data` compressed with flate2 dynamic Huffman (BFINAL on last block).
    /// Produces a valid gzip where the stored prefix is < 100% of total output — the
    /// fixture shape needed to exercise the demotion gate.
    fn gzip_stored_prefix_then_huffman(stored_data: &[u8], huffman_data: &[u8]) -> Vec<u8> {
        use std::io::Write as _;

        let mut full_payload = Vec::with_capacity(stored_data.len() + huffman_data.len());
        full_payload.extend_from_slice(stored_data);
        full_payload.extend_from_slice(huffman_data);
        let crc = crc32(&full_payload);
        let isize_val = full_payload.len() as u32;

        let mut deflate = Vec::new();
        // Non-final stored blocks for the prefix.
        let block_size = 65535usize;
        let mut off = 0;
        while off < stored_data.len() {
            let end = (off + block_size).min(stored_data.len());
            let chunk = &stored_data[off..end];
            deflate.push(0x00); // bfinal=0, btype=00
            let len = chunk.len() as u16;
            deflate.extend_from_slice(&len.to_le_bytes());
            deflate.extend_from_slice(&(!len).to_le_bytes());
            deflate.extend_from_slice(chunk);
            off = end;
        }
        // Dynamic Huffman tail via flate2 raw deflate (BFINAL on last block).
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(6));
        enc.write_all(huffman_data).unwrap();
        let tail = enc.finish().unwrap();
        deflate.extend_from_slice(&tail);

        let mut gz = vec![0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff];
        gz.extend_from_slice(&deflate);
        gz.extend_from_slice(&crc.to_le_bytes());
        gz.extend_from_slice(&isize_val.to_le_bytes());
        gz
    }

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
        let n = decompress_stored_parallel(&gz, &mut out, threads, None).expect("decode");
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
        let n = decompress_stored_parallel(&gz, &mut out, 4, None).expect("decode");
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
        let r = decompress_stored_parallel(&gz, &mut out, 4, None);
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
        let r = decompress_stored_parallel(&gz, &mut out, 8, None);
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
        match decompress_stored_parallel(&gz, &mut out, 4, None) {
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
        match decompress_stored_parallel(&gz, &mut out, 4, None) {
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
        decompress_stored_parallel(&gz, &mut out, 8, None).expect("decode");
        assert_eq!(out, payload);
        assert_eq!(out, oracle);
    }

    /// The streaming CRC (`crc_runs`, parallel partitioned + `combine_crc32`
    /// fold, computed straight from the input slices with NO output buffer) must
    /// equal the serial whole-output CRC32 at every thread count. This is the
    /// load-bearing correctness invariant of the no-monolithic-buffer fix: the
    /// verify-before-write contract relies on `crc_runs == crc32(assembled)`.
    #[test]
    fn crc_runs_matches_whole_crc() {
        let payload: Vec<u8> = (0..3_000_001u64)
            .map(|i| (i.wrapping_mul(2654435761) >> 13) as u8)
            .collect();
        let gz = gzip_stored(&payload, 65535);
        let (_h, header_size) = gzip_format::read_header(&gz).unwrap();
        let deflate = &gz[header_size..gz.len() - 8];
        let (runs, walk_end) = walk_stored_chain(deflate, header_size).unwrap();
        assert!(
            matches!(walk_end, WalkEnd::Final { .. }),
            "fixture must be pure-stored"
        );
        let total: usize = runs.iter().map(|r| r.len).sum();
        assert_eq!(total, payload.len());
        let expected = crc32(&payload);
        // inline path (t=1) and parallel partitioned path (t>1) must both agree.
        for t in [1usize, 2, 3, 4, 7, 8] {
            assert_eq!(
                crc_runs(deflate, header_size, &runs, total, t),
                expected,
                "crc_runs (parallel combine) != serial crc32(whole) at t={t}"
            );
        }
        // The inline helper alone must also agree.
        assert_eq!(crc_runs_inline(deflate, header_size, &runs), expected);
    }

    /// Build a gzip stream from ordered `regions`, forcing a DEFLATE block
    /// boundary between each region via a `Z_SYNC_FLUSH` (`Write::flush`), so
    /// incompressible regions become STORED blocks and compressible regions
    /// become Huffman islands — the SEGMENTED shape. `Z_SYNC_FLUSH` does NOT
    /// reset the 32 KiB window, so a later region identical to an earlier one
    /// back-references it ACROSS the intervening stored block(s) — the exact
    /// cross-island back-reference the segmented decoder must resolve against the
    /// TRUE rolling DECODED output. Returns `(gz_bytes, full_payload)`.
    fn gzip_segmented_from_regions(regions: &[&[u8]]) -> (Vec<u8>, Vec<u8>) {
        use std::io::Write as _;
        let mut payload = Vec::new();
        for r in regions {
            payload.extend_from_slice(r);
        }
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(6));
        for (i, r) in regions.iter().enumerate() {
            enc.write_all(r).unwrap();
            if i + 1 < regions.len() {
                enc.flush().unwrap(); // Z_SYNC_FLUSH → block boundary, window kept
            }
        }
        let deflate = enc.finish().unwrap();

        // Wrap raw deflate in a minimal gzip envelope.
        let mut gz = vec![0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff];
        gz.extend_from_slice(&deflate);
        gz.extend_from_slice(&crc32(&payload).to_le_bytes());
        gz.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        (gz, payload)
    }

    fn oracle_decode(gz: &[u8]) -> Vec<u8> {
        use std::io::Read as _;
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(gz)
            .read_to_end(&mut out)
            .unwrap();
        out
    }

    /// A distinctive, moderately-compressible pattern (a repeated pseudo-text
    /// line). Compresses to a Huffman block whose bytes are recognisable, so a
    /// predecessor-window bug corrupts them visibly against the oracle.
    fn compressible_pattern(len: usize, salt: u8) -> Vec<u8> {
        let base = b"The quick brown fox jumps over the lazy dog 0123456789. ";
        let mut v = Vec::with_capacity(len);
        let mut i = 0usize;
        while v.len() < len {
            let b = base[i % base.len()];
            v.push(b ^ (salt.wrapping_mul((i / base.len()) as u8 + 1) & 0x03));
            i += 1;
        }
        v.truncate(len);
        v
    }

    /// Incompressible run (PRNG) → forces a STORED block.
    fn incompressible(len: usize, seed: u64) -> Vec<u8> {
        let mut v = vec![0u8; len];
        let mut s = seed;
        for b in &mut v {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *b = (s >> 33) as u8;
        }
        v
    }

    /// THE ADVERSARIAL CASE (brief STEP 1 trap): two Huffman islands within
    /// 32 KiB where the SECOND island back-references the FIRST island's DECODED
    /// output ACROSS an intervening stored block. If the segmented decoder built
    /// its predecessor window by RE-READING the compressed input (instead of the
    /// TRUE rolling DECODED output), island 2 would decode to garbage and the
    /// oracle comparison would fail. The stream starts stored with a <50% prefix
    /// (→ routes to `decode_segmented`, not the single-tail path) and stays
    /// stored-dominant (→ never demotes).
    #[test]
    fn adversarial_two_islands_cross_reference() {
        // Layout (output order) — exercises BOTH cross-reference kinds:
        //   R0  4 KiB random  (STORED, deferred-copy — later referenced by an island)
        //   P1  4 KiB pattern  (Huffman island 1)
        //   R1  8 KiB random  (stored)
        //   P2  4 KiB == P1    (island 2, back-refs P1's DECODED output @ dist 12 KiB
        //                       ACROSS R1 — the island→island trap)
        //   RC  4 KiB == R0    (island: a compressible repeat of R0 → back-refs the
        //                       STORED, NOT-YET-COPIED region R0 @ dist 20 KiB — the
        //                       island→deferred-stored trap that needs materialize)
        //   R2 40 KiB random  (stored; keeps stored-dominant, no demote)
        let r0 = incompressible(4 * 1024, 0xA1);
        let p1 = compressible_pattern(4 * 1024, 0x7);
        let r1 = incompressible(8 * 1024, 0xB2);
        let p2 = p1.clone(); // identical → flate2 back-references P1
        let rc = r0.clone(); // identical → flate2 back-references stored R0
        let r2 = incompressible(40 * 1024, 0xC3);

        let (gz, payload) = gzip_segmented_from_regions(&[&r0, &p1, &r1, &p2, &rc, &r2]);
        let oracle = oracle_decode(&gz);
        assert_eq!(oracle, payload, "oracle sanity");

        // Structural: first block stored, and the prefix routes to segmented
        // (0 < prefix_out < expected/2). This confirms the segmented path — not
        // the single-tail path — decodes it.
        assert!(first_block_is_stored(&gz), "fixture must start stored");
        let (_h, hdr) = gzip_format::read_header(&gz).unwrap();
        let deflate = &gz[hdr..gz.len() - 8];
        let (_runs, walk_end) = walk_stored_chain(deflate, hdr).unwrap();
        match walk_end {
            WalkEnd::HuffmanTail { prefix_out, .. } => {
                assert!(prefix_out > 0, "prefix must be non-empty");
                assert!(
                    prefix_out * 2 < payload.len(),
                    "prefix {prefix_out} must be < 50% to route segmented"
                );
            }
            WalkEnd::Final { .. } => panic!("fixture must contain Huffman islands"),
        }

        let before = STORED_SEGMENTED_ISLANDS.load(Ordering::Relaxed);
        let mut out = Vec::new();
        let r = decompress_stored_parallel(&gz, &mut out, 4, None);
        #[cfg(parallel_sm)]
        {
            assert_eq!(r.map(|n| n as usize).unwrap(), payload.len());
            assert_eq!(
                out, payload,
                "ADVERSARIAL cross-island back-ref corrupted — predecessor window \
                 must be the TRUE decoded output, not reconstructed-from-input"
            );
            assert_eq!(out, oracle);
            assert!(
                STORED_SEGMENTED_ISLANDS.load(Ordering::Relaxed) > before,
                "segmented island decode must have run (Gate-0 non-inert witness)"
            );
        }
        #[cfg(not(parallel_sm))]
        {
            let _ = before;
            match r {
                Err(StoredSplitError::NotStoredDominated) => {}
                other => panic!("expected NotStoredDominated on non-parallel_sm, got {other:?}"),
            }
            assert!(out.is_empty());
        }
    }

    /// Many scattered Huffman islands across a large stored-dominant stream (the
    /// storedheavy shape): alternating incompressible (stored) and compressible
    /// (Huffman) regions. Must decode byte-exact vs the flate2 oracle and stay
    /// on the segmented path (no demote — stored dominates).
    #[test]
    fn segmented_scattered_islands_byte_exact() {
        // Stored prefix (< 50% of total) then 10 islands each flanked by a
        // larger stored run (~85% stored overall). Islands repeat earlier
        // patterns → cross-references into earlier decoded output.
        let prefix = incompressible(16 * 1024, 0x11);
        let mut owned: Vec<Vec<u8>> = vec![prefix];
        for k in 0..10u64 {
            let salt = (k as u8) & 0x3;
            owned.push(compressible_pattern(3 * 1024, salt));
            owned.push(incompressible(20 * 1024, 0x1000 + k));
        }
        owned.push(compressible_pattern(3 * 1024, 0)); // trailing repeat of island 0
        let regions: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();

        let (gz, payload) = gzip_segmented_from_regions(&regions);
        let oracle = oracle_decode(&gz);
        assert_eq!(oracle, payload, "oracle sanity");
        assert!(first_block_is_stored(&gz), "fixture must start stored");

        for t in [1usize, 2, 4, 8] {
            let mut out = Vec::new();
            let r = decompress_stored_parallel(&gz, &mut out, t, None);
            #[cfg(parallel_sm)]
            {
                assert_eq!(r.map(|n| n as usize).unwrap(), payload.len(), "t={t}");
                assert_eq!(out, payload, "segmented byte-exact mismatch at t={t}");
            }
            #[cfg(not(parallel_sm))]
            {
                match r {
                    Err(StoredSplitError::NotStoredDominated) => {}
                    other => panic!("expected NotStoredDominated, got {other:?}"),
                }
            }
        }
    }

    /// Demotion still fires (segmented) when Huffman ACTUALLY dominates: a small
    /// stored prefix (routes to segmented) followed by a large compressible body
    /// (> 50% of output) must bail to `NotStoredDominated` with no output.
    #[test]
    fn segmented_huffman_dominant_demotes() {
        let prefix = incompressible(8 * 1024, 0x55); // 8 KiB stored prefix
                                                     // 200 KiB highly compressible (zeros) → Huffman ≫ 50% of the ~208 KiB total.
        let body = vec![0u8; 200 * 1024];
        let (gz, _payload) = gzip_segmented_from_regions(&[&prefix, &body]);
        assert!(first_block_is_stored(&gz), "fixture must start stored");

        let mut out = Vec::new();
        let r = decompress_stored_parallel(&gz, &mut out, 4, None);
        match r {
            Err(StoredSplitError::NotStoredDominated) => {} // demoted (Huffman dominates)
            other => panic!("expected NotStoredDominated (Huffman-dominant), got {other:?}"),
        }
        assert!(out.is_empty(), "no partial output on demotion");
    }

    /// Demotion gate: a stored prefix that is < 50% of total output must fire
    /// `STORED_DEMOTE_TO_PARALLEL_SM` and return `NotStoredDominated`.
    ///
    /// Fixture: 40 KiB pseudo-random (→ stored blocks) + 60 KiB zeros (→ Huffman).
    /// stored_fraction = 40_000 / 100_000 = 40% < 50% → demotion gate fires.
    ///
    /// Existing fixtures for context:
    ///   `stored_prefix_then_huffman_tail`: 1.5 MB stored + 1 MB tail = 60% stored
    ///     (above 50% threshold → NOT demoted, decodes normally).
    ///   `huffman_first_block`: 0% stored → NotStoredDominated for a different reason.
    #[test]
    fn stored_prefix_below_50pct_demotes_to_parallel_sm() {
        // 40 KiB pseudo-random stored prefix.
        let stored_size = 40_000usize;
        let mut stored_data = vec![0u8; stored_size];
        let mut rng = 0xdead_beef_cafe_0123u64;
        for b in &mut stored_data {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (rng >> 33) as u8;
        }
        // 60 KiB zeros → flate2 level 6 emits dynamic-Huffman blocks (highly compressible).
        let huffman_size = 60_000usize;
        let huffman_data = vec![0u8; huffman_size];

        let gz = gzip_stored_prefix_then_huffman(&stored_data, &huffman_data);

        // Sanity: the first deflate block must be stored (btype=00).
        assert!(
            first_block_is_stored(&gz),
            "fixture must start with a stored block"
        );

        // The stored prefix (40_000) * 2 = 80_000 < total (100_000) → demotion gate.
        // expected_size / 2 = 50_000; prefix_out (40_000) < 50_000 → DEMOTE.
        let mut out = Vec::new();
        match decompress_stored_parallel(&gz, &mut out, 4, None) {
            Err(StoredSplitError::NotStoredDominated) => {} // gate fired correctly
            other => panic!(
                "expected NotStoredDominated (demotion gate, prefix={stored_size} < total/2={}), got {other:?}",
                (stored_size + huffman_size) / 2
            ),
        }
        // No partial output on NotStoredDominated.
        assert!(out.is_empty(), "must not write partial output on demotion");
    }
}
