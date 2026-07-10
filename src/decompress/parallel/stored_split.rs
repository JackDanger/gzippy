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

/// Counter: number of stored-DOMINATED-but-interleaved streams (stored runs
/// with scattered Huffman islands, e.g. `storedheavy.gz`: 98.9% stored + 33
/// dynamic-Huffman blocks in 11 island-groups) decoded via the SEGMENTED path —
/// the stored runs stay on the byte-aligned parallel-copy/stream path and only
/// the tiny Huffman islands are decoded in place. Non-zero + `STORED_DEMOTE=0`
/// is the Gate-0 witness that the segmented lever fired (no demote to
/// ParallelSM). Dumped by `GZIPPY_DEBUG=1`.
pub static STORED_SEGMENTED_RUNS: AtomicU64 = AtomicU64::new(0);

/// Counter: total Huffman islands decoded in place across all segmented runs
/// (11 per `storedheavy.gz` decode). Non-inert witness that islands were
/// actually decoded (not skipped). Dumped by `GZIPPY_DEBUG=1`.
pub static STORED_SEGMENT_ISLANDS: AtomicU64 = AtomicU64::new(0);

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
                write_runs(writer, deflate, header_size, &runs)
            })?;
            Ok(total as u64)
        }
        WalkEnd::HuffmanTail {
            tail_byte,
            prefix_out,
        } => {
            // SEGMENTED STORED WALK (2026-07-10 lever): a stored-DOMINATED stream
            // whose Huffman content is scattered ISLANDS between long stored runs
            // (e.g. `storedheavy.gz`: 98.9% stored in 3008 blocks + 1.09% Huffman
            // in 33 blocks / 11 island-groups, stored RESUMING after every island)
            // must NOT demote the whole 100 MB to ParallelSM just because the FIRST
            // island appears early (at 8.6% of output). `walk_stored_chain` stops
            // at that first island, so the prefix-based demote gate below would
            // route the entire — 98.9%-stored — stream through the worker-decode-
            // throughput-bound marker pipeline. Instead, walk the WHOLE member:
            // keep every stored run on the byte-aligned parallel-copy/stream path
            // and decode only the tiny islands in place (against a true rolling
            // 32 KiB OUTPUT window, so an island back-ref into a prior island's
            // DECODED bytes resolves byte-exactly). Only when Huffman actually
            // dominates (> 50% of output) do we demote.
            #[cfg(parallel_sm)]
            {
                match segmented_walk(deflate, header_size, expected_size) {
                    SegOutcome::Segmented {
                        segments,
                        total_out,
                        island_count,
                    } => {
                        return finish_segmented(
                            writer,
                            deflate,
                            header_size,
                            &segments,
                            total_out,
                            island_count,
                            expected_crc,
                            expected_size,
                            num_threads,
                        );
                    }
                    SegOutcome::Demote => {
                        // Huffman > 50% of the WHOLE stream — the sequential island
                        // decode would be the wall; ParallelSM parallelises it.
                        STORED_DEMOTE_TO_PARALLEL_SM.fetch_add(1, Ordering::Relaxed);
                        if crate::utils::debug_enabled() {
                            eprintln!(
                                "[gzippy] StoredParallel demote → ParallelSM: \
                                 Huffman > 50% of output (segmented walk)"
                            );
                        }
                        return Err(StoredSplitError::NotStoredDominated);
                    }
                    // SingleTail (Huffman is one contiguous suffix — the classic
                    // stored-prefix + Huffman-tail shape) OR Decline (unexpected
                    // structure / decode surprise): fall through to the EXISTING
                    // prefix-based demote gate + single-tail decoder, preserving
                    // that path's behaviour EXACTLY.
                    SegOutcome::SingleTail | SegOutcome::Decline => {}
                }
            }

            // DEMOTION GATE: if the Huffman tail is >= 50% of total output,
            // the sequential tail decode is the wall bottleneck — ParallelSM
            // can speculate boundaries across the tail and parallelize it.
            // Return NotStoredDominated so the caller routes to ParallelSM.
            //
            // Counter: STORED_DEMOTE_TO_PARALLEL_SM counts demotion events.
            //
            // Threshold: prefix_out < expected_size * (1/2).
            // storedheavy: prefix_out ~8.2 MB, expected_size ~100 MB → 8.2% < 50%
            // → DEMOTE. A pure-stored or >50% stored stream stays on this path.
            if prefix_out > 0
                && prefix_out * DEMOTE_THRESHOLD_DEN < expected_size * DEMOTE_THRESHOLD_NUM
            {
                STORED_DEMOTE_TO_PARALLEL_SM.fetch_add(1, Ordering::Relaxed);
                if crate::utils::debug_enabled() {
                    eprintln!(
                        "[gzippy] StoredParallel demote → ParallelSM: \
                         prefix_out={prefix_out} < expected_size/2={} \
                         (stored fraction {:.1}%)",
                        expected_size / 2,
                        prefix_out as f64 / expected_size as f64 * 100.0,
                    );
                }
                return Err(StoredSplitError::NotStoredDominated);
            }
            decode_with_huffman_tail(
                writer,
                deflate,
                header_size,
                &runs,
                tail_byte,
                prefix_out,
                expected_crc,
                expected_size,
                num_threads,
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
fn write_runs<W: Write>(
    writer: &mut W,
    deflate: &[u8],
    base_off: usize,
    runs: &[StoredRun],
) -> Result<(), StoredSplitError> {
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

// ═══════════════════════════════════════════════════════════════════════════
// SEGMENTED STORED WALK (2026-07-10)
// ═══════════════════════════════════════════════════════════════════════════

/// One materialised output segment produced by [`segmented_walk`]: either a
/// verbatim STORED run (bytes live in the compressed input — parallel-copy /
/// stream path) or a decoded Huffman ISLAND (bytes owned, decoded in place).
#[cfg(parallel_sm)]
enum Segment {
    Stored {
        /// Byte offset of the first literal byte in the gzip-compressed input.
        src_off: usize,
        /// Output offset where these literals land.
        out_off: usize,
        /// Number of literal bytes (== the block's LEN).
        len: usize,
    },
    Island {
        /// Output offset where the decoded island bytes land.
        out_off: usize,
        /// The decoded island bytes.
        bytes: Vec<u8>,
    },
}

#[cfg(parallel_sm)]
impl Segment {
    #[inline]
    fn out_off(&self) -> usize {
        match self {
            Segment::Stored { out_off, .. } | Segment::Island { out_off, .. } => *out_off,
        }
    }
    #[inline]
    fn out_len(&self) -> usize {
        match self {
            Segment::Stored { len, .. } => *len,
            Segment::Island { bytes, .. } => bytes.len(),
        }
    }
    /// The segment's output bytes, borrowed from the input (Stored) or the
    /// owned decoded buffer (Island).
    #[inline]
    fn bytes<'a>(&'a self, deflate: &'a [u8], base_off: usize) -> &'a [u8] {
        match self {
            Segment::Stored { src_off, len, .. } => {
                let s = *src_off - base_off;
                &deflate[s..s + *len]
            }
            Segment::Island { bytes, .. } => bytes,
        }
    }
}

/// What [`segmented_walk`] decided to do with a stored-first, Huffman-bearing
/// stream.
#[cfg(parallel_sm)]
enum SegOutcome {
    /// Stored-dominant interleaved stream (stored runs + scattered islands):
    /// decode via the segmented path (keep stored on the copy path).
    Segmented {
        segments: Vec<Segment>,
        total_out: usize,
        island_count: usize,
    },
    /// Huffman is > 50% of total output — demote to ParallelSM.
    Demote,
    /// Huffman is a single contiguous suffix (no stored block follows any
    /// Huffman block) — the classic stored-prefix + Huffman-tail shape. Use the
    /// existing single-tail path unchanged.
    SingleTail,
    /// Unexpected structure or a decode surprise — use the safe one-shot path.
    Decline,
}

/// Bytes of a leading contiguous Huffman region we will probe before concluding
/// the stream is a single trailing Huffman tail (and handing it to the existing
/// overlap-decoder). Comfortably exceeds real island-group sizes (storedheavy's
/// are ~98.7 KiB); a mis-classify only costs the parallel-copy win (never
/// correctness — the single-tail decoder is byte-exact for a mixed tail too).
#[cfg(parallel_sm)]
const SINGLE_TAIL_BAIL: usize = 1 << 20; // 1 MiB

/// Walk the WHOLE deflate member, keeping stored runs as [`Segment::Stored`]
/// (byte-aligned, copied later) and decoding each Huffman island in place into
/// a [`Segment::Island`], resolving island back-references against a true
/// rolling 32 KiB OUTPUT window (stored-or-decoded bytes — so an island that
/// references a prior island's DECODED output is byte-exact).
///
/// Writes NOTHING. On any decode error / unexpected structure returns
/// [`SegOutcome::Decline`] (the caller falls to the proven safe path with no
/// partial output).
#[cfg(parallel_sm)]
fn segmented_walk(deflate: &[u8], base_off: usize, expected_size: usize) -> SegOutcome {
    use crate::decompress::inflate::consume_first_decode::Bits;
    use crate::decompress::parallel::lut_bulk_inflate::{
        decode_block, BulkDecodeError, DecoderScratch,
    };

    let n = deflate.len();
    let mut bits = Bits::new(deflate);
    let mut out_off: usize = 0;
    let mut segments: Vec<Segment> = Vec::new();
    let mut scratch = DecoderScratch::new();
    let mut stored_out: usize = 0;
    let mut island_count: usize = 0;
    let mut seen_huff = false;
    let mut stored_after_huff = false;

    loop {
        // Peek the next block header (BFINAL + BTYPE) WITHOUT consuming — a
        // repeated `refill` on an already-full buffer is a no-op.
        bits.refill();
        if bits.available() < 3 {
            // Ran out of bits without a BFINAL block → malformed for our walk.
            return SegOutcome::Decline;
        }
        let header = bits.bitbuf & 0b111;
        let bfinal = (header & 1) != 0;
        let btype = ((header >> 1) & 0b11) as u8;

        if btype == 0 {
            // ── Stored block: parse LEN/NLEN by byte arithmetic (no decode). ──
            if seen_huff {
                stored_after_huff = true;
            }
            bits.consume(3);
            bits.align_to_byte();
            let byte_pos = bits.bit_position() / 8;
            if byte_pos + 4 > n {
                return SegOutcome::Decline;
            }
            let len = u16::from_le_bytes([deflate[byte_pos], deflate[byte_pos + 1]]) as usize;
            let nlen = u16::from_le_bytes([deflate[byte_pos + 2], deflate[byte_pos + 3]]);
            if (len as u16) != !nlen {
                return SegOutcome::Decline;
            }
            let data_start = byte_pos + 4;
            let data_end = data_start + len;
            if data_end > n {
                return SegOutcome::Decline;
            }
            if len > 0 {
                segments.push(Segment::Stored {
                    src_off: base_off + data_start,
                    out_off,
                    len,
                });
                out_off += len;
                stored_out += len;
            }
            // Reposition the bit reader at the next (byte-aligned) block header.
            bits = Bits::at_bit_offset(deflate, data_end * 8);
            if bfinal {
                break;
            }
        } else if btype == 0b11 {
            // Reserved block type — genuine corruption; let the safe path report.
            return SegOutcome::Decline;
        } else {
            // ── Huffman island: decode the maximal run of consecutive Huffman
            //    blocks into one owned buffer, against the rolling output window.
            seen_huff = true;
            island_count += 1;
            let island_out_off = out_off;
            let win = build_window_from_segments(deflate, base_off, &segments, island_out_off);
            let remaining = expected_size.saturating_sub(island_out_off);
            let cap = (128 * 1024).min(remaining + 512).max(4096);
            let mut island_buf = vec![0u8; cap];
            let mut ipos: usize = 0;
            let mut island_final = false;
            loop {
                // Snapshot the bit-reader position so a mid-block OutputOverflow
                // can rewind and retry into a larger buffer (Bits is not Copy;
                // `data` is immutable so restoring the 3 scalar fields is exact).
                let (sp, sb, sl) = (bits.pos, bits.bitbuf, bits.bitsleft);
                let before = ipos;
                match decode_block(&mut bits, &mut island_buf, &mut ipos, &win, &mut scratch) {
                    Ok(r) => {
                        if r.is_final_block {
                            island_final = true;
                            break;
                        }
                        // Peek the following block's type.
                        bits.refill();
                        let next_btype = ((bits.bitbuf >> 1) & 0b11) as u8;
                        if next_btype == 0 {
                            // Stored resumes → this island is complete.
                            break;
                        }
                        // Still in a Huffman run. If we have not yet committed to
                        // the interleaved (segmented) shape and this contiguous
                        // Huffman region is already large, treat it as a single
                        // trailing tail and hand it to the existing decoder.
                        if !stored_after_huff && ipos >= SINGLE_TAIL_BAIL {
                            return SegOutcome::SingleTail;
                        }
                        // Ensure headroom for the next block (capped at the true
                        // remaining-output upper bound).
                        if island_buf.len() - ipos < 64 * 1024 && island_buf.len() < remaining + 512
                        {
                            let newlen = (island_buf.len() * 2).min(remaining + 512);
                            island_buf.resize(newlen.max(island_buf.len() + 1), 0);
                        }
                    }
                    Err(BulkDecodeError::OutputOverflow) => {
                        // Rewind and grow. `remaining + 512` is a hard upper bound
                        // on this island's output (total == expected_size), so a
                        // buffer at that size can always hold it; if we are already
                        // there and still overflow, the structure is unexpected.
                        bits.pos = sp;
                        bits.bitbuf = sb;
                        bits.bitsleft = sl;
                        ipos = before;
                        if island_buf.len() >= remaining + 512 {
                            return SegOutcome::Decline;
                        }
                        let newlen = (island_buf.len() * 2)
                            .min(remaining + 512)
                            .max(island_buf.len() + 64 * 1024);
                        island_buf.resize(newlen, 0);
                        continue;
                    }
                    Err(_) => return SegOutcome::Decline,
                }
            }
            island_buf.truncate(ipos);
            out_off += ipos;
            segments.push(Segment::Island {
                out_off: island_out_off,
                bytes: island_buf,
            });
            if island_final {
                break;
            }
        }
    }

    let total_out = out_off;
    if total_out != expected_size {
        return SegOutcome::Decline;
    }
    if !seen_huff {
        // No Huffman at all — pure stored; shouldn't reach here (WalkEnd::Final),
        // but be safe.
        return SegOutcome::Decline;
    }
    if !stored_after_huff {
        // Huffman is a single contiguous suffix — the classic prefix+tail shape.
        return SegOutcome::SingleTail;
    }
    if stored_out.saturating_mul(2) < total_out {
        return SegOutcome::Demote;
    }
    SegOutcome::Segmented {
        segments,
        total_out,
        island_count,
    }
}

/// Build the 32 KiB predecessor window immediately preceding output offset
/// `island_out_off`, gathered from the already-walked segments (stored bytes
/// from the input, prior-island bytes from their decoded buffers). The window's
/// LAST byte is `output[island_out_off - 1]`, matching `copy_match`'s contract.
/// Because it sources ACTUAL emitted output (including a prior island's DECODED
/// bytes when two islands sit within 32 KiB), island back-references resolve
/// byte-exactly — this is the adversarial-correctness guarantee.
#[cfg(parallel_sm)]
fn build_window_from_segments(
    deflate: &[u8],
    base_off: usize,
    segments: &[Segment],
    island_out_off: usize,
) -> Vec<u8> {
    let w = island_out_off.min(MAX_WINDOW_SIZE);
    let mut win = vec![0u8; w];
    if w == 0 {
        return win;
    }
    let win_start = island_out_off - w;
    for s in segments {
        let soff = s.out_off();
        let s_end = soff + s.out_len();
        if s_end <= win_start {
            continue;
        }
        if soff >= island_out_off {
            break; // segments are ordered by ascending out_off
        }
        let lo = soff.max(win_start);
        let hi = s_end.min(island_out_off);
        if lo >= hi {
            continue;
        }
        let dst = lo - win_start;
        let src_bytes = s.bytes(deflate, base_off);
        let src = lo - soff;
        win[dst..dst + (hi - lo)].copy_from_slice(&src_bytes[src..src + (hi - lo)]);
    }
    win
}

/// Verify the segmented decode against the gzip trailer (CRC32 over the segments
/// in output order, parallel-partitioned + folded), then stream every segment
/// to the writer. Verify-before-write is preserved: nothing reaches the sink
/// until CRC + size are confirmed.
#[cfg(parallel_sm)]
#[allow(clippy::too_many_arguments)]
fn finish_segmented<W: Write>(
    writer: &mut W,
    deflate: &[u8],
    base_off: usize,
    segments: &[Segment],
    total_out: usize,
    island_count: usize,
    expected_crc: u32,
    expected_size: usize,
    num_threads: usize,
) -> Result<u64, StoredSplitError> {
    if total_out != expected_size {
        return Err(StoredSplitError::SizeMismatch {
            expected: expected_size,
            actual: total_out,
        });
    }
    let crc = crc_segments(deflate, base_off, segments, total_out, num_threads);
    if crc != expected_crc {
        return Err(StoredSplitError::CrcMismatch {
            expected: expected_crc,
            actual: crc,
        });
    }
    STORED_SEGMENTED_RUNS.fetch_add(1, Ordering::Relaxed);
    STORED_SEGMENT_ISLANDS.fetch_add(island_count as u64, Ordering::Relaxed);
    for s in segments {
        writer.write_all(s.bytes(deflate, base_off))?;
    }
    writer.flush()?;
    Ok(total_out as u64)
}

/// Whole-output CRC32 over the segments in output order, computed in parallel by
/// partitioning the segment list into contiguous output-byte-balanced groups,
/// hashing each group's bytes (from input or island buffer) on its own thread,
/// and folding the per-group CRCs left-to-right with `combine_crc32`. Equals the
/// serial `crc32(assembled_output)` byte-for-byte (same fold as `crc_runs`).
#[cfg(parallel_sm)]
fn crc_segments(
    deflate: &[u8],
    base_off: usize,
    segments: &[Segment],
    total: usize,
    num_threads: usize,
) -> u32 {
    if segments.is_empty() || total == 0 {
        return 0;
    }
    let threads = num_threads.max(1).min(num_cpus::get_physical().max(1));
    if threads <= 1 || total < 1 << 20 {
        let mut h = Hasher::new();
        for s in segments {
            h.update(s.bytes(deflate, base_off));
        }
        return h.finalize();
    }

    // Partition segments into ≤ threads contiguous groups by output bytes.
    let target = total.div_ceil(threads).max(1);
    let mut parts: Vec<std::ops::Range<usize>> = Vec::with_capacity(threads);
    let mut start = 0usize;
    let mut acc = 0usize;
    for (i, s) in segments.iter().enumerate() {
        acc += s.out_len();
        if acc >= target && parts.len() < threads - 1 {
            parts.push(start..i + 1);
            start = i + 1;
            acc = 0;
        }
    }
    if start < segments.len() {
        parts.push(start..segments.len());
    }

    let mut results: Vec<(u32, usize)> = vec![(0u32, 0usize); parts.len()];
    std::thread::scope(|scope| {
        for (part, result) in parts.iter().zip(results.iter_mut()) {
            let segs = &segments[part.clone()];
            scope.spawn(move || {
                let mut h = Hasher::new();
                let mut len = 0usize;
                for s in segs {
                    let b = s.bytes(deflate, base_off);
                    h.update(b);
                    len += b.len();
                }
                *result = (h.finalize(), len);
            });
        }
    });

    let mut acc_crc = results[0].0;
    for (crc, len) in results.iter().skip(1) {
        acc_crc = combine_crc32(acc_crc, *crc, *len as u64);
    }
    acc_crc
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
        match decompress_stored_parallel(&gz, &mut out, 4) {
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

#[cfg(all(test, parallel_sm))]
mod segmented_tests {
    use super::*;
    use std::io::Read as _;

    /// Minimal DEFLATE bit writer (LSB-first byte packing, MSB-first Huffman
    /// codes) — lets us hand-build the ONE stream shape no off-the-shelf tool
    /// produces: STORED runs interleaved with fixed-Huffman ISLANDS that carry a
    /// cross-island back-reference. Used to construct the adversarial fixture.
    struct BitWriter {
        bytes: Vec<u8>,
        cur: u32,
        nbits: u32,
    }
    impl BitWriter {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                cur: 0,
                nbits: 0,
            }
        }
        /// Append `n` bits of `val`, LSB-first.
        fn write_bits(&mut self, val: u32, n: u32) {
            for i in 0..n {
                let bit = (val >> i) & 1;
                self.cur |= bit << self.nbits;
                self.nbits += 1;
                if self.nbits == 8 {
                    self.bytes.push(self.cur as u8);
                    self.cur = 0;
                    self.nbits = 0;
                }
            }
        }
        /// Append a Huffman code of `len` bits, MSB-first (RFC 1951 §3.1.1).
        fn write_huff(&mut self, code: u32, len: u32) {
            for i in (0..len).rev() {
                let bit = (code >> i) & 1;
                self.cur |= bit << self.nbits;
                self.nbits += 1;
                if self.nbits == 8 {
                    self.bytes.push(self.cur as u8);
                    self.cur = 0;
                    self.nbits = 0;
                }
            }
        }
        fn align(&mut self) {
            if self.nbits > 0 {
                self.bytes.push(self.cur as u8);
                self.cur = 0;
                self.nbits = 0;
            }
        }
        fn block_header(&mut self, bfinal: bool, btype: u32) {
            self.write_bits((bfinal as u32) | (btype << 1), 3);
        }
        /// One or more STORED blocks (byte-aligned LEN/NLEN + raw bytes),
        /// splitting at the 65535-byte max LEN. `bfinal` marks only the LAST.
        fn stored_block(&mut self, data: &[u8], bfinal: bool) {
            if data.is_empty() {
                self.block_header(bfinal, 0);
                self.align();
                self.bytes.extend_from_slice(&0u16.to_le_bytes());
                self.bytes.extend_from_slice(&(!0u16).to_le_bytes());
                return;
            }
            let mut off = 0;
            while off < data.len() {
                let end = (off + 65535).min(data.len());
                let last = end == data.len();
                self.block_header(bfinal && last, 0);
                self.align();
                let chunk = &data[off..end];
                let len = chunk.len() as u16;
                self.bytes.extend_from_slice(&len.to_le_bytes());
                self.bytes.extend_from_slice(&(!len).to_le_bytes());
                self.bytes.extend_from_slice(chunk);
                off = end;
            }
        }
        /// Fixed-Huffman literal (RFC 1951 §3.2.6).
        fn fixed_lit(&mut self, lit: u32) {
            if lit <= 143 {
                self.write_huff(0x30 + lit, 8);
            } else {
                self.write_huff(0x190 + (lit - 144), 9);
            }
        }
        fn fixed_eob(&mut self) {
            self.write_huff(0, 7); // symbol 256
        }
        /// Fixed-Huffman match of length 258 (symbol 285, no extra) at `distance`.
        fn fixed_match_258(&mut self, distance: u32) {
            // length symbol 285 → fixed code 0b11000000 + (285-280) = 0xC5, 8 bits.
            self.write_huff(0xC5, 8);
            // distance symbol + extra (RFC 1951 §3.2.5).
            let (dsym, base, extra_bits) = dist_sym(distance);
            self.write_huff(dsym, 5); // fixed 5-bit distance code = symbol value
            if extra_bits > 0 {
                self.write_bits(distance - base, extra_bits);
            }
        }
        fn finish(mut self) -> Vec<u8> {
            self.align();
            self.bytes
        }
    }

    /// Distance → (symbol, base, extra_bits) for the symbols we exercise.
    fn dist_sym(d: u32) -> (u32, u32, u32) {
        // (base, extra) per symbol; symbol index is the array position.
        const START: [u32; 30] = [
            1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025,
            1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
        ];
        const EXTRA: [u32; 30] = [
            0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12,
            12, 13, 13,
        ];
        let mut sym = 0;
        for i in 0..30 {
            let hi = START[i] + (1u32 << EXTRA[i]);
            if d >= START[i] && d < hi {
                sym = i;
                break;
            }
        }
        (sym as u32, START[sym], EXTRA[sym])
    }

    fn wrap_gz(deflate: &[u8], payload: &[u8]) -> Vec<u8> {
        let mut gz = vec![0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff];
        gz.extend_from_slice(deflate);
        gz.extend_from_slice(&crc32(payload).to_le_bytes());
        gz.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        gz
    }

    /// THE ADVERSARIAL FIXTURE (make-or-break correctness):
    /// two Huffman islands within 32 KiB, where the SECOND island back-references
    /// the FIRST island's DECODED output (across a tiny stored gap). If the
    /// island predecessor window were reconstructed from INPUT stored runs (the
    /// warned-against bug) rather than the true rolling OUTPUT, the copied bytes
    /// would be garbage and the CRC would mismatch.
    ///
    /// Layout (output order):
    ///   S0  : 40000 bytes stored  (routes to StoredParallel, stored-dominant)
    ///   PA  : 4096 bytes fixed-Huffman literals   ← island A (decoded output)
    ///   G   : 16 bytes stored      (tiny gap; keeps A and B within 32 KiB)
    ///   B   : 258 bytes = copy of PA[0..258] via a match(len=258, dist=4112)
    ///                                              ← island B references A
    fn build_adversarial() -> (Vec<u8>, Vec<u8>) {
        let mut s0 = vec![0u8; 40000];
        let mut st = 0x1234_5678_9abc_def0u64;
        for b in &mut s0 {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (st >> 33) as u8;
        }
        let mut pa = vec![0u8; 4096];
        for (i, b) in pa.iter_mut().enumerate() {
            *b = (i as u32).wrapping_mul(37).wrapping_add(11) as u8;
        }
        let g = [0xAAu8; 16];
        let b = pa[0..258].to_vec();

        // distance from island-B start back to PA start:
        //   out_pos at B start = 40000 + 4096 + 16 = 44112; PA start = 40000
        //   distance = 44112 - 40000 = 4112  (reaches PA[0], length 258)
        let distance = 4112u32;

        let mut payload = Vec::new();
        payload.extend_from_slice(&s0);
        payload.extend_from_slice(&pa);
        payload.extend_from_slice(&g);
        payload.extend_from_slice(&b);

        let mut bw = BitWriter::new();
        bw.stored_block(&s0, false); // block 0
        bw.block_header(false, 1); // block 1: fixed Huffman, non-final
        for &byte in &pa {
            bw.fixed_lit(byte as u32);
        }
        bw.fixed_eob();
        bw.stored_block(&g, false); // block 2: tiny stored gap
        bw.block_header(true, 1); // block 3: fixed Huffman, FINAL
        bw.fixed_match_258(distance);
        bw.fixed_eob();
        let deflate = bw.finish();

        (wrap_gz(&deflate, &payload), payload)
    }

    #[test]
    fn adversarial_two_islands_within_32k_byte_exact() {
        let (gz, payload) = build_adversarial();
        // Sanity: routes to the stored path.
        assert!(first_block_is_stored(&gz), "fixture must start stored");

        // Independent oracle: flate2 must decode our hand-built stream identically.
        let mut oracle = Vec::new();
        flate2::read::GzDecoder::new(&gz[..])
            .read_to_end(&mut oracle)
            .expect("flate2 oracle decode");
        assert_eq!(oracle, payload, "oracle sanity (hand-built stream valid)");

        // Deterministic routing check (no shared global counters): this stream
        // must be classified Segmented with exactly two islands.
        let (_h, hsize) = gzip_format::read_header(&gz).unwrap();
        let footer = gzip_format::read_footer(&gz, gz.len() - 8).unwrap();
        let deflate = &gz[hsize..gz.len() - 8];
        match segmented_walk(deflate, hsize, footer.uncompressed_size as usize) {
            SegOutcome::Segmented { island_count, .. } => {
                assert_eq!(island_count, 2, "expected two Huffman islands")
            }
            _ => panic!("adversarial stream must route to Segmented (not demote/tail/decline)"),
        }

        for t in [1usize, 2, 4, 8] {
            let mut out = Vec::new();
            let n = decompress_stored_parallel(&gz, &mut out, t).expect("segmented decode");
            assert_eq!(n as usize, payload.len(), "size mismatch at t={t}");
            assert_eq!(
                out, payload,
                "ADVERSARIAL byte mismatch at t={t} — island-B back-ref into \
                 island-A decoded output corrupted (predecessor window bug)"
            );
        }
    }

    /// Interleaved stored + far-apart islands (storedheavy-like at small scale):
    /// exercises multi-segment resumption, parallel CRC partitioning across
    /// thread counts, and stored runs BOTH before and after islands.
    fn build_interleaved() -> (Vec<u8>, Vec<u8>) {
        // Three stored runs (each > 32 KiB so islands are never within a window)
        // separated by two small fixed-Huffman islands.
        let mk_stored = |seed: u64, len: usize| -> Vec<u8> {
            let mut v = vec![0u8; len];
            let mut st = seed;
            for b in &mut v {
                st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
                *b = (st >> 33) as u8;
            }
            v
        };
        let s0 = mk_stored(0xa1, 50_000);
        let isl0: Vec<u8> = (0..2000u32)
            .map(|i| (i.wrapping_mul(7) + 3) as u8)
            .collect();
        let s1 = mk_stored(0xb2, 60_000);
        let isl1: Vec<u8> = (0..1500u32)
            .map(|i| (i.wrapping_mul(13) + 5) as u8)
            .collect();
        let s2 = mk_stored(0xc3, 55_000);

        let mut payload = Vec::new();
        for part in [&s0, &isl0, &s1, &isl1, &s2] {
            payload.extend_from_slice(part);
        }

        let mut bw = BitWriter::new();
        bw.stored_block(&s0, false);
        bw.block_header(false, 1);
        for &b in &isl0 {
            bw.fixed_lit(b as u32);
        }
        bw.fixed_eob();
        bw.stored_block(&s1, false);
        bw.block_header(false, 1);
        for &b in &isl1 {
            bw.fixed_lit(b as u32);
        }
        bw.fixed_eob();
        bw.stored_block(&s2, true); // final stored block
        let deflate = bw.finish();
        (wrap_gz(&deflate, &payload), payload)
    }

    #[test]
    fn interleaved_stored_islands_byte_exact_all_threads() {
        let (gz, payload) = build_interleaved();
        assert!(first_block_is_stored(&gz));

        let mut oracle = Vec::new();
        flate2::read::GzDecoder::new(&gz[..])
            .read_to_end(&mut oracle)
            .expect("flate2 oracle");
        assert_eq!(oracle, payload, "oracle sanity");

        for t in [1usize, 2, 3, 4, 8] {
            let mut out = Vec::new();
            let n = decompress_stored_parallel(&gz, &mut out, t).expect("decode");
            assert_eq!(n as usize, payload.len());
            assert_eq!(out, payload, "interleaved byte mismatch at t={t}");
        }
    }

    /// A stored-prefix + single contiguous Huffman tail must STILL take the
    /// existing single-tail path (SegOutcome::SingleTail), not the segmented one.
    #[test]
    fn single_tail_still_routes_to_existing_path() {
        // 200 KiB stored + 100 KiB fixed-Huffman tail to end (no stored after).
        let mut s = vec![0u8; 200_000];
        let mut st = 0xfeedu64;
        for b in &mut s {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (st >> 33) as u8;
        }
        let tail: Vec<u8> = (0..100_000u32).map(|i| (i % 251) as u8).collect();
        let mut payload = s.clone();
        payload.extend_from_slice(&tail);

        let mut bw = BitWriter::new();
        bw.stored_block(&s, false);
        bw.block_header(true, 1);
        for &b in &tail {
            bw.fixed_lit(b as u32);
        }
        bw.fixed_eob();
        let deflate = bw.finish();
        let gz = wrap_gz(&deflate, &payload);

        // Deterministic: the segmented walk must classify this as SingleTail
        // (Huffman is one contiguous suffix), routing to the existing decoder.
        let (_h, hsize) = gzip_format::read_header(&gz).unwrap();
        let footer = gzip_format::read_footer(&gz, gz.len() - 8).unwrap();
        let deflate = &gz[hsize..gz.len() - 8];
        assert!(
            matches!(
                segmented_walk(deflate, hsize, footer.uncompressed_size as usize),
                SegOutcome::SingleTail
            ),
            "single contiguous tail must classify as SingleTail (not Segmented)"
        );

        let mut out = Vec::new();
        // stored 200k / total 300k = 66% > 50% → not demoted → single-tail decode.
        let n = decompress_stored_parallel(&gz, &mut out, 4).expect("decode");
        assert_eq!(n as usize, payload.len());
        assert_eq!(out, payload);
    }
}
