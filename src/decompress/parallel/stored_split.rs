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

/// Counter: number of stored-heavy-with-Huffman-islands streams decoded via the
/// multi-island path ([`decode_stored_with_islands`]) — stored runs across the
/// WHOLE stream are parallel-copied and each Huffman island is decoded in place
/// against the true rolling output window, instead of demoting to the ParallelSM
/// grid. Non-zero proves the islands lever fired (Gate-0 non-inert witness).
/// Dumped by `GZIPPY_DEBUG=1`.
pub static STORED_ISLANDS_RUNS: AtomicU64 = AtomicU64::new(0);

/// Threshold: if the TOTAL stored bytes account for < this fraction of total
/// output (numerator/denominator), demote to ParallelSM so the Huffman islands
/// are decoded in parallel by the grid. Currently 50% (1/2). Used only by the
/// `parallel_sm` island decoder.
#[cfg(parallel_sm)]
const DEMOTE_THRESHOLD_NUM: usize = 1;
#[cfg(parallel_sm)]
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
    walk_stored_prefix(deflate, 0, base_off, 0)
}

/// Generalised [`walk_stored_chain`]: walk stored blocks from an ARBITRARY
/// byte-aligned deflate offset `start_p` with the output cursor starting at
/// `start_out`. Used by the multi-island path to resume the fast byte-aligned
/// stored walk after a Huffman island (whose trailing stored blocks are again
/// byte-aligned). `WalkEnd`'s `deflate_end` / `tail_byte` are absolute byte
/// offsets into `deflate`; `prefix_out` is the absolute output offset.
///
/// PRECONDITION: `start_p` is a byte boundary at which a block header begins
/// (true for the first block, and for any block following a stored block).
fn walk_stored_prefix(
    deflate: &[u8],
    start_p: usize,
    base_off: usize,
    start_out: usize,
) -> Result<(Vec<StoredRun>, WalkEnd), StoredSplitError> {
    let mut runs: Vec<StoredRun> = Vec::new();
    let mut p: usize = start_p; // byte cursor into `deflate`
    let mut out_off: usize = start_out;
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
            // Stored data INTERRUPTED BY HUFFMAN ISLANDS (the `storedheavy`
            // shape: ~all-stored with occasional small Huffman blocks). The old
            // path stopped at the FIRST Huffman block and — when its contiguous
            // stored prefix was < 50% of output (storedheavy: ~8.6%) — DEMOTED
            // to the ParallelSM grid, paying the grid's per-chunk alloc+copy.
            //
            // Instead, walk the WHOLE stream across islands: decode each Huffman
            // island in place against the TRUE rolling output window, and
            // parallel-copy the stored runs BETWEEN/AFTER the islands too. The
            // demotion decision now uses the TOTAL stored fraction (computed by
            // the full walk), so a genuinely Huffman-dominant stream still
            // demotes, but stored-throughout data stays on the fast path.
            decode_stored_with_islands(
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

/// Stored data INTERRUPTED BY HUFFMAN ISLANDS. Walk the WHOLE stream: the stored
/// runs between/after the islands are recorded for a parallel bulk copy, and each
/// Huffman island is decoded IN PLACE into the materialised output buffer.
///
/// ⚠ BYTE-EXACTNESS (the make-or-break): a Huffman island's back-references reach
/// up to 32 KiB before its first output byte. Those bytes may be verbatim/stored
/// OR a PRIOR island's DECODED output (two islands within 32 KiB). We NEVER
/// reconstruct the window from input alone — the island is decoded straight into
/// `output` at its ABSOLUTE offset, so `copy_match` resolves every `distance <=
/// out_pos` back-ref against the TRUE preceding output (stored bytes materialised
/// by [`materialize_predecessor`] for the <=32 KiB window, prior islands already
/// present because islands are decoded left-to-right). The adjacent-island case
/// therefore reads the first island's real decoded bytes, not a stale input copy.
///
/// On non-`parallel_sm` builds the bulk decoder is unavailable, so we decline
/// (`NotStoredDominated`) and let the safe one-shot path decode the whole stream
/// — same byte-exact result, just not parallel.
#[cfg(parallel_sm)]
#[allow(clippy::too_many_arguments)]
fn decode_stored_with_islands<W: Write>(
    writer: &mut W,
    deflate: &[u8],
    base_off: usize,
    prefix_runs: &[StoredRun],
    first_tail_byte: usize,
    prefix_out: usize,
    expected_crc: u32,
    expected_size: usize,
    num_threads: usize,
) -> Result<u64, StoredSplitError> {
    use crate::decompress::inflate::consume_first_decode::Bits;
    use crate::decompress::parallel::lut_bulk_inflate::{decode_block, DecoderScratch};

    // MULTI-MEMBER DEFENSE (defense-in-depth; classify catches this up front):
    // `expected_size` is the WHOLE-FILE trailer's ISIZE. If the stored prefix
    // alone already exceeds it we were mis-routed a multi-member stream whose
    // dominant first member is larger than the small last member's ISIZE.
    // DECLINE (no writer bytes) to the safe multi-member-capable path.
    if prefix_out > expected_size {
        return Err(StoredSplitError::NotStoredDominated);
    }

    // Materialised output: stored runs are bulk-copied in, islands decoded in
    // place. `vec![0u8; ..]` matches the old single-tail path's allocation.
    let mut output = time_phase("alloc_zero", || vec![0u8; expected_size]);

    // Every stored run recorded by the fast byte-aligned walk (prefix + the
    // runs between/after islands), for the final parallel copy.
    let mut all_runs: Vec<StoredRun> = prefix_runs.to_vec();
    let mut stored_bytes: usize = prefix_runs.iter().map(|r| r.len).sum();

    let mut scratch = DecoderScratch::new();
    let mut out_pos = prefix_out;

    // Drive a bit reader over the WHOLE deflate slice so `bit_position()` is in
    // absolute deflate coordinates. Start at the first Huffman block (which is
    // byte-aligned — it follows the stored prefix).
    let mut bits = Bits::at_bit_offset(deflate, first_tail_byte * 8);

    'walk: loop {
        // Peek the next block's 3-bit header WITHOUT consuming (decode_block /
        // the fast walk re-read it). `refill` does not consume, so bit_position
        // still points at the header start.
        bits.refill();
        let header = bits.bitbuf & 0b111;
        let btype = ((header >> 1) & 0b11) as u8;
        let bit_pos = bits.bit_position();

        if btype == 0 && bit_pos.is_multiple_of(8) {
            // A BYTE-ALIGNED stored region (the common case after an island's
            // stored blocks re-byte-align). Hand it to the fast walk, which
            // records runs for the parallel copy with no per-byte bit decoding.
            let p = bit_pos / 8;
            let (runs_seg, end) = walk_stored_prefix(deflate, p, base_off, out_pos)?;
            let seg_out: usize = runs_seg.iter().map(|r| r.len).sum();
            stored_bytes += seg_out;
            all_runs.extend_from_slice(&runs_seg);
            out_pos += seg_out;
            if out_pos > expected_size {
                return Err(StoredSplitError::NotStoredDominated);
            }
            match end {
                WalkEnd::Final { .. } => break 'walk,
                WalkEnd::HuffmanTail { tail_byte, .. } => {
                    bits = Bits::at_bit_offset(deflate, tail_byte * 8);
                    continue 'walk;
                }
            }
        }

        // A Huffman block, OR a stored block whose 3-bit header starts mid-byte
        // (the first stored block right after a Huffman block — it re-aligns
        // internally, after which the fast walk takes over on the next lap).
        // Decode ONE block via the bulk decoder into `output` at `out_pos`.
        //
        // Before a Huffman block, materialise its <=32 KiB predecessor from the
        // recorded stored runs so early back-refs into not-yet-bulk-copied
        // stored bytes resolve; prior islands / mid-byte stored blocks are
        // already in `output`. `&[]` predecessor is correct because every legal
        // back-ref has `distance <= out_pos` and reads `output` directly.
        if btype != 0 {
            materialize_predecessor(&mut output, deflate, base_off, &all_runs, out_pos);
        }
        let before = out_pos;
        let result = decode_block(&mut bits, &mut output, &mut out_pos, &[], &mut scratch)
            .map_err(|_| StoredSplitError::Corrupt("huffman island decode failed"))?;
        if btype == 0 {
            // A mid-byte stored block: its bytes are already in `output` (not
            // deferred to the parallel copy). Count them as stored.
            stored_bytes += out_pos - before;
        }
        if out_pos > expected_size {
            return Err(StoredSplitError::NotStoredDominated);
        }
        if result.is_final_block {
            break 'walk;
        }
    }

    // Size agreement: the walk must have tiled the whole output exactly.
    if out_pos != expected_size {
        return Err(StoredSplitError::SizeMismatch {
            expected: expected_size,
            actual: out_pos,
        });
    }

    // DEMOTION GATE (whole-stream stored fraction): if stored is < 50% of the
    // output the Huffman islands dominate and the ParallelSM grid parallelises
    // their decode better than this serial-island path. Discard (no writer
    // bytes) and route to the grid. classify's ratio gate already guarantees
    // stored dominance for real workloads, so this fires only on contrived input.
    if stored_bytes * DEMOTE_THRESHOLD_DEN < expected_size * DEMOTE_THRESHOLD_NUM {
        STORED_DEMOTE_TO_PARALLEL_SM.fetch_add(1, Ordering::Relaxed);
        if crate::utils::debug_enabled() {
            eprintln!(
                "[gzippy] StoredParallel demote → ParallelSM: stored={stored_bytes} \
                 < expected_size/2={} (stored fraction {:.1}%)",
                expected_size / 2,
                stored_bytes as f64 / expected_size as f64 * 100.0,
            );
        }
        return Err(StoredSplitError::NotStoredDominated);
    }

    // Parallel-copy the recorded stored runs into `output`. Island bytes (and
    // any mid-byte stored blocks) were decoded in place and lie in the GAPS
    // between recorded runs; the copy touches only recorded-stored ranges, so
    // those decoded bytes are preserved.
    time_phase("copy_runs", || {
        copy_runs_parallel(&mut output, deflate, base_off, &all_runs, num_threads)
    });

    // Whole-output CRC32 (parallel), verified BEFORE the first byte is written —
    // the verify-before-write / no-partial-output-on-corruption contract.
    let crc = time_phase("crc_whole", || crc32_whole_parallel(&output, num_threads));
    STORED_ISLANDS_RUNS.fetch_add(1, Ordering::Relaxed);
    time_phase("verify_write", || {
        verify_and_write(writer, &output, crc, expected_crc, expected_size)
    })
}

/// Non-`parallel_sm` builds lack the bulk block decoder; decline so the safe
/// one-shot path decodes the whole stream (same bytes, just not parallel).
#[cfg(not(parallel_sm))]
#[allow(clippy::too_many_arguments)]
fn decode_stored_with_islands<W: Write>(
    writer: &mut W,
    deflate: &[u8],
    base_off: usize,
    prefix_runs: &[StoredRun],
    first_tail_byte: usize,
    prefix_out: usize,
    expected_crc: u32,
    expected_size: usize,
    num_threads: usize,
) -> Result<u64, StoredSplitError> {
    let _ = (
        writer,
        deflate,
        base_off,
        prefix_runs,
        first_tail_byte,
        prefix_out,
        expected_crc,
        expected_size,
        num_threads,
    );
    Err(StoredSplitError::NotStoredDominated)
}

/// Materialise the <=32 KiB predecessor window of an in-place island decode:
/// copy the portions of the recorded stored runs that intersect
/// `[out_pos - min(out_pos, 32 KiB), out_pos)` DIRECTLY into `output` at their
/// absolute positions, so `copy_match`'s `distance <= out_pos` back-refs read the
/// true stored bytes. Prior islands (and mid-byte stored blocks) already occupy
/// their output ranges, so this only fills the not-yet-bulk-copied stored bytes.
///
/// `runs` is ascending by `out_off` and every run precedes `out_pos`, so we scan
/// from the end and stop once a run lies fully before the window.
#[cfg(parallel_sm)]
fn materialize_predecessor(
    output: &mut [u8],
    deflate: &[u8],
    base_off: usize,
    runs: &[StoredRun],
    out_pos: usize,
) {
    let w = out_pos.min(MAX_WINDOW_SIZE);
    if w == 0 {
        return;
    }
    let window_start = out_pos - w;
    for r in runs.iter().rev() {
        let r_end = r.out_off + r.len;
        if r_end <= window_start {
            break; // fully before the window; all earlier runs are too.
        }
        let lo = r.out_off.max(window_start);
        let hi = r_end.min(out_pos);
        if lo >= hi {
            continue;
        }
        let src = (r.src_off - base_off) + (lo - r.out_off);
        output[lo..hi].copy_from_slice(&deflate[src..src + (hi - lo)]);
    }
}

/// Copy the recorded stored runs into `output` in parallel. Runs write DISJOINT
/// output ranges separated by island GAPS (decoded in place), so the tiling
/// boundaries are placed at each partition's first run `out_off` — a chunk may
/// contain island bytes, which the workers leave untouched. Mirrors
/// [`fill_and_crc`]'s split-at-mut structure but tolerates the gaps and computes
/// no CRC (the whole-output CRC is taken separately over the tiled buffer).
#[cfg(parallel_sm)]
fn copy_runs_parallel(
    output: &mut [u8],
    deflate: &[u8],
    base_off: usize,
    runs: &[StoredRun],
    num_threads: usize,
) {
    if runs.is_empty() {
        return;
    }
    let total_stored: usize = runs.iter().map(|r| r.len).sum();
    let threads = num_threads.max(1).min(num_cpus::get_physical().max(1));
    if threads <= 1 || total_stored < 1 << 20 {
        for r in runs {
            let src = r.src_off - base_off;
            output[r.out_off..r.out_off + r.len].copy_from_slice(&deflate[src..src + r.len]);
        }
        return;
    }

    // Partition runs into contiguous index groups balanced by stored bytes.
    let parts = partition_runs(runs, total_stored, threads);
    let out_len = output.len();

    // Split `output` into per-partition disjoint slices. Boundary between group
    // i and i+1 is `runs[parts[i+1].start].out_off` (ascending, since runs are
    // ascending by out_off); the last group runs to `output.len()`.
    let mut out_slices: Vec<&mut [u8]> = Vec::with_capacity(parts.len());
    {
        let mut rest = &mut output[..];
        let mut prev = 0usize;
        for (i, _part) in parts.iter().enumerate() {
            let boundary = if i + 1 < parts.len() {
                runs[parts[i + 1].start].out_off
            } else {
                out_len
            };
            let (head, tail) = rest.split_at_mut(boundary - prev);
            out_slices.push(head);
            rest = tail;
            prev = boundary;
        }
        debug_assert!(rest.is_empty(), "partition slices must tile the buffer");
    }

    std::thread::scope(|scope| {
        let mut prev = 0usize;
        for (part, out_slice) in parts.iter().zip(out_slices) {
            let runs_part = &runs[part.clone()];
            let base = prev;
            prev = if part.end < runs.len() {
                // next group's first run out_off, else end (unused on last).
                runs.get(part.end).map(|r| r.out_off).unwrap_or(out_len)
            } else {
                out_len
            };
            scope.spawn(move || {
                for r in runs_part {
                    let dst = r.out_off - base;
                    let s = r.src_off - base_off;
                    out_slice[dst..dst + r.len].copy_from_slice(&deflate[s..s + r.len]);
                }
            });
        }
    });
}

/// Whole-buffer CRC32, parallel over contiguous chunks folded with
/// `combine_crc32` (equals the serial `crc32(output)` byte-for-byte).
#[cfg(parallel_sm)]
fn crc32_whole_parallel(output: &[u8], num_threads: usize) -> u32 {
    let n = output.len();
    if n == 0 {
        return 0;
    }
    let threads = num_threads.max(1).min(num_cpus::get_physical().max(1));
    if threads <= 1 || n < 1 << 20 {
        return crc32(output);
    }
    let chunk = n.div_ceil(threads);
    let chunks: Vec<&[u8]> = output.chunks(chunk).collect();
    let mut results: Vec<(u32, usize)> = vec![(0u32, 0usize); chunks.len()];
    std::thread::scope(|scope| {
        for (c, res) in chunks.iter().zip(results.iter_mut()) {
            let c = *c;
            scope.spawn(move || {
                *res = (crc32(c), c.len());
            });
        }
    });
    let mut acc = results[0].0;
    for (crc, len) in results.iter().skip(1) {
        acc = combine_crc32(acc, *crc, *len as u64);
    }
    acc
}

/// Maximum DEFLATE back-reference distance (RFC 1951 §3.2.5): a tail block can
/// reach at most this far before its first output byte.
#[cfg(parallel_sm)]
const MAX_WINDOW_SIZE: usize = 32 * 1024;

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
        // A real flate2 deflate stream (dynamic Huffman) has NO stored prefix
        // (0% stored). Production never routes such a stream here
        // (`first_block_is_stored` is false). A direct call must DECLINE without
        // writing: the whole-stream demotion gate sees stored fraction 0% < 50%
        // (Huffman dominates) and returns `NotStoredDominated` so the caller
        // routes to the ParallelSM grid — on every platform.
        let payload: Vec<u8> = (0..50_000).map(|i| (i % 7) as u8).collect();
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(6));
        enc.write_all(&payload).unwrap();
        let gz = enc.finish().unwrap();
        // first block must be Huffman (not stored) for this fixture.
        assert!(!first_block_is_stored(&gz));

        let mut out = Vec::new();
        let r = decompress_stored_parallel(&gz, &mut out, 4);
        match r {
            Err(StoredSplitError::NotStoredDominated) => {}
            other => panic!("expected NotStoredDominated (0% stored → demote), got {other:?}"),
        }
        assert!(out.is_empty(), "must not write on NotStoredDominated");
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
    ///   `huffman_first_block`: 0% stored → demotes (Huffman dominates).
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

    // ---- Multi-island fixed-Huffman assembler (for the adversarial test) ----

    /// LSB-first deflate bit writer (bits accumulated then packed low-bit-first).
    struct DeflateBits {
        bits: Vec<u8>,
    }
    impl DeflateBits {
        fn new() -> Self {
            Self { bits: Vec::new() }
        }
        fn bit(&mut self, b: u8) {
            self.bits.push(b & 1);
        }
        fn lsb(&mut self, val: u32, n: u32) {
            for i in 0..n {
                self.bit(((val >> i) & 1) as u8);
            }
        }
        fn huff(&mut self, code: u32, n: u32) {
            // Huffman codes pack most-significant bit first.
            for i in (0..n).rev() {
                self.bit(((code >> i) & 1) as u8);
            }
        }
        fn align(&mut self) {
            while !self.bits.len().is_multiple_of(8) {
                self.bit(0);
            }
        }
        fn into_bytes(self) -> Vec<u8> {
            let mut out = vec![0u8; self.bits.len().div_ceil(8)];
            for (i, &b) in self.bits.iter().enumerate() {
                out[i / 8] |= b << (i % 8);
            }
            out
        }
    }

    /// Fixed-Huffman literal/length symbol code (RFC 1951 §3.2.6).
    fn fixed_lit_code(sym: u32) -> (u32, u32) {
        match sym {
            0..=143 => (0x30 + sym, 8),
            144..=255 => (0x190 + (sym - 144), 9),
            256..=279 => (sym - 256, 7),
            _ => (0xC0 + (sym - 280), 8),
        }
    }

    // (symbol, base, extra_bits) for match lengths 3..=258.
    const LEN_TBL: &[(u32, u32, u32)] = &[
        (257, 3, 0),
        (258, 4, 0),
        (259, 5, 0),
        (260, 6, 0),
        (261, 7, 0),
        (262, 8, 0),
        (263, 9, 0),
        (264, 10, 0),
        (265, 11, 1),
        (266, 13, 1),
        (267, 15, 1),
        (268, 17, 1),
        (269, 19, 2),
        (270, 23, 2),
        (271, 27, 2),
        (272, 31, 2),
        (273, 35, 3),
        (274, 43, 3),
        (275, 51, 3),
        (276, 59, 3),
        (277, 67, 4),
        (278, 83, 4),
        (279, 99, 4),
        (280, 115, 4),
        (281, 131, 5),
        (282, 163, 5),
        (283, 195, 5),
        (284, 227, 5),
        (285, 258, 0),
    ];
    const DIST_TBL: &[(u32, u32, u32)] = &[
        (0, 1, 0),
        (1, 2, 0),
        (2, 3, 0),
        (3, 4, 0),
        (4, 5, 1),
        (5, 7, 1),
        (6, 9, 2),
        (7, 13, 2),
        (8, 17, 3),
        (9, 25, 3),
        (10, 33, 4),
        (11, 49, 4),
        (12, 65, 5),
        (13, 97, 5),
        (14, 129, 6),
        (15, 193, 6),
        (16, 257, 7),
        (17, 385, 7),
        (18, 513, 8),
        (19, 769, 8),
        (20, 1025, 9),
        (21, 1537, 9),
        (22, 2049, 10),
        (23, 3073, 10),
        (24, 4097, 11),
        (25, 6145, 11),
        (26, 8193, 12),
        (27, 12289, 12),
        (28, 16385, 13),
        (29, 24577, 13),
    ];

    /// Token for a fixed-Huffman island: a literal byte or a back-reference.
    enum Tok {
        Lit(u8),
        Match { len: u32, dist: u32 },
    }

    fn emit_stored_blocks(bw: &mut DeflateBits, data: &[u8], bfinal_last: bool) {
        if data.is_empty() {
            bw.bit(bfinal_last as u8);
            bw.lsb(0, 2);
            bw.align();
            bw.lsb(0, 16);
            bw.lsb(0xFFFF, 16);
            return;
        }
        let mut off = 0;
        while off < data.len() {
            let end = (off + 65535).min(data.len());
            let last = end == data.len();
            bw.bit((last && bfinal_last) as u8);
            bw.lsb(0, 2); // BTYPE=00
            bw.align();
            let ln = (end - off) as u32;
            bw.lsb(ln, 16);
            bw.lsb(!ln & 0xFFFF, 16);
            for &b in &data[off..end] {
                bw.lsb(b as u32, 8);
            }
            off = end;
        }
    }

    fn emit_fixed_island(bw: &mut DeflateBits, toks: &[Tok]) {
        bw.bit(0); // bfinal=0
        bw.lsb(1, 2); // BTYPE=01 (fixed Huffman)
        for t in toks {
            match *t {
                Tok::Lit(b) => {
                    let (c, n) = fixed_lit_code(b as u32);
                    bw.huff(c, n);
                }
                Tok::Match { len, dist } => {
                    let (sym, base, extra) = *LEN_TBL
                        .iter()
                        .rev()
                        .find(|&&(_, base, ex)| base <= len && len <= base + ((1 << ex) - 1))
                        .unwrap();
                    let (c, n) = fixed_lit_code(sym);
                    bw.huff(c, n);
                    bw.lsb(len - base, extra);
                    let (dsym, dbase, dextra) = *DIST_TBL
                        .iter()
                        .find(|&&(_, base, ex)| base <= dist && dist <= base + ((1 << ex) - 1))
                        .unwrap();
                    bw.huff(dsym, 5);
                    bw.lsb(dist - dbase, dextra);
                }
            }
        }
        let (c, n) = fixed_lit_code(256); // end-of-block
        bw.huff(c, n);
    }

    fn gzip_wrap(deflate: &[u8], payload: &[u8]) -> Vec<u8> {
        let mut gz = vec![0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff];
        gz.extend_from_slice(deflate);
        gz.extend_from_slice(&crc32(payload).to_le_bytes());
        gz.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        gz
    }

    /// THE ADVERSARIAL CASE (byte-exactness trap): a stored-heavy stream with two
    /// Huffman islands within 32 KiB where the SECOND island back-references into
    /// the FIRST island's DECODED output (bytes that exist nowhere in the input
    /// verbatim). If the island decoder reconstructed its predecessor window from
    /// input instead of the true decoded output, this corrupts. Verifies byte-
    /// exact output AND that the stream stays on StoredParallel (islands counter
    /// increments, demote counter unchanged).
    #[test]
    fn adversarial_adjacent_islands_cross_backref() {
        let mut rng = 0x1234_5678_9abc_def0u64;
        let mut rbytes = |n: usize| -> Vec<u8> {
            (0..n)
                .map(|_| {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (rng >> 33) as u8
                })
                .collect()
        };

        let mut bw = DeflateBits::new();
        let mut payload: Vec<u8> = Vec::new();

        // prefix stored (byte-aligned start → first_block_is_stored true)
        let p = rbytes(40_000);
        payload.extend_from_slice(&p);
        emit_stored_blocks(&mut bw, &p, false);

        // ISLAND 1: a distinctive 300-byte literal run.
        let s1: Vec<u8> = (0..300u32)
            .map(|i| (i.wrapping_mul(37) + 11) as u8)
            .collect();
        payload.extend_from_slice(&s1);
        emit_fixed_island(
            &mut bw,
            &s1.iter().map(|&b| Tok::Lit(b)).collect::<Vec<_>>(),
        );

        // stored B: 2000 bytes (< 32 KiB → island 2 within 32 KiB of island 1)
        let b = rbytes(2000);
        payload.extend_from_slice(&b);
        emit_stored_blocks(&mut bw, &b, false);

        // ISLAND 2: a MATCH reaching back INTO island 1's decoded output.
        // out_pos = 40000 + 300 + 2000 = 42300; dist 2200 → src 40100 ∈ island 1.
        let out_pos = 40_000 + 300 + 2000;
        let (dist, len) = (2200u32, 200u32);
        let src = out_pos - dist as usize;
        let copied = payload[src..src + len as usize].to_vec();
        payload.extend_from_slice(&copied);
        emit_fixed_island(&mut bw, &[Tok::Match { len, dist }]);

        // final stored suffix (multi-block → exercises the fast-walk resume;
        // > 1 MiB total stored so copy_runs_parallel takes its PARALLEL branch
        // at T>1, not just the serial fallback).
        let s = rbytes(1_200_000);
        payload.extend_from_slice(&s);
        emit_stored_blocks(&mut bw, &s, true);

        let gz = gzip_wrap(&bw.into_bytes(), &payload);
        assert!(first_block_is_stored(&gz), "must start stored");

        // Independent oracle.
        let mut oracle = Vec::new();
        {
            use std::io::Read as _;
            flate2::read::GzDecoder::new(&gz[..])
                .read_to_end(&mut oracle)
                .unwrap();
        }
        assert_eq!(
            oracle, payload,
            "oracle sanity (hand-built stream is valid)"
        );

        let demote_before = STORED_DEMOTE_TO_PARALLEL_SM.load(Ordering::Relaxed);
        let islands_before = STORED_ISLANDS_RUNS.load(Ordering::Relaxed);

        for t in [1usize, 2, 4, 8] {
            let mut out = Vec::new();
            let r = decompress_stored_parallel(&gz, &mut out, t);
            #[cfg(parallel_sm)]
            {
                assert_eq!(r.map(|n| n as usize).unwrap(), payload.len(), "t={t}");
                assert_eq!(
                    out, payload,
                    "adversarial cross-island back-ref must be byte-exact at t={t}"
                );
            }
            #[cfg(not(parallel_sm))]
            {
                let _ = t;
                match r {
                    Err(StoredSplitError::NotStoredDominated) => {}
                    other => {
                        panic!("expected NotStoredDominated on non-parallel_sm, got {other:?}")
                    }
                }
            }
        }

        #[cfg(parallel_sm)]
        {
            // Stayed on the stored path (islands fired) and was NOT demoted.
            assert!(
                STORED_ISLANDS_RUNS.load(Ordering::Relaxed) > islands_before,
                "islands path must have run"
            );
            assert_eq!(
                STORED_DEMOTE_TO_PARALLEL_SM.load(Ordering::Relaxed),
                demote_before,
                "stored-dominant islands stream must NOT demote"
            );
        }
        let _ = (demote_before, islands_before);
    }
}
