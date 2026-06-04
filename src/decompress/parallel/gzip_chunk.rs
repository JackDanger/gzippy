#![cfg(parallel_sm)]

//! Per-chunk deflate decode for parallel single-member.
//!
//! - [`decode_chunk_with_rapidgzip`] — vendor `decodeChunkWithRapidgzip` +
//!   `finishDecodeChunkWithInexactOffset` on one [`ChunkData`]:
//!   one outer decode iteration (`worker.decode_chunk`) alternates
//!   `marker_inflate` blocks (u16 markers) until 32 KiB clean, then streaming
//!   inflate on the same [`ChunkData`].
//! - [`decode_chunk`] — production entry (known 32 KiB window or chunk 0).
//! - [`decode_chunk_window_absent`] — marker bootstrap + clean tail (no window).
//!
//! `stop_hint_bits` is an inexact stop hint (vendor `untilOffset`): the
//! decoder runs to the first block boundary at-or-past it, then stops.

use crate::decompress::parallel::chunk_data::{ChunkConfiguration, ChunkData};
use crate::decompress::parallel::inflate_wrapper::InflateError;
#[cfg(parallel_sm)]
use crate::decompress::parallel::inflate_wrapper::{
    DeflateCompressionType, IsalInflateWrapper, StoppingPoints,
};
#[cfg(parallel_sm)]
use crate::decompress::parallel::rpmalloc_alloc::types;
#[cfg(parallel_sm)]
use crate::decompress::parallel::trace;

#[derive(Debug)]
#[allow(dead_code)] // error payloads surfaced via Debug in production
pub enum ChunkDecodeError {
    InflateFailed(InflateError),
    BootstrapFailed(std::io::Error),
    ExactStopMissed {
        requested: usize,
        actual: usize,
    },
    #[allow(dead_code)] // non-SM-build cfg stub
    UnsupportedPlatform,
}

impl From<InflateError> for ChunkDecodeError {
    fn from(e: InflateError) -> Self {
        ChunkDecodeError::InflateFailed(e)
    }
}

impl From<std::io::Error> for ChunkDecodeError {
    fn from(e: std::io::Error) -> Self {
        ChunkDecodeError::BootstrapFailed(e)
    }
}

/// Output buffer size used per `read_stream` iteration. Matches
/// rapidgzip's `ALLOCATION_CHUNK_SIZE` (GzipChunk.hpp uses 128 KiB).
const ALLOCATION_CHUNK_SIZE: usize = 128 * 1024;

// =========================================================================
// Body-failure diagnostics — speculation-accuracy investigation
// =========================================================================
// Per advisor-disproof: gzippy fails ~34 body speculations per silesia run,
// vendor fails 0. The hypothesis is that the WHY of these failures matters:
// where in the body does decode break, what error variant fires, how many
// bytes are wasted before failure. If failures cluster on specific corpus
// regions or specific error types, we can build a precondition heuristic
// that vendor has but we don't.

pub static BODY_FAIL_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static BODY_FAIL_BYTES_WASTED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static BODY_FAIL_BITS_INTO_BODY: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static BODY_FAIL_INVALID_HUFFMAN: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static BODY_FAIL_EXCEEDED_WINDOW: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static BODY_FAIL_INVALID_COMPRESSION: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static BODY_FAIL_INVALID_CODE_LENGTHS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static BODY_FAIL_OTHER_VARIANT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

// C-instrumentation: per-block table-build vs body cost inside
// bootstrap_with_deflate_block. Each chunk's bootstrap iterates blocks,
// for each: read_header (precode + lit/len/dist table build) then body.
// 8.3B flame samples in "marker bootstrap" — need to split.
pub static BOOTSTRAP_HEADER_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static BOOTSTRAP_HEADER_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static BOOTSTRAP_BODY_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static BOOTSTRAP_BODY_BYTES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Commit-1 instrumentation for the bootstrap-unification lever: bytes decoded
/// into the u16 marker ring AFTER `contains_marker_bytes` flips false (the flip
/// is permanent within a chunk — once 32 KiB clean exists, no back-ref can
/// reach the unknown predecessor). These are the bytes that could instead be
/// decoded into a u8 linear buffer with the fast copy path (Design B1). The
/// ratio POST_FLIP / BODY_BYTES sizes the prize. No behavior change.
pub static BOOTSTRAP_POST_FLIP_U16_BYTES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// P2 unified decode: clean literals/backrefs routed to `chunk.data` (u8) after
/// `contains_marker_bytes` flips false — rapidgzip-shaped single output stream.
pub static UNIFIED_ROUTE_CLEAN_U8_BYTES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static UNIFIED_MODE_CLEAN_FLIPS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Clean tail stayed on bulk LUT across a 128 KiB segment boundary (P2 unified).
pub static BULK_TAIL_SEGMENT_CONTINUES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Bulk LUT declined; tail used ResumableInflate2 (should be rare with 32 KiB window).
pub static BULK_TAIL_RESUMABLE_FALLBACK: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Times the reused thread-local marker→clean handoff window buffer had to GROW
/// its capacity. Proves the per-chunk 32 KiB `clean_window: Vec<u8>` allocation
/// is gone: this should settle at ≈ num_worker_threads (one grow per thread on
/// first handoff) and then stay flat — NOT scale with the window-absent chunk
/// count. If it tracks the chunk count, the buffer is being reallocated per
/// chunk (the seam is not actually cut).
pub static HANDOFF_WINDOW_BUF_GROWS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Chunks where the unified marker driver failed at a bad 4 MiB speculative seed
/// and fell to the internal resumable re-sync (the goal's accepted re-sync).
/// Should be ~0 for real-boundary chunks and bounded by the speculative-seed count.
pub static BAD_SEED_RESYNC: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Chunks that took the marker→clean FLIP: a bounded u16 marker prefix then the
/// fast u8 clean tail (vendor setInitialWindow). This is the FAST window-absent
/// path; if it stays 0 while speculative chunks exist, every window-absent chunk
/// is decoding its whole body as u16 markers + full resolve (the slow path).
pub static FLIP_TO_CLEAN_CHUNKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Chunks that finished WITHOUT a flip (BFINAL or stop-hint reached before 32 KiB
/// of clean output accumulated) — the whole decoded body stayed u16 markers.
pub static FINISHED_NO_FLIP_CHUNKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// `resumable_resync` bulk loop saw `Handoff` with no bit advance (stall guard).
#[cfg(pure_inflate_decode)]
pub static BULK_HANDOFF_NO_PROGRESS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Why [`finish_decode_chunk_bulk_lut`] returned [`BulkCleanTailResult::Decline`].
#[cfg(pure_inflate_decode)]
pub static BULK_LUT_ENTERED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(pure_inflate_decode)]
pub static BULK_LUT_DECLINED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(pure_inflate_decode)]
pub static BULK_DECLINE_DISABLED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
#[cfg(pure_inflate_decode)]
pub static BULK_DECLINE_NO_LOOKBACK: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
#[cfg(pure_inflate_decode)]
pub static BULK_DECLINE_INVALID_HUFFMAN: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
#[cfg(pure_inflate_decode)]
pub static BULK_DECLINE_INVALID_LOOKBACK: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
#[cfg(pure_inflate_decode)]
pub static BULK_DECLINE_OUTPUT_OVERFLOW: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
#[cfg(pure_inflate_decode)]
pub static BULK_DECLINE_OTHER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(pure_inflate_decode)]
fn record_bulk_decline(err: crate::decompress::parallel::isal_lut_bulk::BulkDecodeError) {
    use crate::decompress::parallel::isal_lut_bulk::BulkDecodeError;
    use std::sync::atomic::Ordering;
    let c = match err {
        BulkDecodeError::InvalidHuffmanCode => &BULK_DECLINE_INVALID_HUFFMAN,
        BulkDecodeError::InvalidLookback => &BULK_DECLINE_INVALID_LOOKBACK,
        BulkDecodeError::OutputOverflow => &BULK_DECLINE_OUTPUT_OVERFLOW,
        BulkDecodeError::InvalidCodeLengths | BulkDecodeError::BlockTypeReserved => {
            &BULK_DECLINE_OTHER
        }
    };
    c.fetch_add(1, Ordering::Relaxed);
}

/// Per-failure structured log. Writes one JSON line to the path in
/// `GZIPPY_BODY_FAIL_LOG` env var (no-op if unset). Use this for
/// distributional analysis: cluster failures by offset to see if they
/// fall in specific corpus regions.
#[cfg(parallel_sm)]
fn body_fail_log(start_bit: usize, bits_into_body: usize, bytes_wasted: usize, err: &str) {
    use std::io::Write;
    use std::sync::Mutex;
    use std::sync::OnceLock;
    static FILE: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
    let f = FILE.get_or_init(|| {
        let path = std::env::var("GZIPPY_BODY_FAIL_LOG").ok()?;
        let f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok()?;
        Some(Mutex::new(f))
    });
    let Some(mtx) = f else {
        return;
    };
    let line = format!(
        r#"{{"start_bit":{start_bit},"bits_into_body":{bits_into_body},"bytes_wasted":{bytes_wasted},"err":"{}"}}"#,
        err.replace('"', "\\\"")
    );
    let mut g = mtx.lock().unwrap();
    let _ = writeln!(g, "{}", line);
}

#[cfg(not(parallel_sm))]
fn body_fail_log(_: usize, _: usize, _: usize, _: &str) {}

/// Rapidgzip-shaped chunk decode — `GzipChunk.hpp::decodeChunkWithRapidgzip`
/// (outer `while` over deflate blocks) with handoff to
/// `finishDecodeChunkWithInexactOffset` once 32 KiB clean exist at a block
/// boundary (or immediately when `initial_window` is full 32 KiB).
#[cfg(parallel_sm)]
pub fn decode_chunk_with_rapidgzip(
    input: &[u8],
    encoded_offset_bits: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    decode_chunk_with_rapidgzip_impl(
        input,
        encoded_offset_bits,
        stop_hint_bits,
        initial_window,
        configuration,
    )
}

/// Production chunk decode with optional predecessor window (vendor
/// `decodeChunkWithRapidgzip` + `finishDecodeChunkWithInexactOffset`).
#[cfg(parallel_sm)]
pub fn decode_chunk(
    input: &[u8],
    encoded_offset_bits: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    decode_chunk_with_rapidgzip(
        input,
        encoded_offset_bits,
        stop_hint_bits,
        initial_window,
        configuration,
    )
}

/// Option A3+A4 dispatch (`plans/unified-decoder.md` §6, commits
/// `9e14bbe`/`757e7a4`/`a6076e1`). Pre-fills `chunk.data[0..32K]` with
/// the predecessor's window image so every back-reference hits the
/// AVX2 fast path; the consumer skips `chunk.data[..data_prefix_len]`
/// when writing.
///
/// Measured +4.2% T=16 silesia on neurotic, byte-perfect across 11
/// P2 unified clean tail (vendor `finishDecodeChunkWithInexactOffset`,
/// GzipChunk.hpp:280-410): ISA-L LUT bulk loop first; ResumableInflate2 only
/// when lookback is unavailable (< 32 KiB predecessor).
#[cfg(parallel_sm)]
fn resumable_resync(
    chunk: &mut ChunkData,
    input: &[u8],
    mut inflate_start_bit: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
    record_decode_duration: bool,
) -> Result<(), ChunkDecodeError> {
    use crate::decompress::parallel::marker_inflate::MAX_WINDOW_SIZE;

    // `worker.isal_stream_inflate` span — wraps every invocation of
    // gzippy's ISA-L stream-inflate path. Both call sites land here:
    //   (a) the post-bootstrap call at gzip_chunk.rs:549 (after the
    //       pure-Rust phase-1 produces a clean 32 KiB window)
    //   (b) the non-speculative direct path via `decode_chunk`
    //       at gzip_chunk.rs:150 (when the predecessor window is
    //       already known and bootstrap is skipped)
    //
    // Pre-instrumentation the only proxy for ISA-L bulk-inflate time
    // was `decode_chunk dur - bootstrap dur` — a derived value with
    // no per-call visibility. With this span we can answer the
    // post-Fix-#3 question: is the 1.5x per-chunk gap to vendor in
    // bootstrap or in ISA-L bulk inflate?
    //
    // Args: `start_bit` (where decode starts) + `stop_hint` + a
    // `has_window` flag distinguishing the two call paths above.
    let t_decode = std::time::Instant::now();

    // Option A3 (always on): pre-fill segment 0 with the predecessor's
    // 32 KiB window so back-references hit copy_match_fast via
    // `output[..out_pos]`. The consumer skips `data_prefix_len` when writing (A4).
    let full_window_tail = initial_window.len() == MAX_WINDOW_SIZE;
    // A3: seed lookback into `chunk.data[0..32K]` once, before any tail bytes.
    // Resumable `set_window` must stay empty when prefilled — otherwise the
    // 32 KiB window is applied twice (bulk + resumable ring) and stored blocks
    // decode with wrong lengths.
    if full_window_tail && chunk.data.is_empty() {
        chunk.prefill_window_prefix(initial_window);
    }
    let resumable_window: &[u8] = if chunk.data_prefix_len >= MAX_WINDOW_SIZE {
        &[]
    } else {
        initial_window
    };

    // Bad-seed RE-SYNC ONLY: this function is now reached solely as the internal
    // fallback when the unified marker driver fails on a 4 MiB speculative seed.
    // The bulk-LUT clean tail (finish_decode_chunk_bulk_lut) is deleted — it only
    // ever declined here on a non-boundary seed anyway. ResumableInflate2 re-syncs
    // from `inflate_start_bit` (the goal's accepted resumable re-sync).
    BULK_TAIL_RESUMABLE_FALLBACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let _tv2 = crate::decompress::parallel::trace_v2::SpanGuard::begin_with(
        "worker.stream_inflate",
        &format!(
            r#""start_bit":{inflate_start_bit},"stop_hint":{stop_hint_bits},"has_window":{}"#,
            !initial_window.is_empty()
        ),
    );
    // `stop_hint_bits` is an inexact stop *hint* (vendor `untilOffset`), not a hard
    // read cap. Capping `refill_buffer` at a partition guess stops mid-block
    // (e.g. silesia gzip-9 at 33554427 vs hint 33554432 → InvalidBlock on resume).
    let read_cap = input.len() * 8;
    let mut wrapper = IsalInflateWrapper::with_until_bits(input, inflate_start_bit, read_cap)?;
    wrapper.set_window(resumable_window)?;
    wrapper.set_stopping_points(
        StoppingPoints::END_OF_BLOCK
            | StoppingPoints::END_OF_BLOCK_HEADER
            | StoppingPoints::END_OF_STREAM_HEADER
            | StoppingPoints::END_OF_STREAM,
    );

    let mut stopping_point_reached = false;
    let mut last_end_bit = inflate_start_bit;
    let mut last_eob_pos = inflate_start_bit;
    let mut last_eob_decoded_bytes: usize = chunk.data.len();
    let mut already_decoded: usize = 0;
    // Set once the decoder reaches the genuine end of the input (final
    // gzip member's footer consumed, or ISA-L reports `finished`). When
    // true the chunk finalizes at `last_end_bit` (post-footer / post-
    // final-block), NOT at `last_eob_pos` — the latter is the boundary
    // BEFORE the BFINAL block, so finalizing there drops the final block
    // from the chunk's encoded range and the consumer re-decodes it,
    // duplicating its bytes in the output.
    let mut reached_stream_end = false;
    let mut pending_stop_after_flush = false;
    // §5 step 5 (NEW wrapper / ResumableInflate2): the pure-rust backend no
    // longer has a `session` accumulator, so `session_pending()` is always
    // false. The old "keep looping to drain session" behavior (a6c0a8b) would
    // run past the partition boundary, clobbering `last_end_bit` with the
    // bit_position of the NEXT block's header — successor chunks would then
    // resume at a non-block-boundary and decode garbage Huffman tables. With
    // ResumableInflate2 the inner loop must STOP as soon as
    // `pending_stop_after_flush` fires, same as the isal-only build.
    const STOP_INNER_ON_PENDING_FLUSH: bool = true;

    while !stopping_point_reached || wrapper.session_pending() {
        // Write ISA-L output DIRECTLY into chunk.data's spare capacity.
        // Previous version allocated a fresh `ALLOCATION_CHUNK_SIZE`
        // (128 KiB) buffer per outer iteration, ran ISA-L into it, then
        // called `append_owned_buffer` which either replaced
        // chunk.data (first call — losing the 80 MiB pre-allocated
        // pool buffer) or extend_from_slice'd it (subsequent calls —
        // paying a memcpy + grow page faults). Per the bench-sm
        // perf-dwarf profile this accounted for ~3.4% of total cycles
        // (`append_owned_buffer → Vec::extend_from_slice → page fault`).
        //
        // Vendor pattern: `GzipChunk::readBlock` writes directly into
        // the chunk's vector via `next_out`; no intermediate buffer.
        // gzippy's `ChunkData::note_clean_bytes_written_in_place`
        // helper was already in place (chunk_data.rs:405) — this just
        // wires it up.
        let prev_data_len = chunk.data.len();
        // FOOTPRINT-ALIGN: `chunk.data` is a `SegmentedU8`. Acquire a fresh
        // 128 KiB segment tail to decode this outer iteration into. The
        // returned slice is exactly ALLOCATION_CHUNK_SIZE bytes (a brand-new
        // segment if the prior tail was full) — uninitialized but writable.
        // We write into it across the inner `read_stream` calls, then
        // `commit(append_len)` the bytes we keep. Back-refs reaching before
        // this segment resolve via the wrapper's internal 32 KiB window ring
        // (resumable.rs:653 feeds each call's output into `state.window`), so
        // segment boundaries are transparent to correctness — vendor's
        // `DecodedData::append` segments output the same way
        // (DecodedData.hpp:243-289).
        // `writable_tail` returns the CURRENT tail segment's spare — which is
        // a FULL 128 KiB when the prior outer iteration filled (or there was
        // none), or a partial remainder when the prior commit left the tail
        // non-full. Either way `buffer_cap` is the real writable spare; the
        // inner loop respects it and the outer loop simply runs again to fill
        // the next segment. (No fixed-128KiB assumption — that was wrong for
        // partially-committed tails and tripped the 2 MB decode test.)
        let seg_tail: &mut [u8] = chunk.data.writable_tail();
        let seg_ptr = seg_tail.as_mut_ptr();
        let buffer_cap = seg_tail.len();
        let mut n_bytes_read: usize = 0;
        let mut last_per_call: usize = 0;
        let mut last_stopped_at = StoppingPoints::NONE;
        let mut last_finished = false;
        let mut end_of_stream_hit = false;

        let decode_base = already_decoded;
        while n_bytes_read < buffer_cap
            && !stopping_point_reached
            && !(STOP_INNER_ON_PENDING_FLUSH
                && pending_stop_after_flush
                && !wrapper.session_pending())
        {
            let bit_before_read = wrapper.tell_compressed();
            // A3: decode into segment 0's full backing store so back-refs
            // resolve via `output[..out_pos]` (includes the prefilled window).
            // Non-A3 / multi-segment: spare slice on the tail segment only;
            // cross-chunk refs use the wrapper's window ring.
            let r = if full_window_tail && chunk.data.all_in_first_segment() {
                let out_buf = chunk.data.first_segment_a3_output();
                wrapper.read_stream_starting_at(out_buf, prev_data_len + n_bytes_read)?
            } else {
                let spare: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(
                        seg_ptr.add(n_bytes_read),
                        buffer_cap - n_bytes_read,
                    )
                };
                wrapper.read_stream(spare)?
            };
            last_per_call = r.bytes_written;
            n_bytes_read += last_per_call;
            chunk.note_inner_decoded_bytes(last_per_call);

            last_stopped_at = r.stopped_at;
            last_finished = r.finished;
            last_end_bit = r.bit_position;

            // Defense-in-depth termination guard. If `read_stream` made
            // no progress whatsoever — no output, no stopping point, not
            // finished, and the compressed cursor did not advance — then
            // ISA-L is exhausted on this input and every further call
            // would stall identically (the input is fixed; `read_stream`
            // already looped internally until `!made_progress`). Finalize
            // the chunk here instead of spinning forever.
            //
            // In production `consumer_loop`'s sub-byte-tail guard means a
            // chunk too small to hold a deflate block is never scheduled;
            // this is the backstop for any future caller that hands
            // `decode_chunk` such a fragment directly.
            //
            // A genuinely truncated stream also stalls here; its
            // already-decoded prefix is kept (appended by earlier
            // iterations) and the short finalize is caught downstream by
            // CRC32/ISIZE verification.
            if r.bytes_written == 0
                && r.stopped_at == StoppingPoints::NONE
                && !r.finished
                && r.bit_position == bit_before_read
            {
                stopping_point_reached = true;
                break;
            }

            // END_OF_STREAM must be detected even when ISA-L reports
            // `finished` on the same call — `read_stream` returns both
            // flags together at the final block. Checking `finished`
            // first would skip the footer-reading branch below, leaving
            // the chunk finalized at the pre-BFINAL-block EOB.
            if r.stopped_at == StoppingPoints::END_OF_STREAM {
                end_of_stream_hit = true;
                break;
            }
            if r.finished {
                break;
            }

            match r.stopped_at {
                sp if sp == StoppingPoints::END_OF_STREAM_HEADER => {
                    chunk.append_block_boundary_at(r.bit_position, decode_base + n_bytes_read);
                }
                sp if sp == StoppingPoints::END_OF_BLOCK => {
                    if !wrapper.is_final_block() {
                        chunk.append_block_boundary_at(r.bit_position, decode_base + n_bytes_read);
                        if r.bit_position >= stop_hint_bits {
                            // Do not keep filling this buffer from the next block
                            // before HEADER/NONE handling — finalize at pre-header EOB.
                            last_end_bit = r.bit_position;
                            pending_stop_after_flush = true;
                        }
                    }
                    last_eob_pos = r.bit_position;
                    last_eob_decoded_bytes = decode_base + n_bytes_read;
                }
                sp if sp == StoppingPoints::END_OF_BLOCK_HEADER => {
                    let not_final = !wrapper.is_final_block();
                    let not_fixed = wrapper.btype() != Some(DeflateCompressionType::FixedHuffman);
                    if last_eob_pos >= stop_hint_bits && not_final && not_fixed {
                        last_end_bit = last_eob_pos;
                        pending_stop_after_flush = true;
                    }
                }
                sp if sp == StoppingPoints::NONE
                    && last_per_call == 0
                    && last_eob_pos >= stop_hint_bits =>
                {
                    // ISA-L can return 0 bytes between block boundaries.
                    // Do not end the chunk early while still before `stop_hint_bits`.
                    last_end_bit = last_eob_pos;
                    pending_stop_after_flush = true;
                }
                _ => {}
            }
            if end_of_stream_hit || last_finished {
                break;
            }
        }

        let mut append_len = n_bytes_read;
        if stopping_point_reached {
            // Stopped on NONE+0 / HEADER after the prior END_OF_BLOCK was
            // already appended in a previous outer iteration — do not
            // emit bytes from the next block read into this buffer.
            append_len = last_eob_decoded_bytes.saturating_sub(decode_base);
        } else if pending_stop_after_flush {
            // Session-flush outer iterations after the EOB hint fired:
            // all bytes in this buffer are tail output from the same block.
            append_len = n_bytes_read;
        }
        if append_len > 0 {
            // CRC the kept bytes FIRST (read them straight from the segment
            // we just wrote, before `commit`), then commit `append_len` into
            // the segmented buffer. Bytes `[append_len, n_bytes_read)` that
            // the decoder wrote but we're discarding (between EOB and the
            // truncate point) simply never get committed — the next outer
            // iteration overwrites the segment tail from `append_len`.
            //
            // We deliberately do NOT touch `subchunks.last_mut().decoded_size`
            // here because the inner loop already credited the FULL
            // n_bytes_read via `note_inner_decoded_bytes(last_per_call)`.
            //
            // SAFETY: `append_len ≤ n_bytes_read ≤ buffer_cap`; the decoder
            // wrote initialized bytes through `seg_ptr[0..n_bytes_read]`, so
            // `seg_ptr[0..append_len]` is fully initialized.
            if chunk.configuration.crc32_enabled {
                let kept: &[u8] = unsafe { std::slice::from_raw_parts(seg_ptr, append_len) };
                if let Some(last_crc) = chunk.crc32s.last_mut() {
                    last_crc.update(kept);
                }
            }
            chunk.data.commit(append_len);
            chunk.statistics.non_marker_count += append_len as u64;
        }
        let _ = prev_data_len;
        already_decoded = decode_base + append_len;

        if pending_stop_after_flush && !wrapper.session_pending() {
            stopping_point_reached = true;
        }

        if end_of_stream_hit {
            let (crc32, isize_field) = wrapper.read_footer_at_current()?;
            let footer_end_bits = wrapper.tell_compressed();
            chunk.append_footer(crc32, isize_field, footer_end_bits);

            let remaining = wrapper.remaining_input();
            if remaining.is_empty() {
                last_end_bit = footer_end_bits;
                reached_stream_end = true;
                break;
            }
            let (_hdr, hdr_size) = crate::decompress::parallel::gzip_format::read_header(remaining)
                .map_err(|e| {
                    ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("multi-stream gzip header at bit {footer_end_bits}: {e}"),
                    ))
                })?;
            wrapper.advance_input(hdr_size);
            wrapper.reset_for_next_stream();
            wrapper.set_stopping_points(
                StoppingPoints::END_OF_BLOCK
                    | StoppingPoints::END_OF_BLOCK_HEADER
                    | StoppingPoints::END_OF_STREAM_HEADER
                    | StoppingPoints::END_OF_STREAM,
            );
            wrapper.set_window(&[])?;
            last_end_bit = wrapper.tell_compressed();
            last_eob_pos = last_end_bit;
            chunk.append_block_boundary_at(last_end_bit, already_decoded);
            continue;
        }

        if last_finished {
            reached_stream_end = true;
            break;
        }
        if last_stopped_at == StoppingPoints::NONE && last_per_call == 0 {
            if last_eob_pos >= stop_hint_bits {
                last_end_bit = last_eob_pos;
                break;
            }
            continue;
        }
    }

    if record_decode_duration {
        chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
    }
    // A4: window-image prefix STAYS in chunk.data through finalize
    // and into the consumer. The consumer write site reads
    // &chunk.data[chunk.data_prefix_len..] so the prefix never
    // reaches the user's output. Downstream chunk methods that
    // need the decoded portion (last_32kib_window, get_last_window,
    // populate_subchunk_windows) skip data_prefix_len bytes;
    // `resumable_resync` reads from
    // `tail.data[tail.data_prefix_len..]`. Eliminating the trim
    // memmove was measured at -3.81pp `__memmove_avx_unaligned_erms`
    // CPU share vs A3-with-trim, lifting net throughput to +4.2%
    // vs default at T=16 on neurotic silesia-gzip9.
    let _ = full_window_tail; // consumed at decode-start (A3 prefill + bulk path)
                              // When the reader runs past `stop_hint_bits` without an inexact
                              // stop, `tell_compressed()` can land mid-block. Successor chunks must
                              // resume at the last END_OF_BLOCK position (pre-header), not mid-block.
    let final_bit = if stopping_point_reached || reached_stream_end {
        last_end_bit
    } else if last_eob_pos > inflate_start_bit {
        // Hit the until read cap without an inexact header-stop — finalize at
        // the last pre-header EOB, never at a mid-block bit cursor.
        last_eob_pos
    } else {
        wrapper.tell_compressed()
    };
    chunk.finalize(final_bit);
    Ok(())
}

/// Window-absent chunk decode (speculative prefetch / boundary search).
/// Same unified [`decode_chunk_with_rapidgzip_impl`] as [`decode_chunk`], with
/// an empty initial window so the marker phase runs until 32 KiB clean.
#[cfg(parallel_sm)]
pub fn decode_chunk_window_absent(
    input: &[u8],
    encoded_offset_bits: usize,
    stop_hint_bits: usize,
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    decode_chunk_with_rapidgzip_impl(
        input,
        encoded_offset_bits,
        stop_hint_bits,
        &[],
        configuration,
    )
}

/// `decodeChunkWithRapidgzip` body (GzipChunk.hpp:468-654).
#[cfg(parallel_sm)]
fn decode_chunk_with_rapidgzip_impl(
    input: &[u8],
    encoded_offset_bits: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    use crate::decompress::parallel::marker_inflate::MAX_WINDOW_SIZE;

    // Envelope span: `chunk_fetcher::run_decode_task` (`worker.decode_chunk`).
    let t_decode = std::time::Instant::now();

    // ONE unified decode driver (vendor single deflate::Block shape): the marker
    // engine decodes the WHOLE chunk into data_with_markers — emitting u16 markers
    // while the predecessor window is absent and flipping mid-stream to clean u16
    // without restarting — resolved later on the publish chain. There is no
    // separate clean-tail entry. On a bad speculative-seed decode failure (4 MiB
    // partition guesses that aren't real block boundaries), fall to the resumable
    // re-sync, kept ONLY as an internal correctness strategy (the goal's accepted
    // re-sync), never an outer fallback ladder.
    let _ = &initial_window;
    match decode_chunk_unified_marker(input, encoded_offset_bits, stop_hint_bits, configuration) {
        Ok(mut chunk) => {
            chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
            Ok(chunk)
        }
        Err(marker_err) => {
            BAD_SEED_RESYNC.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut chunk = ChunkData::new(encoded_offset_bits, configuration);
            resumable_resync(
                &mut chunk,
                input,
                encoded_offset_bits,
                stop_hint_bits,
                initial_window,
                false,
            )
            .map_err(|_| marker_err)?;
            chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
            Ok(chunk)
        }
    }
}

/// The ONE unified decode driver. `marker_inflate::Block` emits u16 markers while
/// the predecessor window is absent and flips mid-stream to clean u16 (no restart),
/// decoding the whole chunk into `data_with_markers`; the chunk ends at BFINAL or
/// the `stop_hint` partition boundary. Returns `Err` on a bad speculative seed so
/// the caller can fall to the internal resumable re-sync.
#[cfg(parallel_sm)]
fn decode_chunk_unified_marker(
    input: &[u8],
    encoded_offset_bits: usize,
    stop_hint_bits: usize,
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    let mut chunk = ChunkData::new(encoded_offset_bits, configuration);
    chunk.data_with_markers.reserve(128 * 1024);
    let mut marker_ctx = MarkerDecodeCtx::new(input, encoded_offset_bits)?;
    loop {
        let (step, flipped_clean) = marker_decode_step(
            &mut marker_ctx,
            input,
            stop_hint_bits,
            &mut chunk.data_with_markers,
        )?;
        if flipped_clean {
            UNIFIED_MODE_CLEAN_FLIPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        match step {
            MarkerStep::Continue => {}
            MarkerStep::FlipToClean {
                end_bit_offset,
                window_len,
            } => {
                FLIP_TO_CLEAN_CHUNKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // FLIP (vendor setInitialWindow): the marker prefix (u16, ≤ ~32 KiB)
                // is done; decode the REST as the u8 clean tail into chunk.data on
                // the same cursor, with the flipped 32 KiB clean window narrowed
                // from the marker-prefix tail. The clean tail is u8 (NOT u16) and is
                // never resolved — so the marker work stays bounded.
                use crate::decompress::parallel::marker_inflate::MarkerSink;
                use std::cell::RefCell;
                thread_local! {
                    static FLIP_WINDOW: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(32 * 1024));
                }
                chunk.statistics.non_marker_count += chunk.data_with_markers.len() as u64;
                FLIP_WINDOW.with(|cell| -> Result<(), ChunkDecodeError> {
                    let mut win = cell.borrow_mut();
                    let ok = chunk
                        .data_with_markers
                        .copy_last_n_clean_u8(window_len, &mut win);
                    debug_assert!(ok, "flip window tail not all-clean");
                    resumable_resync(
                        &mut chunk,
                        input,
                        end_bit_offset,
                        stop_hint_bits,
                        &win,
                        false,
                    )
                })?;
                return Ok(chunk);
            }
            MarkerStep::Finished { end_bit_offset, .. } => {
                FINISHED_NO_FLIP_CHUNKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                chunk.statistics.non_marker_count += chunk.data_with_markers.len() as u64;
                chunk.finalize(end_bit_offset);
                return Ok(chunk);
            }
        }
    }
}

/// One iteration of the vendor `decodeChunkWithRapidgzip` block loop.
#[cfg(parallel_sm)]
enum MarkerStep {
    /// Another deflate block was decoded; call again.
    Continue,
    /// 32 KiB of clean output reached at a block boundary — FLIP to the u8 clean
    /// tail (vendor setInitialWindow). The clean 32 KiB window is the tail of
    /// `data_with_markers`; the caller decodes the rest as u8 into `chunk.data`.
    FlipToClean {
        end_bit_offset: usize,
        window_len: usize,
    },
    /// Chunk ends in the marker path (BFINAL, stop hint, or no clean dict).
    Finished {
        end_bit_offset: usize,
        bfinal_hit: bool,
    },
}

/// Persistent state for [`marker_decode_step`] (one block per call).
#[cfg(parallel_sm)]
struct MarkerDecodeCtx {
    /// `Bits` slice base in `data` (byte index).
    data_base_byte: usize,
    current_bit_offset: usize,
    trailing_clean: usize,
    block_primed: bool,
}

#[cfg(parallel_sm)]
impl MarkerDecodeCtx {
    fn new(_data: &[u8], start_bit_offset: usize) -> Result<Self, ChunkDecodeError> {
        let data_base_byte = start_bit_offset / 8;
        if data_base_byte >= _data.len() {
            return Err(ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "start_bit_offset past end of data",
            )));
        }
        Ok(Self {
            data_base_byte,
            current_bit_offset: start_bit_offset,
            trailing_clean: 0,
            block_primed: false,
        })
    }

    fn open_bits<'a>(
        &self,
        data: &'a [u8],
    ) -> crate::decompress::inflate::consume_first_decode::Bits<'a> {
        let slice_byte = self.current_bit_offset / 8;
        let mut bits =
            crate::decompress::inflate::consume_first_decode::Bits::new(&data[slice_byte..]);
        let bit_in_byte = (self.current_bit_offset % 8) as u32;
        if bit_in_byte > 0 {
            bits.consume(bit_in_byte);
        }
        bits
    }
}

/// Counter: fresh `Vec::reserve(128*1024)` events from
/// `take_bootstrap_output_from_pool`. Each is an mmap-eligible
/// allocation (256 KiB u16 buffer > glibc mmap threshold). With
/// pool reuse working, expected ≈ num_worker_threads (one allocation
/// per thread, then reused on every subsequent call). If this counter
/// approaches the bootstrap call count, the pool is broken.
#[cfg(parallel_sm)]
pub static BOOTSTRAP_OUTPUT_ALLOCS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Counter: total `take` calls. Pool effectiveness =
/// `1 - (allocs / takes)`. Vendor-equivalent: rpmalloc per-thread arena
/// reuse rate, which approaches 1.0 after warm-up.
#[cfg(parallel_sm)]
pub static BOOTSTRAP_OUTPUT_TAKES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Counter: sum of capacity (in BYTES) of Vecs returned from the
/// pool with cap ≥ 128 KiB. Total bytes the pool "saved" from a fresh
/// allocation. A run-end value of `~BOOTSTRAP_OUTPUT_TAKES *
/// 256 KiB` means every take hit the pool path.
#[cfg(parallel_sm)]
pub static BOOTSTRAP_OUTPUT_REUSED_BYTES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Counter: every call to `return_bootstrap_output_to_pool`. Mirrors
/// TAKES on the success path; if RETURNS << TAKES, bootstraps are
/// erroring out and not returning their buffer (it's still moved via
/// the chunk_fetcher.rs path at line 508, so should always match).
#[cfg(parallel_sm)]
pub static BOOTSTRAP_OUTPUT_RETURNS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Counter: returns where the incoming Vec was DROPPED instead of
/// retained (because the pool's slot already had a larger cap).
/// Non-zero means we threw away a hot allocation — should be 0 with
/// 1-Vec-per-thread pool when caps are stable.
#[cfg(parallel_sm)]
pub static BOOTSTRAP_OUTPUT_DROPPED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Mirror of the `while ( true )` loop in
/// `decodeChunkWithRapidgzip` (GzipChunk.hpp:468-654), restricted to the
/// single-member case (no multi-stream loop) and with the handoff
/// triggered exclusively by `cleanDataCount` (GzipChunk.hpp:520-525).

/// Decode one deflate block into `output` (vendor block loop body).
#[cfg(parallel_sm)]
fn marker_decode_step(
    ctx: &mut MarkerDecodeCtx,
    data: &[u8],
    stop_hint_bits: usize,
    output: &mut impl crate::decompress::parallel::marker_inflate::MarkerSink,
) -> Result<(MarkerStep, bool), ChunkDecodeError> {
    use crate::decompress::parallel::isal_lut_bulk::MarkerRing;
    use crate::decompress::parallel::marker_inflate::MAX_WINDOW_SIZE;
    use crate::decompress::parallel::trace_v2;
    use std::cell::RefCell;

    // The ONE unified decoder's marker arm — the fast LUT loop emitting u16
    // markers, folded into isal_lut_bulk. Replaces the retired
    // marker_inflate::Block bootstrap engine.
    thread_local! {
        static BOOTSTRAP_BLOCK: RefCell<MarkerRing> = RefCell::new(MarkerRing::new());
    }

    BOOTSTRAP_BLOCK.with(|cell_block| {
        let mut block = cell_block.borrow_mut();
        if !ctx.block_primed {
            block.reset();
            ctx.block_primed = true;
        }
        let block = &mut *block;

        // P2: vendor `decodeChunkWithRapidgzip` decodes many blocks per outer
        // iteration; batch here to avoid per-block consumer-loop overhead.
        loop {
            let slice_byte = ctx.current_bit_offset / 8;
            let mut bits = ctx.open_bits(data);

            let next_block_offset = absolute_bit_pos(slice_byte, &bits);

            // UNIFIED DECODE, marker(u16) -> clean(u8) FLIP on ONE cursor (vendor
            // deflate::Block setInitialWindow, deflate.hpp:1282-1292). Once 32 KiB
            // of clean output has accumulated at a block boundary, FLIP: the caller
            // decodes the rest as the fast u8 clean tail into chunk.data (vendor's
            // "~400 MB/s -> ~6 GB/s" path). The marker prefix (u16, <= ~32 KiB)
            // stays in data_with_markers (resolved on the publish chain); the clean
            // tail (the bulk) is u8 and is never resolved. One driver, no separate
            // bootstrap engine and no separate clean-tail entry.
            if ctx.trailing_clean >= MAX_WINDOW_SIZE && output.is_last_n_clean(MAX_WINDOW_SIZE) {
                let end_bit_offset = next_block_offset;
                ctx.current_bit_offset = end_bit_offset;
                return Ok((
                    MarkerStep::FlipToClean {
                        end_bit_offset,
                        window_len: MAX_WINDOW_SIZE,
                    },
                    false,
                ));
            }

        let header_res = {
            let _tv2 = trace_v2::SpanGuard::begin("worker.block_header");
            let t_header = std::time::Instant::now();
            let r = block.read_header(&mut bits);
            BOOTSTRAP_HEADER_US.fetch_add(
                t_header.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            BOOTSTRAP_HEADER_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            r
        };
        if let Err(e) = header_res {
            return Err(ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("deflate header at bit {next_block_offset}: {e:?}"),
            )));
        }

        if next_block_offset >= stop_hint_bits && !block.is_last_block() {
            let end_bit_offset = next_block_offset;
            ctx.current_bit_offset = end_bit_offset;
            return Ok((
                MarkerStep::Finished {
                    end_bit_offset,
                    bfinal_hit: false,
                },
                false,
            ));
        }

        let before_len = output.sink_len();
        let t_body = std::time::Instant::now();
        let _tv2_body = trace_v2::SpanGuard::begin("worker.block_body");
        while !block.eob() {
            let res = block.read(&mut bits, &mut *output);
            if let Err(e) = res {
                let bits_at_fail = absolute_bit_pos(slice_byte, &bits);
                let bytes_wasted = output.sink_len() - before_len;
                let bits_into_body = bits_at_fail.saturating_sub(next_block_offset);
                BODY_FAIL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                BODY_FAIL_BYTES_WASTED
                    .fetch_add(bytes_wasted as u64, std::sync::atomic::Ordering::Relaxed);
                BODY_FAIL_BITS_INTO_BODY
                    .fetch_add(bits_into_body as u64, std::sync::atomic::Ordering::Relaxed);
                use crate::decompress::parallel::isal_lut_bulk::BulkDecodeError;
                match &e {
                    BulkDecodeError::InvalidHuffmanCode => {
                        BODY_FAIL_INVALID_HUFFMAN
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    BulkDecodeError::InvalidLookback => {
                        BODY_FAIL_EXCEEDED_WINDOW
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    BulkDecodeError::BlockTypeReserved => {
                        BODY_FAIL_INVALID_COMPRESSION
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    BulkDecodeError::InvalidCodeLengths => {
                        BODY_FAIL_INVALID_CODE_LENGTHS
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    _ => {
                        BODY_FAIL_OTHER_VARIANT
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                body_fail_log(
                    next_block_offset,
                    bits_into_body,
                    bytes_wasted,
                    &format!("{e:?}"),
                );
                return Err(ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "deflate body at bit {next_block_offset} (failed +{bits_into_body} bits, wasted {bytes_wasted} bytes): {e:?}"
                    ),
                )));
            }
        }
        BOOTSTRAP_BODY_US.fetch_add(
            t_body.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        BOOTSTRAP_BODY_BYTES.fetch_add(
            (output.sink_len() - before_len) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        if output.sink_len() > before_len {
            let block_len = output.sink_len() - before_len;
            let trailing_this_block = output.trailing_clean_since(before_len);
            if trailing_this_block == block_len {
                ctx.trailing_clean =
                    (ctx.trailing_clean + trailing_this_block).min(MAX_WINDOW_SIZE);
            } else {
                ctx.trailing_clean = trailing_this_block.min(MAX_WINDOW_SIZE);
            }
        }

        let flipped_clean = !block.contains_marker_bytes();
        if flipped_clean {
            BOOTSTRAP_POST_FLIP_U16_BYTES.fetch_add(
                (output.sink_len().saturating_sub(before_len)) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        let end_bit_offset = absolute_bit_pos(slice_byte, &bits);
        ctx.current_bit_offset = end_bit_offset;

            if block.is_last_block() {
                return Ok((
                    MarkerStep::Finished {
                        end_bit_offset,
                        bfinal_hit: true,
                    },
                    flipped_clean,
                ));
            }
        }
    })
}

/// Compute the absolute bit position within `data` given that `bits`
/// was constructed from `&data[byte_offset..]`. The Bits buffer
/// pre-loads bytes from its slice, so the actual consumed-from-slice
/// count is `bits.pos * 8 - bits.available()`.
#[cfg(parallel_sm)]
#[inline]
fn absolute_bit_pos(
    byte_offset: usize,
    bits: &crate::decompress::inflate::consume_first_decode::Bits,
) -> usize {
    let consumed_bytes_from_slice = bits.pos;
    let bits_in_buf = bits.available();
    let bits_consumed_from_slice = consumed_bytes_from_slice
        .saturating_mul(8)
        .saturating_sub(bits_in_buf as usize);
    byte_offset * 8 + bits_consumed_from_slice
}

#[cfg(not(parallel_sm))]
pub fn decode_chunk_window_absent(
    _input: &[u8],
    _encoded_offset_bits: usize,
    _stop_hint_bits: usize,
    _configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    Err(ChunkDecodeError::UnsupportedPlatform)
}

#[cfg(not(parallel_sm))]
pub fn decode_chunk(
    _input: &[u8],
    _encoded_offset_bits: usize,
    _stop_hint_bits: usize,
    _initial_window: &[u8],
    _configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    Err(ChunkDecodeError::UnsupportedPlatform)
}

#[cfg(test)]
#[cfg(parallel_sm)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_deflate(payload: &[u8]) -> Vec<u8> {
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(6));
        enc.write_all(payload).unwrap();
        enc.finish().unwrap()
    }

    /// Force multiple deflate blocks (Sync flush every 32 KiB) so
    /// inexact-stop and END_OF_BLOCK probing tests have real boundaries.
    fn make_multi_block_deflate(payload: &[u8]) -> Vec<u8> {
        use flate2::{Compress, Compression, FlushCompress, Status};
        let mut compress = Compress::new(Compression::new(6), false);
        let mut out = Vec::new();
        let mut scratch = vec![0u8; 64 * 1024];
        for piece in payload.chunks(32 * 1024) {
            let mut block_data = piece;
            loop {
                let before_in = compress.total_in();
                let before_out = compress.total_out();
                let status = compress
                    .compress(block_data, &mut scratch, FlushCompress::None)
                    .unwrap();
                let consumed = (compress.total_in() - before_in) as usize;
                let produced = (compress.total_out() - before_out) as usize;
                out.extend_from_slice(&scratch[..produced]);
                block_data = &block_data[consumed..];
                if block_data.is_empty() {
                    break;
                }
                if matches!(status, Status::BufError) && produced == 0 {
                    break;
                }
            }
            loop {
                let before_out = compress.total_out();
                let status = compress
                    .compress(&[], &mut scratch, FlushCompress::Sync)
                    .unwrap();
                let produced = (compress.total_out() - before_out) as usize;
                out.extend_from_slice(&scratch[..produced]);
                if produced == 0 || matches!(status, Status::StreamEnd) {
                    break;
                }
            }
        }
        loop {
            let before_out = compress.total_out();
            let status = compress
                .compress(&[], &mut scratch, FlushCompress::Finish)
                .unwrap();
            let produced = (compress.total_out() - before_out) as usize;
            out.extend_from_slice(&scratch[..produced]);
            if matches!(status, Status::StreamEnd) || produced == 0 {
                break;
            }
        }
        out
    }

    fn flatten(chunk: &ChunkData) -> Vec<u8> {
        let mut out = Vec::with_capacity(chunk.decoded_size());
        for v in chunk.data_with_markers.iter() {
            out.push(v as u8);
        }
        for seg in chunk.data.segments() {
            out.extend_from_slice(seg);
        }
        out
    }

    /// MICROBENCH (advisor-prescribed disambiguation): is the ISA-L multi-symbol
    /// bulk-LUT (`decode_block`) actually faster than the single-symbol resumable
    /// (`ResumableInflate2` via `IsalInflateWrapper`) on THIS code state, over a
    /// REAL silesia clean span (dynamic Huffman)? The locked-Fulcrum lever says
    /// the clean decode tail is the wall (798ms gzippy vs 305ms rapidgzip, T8);
    /// the FlipToClean tail currently runs 100% through resumable. Only integrate
    /// a bulk-LUT clean tail if it wins here. Run:
    ///   cargo test --release --lib --target x86_64-apple-darwin \
    ///     --no-default-features --features pure-rust-inflate \
    ///     -- --ignored --nocapture clean_tail_engine_microbench
    #[cfg(pure_inflate_decode)]
    #[test]
    #[ignore = "manual perf microbench; needs /tmp/ref_silesia.bin"]
    fn clean_tail_engine_microbench() {
        use crate::decompress::inflate::consume_first_decode::Bits;
        use crate::decompress::parallel::inflate_wrapper::{IsalInflateWrapper, StoppingPoints};
        use crate::decompress::parallel::isal_lut_bulk::{decode_block, DecoderScratch};
        use std::time::Instant;

        let raw = match std::fs::read("/tmp/ref_silesia.bin") {
            Ok(v) => v,
            Err(_) => {
                eprintln!("SKIP: /tmp/ref_silesia.bin absent");
                return;
            }
        };
        // ~8 MiB of real silesia → dynamic-Huffman blocks, realistic literal/
        // backref mix (NOT synthetic repeat() which is backref-degenerate).
        // Multi-block (sync flush every 32 KiB) to mirror silesia's block density
        // so the per-block stopping-point overhead is exercised.
        let payload = &raw[..(8 * 1024 * 1024).min(raw.len())];
        // Single-stream for the engine comparison (apples-to-apples with the
        // first run). Multi-block (sync flush every 32 KiB) for the per-block
        // stopping-point comparison — only ResumableInflate2 is exercised there
        // (decode_block doesn't decode flate2 sync-flush empty stored blocks).
        let deflate = make_deflate(payload);
        let deflate_mb = make_multi_block_deflate(payload);
        let n = payload.len();
        let iters = 7;

        // ── bulk-LUT (decode_block) ──
        let mut bulk_out = vec![0u8; n + 64];
        let mut best_bulk = f64::MAX;
        for _ in 0..iters {
            let mut bits = Bits::new(&deflate);
            let mut out_pos = 0usize;
            let mut scratch = DecoderScratch::new();
            let t = Instant::now();
            loop {
                let r = decode_block(&mut bits, &mut bulk_out, &mut out_pos, &[], &mut scratch)
                    .expect("bulk decode");
                if r.is_final_block {
                    break;
                }
            }
            best_bulk = best_bulk.min(t.elapsed().as_secs_f64());
            assert_eq!(&bulk_out[..n], payload, "bulk output mismatch");
        }

        // ── resumable FREE-RUN (no stopping points) on the multi-block stream ──
        let mut res_out = vec![0u8; n + 64];
        let mut best_res = f64::MAX;
        for _ in 0..iters {
            let mut wrapper =
                IsalInflateWrapper::with_until_bits(&deflate_mb, 0, deflate_mb.len() * 8).unwrap();
            wrapper.set_window(&[]).unwrap();
            let mut out_pos = 0usize;
            let t = Instant::now();
            loop {
                let r = wrapper
                    .read_stream(&mut res_out[out_pos..])
                    .expect("resumable decode");
                out_pos += r.bytes_written;
                if r.finished || (r.bytes_written == 0 && out_pos >= n) {
                    break;
                }
                if r.bytes_written == 0 {
                    break;
                }
            }
            best_res = best_res.min(t.elapsed().as_secs_f64());
            assert_eq!(&res_out[..n], payload, "resumable output mismatch");
        }

        // ── resumable WITH production stopping points (returns at EVERY block
        // boundary, as resumable_resync does) — isolates the per-block stop +
        // boundary-bookkeeping overhead from the raw decode rate. ──
        let mut sp_out = vec![0u8; n + 64];
        let mut best_sp = f64::MAX;
        let mut block_stops = 0u64;
        for it in 0..iters {
            let mut wrapper =
                IsalInflateWrapper::with_until_bits(&deflate_mb, 0, deflate_mb.len() * 8).unwrap();
            wrapper.set_window(&[]).unwrap();
            wrapper.set_stopping_points(
                StoppingPoints::END_OF_BLOCK
                    | StoppingPoints::END_OF_BLOCK_HEADER
                    | StoppingPoints::END_OF_STREAM_HEADER
                    | StoppingPoints::END_OF_STREAM,
            );
            let mut out_pos = 0usize;
            let mut stops = 0u64;
            let t = Instant::now();
            loop {
                let r = wrapper
                    .read_stream(&mut sp_out[out_pos..])
                    .expect("sp decode");
                out_pos += r.bytes_written;
                stops += 1;
                if r.finished || (r.bytes_written == 0 && r.stopped_at == StoppingPoints::NONE) {
                    break;
                }
            }
            best_sp = best_sp.min(t.elapsed().as_secs_f64());
            if it == 0 {
                block_stops = stops;
            }
            assert_eq!(&sp_out[..n], payload, "sp output mismatch");
        }

        let mbps = |s: f64| (n as f64 / (1024.0 * 1024.0)) / s;
        eprintln!(
            "CLEAN-TAIL MICROBENCH (n={} MiB, best-of-{}, {} block-stops):\n  bulk-LUT             {:.3}ms  {:.0} MB/s\n  resumable free-run   {:.3}ms  {:.0} MB/s\n  resumable +stoppts   {:.3}ms  {:.0} MB/s  <- production driver shape\n  bulk/resumable = {:.2}x   stoppts-overhead = {:.2}x (free/stop)",
            n / 1024 / 1024,
            iters,
            block_stops,
            best_bulk * 1e3,
            mbps(best_bulk),
            best_res * 1e3,
            mbps(best_res),
            best_sp * 1e3,
            mbps(best_sp),
            best_res / best_bulk,
            best_res / best_sp,
        );
    }

    #[test]
    fn decode_chunk_from_bit_0_matches_payload() {
        let payload = b"abcdefghij".repeat(200_000);
        let deflate = make_deflate(&payload);
        let cfg = ChunkConfiguration {
            split_chunk_size: 512 * 1024,
            max_decoded_chunk_size: 20 * 512 * 1024,
            crc32_enabled: true,
        };
        let stop_hint_bits = deflate.len() * 8;
        let chunk = decode_chunk(&deflate, 0, stop_hint_bits, &[], cfg).unwrap();
        assert_eq!(flatten(&chunk), payload);
    }

    #[test]
    fn decode_chunk_stops_before_eof_when_stop_hint_bits_set() {
        let payload: Vec<u8> = (0u32..500_000)
            .map(|i| (i.wrapping_mul(31) as u8).wrapping_add(7))
            .collect();
        let deflate = make_multi_block_deflate(&payload);
        let cfg = ChunkConfiguration {
            split_chunk_size: 256 * 1024,
            max_decoded_chunk_size: 20 * 256 * 1024,
            crc32_enabled: false,
        };
        let stop_hint_bits = deflate.len() * 8 / 2;
        let chunk = decode_chunk(&deflate, 0, stop_hint_bits, &[], cfg).unwrap();
        assert!(chunk.decoded_size() < payload.len());
        let chunk_end = chunk.encoded_offset_bits + chunk.encoded_size_bits;
        assert!(chunk_end >= stop_hint_bits);
    }

    /// Neurotic profile fixture: gzip(1) -9 on 64 MiB silesia head. Chunk 0
    /// stops at a non-byte-aligned bit; chunk 1 must resume with the published
    /// 32 KiB window. Fails with `InvalidBlock` when handoff is wrong.
    ///
    /// Gated to `isal-compression`: this exercises the one-shot
    /// `inflate_bit::decompress_deflate_from_bit` primitive, which supports
    /// arbitrary-bit-offset resume only on the ISA-L backend. The production
    /// pure-rust parallel-SM path resumes via `ResumableInflate2` (bit-offset
    /// capable; `decompress_deflate_from_bit` has no production caller), and
    /// that path is covered by the silesia differential + `resumable_isal_oracle`.
    /// The `not(isal)` zng fallback's `inflatePrime` convention doesn't match
    /// this primitive's contract — tracked future work for arm64 parallel-SM,
    /// which will itself use `ResumableInflate2`, not zng.
    #[cfg(feature = "isal-compression")]
    #[test]
    fn cross_chunk_resume_silesia_gzip9_chunk0_handoff() {
        use std::io::Read;

        let gz = if std::path::Path::new("/tmp/silesia64.gz").exists() {
            std::fs::read("/tmp/silesia64.gz").expect("read cached gzip")
        } else {
            let path = std::path::Path::new("benchmark_data/silesia-large.bin");
            if !path.exists() {
                return;
            }
            let raw = std::fs::read(path).expect("read silesia");
            let head_len = (64 * 1024 * 1024).min(raw.len());
            let head = &raw[..head_len];
            let mut child = std::process::Command::new("gzip")
                .args(["-9", "-c", "-n"])
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .spawn()
                .expect("spawn gzip");
            // Drain stdout on the main thread while a worker feeds stdin —
            // otherwise gzip's 64 KiB stdout pipe fills mid-write and both
            // sides deadlock (`write_all` 64 MiB stdin vs unread stdout).
            let mut stdin = child.stdin.take().expect("stdin");
            let head_owned = head.to_vec();
            let writer = std::thread::spawn(move || {
                let _ = std::io::Write::write_all(&mut stdin, &head_owned);
                // drop(stdin) here closes the pipe → gzip sees EOF
            });
            let mut gz = Vec::new();
            child
                .stdout
                .as_mut()
                .expect("stdout")
                .read_to_end(&mut gz)
                .expect("read gzip stdout");
            writer.join().expect("stdin writer thread");
            let _ = child.wait();
            gz
        };
        let head_len = {
            let _ = crate::decompress::parallel::gzip_format::read_header(&gz).expect("gzip hdr");
            let footer = crate::decompress::parallel::gzip_format::read_footer(&gz, gz.len() - 8)
                .expect("footer");
            footer.uncompressed_size as usize
        };

        let (_hdr, hdr_len) =
            crate::decompress::parallel::gzip_format::read_header(&gz).expect("gzip hdr");
        let deflate = &gz[hdr_len..gz.len() - 8];

        let spacing_bits = 4 * 1024 * 1024 * 8;
        let cfg = ChunkConfiguration {
            split_chunk_size: 4 * 1024 * 1024,
            max_decoded_chunk_size: 20 * 4 * 1024 * 1024,
            crc32_enabled: false,
        };
        let zero = [0u8; 32768];
        let chunk0 = decode_chunk(deflate, 0, spacing_bits, &zero, cfg).expect("chunk0");
        let resume_at = chunk0.encoded_offset_bits + chunk0.encoded_size_bits;
        assert!(
            resume_at > 0 && !resume_at.is_multiple_of(8),
            "expected non-zero non-byte-aligned handoff, got {resume_at}"
        );
        let tail = chunk0
            .last_32kib_window()
            .unwrap_or_else(|| chunk0.get_last_window(&zero));

        crate::backends::inflate_bit::decompress_deflate_from_bit(
            deflate, resume_at, &tail, head_len,
        )
        .unwrap_or_else(|| {
            panic!("resume at chunk0 end bit {resume_at} must succeed");
        });
        let chunk1 = decode_chunk(deflate, resume_at, resume_at + spacing_bits, &tail, cfg)
            .expect("chunk1 at chunk0 end");
        assert!(!chunk1.is_empty());
    }

    /// Regression for the parallel-SM hang. Given a sub-block input
    /// fragment — the gzip end-of-stream byte-alignment padding (0-7
    /// zero bits before the footer) — `read_stream` can make no
    /// progress, and the streaming inflate loop used to call it forever.
    /// `decode_chunk` must instead return.
    #[test]
    fn decode_chunk_terminates_on_sub_byte_eof_padding() {
        let worker = std::thread::spawn(|| {
            let cfg = ChunkConfiguration {
                split_chunk_size: 512 * 1024,
                max_decoded_chunk_size: 20 * 512 * 1024,
                crc32_enabled: true,
            };
            // One zero byte; decode the sub-byte span [4, 8) — four
            // zero bits, exactly an EOF byte-alignment padding tail.
            // Fresh wrapper, NEW_HDR, stopping points set — same shape
            // as the production chunk in the gdb backtrace.
            let _ = decode_chunk(&[0u8], 4, 8, &[], cfg);
        });

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        while !worker.is_finished() {
            assert!(
                std::time::Instant::now() < deadline,
                "decode_chunk did not return on a sub-byte EOF-padding \
                 fragment: isal_inflate is spinning"
            );
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        worker.join().expect("decode worker panicked");
    }
}
