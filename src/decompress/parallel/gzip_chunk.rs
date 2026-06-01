#![cfg(parallel_sm)]

//! Per-chunk deflate decode for parallel single-member.
//!
//! - [`decode_chunk_isal`] — on-demand decode when the predecessor
//!   window is known (or chunk 0 with a zero dict). Clean bytes only.
//! - [`decode_chunk_marker_bootstrap_then_isal`] — speculative prefetch when
//!   no window yet: marker bootstrap for cross-chunk refs, then ISA-L bulk.
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
#[allow(dead_code)] // used by x86_64+isal-compression decode_chunk_isal path
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

/// ISA-L decode of one chunk when the predecessor window is known.
///
/// Stops at the first block boundary at-or-past `stop_hint_bits` (an
/// inexact hint — the decoder may overshoot), or at end-of-stream.
#[cfg(parallel_sm)]
pub fn decode_chunk_isal(
    input: &[u8],
    encoded_offset_bits: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    decode_chunk_isal_impl(
        input,
        encoded_offset_bits,
        stop_hint_bits,
        initial_window,
        configuration,
    )
}

/// Env-flag dispatch helper. When `GZIPPY_ISAL_PURE_BULK=1` is set,
/// the windowed-decode path routes through `decode_chunk_pure_bulk_impl`
/// (the stateless ISA-L-LUT bulk decoder from
/// `isal_lut_bulk::decode_block`) instead of `IsalInflateWrapper`.
///
/// Default OFF for controlled rollout: ResumableInflate2 (the heavily-
/// optimized incumbent) stays the production path until the bulk
/// decoder proves out on neurotic. Once the win is confirmed, flip the
/// default.
///
/// Read once at process start via OnceLock so the env-var lookup
/// doesn't tax the hot path.
#[cfg(pure_inflate_decode)]
fn use_pure_bulk_path() -> bool {
    use std::sync::OnceLock;
    static USE_BULK: OnceLock<bool> = OnceLock::new();
    *USE_BULK.get_or_init(|| std::env::var_os("GZIPPY_ISAL_PURE_BULK").is_some())
}

/// Option A3+A4 dispatch (`plans/unified-decoder.md` §6, commits
/// `9e14bbe`/`757e7a4`/`a6076e1`). Pre-fills `chunk.data[0..32K]` with
/// the predecessor's window image so every back-reference hits the
/// AVX2 fast path; the consumer skips `chunk.data[..data_prefix_len]`
/// when writing.
///
/// Measured +4.2% T=16 silesia on neurotic, byte-perfect across 11
/// corpora (silesia/logs/software in gzip/pigz/bgzf flavors +
/// urandom-100M) at T∈{1,4,16} — 33/33 pass.
///
/// **Default ON** as of the cross-corpus validation gate. To disable
/// for A/B comparison: `GZIPPY_OPTION_A_PREFILL=0`.
#[cfg(pure_inflate_decode)]
fn use_option_a_prefill_path() -> bool {
    use std::sync::OnceLock;
    static USE_A: OnceLock<bool> = OnceLock::new();
    *USE_A.get_or_init(|| match std::env::var("GZIPPY_OPTION_A_PREFILL") {
        Ok(v) if v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false") => {
            false
        }
        _ => true,
    })
}

#[cfg(parallel_sm)]
fn decode_chunk_isal_impl(
    input: &[u8],
    encoded_offset_bits: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    // Env-flag dispatch: route windowed-decode through the new pure-Rust
    // bulk decoder when GZIPPY_ISAL_PURE_BULK is set. Available only on
    // the pure-rust-inflate build (the isal-compression build uses real
    // ISA-L FFI, which is fast enough we don't override it from here).
    #[cfg(feature = "pure-rust-inflate")]
    if use_pure_bulk_path() {
        return decode_chunk_pure_bulk_impl(
            input,
            encoded_offset_bits,
            stop_hint_bits,
            initial_window,
            configuration,
        );
    }
    // `worker.isal_stream_inflate` span — wraps every invocation of
    // gzippy's ISA-L stream-inflate path. Both call sites land here:
    //   (a) the post-bootstrap call at gzip_chunk.rs:549 (after the
    //       pure-Rust phase-1 produces a clean 32 KiB window)
    //   (b) the non-speculative direct path via `decode_chunk_isal`
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
    let _tv2 = crate::decompress::parallel::trace_v2::SpanGuard::begin_with(
        "worker.isal_stream_inflate",
        &format!(
            r#""start_bit":{encoded_offset_bits},"stop_hint":{stop_hint_bits},"has_window":{}"#,
            !initial_window.is_empty()
        ),
    );
    let t_decode = std::time::Instant::now();
    // `stop_hint_bits` is an inexact stop *hint* (vendor `untilOffset`), not a hard
    // read cap. Capping `refill_buffer` at a partition guess stops mid-block
    // (e.g. silesia gzip-9 at 33554427 vs hint 33554432 → InvalidBlock on resume).
    let read_cap = input.len() * 8;
    let mut wrapper = IsalInflateWrapper::with_until_bits(input, encoded_offset_bits, read_cap)?;
    wrapper.set_window(initial_window)?;
    wrapper.set_stopping_points(
        StoppingPoints::END_OF_BLOCK
            | StoppingPoints::END_OF_BLOCK_HEADER
            | StoppingPoints::END_OF_STREAM_HEADER
            | StoppingPoints::END_OF_STREAM,
    );

    let mut chunk = ChunkData::new(encoded_offset_bits, configuration);

    // Option A3 (env-flag-gated): pre-fill chunk.data[0..32K] with the
    // predecessor's sliding-window image so back-references in the
    // wrapper's decode hit copy_match_fast's SIMD path instead of the
    // copy_match_windowed slow path (the 3.5pp delta vs ISA-L per the
    // perf attribution log). Default OFF — `GZIPPY_OPTION_A_PREFILL=1`
    // to opt in. Trim happens before `chunk.finalize()` at the bottom
    // of this function so the chunk leaves with only decoded bytes in
    // `data` (the A1 debug_asserts enforce this).
    #[cfg(feature = "pure-rust-inflate")]
    let a3_prefill_active = use_option_a_prefill_path() && initial_window.len() == 32768;
    #[cfg(not(feature = "pure-rust-inflate"))]
    let a3_prefill_active = false;
    if a3_prefill_active {
        chunk.prefill_window_prefix(initial_window);
    }

    let mut stopping_point_reached = false;
    let mut last_end_bit = encoded_offset_bits;
    let mut last_eob_pos = encoded_offset_bits;
    let mut last_eob_decoded_bytes: usize = 0;
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
        // Reserve at least ALLOCATION_CHUNK_SIZE of spare capacity.
        // chunk.data starts with `max_decoded_chunk_size` (80 MiB)
        // capacity from `take_u8`, so this is a no-op until we exceed
        // that — at which point amortized growth handles it.
        chunk.data.reserve(ALLOCATION_CHUNK_SIZE);
        let buffer_cap = ALLOCATION_CHUNK_SIZE;
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
            // SAFETY: we reserved `ALLOCATION_CHUNK_SIZE` of spare
            // capacity above; `prev_data_len + n_bytes_read` is within
            // the allocation and the slice covers
            // `[prev_data_len + n_bytes_read, prev_data_len + buffer_cap)`
            // bytes which are uninitialized but writable. ISA-L writes
            // monotonically forward; we only `set_len` AFTER the inner
            // loop has decided how many bytes are valid.
            // Option A3 path: hand the wrapper the FULL chunk.data
            // buffer (extended through the spare capacity) with
            // `out_pos_start = prev_data_len + n_bytes_read`. Back-refs
            // resolve via output[..out_pos] (which includes the 32 KiB
            // window prefix at the front of chunk.data), keeping every
            // dist ≤ 32K within copy_match_fast's fast path.
            //
            // Non-A3 path: existing spare-slice handoff; the wrapper
            // sees out_pos=0 and falls through to state.window for
            // every cross-chunk back-ref.
            //
            // SAFETY (A3): `chunk.data.reserve(ALLOCATION_CHUNK_SIZE)`
            // above guarantees the allocation covers
            // `[0, prev_data_len + buffer_cap)`. The first
            // `prev_data_len` bytes are initialized (prefill + prior
            // outer-iter writes); the tail `[prev_data_len + n_bytes_read,
            // prev_data_len + buffer_cap)` is uninitialized but
            // writable. `read_stream_starting_at` only WRITES at
            // out_pos_start onward and only READS at indices < out_pos
            // (i.e. into the initialized portion). The `set_len` at
            // the bottom of the outer iter records the actual bytes
            // written.
            //
            // SAFETY (non-A3): unchanged from the original.
            #[cfg(feature = "pure-rust-inflate")]
            let r = if a3_prefill_active {
                let total_len = prev_data_len + buffer_cap;
                let output_slice: &mut [u8] =
                    unsafe { std::slice::from_raw_parts_mut(chunk.data.as_mut_ptr(), total_len) };
                wrapper.read_stream_starting_at(output_slice, prev_data_len + n_bytes_read)?
            } else {
                let spare: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(
                        chunk.data.as_mut_ptr().add(prev_data_len + n_bytes_read),
                        buffer_cap - n_bytes_read,
                    )
                };
                wrapper.read_stream(spare)?
            };
            #[cfg(not(feature = "pure-rust-inflate"))]
            let r = {
                let spare: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(
                        chunk.data.as_mut_ptr().add(prev_data_len + n_bytes_read),
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
            // `decode_chunk_isal` such a fragment directly.
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
            // Commit the bytes ISA-L wrote into chunk.data's spare
            // capacity. Bytes beyond `prev_data_len + append_len`
            // (between EOB and the truncate point) are left
            // uninitialized in the Vec — the NEXT outer iteration
            // will overwrite them starting at
            // `chunk.data.len() = prev_data_len + append_len`.
            //
            // We deliberately do NOT call
            // `chunk.note_clean_bytes_written_in_place` here because
            // its `subchunks.last_mut().decoded_size += written`
            // would double-count: the inner loop already updated
            // subchunks via `note_inner_decoded_bytes(last_per_call)`
            // for the FULL n_bytes_read (before truncation).
            //
            // SAFETY: `prev_data_len + append_len ≤ prev_data_len +
            // n_bytes_read ≤ prev_data_len + buffer_cap`, which is
            // within the capacity reserved at the start of the
            // outer iteration; ISA-L wrote initialized bytes through
            // `prev_data_len + n_bytes_read`.
            unsafe { chunk.data.set_len(prev_data_len + append_len) };
            if chunk.configuration.crc32_enabled {
                if let Some(last_crc) = chunk.crc32s.last_mut() {
                    last_crc.update(&chunk.data[prev_data_len..prev_data_len + append_len]);
                }
            }
            chunk.statistics.non_marker_count += append_len as u64;
        }
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

    chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
    // A4: window-image prefix STAYS in chunk.data through finalize
    // and into the consumer. The consumer write site reads
    // &chunk.data[chunk.data_prefix_len..] so the prefix never
    // reaches the user's output. Downstream chunk methods that
    // need the decoded portion (last_32kib_window, get_last_window,
    // populate_subchunk_windows) skip data_prefix_len bytes;
    // absorb_isal_tail (the bootstrap merge path) reads from
    // `tail.data[tail.data_prefix_len..]`. Eliminating the trim
    // memmove was measured at -3.81pp `__memmove_avx_unaligned_erms`
    // CPU share vs A3-with-trim, lifting net throughput to +4.2%
    // vs default at T=16 on neurotic silesia-gzip9.
    let _ = a3_prefill_active; // value consumed at decode-start
                               // When the reader runs past `stop_hint_bits` without an inexact
                               // stop, `tell_compressed()` can land mid-block. Successor chunks must
                               // resume at the last END_OF_BLOCK position (pre-header), not mid-block.
    let final_bit = if stopping_point_reached || reached_stream_end {
        last_end_bit
    } else if last_eob_pos > encoded_offset_bits {
        // Hit the until read cap without an inexact header-stop — finalize at
        // the last pre-header EOB, never at a mid-block bit cursor.
        last_eob_pos
    } else {
        wrapper.tell_compressed()
    };
    chunk.finalize(final_bit);
    Ok(chunk)
}

/// Pure-Rust bulk-decode path. Used when `GZIPPY_ISAL_PURE_BULK=1`.
///
/// Drives [`crate::decompress::parallel::isal_lut_bulk::decode_block`]
/// in a multi-block loop, replacing the IsalInflateWrapper/
/// ResumableInflate2 path. Skips the stopping-point state machine,
/// session accumulator, and yield-on-output-fill machinery — all
/// unnecessary for the windowed bulk phase.
///
/// Correctness gated by the silesia byte-perfect test in
/// `isal_lut_bulk` (162 MB byte-equal vs flate2 GzDecoder) and the
/// existing routing tests (which will additionally run with the env
/// flag set in `tests::routing`).
#[cfg(pure_inflate_decode)]
fn decode_chunk_pure_bulk_impl(
    input: &[u8],
    encoded_offset_bits: usize,
    stop_hint_bits: usize,
    initial_window: &[u8],
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    use crate::decompress::inflate::consume_first_decode::Bits;
    use crate::decompress::parallel::isal_lut_bulk::{
        decode_block, BulkDecodeError, DecoderScratch,
    };

    let _tv2 = crate::decompress::parallel::trace_v2::SpanGuard::begin_with(
        "worker.pure_bulk_inflate",
        &format!(
            r#""start_bit":{encoded_offset_bits},"stop_hint":{stop_hint_bits},"has_window":{}"#,
            !initial_window.is_empty()
        ),
    );
    let t_decode = std::time::Instant::now();

    let mut chunk = ChunkData::new(encoded_offset_bits, configuration);
    let mut bits = Bits::at_bit_offset(input, encoded_offset_bits);
    let mut scratch = DecoderScratch::new();

    // `predecessor_window` is the bytes immediately before this gzip
    // stream's first decoded byte (i.e., the prior chunk's tail for
    // the first member; empty for any subsequent gzip member started
    // mid-chunk after a footer).
    let mut predecessor: &[u8] = initial_window;
    // Offset in chunk.data where the CURRENT gzip stream's bytes start.
    // Back-references must not reach across this boundary (each gzip
    // member has its own sliding window per RFC 1952).
    let mut stream_data_start: usize = 0;

    let mut last_end_bit = encoded_offset_bits;
    let mut last_eob_pos = encoded_offset_bits;
    let mut reached_stream_end = false;
    let mut stopped_at_block_boundary = false;

    PURE_BULK_RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    loop {
        // Reserve spare for one max-size DEFLATE block. Stored blocks
        // are bounded at 65 KiB by RFC 1951 §3.2.4 (LEN is u16);
        // dynamic-Huffman has no hard bound but rarely exceeds 64 KiB
        // in practice. 128 KiB headroom covers both.
        const PER_BLOCK_RESERVE: usize = 128 * 1024;
        let prev_len = chunk.data.len();
        chunk.data.reserve(PER_BLOCK_RESERVE);

        // SAFETY: chunk.data.capacity() bytes are allocated. The
        // initialized prefix is [0, prev_len); the rest is uninit but
        // writable. The bulk decoder reads only [stream_data_start,
        // out_pos) (the current stream's already-decoded bytes), which
        // is a subset of [0, prev_len). It writes only to [out_pos, ..).
        // We update Vec's len via note_clean_bytes_written_in_place
        // after decode_block returns.
        let buf_ptr = chunk.data.as_mut_ptr();
        let buf_cap = chunk.data.capacity();
        let stream_view: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(
                buf_ptr.add(stream_data_start),
                buf_cap - stream_data_start,
            )
        };
        let mut out_pos_in_view = prev_len - stream_data_start;

        let block_start_bit = bits.bit_position();
        let result = decode_block(
            &mut bits,
            stream_view,
            &mut out_pos_in_view,
            predecessor,
            &mut scratch,
        )
        .map_err(|e| {
            ChunkDecodeError::InflateFailed(InflateError::Internal(match e {
                BulkDecodeError::InvalidHuffmanCode => -101,
                BulkDecodeError::InvalidLookback => -102,
                BulkDecodeError::OutputOverflow => -103,
                BulkDecodeError::BlockTypeReserved => -104,
                BulkDecodeError::InvalidCodeLengths => -105,
            }))
        })?;

        let bytes_written = (stream_data_start + out_pos_in_view) - prev_len;
        chunk.note_clean_bytes_written_in_place(prev_len, bytes_written, true);

        let block_end_bit = bits.bit_position();

        // No-progress guard: if a block produces zero output AND the
        // bit cursor didn't advance AND it's not a final block, we're
        // at sub-byte EOF padding (or some other no-progress state).
        // Without this guard the bulk loop spins forever on inputs
        // like the sub-byte EOF padding fragment exercised by
        // `decode_chunk_isal_terminates_on_sub_byte_eof_padding`.
        if bytes_written == 0 && block_end_bit == block_start_bit && !result.is_final_block {
            reached_stream_end = true;
            break;
        }

        last_end_bit = block_end_bit;
        last_eob_pos = block_end_bit;

        if !result.is_final_block {
            chunk.append_block_boundary_at(block_end_bit, chunk.data.len());
        }

        if result.is_final_block {
            // Byte-align then optionally read 8-byte gzip footer (CRC32+ISIZE).
            // The bulk path is invoked both for full gzip streams (footer
            // present) and for tests passing raw deflate (no footer). Match
            // the wrapper-based impl's tolerant behavior: if remaining
            // input is < 8 bytes after byte-align, end the stream cleanly
            // without footer (the wrapper finishes via `r.finished` in
            // that path; gzip_chunk.rs line 333).
            let byte_align = (8 - (block_end_bit % 8)) % 8;
            if byte_align != 0 {
                bits.refill();
                bits.consume(byte_align as u32);
            }
            let footer_byte = bits.bit_position() / 8;
            if footer_byte + 8 > input.len() {
                last_end_bit = bits.bit_position();
                reached_stream_end = true;
                break;
            }
            let crc32 = u32::from_le_bytes([
                input[footer_byte],
                input[footer_byte + 1],
                input[footer_byte + 2],
                input[footer_byte + 3],
            ]);
            let isize_field = u32::from_le_bytes([
                input[footer_byte + 4],
                input[footer_byte + 5],
                input[footer_byte + 6],
                input[footer_byte + 7],
            ]);
            let footer_end_bits = (footer_byte + 8) * 8;
            chunk.append_footer(crc32, isize_field, footer_end_bits);
            last_end_bit = footer_end_bits;

            // Multi-stream gzip: next member's header follows the footer.
            if footer_byte + 8 >= input.len() {
                reached_stream_end = true;
                break;
            }
            let next_start_byte = footer_byte + 8;
            let remaining = &input[next_start_byte..];
            match crate::decompress::parallel::gzip_format::read_header(remaining) {
                Ok((_hdr, hdr_size)) => {
                    let next_stream_byte = next_start_byte + hdr_size;
                    bits = Bits::at_bit_offset(input, next_stream_byte * 8);
                    // New gzip member: fresh sliding window. Reset
                    // both the back-ref reach base and the predecessor.
                    stream_data_start = chunk.data.len();
                    predecessor = &[];
                    last_end_bit = bits.bit_position();
                    last_eob_pos = last_end_bit;
                    chunk.append_block_boundary_at(last_end_bit, chunk.data.len());
                    continue;
                }
                Err(e) => {
                    return Err(ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("multi-stream gzip header at bit {footer_end_bits}: {e}"),
                    )));
                }
            }
        }

        // Stop-hint logic: stop at the next EOB at-or-past stop_hint_bits,
        // BUT not when the next block is fixed Huffman (matches the
        // END_OF_BLOCK_HEADER + not_fixed gate in the wrapper-based impl
        // at gzip_chunk.rs ~354-360). Fixed-Huffman blocks aren't
        // discoverable by the dynamic-Huffman block finder; the
        // consumer would re-decode them anyway, so we extend the
        // current chunk past them.
        if block_end_bit >= stop_hint_bits {
            // Peek next block's BFINAL+BTYPE without consuming.
            let saved_buf = bits.bitbuf;
            let saved_left = bits.bitsleft;
            let saved_pos = bits.pos;
            bits.refill();
            let next_btype = ((bits.bitbuf >> 1) & 0b11) as u32;
            bits.bitbuf = saved_buf;
            bits.bitsleft = saved_left;
            bits.pos = saved_pos;
            if next_btype != 1 {
                stopped_at_block_boundary = true;
                break;
            }
        }
    }

    chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
    let final_bit = if stopped_at_block_boundary || reached_stream_end {
        last_end_bit
    } else if last_eob_pos > encoded_offset_bits {
        last_eob_pos
    } else {
        bits.bit_position()
    };
    chunk.finalize(final_bit);
    Ok(chunk)
}

/// Counter for deletion-trap tests that need to assert the pure-bulk
/// path actually executed (mirrors `UNIFIED_INFLATE_RUNS` /
/// `MARKER_PIPELINE_RUNS` pattern).
#[cfg(pure_inflate_decode)]
pub static PURE_BULK_RUNS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Marker-bootstrap then ISA-L for speculative prefetch (no predecessor window).
/// Requires a real deflate block boundary at `encoded_offset_bits`.
///
/// Returns a `ChunkData` whose `data_with_markers` covers the bootstrap
/// prefix (still containing markers; resolved by `apply_window` in the
/// consumer) and whose `data` covers the ISA-L bulk (already clean).
/// The chunk stops at the next deflate block boundary at-or-past
/// `stop_hint_bits`, or at BFINAL, or when accumulated `decoded_size`
/// crosses `max_decoded_chunk_size`.
#[cfg(parallel_sm)]
pub fn decode_chunk_marker_bootstrap_then_isal(
    input: &[u8],
    encoded_offset_bits: usize,
    stop_hint_bits: usize,
    _initial_window_unused: &[u8],
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    let t_decode = std::time::Instant::now();

    // Phase 1 — bootstrap with deflate_block::Block, the rapidgzip-
    // faithful port of vendor/.../gzip/deflate.hpp's deflate::Block<>.
    // Decodes deflate blocks one-by-one until any of:
    //   (a) cumulative clean (non-marker) bytes reach MAX_WINDOW_SIZE
    //       AND the next block header is past `stop_hint_bits` is not yet
    //       required — phase 2 (ISA-L) can take over;
    //   (b) BFINAL block is decoded (single-member tail);
    //   (c) we're at a non-fixed-Huffman block boundary at-or-past
    //       `stop_hint_bits` (chunk's end condition).
    //
    // Mirrors GzipChunk.hpp::decodeChunkWithRapidgzip's main loop
    // (vendor/.../chunkdecoding/GzipChunk.hpp:468-654), in particular
    // the `cleanDataCount >= deflate::MAX_WINDOW_SIZE` handoff at
    // L520-525.
    let t_bootstrap = std::time::Instant::now();
    // ZERO-COPY MERGE (DecodedData convergence): decode the window-absent
    // marker bootstrap DIRECTLY into the chunk's pooled `data_with_markers`
    // (U16) — no separate std-Vec pool, no `append_markered` copy. The chunk
    // owns the buffer the decoder fills; rpmalloc's per-thread arena keeps the
    // pages warm across chunks (the std-Vec `BOOTSTRAP_OUTPUT_POOL` it replaces
    // was hand-rolling exactly that and is now deleted).
    let mut chunk = ChunkData::new(encoded_offset_bits, configuration);
    // `ChunkData::new` takes a capacity-0 pooled U16 (lazy: the clean fast path
    // never emits markers). The bootstrap WILL fill it, so pre-reserve the same
    // warm capacity the deleted std-Vec `BOOTSTRAP_OUTPUT_POOL` held (128 Ki
    // u16) — without it, decoding into a 0-cap buffer reallocates repeatedly and
    // is ~5% SLOWER than the old warm-pool-then-copy (measured, frozen box).
    chunk.data_with_markers.reserve(128 * 1024);
    let bootstrap = {
        let _coz = crate::coz_probe::scope("marker_bootstrap");
        bootstrap_with_deflate_block(
            input,
            encoded_offset_bits,
            stop_hint_bits,
            &mut chunk.data_with_markers,
        )
    }?;
    let bootstrap_dur_us = t_bootstrap.elapsed().as_micros();
    // CAUSAL PROBE (GZIPPY_SLOW_BOOTSTRAP=N percent): coz is unavailable in this
    // container (perf-event sampling restricted), so measure the bootstrap's
    // causal wall impact directly — spin N% of the bootstrap's own duration to
    // simulate it being (1 + N/100)x slower, then A/B the interleaved wall with
    // the probe off vs on. Byte-identical (pure delay). If 2x-slower-bootstrap
    // (N=100) barely moves the wall, the bootstrap is overlapped/wall-dead;
    // if it adds ~its critical share, it is the lever. Off by default.
    // GZIPPY_SLOW_BOOTSTRAP_US=K injects a FIXED K microseconds per chunk (the
    // ABSOLUTE-ms calibration the standing protocol mandates); it takes
    // precedence over the percent knob GZIPPY_SLOW_BOOTSTRAP=N.
    let slow_us: Option<u64> = std::env::var("GZIPPY_SLOW_BOOTSTRAP_US")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&k| k > 0)
        .or_else(|| {
            std::env::var("GZIPPY_SLOW_BOOTSTRAP")
                .ok()
                .and_then(|s| s.parse::<u128>().ok())
                .map(|pct| (bootstrap_dur_us * pct / 100) as u64)
        });
    if let Some(extra_us) = slow_us {
        let extra = std::time::Duration::from_micros(extra_us);
        // Frequency-neutral control (disproof of the turbo-depression confound):
        // GZIPPY_SLOW_BOOTSTRAP_SLEEP=1 yields the core via sleep instead of a
        // busy-spin. A sleeping worker can't depress all-core turbo, so if the
        // wall delta survives with sleep it is genuine bootstrap criticality,
        // not a spin artifact.
        if std::env::var_os("GZIPPY_SLOW_BOOTSTRAP_SLEEP").is_some() {
            std::thread::sleep(extra);
        } else {
            let until = std::time::Instant::now() + extra;
            while std::time::Instant::now() < until {
                std::hint::spin_loop();
            }
        }
    }
    // Statistic previously bumped inside `append_markered` (now gone): the
    // count of u16 values the bootstrap emitted into data_with_markers.
    chunk.statistics.non_marker_count += chunk.data_with_markers.len() as u64;

    // Whenever the bootstrap reached a block boundary at-or-past
    // stop_hint_bits (rare on chunks > 32 KiB), or BFINAL fired, or no
    // clean window accumulated — we're done. The chunk is marker-only.
    let Some(clean_window) = bootstrap.clean_window else {
        if trace::is_enabled() {
            trace::emit(
                "worker",
                "decode_span",
                &format!(
                    r#""start_bit":{encoded_offset_bits},"bootstrap_us":{bootstrap_dur_us},"inflate_us":0,"phase":"bootstrap_only","markers":{}"#,
                    chunk.data_with_markers.len(),
                ),
            );
        }
        chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
        chunk.finalize(bootstrap.end_bit_offset);
        return Ok(chunk);
    };
    if bootstrap.bfinal_hit || bootstrap.end_bit_offset >= stop_hint_bits {
        if trace::is_enabled() {
            trace::emit(
                "worker",
                "decode_span",
                &format!(
                    r#""start_bit":{encoded_offset_bits},"bootstrap_us":{bootstrap_dur_us},"inflate_us":0,"phase":"bootstrap_terminal","markers":{}"#,
                    chunk.data_with_markers.len(),
                ),
            );
        }
        chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
        chunk.finalize(bootstrap.end_bit_offset);
        return Ok(chunk);
    }

    // Phase 2 — ISA-L bulk decode from the bootstrap's block boundary
    // with the clean 32 KiB tail as dict. Uses the same production
    // `IsalInflateWrapper` path as on-demand decode.
    let bit_offset = bootstrap.end_bit_offset;
    let t_inflate = std::time::Instant::now();
    let tail = {
        // Coz region: the fast ISA-L clean-window bulk decode (the tail).
        let _coz = crate::coz_probe::scope("clean_isal");
        decode_chunk_isal_impl(
            input,
            bit_offset,
            stop_hint_bits,
            &clean_window,
            configuration,
        )
    }?;
    let inflate_dur_us = t_inflate.elapsed().as_micros();
    let tail_bytes = tail.data.len();
    absorb_isal_tail(&mut chunk, tail);
    if trace::is_enabled() {
        trace::emit(
            "worker",
            "decode_span",
            &format!(
                r#""start_bit":{encoded_offset_bits},"bootstrap_us":{bootstrap_dur_us},"inflate_us":{inflate_dur_us},"phase":"bootstrap+inflate","markers":{},"tail_bytes":{tail_bytes}"#,
                chunk.data_with_markers.len(),
            ),
        );
    }
    chunk.statistics.decode_duration_ns += t_decode.elapsed().as_nanos() as u64;
    Ok(chunk)
}

/// Merge an ISA-L tail segment (clean bytes + block boundaries) into a
/// chunk that already holds a marker bootstrap prefix.
#[cfg(parallel_sm)]
fn absorb_isal_tail(dst: &mut ChunkData, mut tail: ChunkData) {
    // `worker.absorb_isal_tail` span — wraps the merge of the
    // ISA-L bulk-decode result into the chunk that holds the
    // bootstrap marker prefix. Copies `tail.data` into `dst.data`,
    // appends the tail's CRC into `dst.crc32s` via the constant-time
    // polynomial-append path (vendor `crc32.hpp:214-258`), and
    // merges subchunk metadata.
    //
    // For a typical 7.5 MB silesia chunk where most of the chunk
    // came through ISA-L (bootstrap was small), tail.data can be
    // several MB — large memcpy + statistics merge.
    let _tv2 = crate::decompress::parallel::trace_v2::SpanGuard::begin_with(
        "worker.absorb_isal_tail",
        &format!(
            r#""tail_bytes":{},"tail_crcs":{},"tail_footers":{}"#,
            tail.data.len(),
            tail.crc32s.len(),
            tail.footers.len()
        ),
    );
    let end_bit = tail.encoded_offset_bits + tail.encoded_size_bits;
    let decoded_base = dst.decoded_size();

    // A4: skip the tail's window-image prefix when absorbing — it's
    // the predecessor's window image installed by A3 prefill, not
    // decoded output. For non-A3 tails, `data_prefix_len == 0` and
    // this is a no-op.
    let tail_payload_offset = tail.data_prefix_len;
    let tail_payload_len = tail.data.len().saturating_sub(tail_payload_offset);
    // Captured BEFORE the buffer swap below: the footer branch at the end
    // of this fn reads `tail.data.is_empty()`, but the swap empties
    // `tail.data`, so the live value would be wrong post-swap.
    let tail_data_was_empty = tail.data.is_empty();
    if tail_payload_len > 0 {
        // Per the post-inline-always bench-sm profile, the prior
        // `dst.append_clean(&tail.data)` re-CRC'd `tail.data` from
        // scratch — 2.45% of total cycles in
        // `crc32fast::specialized::pclmulqdq::calculate` traced
        // back here. `decode_chunk_isal_impl` had ALREADY computed
        // those bytes' CRC into `tail.crc32s.last_mut()` during
        // the in-place commit step. Combine that CRC into
        // `dst.crc32s` via the constant-time polynomial `append`
        // (vendor `crc32.hpp:214-258`) and skip the bytewise
        // pclmulqdq pass.
        //
        // We still need the bytes in `dst.data`, the statistics
        // update, and the subchunk size bump — inline those bits
        // of `append_clean` directly.
        if dst.configuration.crc32_enabled {
            // For parallel-SM (single-member at routing time),
            // tail's crc32s has exactly one entry. If `tail` ever
            // crosses a stream boundary mid-decode (multi-member
            // input that slipped past `is_likely_multi_member`'s
            // 16 MiB scan and was misrouted as single-member),
            // tail.crc32s.len() > 1 and the per-stream entries
            // need to be split by the corresponding footers.
            // Handle that explicitly below; the common case is
            // the fast single-`append` path.
            if tail.crc32s.len() == 1 && tail.footers.is_empty() {
                if let (Some(dst_last), Some(tail_only)) =
                    (dst.crc32s.last_mut(), tail.crc32s.first())
                {
                    dst_last.append(tail_only);
                }
            } else {
                // Multi-stream tail: walk the (crc, footer) pairs
                // in order. Each crc[i] covers bytes between
                // footer[i-1] and footer[i] (with footer[0] being
                // the first footer if i == 0, etc). Approximate by
                // appending crc[0] into dst's current trailing
                // hasher, then for each footer push a fresh
                // hasher and append the next tail crc into it.
                for (i, footer) in tail.footers.iter().enumerate() {
                    if let (Some(dst_last), Some(tail_crc)) =
                        (dst.crc32s.last_mut(), tail.crc32s.get(i))
                    {
                        dst_last.append(tail_crc);
                    }
                    dst.append_footer(
                        footer.crc32,
                        footer.uncompressed_size,
                        footer.end_bit_offset,
                    );
                }
                // The crc entry AFTER the last footer (if any) goes
                // into the freshly-pushed hasher from append_footer.
                if let (Some(dst_last), Some(tail_trailing)) =
                    (dst.crc32s.last_mut(), tail.crc32s.get(tail.footers.len()))
                {
                    dst_last.append(tail_trailing);
                }
            }
        }

        // Bytes + statistics + subchunk size — the non-CRC half of
        // `append_clean`. We extend in place rather than via
        // `append_clean` because that path would re-CRC.
        //
        // `allocator_api2::vec::Vec::extend_from_slice` does NOT
        // specialize for `Copy` source types (unlike `std::vec::Vec`
        // which has `SpecExtend`) — it falls back to the generic
        // `extend → Cloned::next → Option::cloned → Clone::clone`
        // iterator chain. Per the bench-sm profile post-absorb-fix,
        // that chain accounted for 2.47% of total cycles for u8
        // bytes (where `Clone::clone` is a no-op load that should
        // have been folded into a memcpy). Replace with an explicit
        // `reserve` + `copy_nonoverlapping` + `set_len` — the same
        // shape `std::vec::Vec`'s specialization would have used.
        let added = tail_payload_len;
        // FAST PATH (the common bootstrap-then-ISA-L chunk): the marker
        // prefix lives in `data_with_markers`, so `dst.data` is still
        // empty here, and without an A3 window-image prefix the tail's
        // buffer IS the chunk's clean data verbatim. Swap the buffers
        // (O(1) pointer move) instead of a multi-MB `copy_nonoverlapping`
        // — measured at ~212 ms / silesia-large T8 in
        // `worker.absorb_isal_tail` (the dominant clean-tail copy). Use
        // `mem::swap` rather than `take`: the rpmalloc-backed `U8` has no
        // `Default`, and swapping parks `dst`'s empty buffer in `tail` to
        // be dropped. Byte-identical to the copy (same source bytes, same
        // order, prefix-free).
        if tail_payload_offset == 0 && dst.data.is_empty() {
            std::mem::swap(&mut dst.data, &mut tail.data);
        } else {
            let prev_len = dst.data.len();
            dst.data.reserve(added);
            // SAFETY: `reserve(added)` ensures capacity covers
            // `prev_len + added`; `tail.data` and `dst.data` are
            // distinct allocations (separate Vecs); set_len with the
            // post-copy length is legal because all `added` bytes are
            // initialized from `tail.data[tail_payload_offset..]`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    tail.data.as_ptr().add(tail_payload_offset),
                    dst.data.as_mut_ptr().add(prev_len),
                    added,
                );
                dst.data.set_len(prev_len + added);
            }
        }
        dst.statistics.non_marker_count += added as u64;
        if let Some(last) = dst.subchunks.last_mut() {
            last.decoded_size += added;
        }
    }

    // Multi-stream case already emitted footers above (interleaved
    // with the per-stream CRC propagation). For the
    // single-stream-or-empty case, footers still need pushing.
    if tail.crc32s.len() == 1 || tail_data_was_empty {
        for f in &tail.footers {
            dst.append_footer(f.crc32, f.uncompressed_size, f.end_bit_offset);
        }
    }

    for sc in tail.subchunks.iter().skip(1) {
        let _ =
            dst.append_block_boundary_at(sc.encoded_offset_bits, decoded_base + sc.decoded_offset);
    }

    dst.stopped_preemptively = tail.stopped_preemptively;
    dst.finalize(end_bit);
}

/// Result of one bootstrap pass over deflate blocks via
/// [`deflate_block::Block`]. Mirrors the early-exit contract of
/// rapidgzip's `decodeChunkWithRapidgzip` main loop
/// (vendor/.../chunkdecoding/GzipChunk.hpp:468-654).
#[cfg(parallel_sm)]
struct DeflateBootstrap {
    // The u16 marker/literal output is now written DIRECTLY into the caller's
    // `data_with_markers` buffer (passed as `&mut impl MarkerSink`), not
    // returned here — that's the zero-copy merge. This struct carries only the
    // post-decode metadata the caller needs.
    /// Bit position immediately after the last fully-decoded block.
    /// Always at a real deflate block boundary, suitable as a resume
    /// point for ISA-L's `isal_inflate_set_dict` + decode.
    end_bit_offset: usize,
    /// Last 32 KiB of clean (non-marker) output, present only when the
    /// bootstrap saw ≥ MAX_WINDOW_SIZE cumulative clean bytes ending at
    /// a block boundary (rapidgzip's `cleanDataCount >= MAX_WINDOW_SIZE`
    /// at GzipChunk.hpp:521). When `None`, phase 2 cannot run because we
    /// have no clean dict — either BFINAL fired first, `stop_hint_bits` was
    /// reached, or no block produced enough clean data.
    clean_window: Option<Vec<u8>>,
    /// True when the bootstrap exited because it decoded a BFINAL=1
    /// block. Single-stream tail; no further deflate data follows.
    bfinal_hit: bool,
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
#[cfg(parallel_sm)]
fn bootstrap_with_deflate_block(
    data: &[u8],
    start_bit_offset: usize,
    stop_hint_bits: usize,
    output: &mut impl crate::decompress::parallel::deflate_block::MarkerSink,
) -> Result<DeflateBootstrap, ChunkDecodeError> {
    use crate::decompress::parallel::trace_v2;
    let _tv2 = trace_v2::SpanGuard::begin_with(
        "worker.bootstrap",
        &format!(r#""start_bit":{start_bit_offset},"stop_hint":{stop_hint_bits}"#),
    );
    // Inner function so we can capture the Result and emit an outcome
    // instant event before the SpanGuard drops. Per-attempt outcome
    // attribution lets `scripts/timeline_analyze.py` correlate the
    // 29 retries seen on silesia-large with their specific failure
    // variant (header parse vs body invalid-huffman vs
    // exceeded-window-range vs ...).
    let result = bootstrap_with_deflate_block_inner(data, start_bit_offset, stop_hint_bits, output);
    let markers_len = output.sink_len();
    match &result {
        Ok(b) => {
            // handoff_reason: disambiguates the three Ok exit paths in
            // `bootstrap_with_deflate_block_inner`:
            //   - clean_window_armed: trailing_clean reached
            //     MAX_WINDOW_SIZE at a block boundary; ISA-L will
            //     take over (the "good" path).
            //   - bfinal_hit: BFINAL block decoded; chunk is complete
            //     in pure-Rust phase-1 alone.
            //   - stop_hint_reached: the last block header was at-or-past
            //     stop_hint_bits AND non-FixedHuffman AND not BFINAL;
            //     decode stopped on the upcoming block (caller's
            //     successor will re-decode that block).
            // For the 6 heavy-tail chunks per silesia-large run that
            // burn 200+ ms in bootstrap, the suspicion is they all hit
            // either `bfinal_hit` or `stop_hint_reached` after running
            // through the entire chunk without arming a clean window —
            // this arg confirms or falsifies that prior.
            let handoff_reason = if b.clean_window.is_some() {
                "clean_window_armed"
            } else if b.bfinal_hit {
                "bfinal_hit"
            } else {
                "stop_hint_reached"
            };
            // bytes_decoded: the bootstrap-decoded byte count. Each u16
            // in `b.markers` represents ONE decoded byte (high bits
            // distinguish markers from clean bytes). Join with the
            // worker.bootstrap span's duration to compute per-call MB/s
            // for the Step-2 analyzer.
            trace_v2::emit_instant(
                "worker.bootstrap.outcome",
                &format!(
                    r#""result":"ok","markers_len":{},"end_bit":{},"clean_window":{},"bfinal":{},"handoff_reason":"{}","bytes_decoded":{}"#,
                    markers_len,
                    b.end_bit_offset,
                    b.clean_window.is_some(),
                    b.bfinal_hit,
                    handoff_reason,
                    markers_len,
                ),
                "t",
            );
        }
        Err(e) => {
            let kind = match e {
                ChunkDecodeError::BootstrapFailed(io_err) => {
                    let msg = io_err.to_string();
                    // Extract the first identifying token of the inner
                    // BlockError (e.g. "InvalidHuffmanCode"). The error
                    // strings are formatted in bootstrap as
                    // `"deflate body at bit ... : InvalidHuffmanCode"`
                    // or `"deflate header at bit ...: InvalidHuffmanCode"`.
                    if msg.contains("InvalidHuffmanCode") {
                        "InvalidHuffmanCode"
                    } else if msg.contains("ExceededWindowRange") {
                        "ExceededWindowRange"
                    } else if msg.contains("InvalidCompression") {
                        "InvalidCompression"
                    } else if msg.contains("InvalidCodeLengths") {
                        "InvalidCodeLengths"
                    } else if msg.contains("EndOfFile") {
                        "EndOfFile"
                    } else if msg.contains("past end of data") {
                        "past_eof"
                    } else if msg.contains("header") {
                        "header_other"
                    } else if msg.contains("body") {
                        "body_other"
                    } else {
                        "other"
                    }
                }
                _ => "non_bootstrap_err",
            };
            trace_v2::emit_instant(
                "worker.bootstrap.outcome",
                &format!(r#""result":"err","kind":"{kind}""#),
                "t",
            );
        }
    }
    result
}

#[cfg(parallel_sm)]
fn bootstrap_with_deflate_block_inner(
    data: &[u8],
    start_bit_offset: usize,
    stop_hint_bits: usize,
    output: &mut impl crate::decompress::parallel::deflate_block::MarkerSink,
) -> Result<DeflateBootstrap, ChunkDecodeError> {
    use crate::decompress::inflate::consume_first_decode::Bits;
    use crate::decompress::parallel::deflate_block::{Block, MAX_WINDOW_SIZE};
    use crate::decompress::parallel::replace_markers::MARKER_BASE;
    use crate::decompress::parallel::trace_v2;
    use std::cell::RefCell;

    // Per-thread Block recycling. Block::new() allocates a 128 KiB
    // ring + initializes the marker zone (64 KiB writes). Doing that
    // per chunk was a measured ~4 pp of CPU in `clear_page_erms` on
    // silesia. Per-thread reuse amortizes the cost to once-per-worker.
    // `Block::reset` re-primes marker zone but skips the alloc.
    thread_local! {
        static BOOTSTRAP_BLOCK: RefCell<Block> = RefCell::new(Block::new());
    }

    let byte_offset = start_bit_offset / 8;
    let bit_in_byte = (start_bit_offset % 8) as u32;
    if byte_offset >= data.len() {
        return Err(ChunkDecodeError::BootstrapFailed(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "start_bit_offset past end of data",
        )));
    }
    let mut bits = Bits::new(&data[byte_offset..]);
    if bit_in_byte > 0 {
        bits.consume(bit_in_byte);
    }

    // Zero-copy merge: the caller passes the chunk's own `data_with_markers`
    // (U16) as `output`; the decoder fills it directly. No separate pooled
    // std-Vec, no `append_markered` copy. (rpmalloc's per-thread arena keeps
    // the chunk buffer's pages warm across chunks.)
    BOOTSTRAP_BLOCK.with(|cell_block| {
        let mut block = cell_block.borrow_mut();
        block.reset(None, None);
        let block = &mut *block;

        // Tracks trailing CLEAN-byte run length. When it reaches
        // MAX_WINDOW_SIZE AT a block boundary, the next 32 KiB of `output`
        // forms a clean dict for ISA-L. Cumulative count saturates at
        // MAX_WINDOW_SIZE (we only need the latest 32 KiB).
        let mut trailing_clean: usize = 0;
        // True once `trailing_clean >= MAX_WINDOW_SIZE` AND the most recent
        // block has just finished (so the next bit position is a real block
        // header start that ISA-L can resume from).
        let mut clean_handoff_armed: bool;
        let mut bfinal_hit = false;
        // `end_bit_offset` always points just past the last completed block.
        // Always assigned before the post-loop read (every `break` is
        // preceded by an assignment), so no initializer is needed.
        let mut end_bit_offset;

        loop {
            // Snapshot the bit position BEFORE reading this block's header.
            // This is what the next chunk's worker resumes from when this
            // chunk hands off to ISA-L (or when this block becomes BFINAL).
            let next_block_offset = absolute_bit_pos(byte_offset, &bits);

            // Handoff check (rapidgzip GzipChunk.hpp:520-525): if we have a
            // clean 32 KiB tail AND we're at a real block boundary, stop
            // bootstrap and let phase 2 run. We do this BEFORE reading the
            // next header so `end_bit_offset` is at a clean boundary.
            clean_handoff_armed = trailing_clean >= MAX_WINDOW_SIZE;
            if clean_handoff_armed {
                end_bit_offset = next_block_offset;
                break;
            }

            // Read this block's header. `treat_last_block_as_error = false`
            // (we WANT to see BFINAL so we can stop).
            let header_res = {
                let _tv2 = trace_v2::SpanGuard::begin("worker.block_header");
                let t_header = std::time::Instant::now();
                let r = block.read_header(&mut bits, false);
                BOOTSTRAP_HEADER_US.fetch_add(
                    t_header.elapsed().as_micros() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                BOOTSTRAP_HEADER_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                r
            };
            match header_res {
                Ok(()) => {}
                Err(e) => {
                    return Err(ChunkDecodeError::BootstrapFailed(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("deflate header at bit {next_block_offset}: {e:?}"),
                    )));
                }
            }

            // Preemptive stop condition (rapidgzip GzipChunk.hpp:550-555):
            // if this block's header sits at-or-past stop_hint_bits AND it is
            // non-fixed AND not BFINAL, stop here so the chunk ends on a
            // boundary the successor's BlockFinder can re-find.
            if next_block_offset >= stop_hint_bits && !block.is_last_block() {
                // We've already advanced `bits` past this header. Rewind
                // logically by reporting `end_bit_offset = next_block_offset`
                // — the successor will re-parse the same header.
                end_bit_offset = next_block_offset;
                break;
            }

            // Decode the block's body, one read() call at a time, until
            // EOB. `n_max_to_decode = usize::MAX` matches rapidgzip's
            // `std::numeric_limits<size_t>::max()` at GzipChunk.hpp:568.
            let before_len = output.sink_len();
            let t_body = std::time::Instant::now();
            let _tv2_body = trace_v2::SpanGuard::begin("worker.block_body");
            while !block.eob() {
                let res = block.read(&mut bits, &mut *output, usize::MAX);
                if let Err(e) = res {
                    // Body-failure diagnostics for the speculation-accuracy
                    // investigation (per advisor recommendation). Captures
                    // (a) candidate start_bit, (b) bit position when body
                    // failed (= bits consumed past block start), (c) bytes
                    // emitted before failure (= waste), (d) error variant.
                    let bits_at_fail = absolute_bit_pos(byte_offset, &bits);
                    let bytes_wasted = output.sink_len() - before_len;
                    let bits_into_body = bits_at_fail.saturating_sub(next_block_offset);
                    BODY_FAIL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    BODY_FAIL_BYTES_WASTED
                        .fetch_add(bytes_wasted as u64, std::sync::atomic::Ordering::Relaxed);
                    BODY_FAIL_BITS_INTO_BODY
                        .fetch_add(bits_into_body as u64, std::sync::atomic::Ordering::Relaxed);
                    // Per-error-variant counter so we can see which
                    // block_error dominates.
                    use crate::decompress::parallel::deflate_block::BlockError;
                    match &e {
                        BlockError::InvalidHuffmanCode => {
                            BODY_FAIL_INVALID_HUFFMAN
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        BlockError::ExceededWindowRange => {
                            BODY_FAIL_EXCEEDED_WINDOW
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        BlockError::InvalidCompression => {
                            BODY_FAIL_INVALID_COMPRESSION
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        BlockError::InvalidCodeLengths => {
                            BODY_FAIL_INVALID_CODE_LENGTHS
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        _ => {
                            BODY_FAIL_OTHER_VARIANT
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                    // Detailed per-failure dump (one JSON line) if the
                    // GZIPPY_BODY_FAIL_LOG env var points at a writable
                    // path. Mirror of trace.rs pattern.
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

            // Block fully decoded. Update trailing_clean from the bytes
            // just produced. Markers (≥ MARKER_BASE) reset the run; clean
            // bytes extend it (saturating at MAX_WINDOW_SIZE).
            let block_slice = &output.as_slice()[before_len..];
            if !block_slice.is_empty() {
                let trailing_this_block = block_slice
                    .iter()
                    .rev()
                    .take_while(|&&v| v < MARKER_BASE)
                    .count();
                if trailing_this_block == block_slice.len() {
                    // Entire block was clean — extend the prior run.
                    trailing_clean = (trailing_clean + trailing_this_block).min(MAX_WINDOW_SIZE);
                } else {
                    // A marker appeared mid-block; the trailing clean run
                    // restarts from after that last marker.
                    trailing_clean = trailing_this_block.min(MAX_WINDOW_SIZE);
                }
            }

            // Commit-1 instrumentation (no behavior change): once the block's
            // marker mode has flipped off, every byte this block produced is a
            // clean u16-ring write that Design B1 would route to a u8 buffer.
            // The flip is permanent within a chunk, so counting whole post-flip
            // blocks over-counts only the pre-flip prefix of the single
            // flip-block (≤32 KiB, negligible vs multi-MB blocks).
            if !block.contains_marker_bytes() {
                BOOTSTRAP_POST_FLIP_U16_BYTES.fetch_add(
                    (output.sink_len() - before_len) as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }

            end_bit_offset = absolute_bit_pos(byte_offset, &bits);

            if block.is_last_block() {
                bfinal_hit = true;
                break;
            }
        }

        // Build the clean dict if we have one.
        let clean_window = if clean_handoff_armed && output.sink_len() >= MAX_WINDOW_SIZE {
            let start = output.sink_len() - MAX_WINDOW_SIZE;
            // Invariant: the trailing MAX_WINDOW_SIZE values of `output` are
            // < MARKER_BASE (clean). Assert this — corruption here would
            // seed ISA-L with garbage and produce a wrong-CRC chunk.
            let window: Vec<u8> = output.as_slice()[start..]
                .iter()
                .map(|&v| {
                    assert!(
                        v < MARKER_BASE,
                        "bootstrap clean window contained marker at offset {}; \
                     trailing_clean tracker broken",
                        v.saturating_sub(MARKER_BASE)
                    );
                    v as u8
                })
                .collect();
            Some(window)
        } else {
            None
        };

        Ok(DeflateBootstrap {
            end_bit_offset,
            clean_window,
            bfinal_hit,
        })
    }) // BOOTSTRAP_BLOCK.with closure
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
pub fn decode_chunk_marker_bootstrap_then_isal(
    _input: &[u8],
    _encoded_offset_bits: usize,
    _stop_hint_bits: usize,
    _initial_window: &[u8],
    _configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    Err(ChunkDecodeError::UnsupportedPlatform)
}

#[cfg(not(parallel_sm))]
pub fn decode_chunk_isal(
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
        for v in &chunk.data_with_markers {
            out.push(*v as u8);
        }
        out.extend_from_slice(&chunk.data);
        out
    }

    /// KNOWN-LIMITATION: the bulk impl errors with OutputOverflow when
    /// a single DEFLATE block's output exceeds `chunk.data`'s capacity.
    /// zlib L1 on btype01-heavy data emits ONE huge fixed-Huffman block
    /// per stream (~12 MiB output from 5.7 MiB compressed), which won't
    /// fit in chunk.data's typical 10 MiB capacity. The bulk decoder is
    /// stateless and cannot yield mid-block; ResumableInflate2 yields
    /// naturally. Per advisor 2026-05-27: the right fix is to swap
    /// ResumableInflate2's INNER Huffman hot loop to use
    /// IsalLitLenCodePure, NOT to replace ResumableInflate2 entirely.
    /// Until that lands, the bulk impl is gated behind
    /// `GZIPPY_ISAL_PURE_BULK=1` and not the production default.
    ///
    /// This test documents the limitation; it's `#[ignore]`d because it
    /// will fail on this fixture by design until the inner-primitive
    /// migration completes.
    #[test]
    #[ignore = "known limitation: bulk decoder requires output buffer >= single-block size; awaits inner-primitive migration into ResumableInflate2"]
    #[cfg(feature = "pure-rust-inflate")]
    fn decode_chunk_pure_bulk_btype01_heavy_l1_full() {
        let phrases: &[&[u8]] = &[b"abc", b"foo bar ", b"the quick brown ", b"hello ", b"xyz "];
        let target = 12 * 1024 * 1024;
        let mut payload = Vec::with_capacity(target);
        let mut rng: u64 = 0xb0bd1ec0de;
        while payload.len() < target {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32) % 100 < 70 {
                payload.push((rng >> 16) as u8);
            } else {
                let phrase = phrases[(rng as usize) % phrases.len()];
                let take = phrase.len().min(target - payload.len());
                payload.extend_from_slice(&phrase[..take]);
            }
        }
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(1));
        enc.write_all(&payload).unwrap();
        let gz = enc.finish().unwrap();
        let (_hdr, hdr_size) =
            crate::decompress::parallel::gzip_format::read_header(&gz).expect("gz header");
        let deflate_stop_bits = (gz.len() - 8) * 8 - hdr_size * 8;
        let deflate = &gz[hdr_size..gz.len() - 8];

        let cfg = ChunkConfiguration {
            split_chunk_size: 512 * 1024,
            max_decoded_chunk_size: 20 * 512 * 1024,
            crc32_enabled: false,
        };
        // BISECT: directly invoke decode_block with a Vec-backed output
        // that mirrors chunk.data's capacity (10 MiB), bypassing the
        // chunk-level orchestration. This isolates "is the bug in the
        // chunk-level wrapper, or in decode_block on a 10-MiB output?"
        {
            use crate::decompress::inflate::consume_first_decode::Bits;
            use crate::decompress::parallel::isal_lut_bulk::{decode_block, DecoderScratch};
            let mut output = vec![0u8; cfg.max_decoded_chunk_size];
            let mut bits = Bits::new(deflate);
            bits.refill();
            let mut scratch = DecoderScratch::new();
            let predecessor = [0u8; 32768];
            let mut out_pos = 0usize;
            let mut block_count = 0;
            loop {
                let r = decode_block(
                    &mut bits,
                    &mut output,
                    &mut out_pos,
                    &predecessor[..],
                    &mut scratch,
                );
                match r {
                    Ok(res) => {
                        block_count += 1;
                        if res.is_final_block {
                            eprintln!(
                                "[direct-vec] decoded {block_count} blocks, total {out_pos} bytes (final)"
                            );
                            break;
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "[direct-vec] decode_block error={e:?} after {block_count} blocks, out_pos={out_pos}"
                        );
                        break;
                    }
                }
            }
        }

        // Call the bulk impl directly to bypass the OnceLock env-var check.
        let chunk_result =
            decode_chunk_pure_bulk_impl(deflate, 0, deflate_stop_bits, &[0u8; 32768][..], cfg);
        match chunk_result {
            Ok(chunk) => {
                let flat = flatten(&chunk);
                let same = flat
                    .iter()
                    .zip(payload.iter())
                    .take_while(|(a, b)| a == b)
                    .count();
                eprintln!(
                    "[chunk-repro] decoded {} bytes; matches flate2 for first {} bytes",
                    flat.len(),
                    same
                );
                assert_eq!(flat, payload[..flat.len()], "first {} bytes diverge", same);
            }
            Err(e) => {
                eprintln!("[chunk-repro] ERR: {e:?}");
                panic!("decode failed: {e:?}");
            }
        }
    }

    #[test]
    fn decode_chunk_isal_from_bit_0_matches_payload() {
        let payload = b"abcdefghij".repeat(200_000);
        let deflate = make_deflate(&payload);
        let cfg = ChunkConfiguration {
            split_chunk_size: 512 * 1024,
            max_decoded_chunk_size: 20 * 512 * 1024,
            crc32_enabled: true,
        };
        let stop_hint_bits = deflate.len() * 8;
        let chunk = decode_chunk_isal(&deflate, 0, stop_hint_bits, &[], cfg).unwrap();
        assert_eq!(flatten(&chunk), payload);
    }

    #[test]
    fn decode_chunk_isal_stops_before_eof_when_stop_hint_bits_set() {
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
        let chunk = decode_chunk_isal(&deflate, 0, stop_hint_bits, &[], cfg).unwrap();
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
        let chunk0 = decode_chunk_isal(deflate, 0, spacing_bits, &zero, cfg).expect("chunk0");
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
        let chunk1 = decode_chunk_isal(deflate, resume_at, resume_at + spacing_bits, &tail, cfg)
            .expect("chunk1 at chunk0 end");
        assert!(!chunk1.is_empty());
    }

    /// Regression for the parallel-SM hang. Given a sub-block input
    /// fragment — the gzip end-of-stream byte-alignment padding (0-7
    /// zero bits before the footer) — `read_stream` can make no
    /// progress, and `decode_chunk_isal_impl`'s decode loop used to
    /// call it forever. `decode_chunk_isal` must instead return.
    ///
    /// The decode runs in a child thread joined against a deadline, so
    /// a regression fails this test instead of hanging the whole suite.
    #[test]
    fn decode_chunk_isal_terminates_on_sub_byte_eof_padding() {
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
            let _ = decode_chunk_isal(&[0u8], 4, 8, &[], cfg);
        });

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        while !worker.is_finished() {
            assert!(
                std::time::Instant::now() < deadline,
                "decode_chunk_isal did not return on a sub-byte EOF-padding \
                 fragment: isal_inflate is spinning"
            );
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        worker.join().expect("decode worker panicked");
    }
}
