//! Port of `rapidgzip::GzipChunkFetcher` + `rapidgzip::BlockFetcher`
//! (vendor/rapidgzip/.../GzipChunkFetcher.hpp + BlockFetcher.hpp).
//!
//! Architecture
//! ------------
//! A persistent worker pool (one thread per `parallelization`) pulls
//! decode jobs from a shared MPMC-style work queue. Two job sources:
//!
//!   1. **Speculative prefetch** — the dispatcher submits one decode
//!      per partition seed up to `parallelization * 2` outstanding so
//!      workers are always busy ahead of the consumer.
//!
//!   2. **Authoritative re-dispatch** — when the consumer detects that
//!      a speculative chunk's start doesn't match the predecessor's
//!      actual end (`matches_encoded_offset`), it submits a fresh
//!      decode at the correct start to the same pool. The previously-
//!      submitted speculative result is discarded.
//!
//! Workers prefer the FAST PATH: before decoding, look up the
//! predecessor's window in the shared [`WindowMap`] (with a short
//! wait). If available, call
//! [`crate::decompress::parallel::gzip_chunk::decode_chunk_with_window`]
//! which seeds the ISA-L dict directly and skips the marker-decoder
//! bootstrap. If the wait times out (chunk 0, predecessor still busy)
//! the worker falls back to the SLOW PATH
//! ([`crate::decompress::parallel::gzip_chunk::finish_decode_chunk_with_inexact_offset`]).
//!
//! Workers insert their tail window into the shared `WindowMap` as
//! soon as it's clean (always true on the fast path; for slow-path
//! chunks the consumer handles the insert after `apply_window`
//! resolves markers).
//!
//! The consumer thread drives prefetch, drains chunks in stream order
//! (waiting on per-partition result channels), and writes bytes +
//! combines CRCs. Re-dispatches go to the pool, not the consumer
//! thread — the consumer can overlap `apply_window` work on chunk N
//! with a re-decode of chunk N+1 in the pool.

#![allow(dead_code)]

use crate::decompress::parallel::chunk_data::ChunkConfiguration;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::chunk_data::ChunkData;
use crate::decompress::parallel::gzip_chunk::ChunkDecodeError;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use std::sync::Arc;

#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::apply_window::apply_window;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::block_finder::BlockFinder;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::gzip_chunk::{
    decode_chunk_with_window, finish_decode_chunk_with_inexact_offset,
};
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::trace;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use crate::decompress::parallel::window_map::WindowMap;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use std::sync::mpsc;
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
use std::time::Duration;

#[derive(Debug)]
pub enum FetchError {
    Decode(ChunkDecodeError),
    UnsupportedPlatform,
}

impl From<ChunkDecodeError> for FetchError {
    fn from(e: ChunkDecodeError) -> Self {
        FetchError::Decode(e)
    }
}

/// How far past a partition seed we'll search for a deflate block
/// boundary candidate. Matches rapidgzip's tryToDecode scan budget
/// (vendor/rapidgzip/.../chunkdecoding/GzipChunk.hpp ~ L800).
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
const BOUNDARY_SEARCH_RADIUS_BYTES: usize = 512 * 1024;

/// How long a worker will block waiting for the predecessor's window
/// before falling back to the slow path. Short enough that chunk-0
/// (which never has a predecessor) doesn't waste time; long enough
/// that the predecessor's decode plus apply_window can land for the
/// common case.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
const WINDOW_WAIT_TIMEOUT: Duration = Duration::from_millis(50);

/// One unit of decode work submitted to the pool. The reply channel
/// carries the worker's result. Workers exit when the work-queue
/// sender is dropped (the scope's main thread, which holds the
/// sender, exits at end of `drive`).
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
struct DecodeJob {
    partition_idx: usize,
    start_bit: usize,
    until_bit: usize,
    /// True for the consumer's authoritative re-dispatch: we KNOW the
    /// predecessor's tail is available at `start_bit`, so workers must
    /// take the fast path (no fallback). For false (speculative
    /// prefetch) the worker waits a short timeout and falls back to
    /// the slow path on miss.
    authoritative: bool,
    reply: mpsc::Sender<Result<ChunkData, ChunkDecodeError>>,
}

/// Public driver function. Replaces the old `GzipChunkFetcher` struct
/// API with a single entry point: decompress the entire raw deflate
/// stream, writing output bytes to `writer`, and returning the per-
/// chunk-combined CRC32 + total decoded size.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
pub fn drive<W: std::io::Write>(
    input: &[u8],
    writer: &mut W,
    parallelization: usize,
    configuration: ChunkConfiguration,
) -> Result<(u32, usize), FetchError> {
    let chunk_size_bits = configuration.split_chunk_size * 8;
    let total_bits = input.len() * 8;
    let n_partitions = total_bits.div_ceil(chunk_size_bits).max(1);
    let partition_offsets: Vec<usize> = (0..n_partitions).map(|i| i * chunk_size_bits).collect();

    let window_map = WindowMap::new();
    // Chunk 0's input window is empty by definition (start of stream).
    window_map.insert(0, Arc::new([0u8; 32768]));

    let (job_tx, job_rx) = mpsc::channel::<DecodeJob>();
    let job_rx = Arc::new(std::sync::Mutex::new(job_rx));

    let mut total_crc = crc32fast::Hasher::new();
    let mut total_size: usize = 0;

    let pool_size = parallelization.max(1);

    // The pre-scan was deleted. Workers receive jobs at partition
    // offsets directly; they try to decode AT the partition offset
    // first (cheap if it happens to be a real block boundary), then
    // iterate BlockFinder candidates INSIDE the worker if not. This
    // matches rapidgzip's decodeChunk + tryToDecode pattern at
    // vendor/rapidgzip/.../chunkdecoding/GzipChunk.hpp:712-741.
    let _ = total_bits;
    let _ = partition_offsets;

    let result = std::thread::scope(|s| -> Result<(), FetchError> {
        // Spawn worker pool.
        for _ in 0..pool_size {
            let job_rx = Arc::clone(&job_rx);
            let window_map = window_map.clone();
            let configuration = configuration;
            s.spawn(move || worker_loop(input, job_rx, window_map, configuration));
        }

        consumer_loop(
            input,
            writer,
            n_partitions,
            &partition_offsets,
            total_bits,
            &job_tx,
            &window_map,
            configuration,
            pool_size,
            &mut total_crc,
            &mut total_size,
        )?;

        // Dropping job_tx signals workers to exit.
        drop(job_tx);
        Ok(())
    });
    result?;

    let crc = total_crc.finalize();
    Ok((crc, total_size))
}

/// Try to decode at `start_bit` directly; if that fails, iterate
/// BlockFinder candidates in [start_bit, start_bit + 512 KiB] until
/// one succeeds. Returns a chunk whose `encoded_offset_bits` is the
/// requested `start_bit` (matches the cache key) but whose
/// `max_encoded_offset_bits` is the actual candidate used.
/// Mirror of rapidgzip's tryToDecode + candidate iteration at
/// vendor/rapidgzip/.../GzipChunk.hpp:712-841.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
fn decode_or_iterate(
    input: &[u8],
    start_bit: usize,
    until_bit: usize,
    configuration: ChunkConfiguration,
) -> Result<ChunkData, ChunkDecodeError> {
    // Direct port of rapidgzip's tryToDecode + alternating BlockFinder
    // search (GzipChunk.hpp:712-846). We DON'T try direct decode at
    // start_bit blindly — our finish_decode_chunk_with_inexact_offset's
    // marker bootstrap is too lenient with malformed headers (false
    // "success" on random bits produces a tiny chunk that misses
    // expected_start downstream). Instead, run BlockFinder to get
    // validated candidates (find_blocks does Kraft-precode +
    // Huffman-code validation matching rapidgzip's
    // seekToNonFinalDynamicDeflateBlock), iterate them — including a
    // candidate AT start_bit if find_blocks emits one — and accept the
    // first that try-decodes cleanly.
    let finder = BlockFinder::new(input);
    let scan_end = (start_bit + 512 * 1024 * 8).min(input.len() * 8);
    let candidates = finder.find_blocks(start_bit, scan_end);
    let mut last_err: Option<ChunkDecodeError> = None;
    for candidate in candidates {
        if candidate.bit_offset < start_bit {
            continue;
        }
        match finish_decode_chunk_with_inexact_offset(
            input,
            candidate.bit_offset,
            until_bit,
            &[],
            configuration,
        ) {
            Ok(mut c) => {
                // Adopt rapidgzip's semantic: encoded_offset = requested,
                // max_encoded_offset = actual candidate used.
                let actual = c.encoded_offset_bits;
                c.encoded_offset_bits = start_bit;
                c.max_encoded_offset_bits = actual;
                // encoded_size_bits is relative to encoded_offset_bits.
                // Decoder set it relative to `actual`; adjust so the
                // chunk reports its total span from the requested start.
                c.encoded_size_bits += actual - start_bit;
                // Subchunks were recorded with absolute encoded_offset_bits
                // — those are still valid (consumer's decoded_offset_for
                // looks up by absolute bit position).
                return Ok(c);
            }
            Err(e) => {
                last_err = Some(e);
                continue;
            }
        }
    }
    Err(last_err.unwrap_or(ChunkDecodeError::ExactStopMissed {
        requested: start_bit,
        actual: scan_end,
    }))
}

#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
fn worker_loop(
    input: &[u8],
    job_rx: Arc<std::sync::Mutex<mpsc::Receiver<DecodeJob>>>,
    window_map: WindowMap,
    configuration: ChunkConfiguration,
) {
    loop {
        // Pull one job. Mutex serializes only the recv() — the actual
        // decode runs in parallel across workers.
        let job = {
            let rx = job_rx.lock().unwrap();
            match rx.recv() {
                Ok(j) => j,
                Err(_) => return, // sender dropped → shutdown
            }
        };

        let label = trace::worker_label(job.partition_idx);
        let t0 = std::time::Instant::now();

        // For chunk 0 (start_bit==0) the empty window is the right
        // initial dict; insert is a no-op since we pre-seeded it.
        let window = if job.start_bit == 0 {
            window_map.get(0)
        } else if job.authoritative {
            // Consumer guarantees predecessor's window is in the map.
            window_map.get_or_wait(job.start_bit, Duration::from_secs(60))
        } else {
            // Speculative job: short wait. If miss, take slow path
            // with the empty window. Slow path is bounded (see
            // decode_chunk_bootstrap end_bit_limit) so a phantom
            // boundary returns Err quickly rather than hanging.
            window_map.get_or_wait(job.start_bit, WINDOW_WAIT_TIMEOUT)
        };

        let (result, path) = match window {
            Some(w) => {
                trace::emit(
                    &label,
                    "fast_path_start",
                    &format!(
                        r#""partition_idx":{},"start_bit":{},"until_bit":{},"authoritative":{},"rss_kib":{}"#,
                        job.partition_idx,
                        job.start_bit,
                        job.until_bit,
                        job.authoritative,
                        trace::rss_kib(),
                    ),
                );
                let r = decode_chunk_with_window(
                    input,
                    job.start_bit,
                    job.until_bit,
                    &w,
                    configuration,
                );
                (r, "fast")
            }
            None => {
                trace::emit(
                    &label,
                    "slow_path_start",
                    &format!(
                        r#""partition_idx":{},"start_bit":{},"until_bit":{},"rss_kib":{}"#,
                        job.partition_idx,
                        job.start_bit,
                        job.until_bit,
                        trace::rss_kib(),
                    ),
                );
                // Try direct at start_bit first (rapidgzip's tryToDecode
                // at GzipChunk.hpp:739). If it fails, iterate BlockFinder
                // candidates inside the worker, up to a 512 KiB scan.
                // The returned chunk reports encoded_offset_bits =
                // job.start_bit (the requested position) and
                // max_encoded_offset_bits = the actual candidate used.
                let r = decode_or_iterate(input, job.start_bit, job.until_bit, configuration);
                (r, "slow")
            }
        };

        let dur_us = t0.elapsed().as_micros();
        match &result {
            Ok(c) => {
                trace::emit(
                    &label,
                    "decode_ok",
                    &format!(
                        r#""partition_idx":{},"path":"{}","start_bit":{},"end_bit":{},"decoded":{},"markers":{},"clean":{},"preemptive":{},"duration_us":{dur_us}"#,
                        job.partition_idx,
                        path,
                        job.start_bit,
                        c.encoded_offset_bits + c.encoded_size_bits,
                        c.decoded_size(),
                        c.data_with_markers.len(),
                        c.data.len(),
                        c.stopped_preemptively,
                    ),
                );
                // Publish tail window if we can do it without waiting
                // for apply_window. Fast-path chunks always can. Slow-
                // path chunks whose trailing 32 KiB is in the clean
                // segment also can; otherwise the consumer publishes
                // after apply_window.
                if let Some(tail) = c.last_32kib_window() {
                    let end_bit = c.encoded_offset_bits + c.encoded_size_bits;
                    window_map.insert(end_bit, Arc::new(tail));
                }
            }
            Err(e) => {
                trace::emit(
                    &label,
                    "decode_err",
                    &format!(
                        r#""partition_idx":{},"path":"{}","start_bit":{},"until_bit":{},"err":"{}","duration_us":{dur_us}"#,
                        job.partition_idx,
                        path,
                        job.start_bit,
                        job.until_bit,
                        trace::esc(&format!("{e:?}")),
                    ),
                );
            }
        }
        let _ = job.reply.send(result);
    }
}

#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn consumer_loop<W: std::io::Write>(
    input: &[u8],
    writer: &mut W,
    n_partitions: usize,
    partition_offsets: &[usize],
    total_bits: usize,
    job_tx: &mpsc::Sender<DecodeJob>,
    window_map: &WindowMap,
    configuration: ChunkConfiguration,
    pool_size: usize,
    total_crc: &mut crc32fast::Hasher,
    total_size: &mut usize,
) -> Result<(), FetchError> {
    let _ = (input, configuration);

    // Partition-seed-keyed speculative prefetch (port of rapidgzip's
    // BlockFetcher::get / GzipChunkFetcher::processNextChunk).
    //
    // Each partition idx has a speculative job in flight, dispatched at
    // partition_offsets[idx] (NOT at predecessor's actual end). The
    // worker tries to decode at the seed, then iterates BlockFinder
    // candidates within 512 KiB until one succeeds. The returned chunk
    // carries:
    //   encoded_offset_bits = partition_offsets[idx]   (requested seed)
    //   max_encoded_offset_bits = actual candidate bit-position used.
    //
    // Consumer at expected_start checks the speculative result:
    //   matches_encoded_offset(expected_start)
    //     ⇔ seed ≤ expected_start ≤ actual_candidate
    //     AND decoded_offset_for(expected_start).is_some()
    //     ⇔ there's a subchunk boundary at expected_start.
    // On hit, we trim trim_bytes and consume. On miss, we re-dispatch
    // an AUTHORITATIVE job at expected_start (worker fast-paths because
    // the predecessor's window is in the WindowMap), and discard the
    // speculative.
    //
    // We keep `prefetch_count = 2 * pool_size` speculatives in flight at
    // all times (matches rapidgzip's PREFETCH_COUNT default).
    let prefetch_count = (pool_size * 2).max(1);

    let mut spec: Vec<Option<mpsc::Receiver<Result<ChunkData, ChunkDecodeError>>>> =
        (0..n_partitions).map(|_| None).collect();

    let until_for = |idx: usize| -> usize {
        partition_offsets
            .get(idx + 1)
            .copied()
            .unwrap_or(total_bits)
    };

    let submit_job = |idx: usize,
                      start: usize,
                      until: usize,
                      authoritative: bool|
     -> mpsc::Receiver<Result<ChunkData, ChunkDecodeError>> {
        let (tx, rx) = mpsc::channel();
        let event = if authoritative {
            "authoritative_prefetch"
        } else {
            "speculative_prefetch"
        };
        trace::emit(
            "consumer",
            event,
            &format!(r#""partition_idx":{idx},"start_bit":{start},"until_bit":{until}"#),
        );
        job_tx
            .send(DecodeJob {
                partition_idx: idx,
                start_bit: start,
                until_bit: until,
                authoritative,
                reply: tx,
            })
            .expect("worker pool dropped");
        rx
    };

    // Submit chunk 0 authoritative (start is known: bit 0). Also pre-fill
    // up to prefetch_count speculatives starting at partition 1.
    spec[0] = Some(submit_job(0, 0, until_for(0), true));
    for i in 1..(1 + prefetch_count).min(n_partitions) {
        spec[i] = Some(submit_job(i, partition_offsets[i], until_for(i), false));
    }

    let mut expected_start: usize = 0;
    let mut next_spec_to_dispatch: usize = (1 + prefetch_count).min(n_partitions);

    for idx in 0..n_partitions {
        let rx = spec[idx].take().expect("spec slot was filled");
        trace::emit(
            "consumer",
            "speculative_wait",
            &format!(r#""partition_idx":{idx},"expected_start":{expected_start}"#),
        );
        let spec_result = match rx.recv() {
            Ok(r) => r,
            Err(_) => {
                return Err(FetchError::Decode(ChunkDecodeError::ExactStopMissed {
                    requested: expected_start,
                    actual: 0,
                }));
            }
        };

        // Hit test: speculative chunk must (a) decode successfully,
        // (b) cover expected_start, (c) have a subchunk boundary at
        // expected_start (so decoded_offset_for resolves). Chunk 0 is
        // authoritative (submitted with start = 0); for it, a hit is
        // guaranteed if Ok.
        let hit_chunk: Option<ChunkData> = match spec_result {
            Ok(c) => {
                if c.matches_encoded_offset(expected_start)
                    && c.decoded_offset_for(expected_start).is_some()
                {
                    trace::emit(
                        "consumer",
                        "speculative_hit",
                        &format!(
                            r#""partition_idx":{idx},"expected_start":{expected_start},"seed":{},"actual":{}"#,
                            c.encoded_offset_bits, c.max_encoded_offset_bits,
                        ),
                    );
                    Some(c)
                } else {
                    trace::emit(
                        "consumer",
                        "speculative_miss",
                        &format!(
                            r#""partition_idx":{idx},"expected_start":{expected_start},"seed":{},"actual":{}"#,
                            c.encoded_offset_bits, c.max_encoded_offset_bits,
                        ),
                    );
                    None
                }
            }
            Err(e) => {
                // Chunk 0 was dispatched authoritative; if it fails the
                // input is invalid. Speculative chunks may fail on
                // phantom boundaries — fall through to authoritative.
                if idx == 0 {
                    return Err(FetchError::Decode(e));
                }
                trace::emit(
                    "consumer",
                    "speculative_err",
                    &format!(
                        r#""partition_idx":{idx},"expected_start":{expected_start},"err":"{}""#,
                        trace::esc(&format!("{e:?}")),
                    ),
                );
                None
            }
        };

        let chunk = match hit_chunk {
            Some(c) => c,
            None => {
                // Re-dispatch authoritative at expected_start. Worker
                // will fast-path because predecessor's window is in
                // the WindowMap (just inserted after consume of N-1).
                let until = partition_offsets
                    .iter()
                    .skip(idx + 1)
                    .find(|&&s| s > expected_start)
                    .copied()
                    .unwrap_or(total_bits);
                let auth_rx = submit_job(idx, expected_start, until, true);
                match auth_rx.recv() {
                    Ok(Ok(c)) => c,
                    Ok(Err(e)) => return Err(FetchError::Decode(e)),
                    Err(_) => {
                        return Err(FetchError::Decode(ChunkDecodeError::ExactStopMissed {
                            requested: expected_start,
                            actual: 0,
                        }));
                    }
                }
            }
        };

        let trim_bytes = chunk.decoded_offset_for(expected_start).unwrap_or(0);

        // Process the chunk: apply_window if markers; write bytes;
        // combine CRC; publish tail window.
        //
        // Window lookup is keyed by the ACTUAL decode start
        // (max_encoded_offset_bits), not the requested seed
        // (encoded_offset_bits). For speculative chunks these differ:
        // data corresponds to compressed range [actual, end] and the
        // markers in the leading region are back-refs into bytes ending
        // at compressed position `actual`. Predecessor publishes its
        // tail at its actual_end, which on a hit equals our `actual`.
        let mut chunk = chunk;
        if !chunk.data_with_markers.is_empty() {
            let window_key = chunk.max_encoded_offset_bits;
            let window = window_map
                .get_or_wait(window_key, Duration::from_secs(60))
                .ok_or(FetchError::Decode(ChunkDecodeError::ExactStopMissed {
                    requested: window_key,
                    actual: expected_start,
                }))?;
            let aw_t0 = std::time::Instant::now();
            let marker_count = chunk.data_with_markers.len();
            apply_window(&mut chunk, &window[..]);
            trace::emit(
                "consumer",
                "apply_window_done",
                &format!(
                    r#""partition_idx":{idx},"marker_bytes":{marker_count},"duration_us":{}"#,
                    aw_t0.elapsed().as_micros()
                ),
            );
            if let Some(tail) = chunk.last_32kib_window() {
                let end_bit = chunk.encoded_offset_bits + chunk.encoded_size_bits;
                window_map.insert(end_bit, Arc::new(tail));
            }
        }

        // Write bytes, skipping `trim_bytes` leading bytes that belong
        // to the predecessor's range.
        let mut written_crc = crc32fast::Hasher::new();
        let mut remaining_skip = trim_bytes;
        if !chunk.data_with_markers.is_empty() {
            let dwm_len = chunk.data_with_markers.len();
            let skip_in_dwm = remaining_skip.min(dwm_len);
            remaining_skip -= skip_in_dwm;
            if skip_in_dwm < dwm_len {
                let mut narrowed: Vec<u8> = Vec::with_capacity(dwm_len - skip_in_dwm);
                for v in &chunk.data_with_markers[skip_in_dwm..] {
                    narrowed.push(*v as u8);
                }
                written_crc.update(&narrowed);
                writer
                    .write_all(&narrowed)
                    .map_err(|e| FetchError::Decode(ChunkDecodeError::BootstrapFailed(e)))?;
                *total_size += narrowed.len();
            }
        }
        if !chunk.data.is_empty() {
            let skip_in_data = remaining_skip.min(chunk.data.len());
            if skip_in_data < chunk.data.len() {
                let slice = &chunk.data[skip_in_data..];
                written_crc.update(slice);
                writer
                    .write_all(slice)
                    .map_err(|e| FetchError::Decode(ChunkDecodeError::BootstrapFailed(e)))?;
                *total_size += slice.len();
            }
        }
        total_crc.combine(&written_crc);
        expected_start = chunk.encoded_offset_bits + chunk.encoded_size_bits;
        trace::emit(
            "consumer",
            "consume_done",
            &format!(
                r#""partition_idx":{idx},"end_bit":{expected_start},"decoded":{},"trim_bytes":{trim_bytes},"rss_kib":{}"#,
                chunk.decoded_size(),
                trace::rss_kib(),
            ),
        );

        // Refill the speculative pipeline: keep prefetch_count chunks
        // outstanding ahead of the consumer.
        if next_spec_to_dispatch < n_partitions {
            spec[next_spec_to_dispatch] = Some(submit_job(
                next_spec_to_dispatch,
                partition_offsets[next_spec_to_dispatch],
                until_for(next_spec_to_dispatch),
                false,
            ));
            next_spec_to_dispatch += 1;
        }
    }

    Ok(())
}

// Non-x86_64 / non-isal stub.
#[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
pub fn drive<W: std::io::Write>(
    _input: &[u8],
    _writer: &mut W,
    _parallelization: usize,
    _configuration: ChunkConfiguration,
) -> Result<(u32, usize), FetchError> {
    Err(FetchError::UnsupportedPlatform)
}

// ── Integration tests ────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_deflate(payload: &[u8], level: u32) -> Vec<u8> {
        let mut enc =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(level));
        enc.write_all(payload).unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn drive_round_trips_2mb_level6() {
        let payload = b"abcdefghij".repeat(200_000);
        let deflate = make_deflate(&payload, 6);
        let cfg = ChunkConfiguration {
            split_chunk_size: 512 * 1024,
            max_decoded_chunk_size: 20 * 512 * 1024,
            crc32_enabled: true,
        };
        let mut out = Vec::new();
        let (_crc, size) = drive(&deflate, &mut out, 4, cfg).expect("drive");
        assert_eq!(size, payload.len());
        assert_eq!(out, payload);
    }

    #[test]
    fn drive_round_trips_8mb_level9() {
        let payload = b"the quick brown fox jumps over the lazy dog ".repeat(200_000);
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(9));
        enc.write_all(&payload).unwrap();
        let deflate = enc.finish().unwrap();
        let cfg = ChunkConfiguration {
            split_chunk_size: 512 * 1024,
            max_decoded_chunk_size: 20 * 512 * 1024,
            crc32_enabled: true,
        };
        let mut out = Vec::new();
        let (_crc, size) = drive(&deflate, &mut out, 8, cfg).expect("drive");
        assert_eq!(size, payload.len());
        assert_eq!(out, payload);
    }
}
