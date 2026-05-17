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

    // Pre-compute speculative boundaries for every partition in parallel.
    // No trial decode (rapidgzip principle): just the BlockFinder candidate.
    let speculative_boundaries = compute_speculative_boundaries(input, &partition_offsets);

    let (job_tx, job_rx) = mpsc::channel::<DecodeJob>();
    let job_rx = Arc::new(std::sync::Mutex::new(job_rx));

    let mut total_crc = crc32fast::Hasher::new();
    let mut total_size: usize = 0;

    let pool_size = parallelization.max(1);

    let result = std::thread::scope(|s| -> Result<(), FetchError> {
        // Spawn worker pool.
        for _ in 0..pool_size {
            let job_rx = Arc::clone(&job_rx);
            let window_map = window_map.clone();
            let configuration = configuration;
            s.spawn(move || worker_loop(input, job_rx, window_map, configuration));
        }

        // Consumer loop. Submits speculative jobs, drains in order,
        // re-dispatches on mismatch, writes bytes, propagates windows.
        consumer_loop(
            input,
            writer,
            n_partitions,
            &partition_offsets,
            &speculative_boundaries,
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

#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
fn compute_speculative_boundaries(input: &[u8], partition_offsets: &[usize]) -> Vec<Option<usize>> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let n = partition_offsets.len();
    let total_bits = input.len() * 8;
    let results: Vec<std::sync::Mutex<Option<Option<usize>>>> =
        (0..n).map(|_| std::sync::Mutex::new(None)).collect();
    let next_task = AtomicUsize::new(0);

    std::thread::scope(|s| {
        let workers = num_cpus_for_boundary_scan().min(n);
        for _ in 0..workers {
            let results = &results;
            let next_task = &next_task;
            s.spawn(move || loop {
                let idx = next_task.fetch_add(1, Ordering::Relaxed);
                if idx >= n {
                    break;
                }
                let seed = partition_offsets[idx];
                let label = trace::boundary_label(idx);
                let t0 = std::time::Instant::now();
                let boundary = if seed >= total_bits {
                    None
                } else if seed == 0 {
                    Some(0)
                } else {
                    BlockFinder::new(input).find_first_candidate(
                        seed,
                        (BOUNDARY_SEARCH_RADIUS_BYTES * 8).min(total_bits - seed),
                    )
                };
                let dur_us = t0.elapsed().as_micros();
                trace::emit(
                    &label,
                    "boundary_done",
                    &format!(
                        r#""partition_idx":{idx},"seed_bit":{seed},"found_bit":{},"duration_us":{dur_us}"#,
                        match boundary {
                            Some(b) => format!("{b}"),
                            None => "null".to_string(),
                        }
                    ),
                );
                *results[idx].lock().unwrap() = Some(boundary);
            });
        }
    });

    results
        .into_iter()
        .map(|m| m.into_inner().unwrap().flatten())
        .collect()
}

#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
fn num_cpus_for_boundary_scan() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
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
            // Consumer guarantees the predecessor's window is in the
            // map at start_bit. Wait however long it takes.
            window_map.get_or_wait(job.start_bit, Duration::from_secs(60))
        } else {
            // Speculative: short wait, then fall back to slow path.
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
                let r = finish_decode_chunk_with_inexact_offset(
                    input,
                    job.start_bit,
                    job.until_bit,
                    &[],
                    configuration,
                );
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
    speculative_boundaries: &[Option<usize>],
    total_bits: usize,
    job_tx: &mpsc::Sender<DecodeJob>,
    window_map: &WindowMap,
    configuration: ChunkConfiguration,
    pool_size: usize,
    total_crc: &mut crc32fast::Hasher,
    total_size: &mut usize,
) -> Result<(), FetchError> {
    // Per-partition speculative receivers. Each partition's worker
    // sends its result here when done. Authoritative re-dispatch
    // creates fresh oneshot channels.
    let mut speculative_rx: Vec<Option<mpsc::Receiver<Result<ChunkData, ChunkDecodeError>>>> =
        (0..n_partitions).map(|_| None).collect();

    // Prefetch target: keep at most 2*pool_size speculative jobs in
    // flight ahead of the consumer.
    let prefetch_window = (pool_size * 2).max(2);
    let mut next_to_dispatch = 0usize;

    let mut dispatch = |next_to_dispatch: &mut usize,
                        speculative_rx: &mut Vec<
        Option<mpsc::Receiver<Result<ChunkData, ChunkDecodeError>>>,
    >| {
        while *next_to_dispatch < n_partitions
            && speculative_rx.iter().filter(|r| r.is_some()).count() < prefetch_window
        {
            let i = *next_to_dispatch;
            *next_to_dispatch += 1;
            let Some(start) = speculative_boundaries[i] else {
                continue;
            };
            let until = partition_offsets.get(i + 1).copied().unwrap_or(total_bits);
            let (tx, rx) = mpsc::channel();
            trace::emit(
                "dispatcher",
                "speculative_submit",
                &format!(r#""partition_idx":{i},"start_bit":{start},"until_bit":{until}"#),
            );
            job_tx
                .send(DecodeJob {
                    partition_idx: i,
                    start_bit: start,
                    until_bit: until,
                    authoritative: false,
                    reply: tx,
                })
                .expect("worker pool dropped");
            speculative_rx[i] = Some(rx);
        }
    };

    // Initial prefetch fill.
    dispatch(&mut next_to_dispatch, &mut speculative_rx);

    let mut expected_start: usize = 0;
    let mut next_to_consume: usize = 0;

    while next_to_consume < n_partitions {
        // Try to find a speculative result that matches expected_start.
        // Scan partitions in order from next_to_consume forward; first
        // match wins. If found, drain it.
        let mut chosen: Option<(usize, ChunkData)> = None;
        for i in next_to_consume..n_partitions {
            if let Some(rx_ref) = speculative_rx[i].as_ref() {
                // Non-blocking poll: only accept results that are
                // already ready AND match. Skip in-flight chunks.
                match rx_ref.try_recv() {
                    Ok(Ok(c)) if c.matches_encoded_offset(expected_start) => {
                        let _ = speculative_rx[i].take();
                        chosen = Some((i, c));
                        break;
                    }
                    Ok(Ok(_c)) => {
                        // Speculative result didn't match; discard.
                        let _ = speculative_rx[i].take();
                        trace::emit(
                            "consumer",
                            "speculative_discard",
                            &format!(r#""partition_idx":{i},"expected_start":{expected_start}"#),
                        );
                    }
                    Ok(Err(e)) => {
                        // Speculative result errored; discard.
                        let _ = speculative_rx[i].take();
                        trace::emit(
                            "consumer",
                            "speculative_err_discard",
                            &format!(
                                r#""partition_idx":{i},"err":"{}""#,
                                trace::esc(&format!("{e:?}"))
                            ),
                        );
                    }
                    Err(mpsc::TryRecvError::Empty) => {
                        // Still in flight; can't decide yet. Don't
                        // wait — keep scanning, then if nothing
                        // matches, block-wait on the i==next_to_consume
                        // slot below.
                    }
                    Err(mpsc::TryRecvError::Disconnected) => {
                        let _ = speculative_rx[i].take();
                    }
                }
            }
            if i == next_to_consume + prefetch_window {
                // Don't scan too far ahead; results for partitions much
                // further out are unlikely to be ready.
                break;
            }
        }

        let (idx, chunk) = match chosen {
            Some(c) => c,
            None => {
                // Nothing matched in the prefetch cache. Either we have
                // a speculative result at next_to_consume that's still
                // in flight (block-wait on it), or none. Block-wait on
                // next_to_consume if present; otherwise dispatch
                // authoritative.
                if let Some(rx) = speculative_rx[next_to_consume].take() {
                    match rx.recv() {
                        Ok(Ok(c)) if c.matches_encoded_offset(expected_start) => {
                            (next_to_consume, c)
                        }
                        Ok(Ok(_)) | Ok(Err(_)) | Err(_) => {
                            // Speculative miss → authoritative dispatch.
                            authoritative_dispatch(
                                input,
                                next_to_consume,
                                expected_start,
                                partition_offsets,
                                total_bits,
                                job_tx,
                                window_map,
                                configuration,
                            )?
                        }
                    }
                } else {
                    authoritative_dispatch(
                        input,
                        next_to_consume,
                        expected_start,
                        partition_offsets,
                        total_bits,
                        job_tx,
                        window_map,
                        configuration,
                    )?
                }
            }
        };

        // Process the chosen chunk: apply_window if slow-path produced
        // markers; write bytes; combine CRC; update expected_start;
        // publish tail window if not already published.
        let mut chunk = chunk;
        if !chunk.data_with_markers.is_empty() {
            let chunk_start = chunk.encoded_offset_bits;
            let window = window_map
                .get_or_wait(chunk_start, Duration::from_secs(60))
                .ok_or(FetchError::Decode(ChunkDecodeError::ExactStopMissed {
                    requested: chunk_start,
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
            // Publish tail now that markers are resolved.
            if let Some(tail) = chunk.last_32kib_window() {
                let end_bit = chunk.encoded_offset_bits + chunk.encoded_size_bits;
                window_map.insert(end_bit, Arc::new(tail));
            }
        }

        // Write bytes in stream order.
        if !chunk.data_with_markers.is_empty() {
            let mut narrowed: Vec<u8> = Vec::with_capacity(chunk.data_with_markers.len());
            for v in &chunk.data_with_markers {
                narrowed.push(*v as u8);
            }
            writer
                .write_all(&narrowed)
                .map_err(|e| FetchError::Decode(ChunkDecodeError::BootstrapFailed(e)))?;
            *total_size += narrowed.len();
        }
        if !chunk.data.is_empty() {
            writer
                .write_all(&chunk.data)
                .map_err(|e| FetchError::Decode(ChunkDecodeError::BootstrapFailed(e)))?;
            *total_size += chunk.data.len();
        }
        total_crc.combine(&chunk.crc);
        expected_start = chunk.encoded_offset_bits + chunk.encoded_size_bits;
        trace::emit(
            "consumer",
            "consume_done",
            &format!(
                r#""partition_idx":{idx},"end_bit":{expected_start},"decoded":{},"rss_kib":{},"speculative_in_flight":{}"#,
                chunk.decoded_size(),
                trace::rss_kib(),
                speculative_rx.iter().filter(|r| r.is_some()).count(),
            ),
        );
        next_to_consume += 1;

        // Refill prefetch pipeline.
        dispatch(&mut next_to_dispatch, &mut speculative_rx);
    }

    Ok(())
}

#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn authoritative_dispatch(
    _input: &[u8],
    partition_idx: usize,
    expected_start: usize,
    partition_offsets: &[usize],
    total_bits: usize,
    job_tx: &mpsc::Sender<DecodeJob>,
    _window_map: &WindowMap,
    _configuration: ChunkConfiguration,
) -> Result<(usize, ChunkData), FetchError> {
    let until = partition_offsets
        .iter()
        .skip(partition_idx + 1)
        .find(|&&seed| seed > expected_start)
        .copied()
        .unwrap_or(total_bits);
    let (tx, rx) = mpsc::channel();
    trace::emit(
        "consumer",
        "authoritative_submit",
        &format!(
            r#""partition_idx":{partition_idx},"expected_start":{expected_start},"until_bit":{until}"#
        ),
    );
    job_tx
        .send(DecodeJob {
            partition_idx,
            start_bit: expected_start,
            until_bit: until,
            authoritative: true,
            reply: tx,
        })
        .expect("worker pool dropped");
    let chunk = rx.recv().expect("worker dropped reply")?;
    Ok((partition_idx, chunk))
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
