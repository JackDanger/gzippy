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
    // Step 1: try the requested offset directly.
    if let Ok(mut c) =
        finish_decode_chunk_with_inexact_offset(input, start_bit, until_bit, &[], configuration)
    {
        // Worker decoded successfully at start_bit. Record the requested
        // position as encoded_offset (so the cache key matches consumer
        // expectations) and max_encoded_offset = the actual decoded
        // start (= start_bit, since direct succeeded).
        c.max_encoded_offset_bits = c.encoded_offset_bits;
        return Ok(c);
    }
    // Step 2: iterate candidates inside the worker. 512 KiB matches
    // rapidgzip's per-chunk scan budget.
    let finder = BlockFinder::new(input);
    let scan_end = (start_bit + 512 * 1024 * 8).min(input.len() * 8);
    let candidates = finder.find_blocks(start_bit, scan_end);
    let mut last_err: Option<ChunkDecodeError> = None;
    for candidate in candidates {
        if candidate.bit_offset <= start_bit {
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
    let _ = (input, partition_offsets, pool_size);

    // Pure depth-2 authoritative chain. No speculative pre-scan.
    //
    //   - pending_auth[0] is submitted at start_bit = 0 right away.
    //   - On consume of chunk N, we know chunk N's actual end = the next
    //     chunk's authoritative start. Submit pending_auth[N+1] before
    //     processing chunk N's bytes so the worker decodes in parallel
    //     with consume work (apply_window / write / window insert).
    //   - When chunk N+1's iteration starts, pending_auth[N+1] is
    //     usually already done. Wait, accept, consume.
    //
    // Workers handle the "no real block at start_bit" case INSIDE the
    // worker by iterating BlockFinder candidates within 512 KiB
    // (mirrors rapidgzip's tryToDecode at GzipChunk.hpp:712-741). The
    // returned chunk has encoded_offset_bits = requested start; the
    // actual successful candidate is recorded in max_encoded_offset_bits.
    let mut pending_auth: Vec<Option<mpsc::Receiver<Result<ChunkData, ChunkDecodeError>>>> =
        (0..n_partitions).map(|_| None).collect();

    let mut submit_auth =
        |idx: usize,
         start: usize,
         pending: &mut Vec<Option<mpsc::Receiver<Result<ChunkData, ChunkDecodeError>>>>|
         -> Result<(), FetchError> {
            if idx >= n_partitions || pending[idx].is_some() {
                return Ok(());
            }
            let until = partition_offsets
                .iter()
                .skip(idx + 1)
                .find(|&&s| s > start)
                .copied()
                .unwrap_or(total_bits);
            let (tx, rx) = mpsc::channel();
            trace::emit(
                "consumer",
                "authoritative_prefetch",
                &format!(r#""partition_idx":{idx},"start_bit":{start},"until_bit":{until}"#),
            );
            job_tx
                .send(DecodeJob {
                    partition_idx: idx,
                    start_bit: start,
                    until_bit: until,
                    authoritative: true,
                    reply: tx,
                })
                .expect("worker pool dropped");
            pending[idx] = Some(rx);
            Ok(())
        };

    // Initial dispatch: chunk 0 at bit 0.
    submit_auth(0, 0, &mut pending_auth)?;

    let mut expected_start: usize = 0;
    let mut next_to_consume: usize = 0;

    while next_to_consume < n_partitions {
        let rx = match pending_auth[next_to_consume].take() {
            Some(rx) => rx,
            None => {
                // Should only happen if we never submitted; submit now.
                submit_auth(next_to_consume, expected_start, &mut pending_auth)?;
                pending_auth[next_to_consume]
                    .take()
                    .expect("just submitted")
            }
        };
        trace::emit(
            "consumer",
            "authoritative_prefetch_wait",
            &format!(r#""partition_idx":{next_to_consume},"expected_start":{expected_start}"#),
        );
        let chunk = match rx.recv() {
            Ok(Ok(c)) => c,
            Ok(Err(e)) => return Err(FetchError::Decode(e)),
            Err(_) => {
                return Err(FetchError::Decode(ChunkDecodeError::ExactStopMissed {
                    requested: expected_start,
                    actual: 0,
                }));
            }
        };

        // Workers return chunk.encoded_offset_bits = the REQUESTED start.
        // For range matching: matches_encoded_offset(expected_start) is
        // always true here because we requested expected_start.
        // decoded_offset_for(expected_start) returns Some(0) for the
        // exact-match case (encoded_offset == expected_start) which is
        // the rapidgzip authoritative pattern.
        let trim_bytes = chunk.decoded_offset_for(expected_start).unwrap_or(0);
        let idx = next_to_consume;

        // Process the chunk: apply_window if markers; write bytes;
        // combine CRC; publish tail window.
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
        next_to_consume += 1;

        // Depth-2 prefetch: submit chunk N+1's authoritative NOW so
        // its decode overlaps with consume work for the chunk we just
        // wrote.
        if next_to_consume < n_partitions {
            submit_auth(next_to_consume, expected_start, &mut pending_auth)?;
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
