#![cfg(parallel_sm)]

//! Production entry for parallel single-member gzip decompression.
//!
//! Parses the gzip envelope, runs [`super::chunk_fetcher::drive`] on the
//! raw deflate body, then verifies CRC32 + ISIZE. `single_member` is a thin
//! classifier-routed wrapper around [`read_parallel_sm`].
//!
//! # Window-sparsity configuration (keepIndex=false faithful port)
//!
//! Vendor (`ParallelGzipReader.hpp:1320-1330`, `tools/rapidgzip.cpp:47,167`):
//! the CLI default is `keepIndex{false}`. `applyChunkDataConfiguration` then sets:
//!   ```cpp
//!   m_chunkConfiguration.windowSparsity = m_keepIndex && m_windowSparsity;  // → false
//!   m_chunkConfiguration.windowCompressionType =
//!       m_keepIndex ? m_windowCompressionType
//!                   : std::make_optional(CompressionType::NONE);             // → Some(NONE)
//!   ```
//! With sparsity off the 32 KiB `getUsedWindowSymbols` scan is skipped on every
//! chunk finalize — per mechanism measurement: ~8.5 ms × 51 chunks ~= 430 ms
//! thread-summed at T8, ~⅔ of the decodeBlock CPU gap vs rapidgzip.
//!
//! The `GZIPPY_WINDOW_SPARSITY=1` kill-switch (which restored the old
//! always-on behavior) was removed 2026-07-07 (batch 4f) — sparsity is always
//! OFF (the faithful keepIndex=false default). Effect counter
//! `SPARSITY_DECODE_COUNT` in `chunk_data` tracks actual executions.

/// Whether the pre-port always-on window-sparsity behavior is active.
/// Hardcoded OFF (shipped default; the `GZIPPY_WINDOW_SPARSITY=1` kill-switch
/// that used to restore it was removed as dead — batch 4f, byte-transparent).
fn window_sparsity_kill_switch() -> bool {
    false
}

/// Decompress one single-member gzip buffer with the parallel chunk pipeline.
#[cfg(parallel_sm)]
pub fn read_parallel_sm<W: std::io::Write>(
    gzip_data: &[u8],
    writer: &mut W,
    out_fd: Option<i32>,
    parallelization: usize,
    target_compressed_chunk_bytes: usize,
    verbose: bool,
) -> Result<ReadResult, ReadParallelSmError> {
    read_parallel_sm_inner(
        gzip_data,
        writer,
        out_fd,
        parallelization,
        target_compressed_chunk_bytes,
        None,
        verbose,
    )
}

/// Single-member parallel decode that also reports how many output bytes were
/// streamed — even on the error path. The single-member driver uses the count
/// to resume the remaining members of a misrouted multi-member stream past the
/// validated prefix already written.
#[cfg(parallel_sm)]
pub fn read_parallel_sm_capturing<W: std::io::Write>(
    gzip_data: &[u8],
    writer: &mut W,
    out_fd: Option<i32>,
    parallelization: usize,
    target_compressed_chunk_bytes: usize,
    bytes_written_out: &mut usize,
    verbose: bool,
) -> Result<ReadResult, ReadParallelSmError> {
    read_parallel_sm_inner(
        gzip_data,
        writer,
        out_fd,
        parallelization,
        target_compressed_chunk_bytes,
        Some(bytes_written_out),
        verbose,
    )
}

/// Single-member parallel decode, optionally reporting how many output bytes
/// were streamed even on the error path (`bytes_written_out`). The multi-member
/// driver uses that count to resume past an already-streamed prefix when a
/// misrouted multi-member stream fails the single-stream decode.
#[cfg(parallel_sm)]
fn read_parallel_sm_inner<W: std::io::Write>(
    gzip_data: &[u8],
    writer: &mut W,
    out_fd: Option<i32>,
    parallelization: usize,
    target_compressed_chunk_bytes: usize,
    bytes_written_out: Option<&mut usize>,
    verbose: bool,
) -> Result<ReadResult, ReadParallelSmError> {
    use crate::decompress::parallel::chunk_data::ChunkConfiguration;
    use crate::decompress::parallel::chunk_fetcher;
    use crate::decompress::parallel::compressed_vector::CompressionType;
    use crate::decompress::parallel::gzip_format;

    // Slab auto-gate input: stored at EVERY decode entry (atomic, never
    // once-cached) so consecutive decodes with different T re-gate correctly.
    crate::decompress::parallel::rpmalloc_alloc::set_decode_threads(parallelization);

    let (_hdr, header_size) =
        gzip_format::read_header(gzip_data).map_err(|_| ReadParallelSmError::InvalidHeader)?;
    let trailer_size = 8;
    if gzip_data.len() < header_size + trailer_size {
        return Err(ReadParallelSmError::InvalidFormat);
    }
    let deflate_data = &gzip_data[header_size..gzip_data.len() - trailer_size];

    let footer = gzip_format::read_footer(gzip_data, gzip_data.len() - trailer_size)
        .map_err(|_| ReadParallelSmError::InvalidFormat)?;
    let expected_crc = footer.crc32;
    let expected_size = footer.uncompressed_size as usize;

    // Ratio-informed upfront reserve (DIS-14/DIS-17 / box-proven +41% model-T8).
    // ratio_ceiling = ceil((ISIZE / compressed_len) × 1.25), minimum 2.
    // = ceil(ISIZE × 5 / (compressed_len × 4)), minimum 2.
    // ISIZE is mod 2^32 so files >4 GiB raw may wrap → under-ratio → safe
    // regrow via GROW_BYTES; no correctness risk, just a few extra grows.
    // 0 = unknown (compressed_len == 0 or ISIZE == 0) → 8× fallback in
    // finish_decode_chunk_isal_oracle.
    let expansion_ratio_ceil: u16 = {
        let isize_bytes = footer.uncompressed_size as u64;
        let compressed_bytes = deflate_data.len() as u64;
        if compressed_bytes == 0 || isize_bytes == 0 {
            0
        } else {
            let numer = isize_bytes.saturating_mul(5);
            let denom = compressed_bytes.saturating_mul(4);
            numer.div_ceil(denom).max(2).min(u16::MAX as u64) as u16
        }
    };

    // Faithful port of vendor `applyChunkDataConfiguration` at keepIndex=false
    // (ParallelGzipReader.hpp:1320-1330, tools/rapidgzip.cpp:47):
    //   windowSparsity   = keepIndex && windowSparsity = false
    //   windowCompressionType = keepIndex ? userValue : Some(NONE) = Some(NONE)
    let sparsity = window_sparsity_kill_switch();
    let configuration = ChunkConfiguration {
        split_chunk_size: target_compressed_chunk_bytes,
        max_decoded_chunk_size: 20 * target_compressed_chunk_bytes,
        crc32_enabled: true,
        // Default (keepIndex=false, vendor benchmark): sparsity OFF — skip the
        // 32 KiB `getUsedWindowSymbols` scan per chunk finalize.
        window_sparsity: sparsity,
        // Default (keepIndex=false): store windows uncompressed (CompressionType::None).
        window_compression_type: if sparsity {
            None
        } else {
            Some(CompressionType::None)
        },
        expansion_ratio_ceil,
        // Single-member driver: never walk member boundaries.
        multi_member: false,
    };

    // THIN-T1 PRODUCTION PATH (route-A scaffold shed). At T==1 the decode is
    // strictly sequential front-to-back, so EVERY block already has its 32 KiB
    // predecessor window — the parallel block-finder / WindowMap / marker arming
    // / prefetch / threadpool scaffold is pure overhead. The thin serial rolling-
    // window driver over the SAME shared `decode_chunk` kernel sheds it (route-A
    // oracle: thin/libdeflate≈1.08 vs prod/libdeflate≈1.22 on this box). CRC32 +
    // ISIZE are verified below exactly as for the parallel path, and bytes_written
    // is captured so the multi-member-misroute resume net still works.
    // T1 serial-eligible: strictly T==1.
    let t1_serial = parallelization <= 1;
    // T1-MONOLITH (igzip-shaped single-buffer path) and T1-MONOLITH-STREAMING
    // were deliberate, T1-gated divergences from the rapidgzip chunk pipeline
    // toward an igzip-shaped monolith, both opt-in only (`GZIPPY_MONOLITH=1` /
    // `GZIPPY_STREAM_MONOLITH=1`). Both were gate-measured and never won: the
    // full-ISIZE monolith FALSIFIED (regresses past thin-T1 — the single ISIZE
    // buffer first-touches the whole output, ~4x igzip's page faults, the cost
    // igzip's streaming small reused buffer avoids); the streaming variant fixed
    // the fault-storm and shed per-chunk scaffold INSTRUCTIONS but fulcrum optgate
    // REFUSED the wall win as INSTRUCTION-ONLY (cyc/byte did not improve beyond
    // spread). Removed 2026-07-07 (batch 4f) — production T1 stays thin-T1.
    let use_thin_t1 = t1_serial;
    // Tiny-file lever (#189/#199 lever-2b): a tiny thin-T1 decode makes exactly
    // ONE rpmalloc-backed allocation (its huge chunk-output reserve), and that
    // allocation triggers rpmalloc process+thread init — pure overhead for a
    // single-buffer single-thread decode (MEASURED: instr/byte RESOLVED-improved
    // on 40-400 KiB files, both arches, when the decode runs on the system
    // allocator). While this RAII scope is alive on THIS thread, new huge
    // slab-routed allocations are system-backed, so the tiny decode never
    // initializes rpmalloc. PER-DECODE and thread-local (no process latch, no
    // cross-decode state): a big decode after a tiny one behaves exactly as if
    // it were the process's first. Large thin-T1 decodes (> 8 MiB output:
    // weights, silesia-T1) keep rpmalloc + the resident slab (its measured win);
    // T>1 workers never enter the scope, so the parallel path is unchanged by
    // construction. See `rpmalloc_alloc::SystemHugeScope`.
    const SYSTEM_ALLOC_MAX_OUTPUT: usize = 8 * 1024 * 1024;
    let _sys_huge_scope = if use_thin_t1 && expected_size <= SYSTEM_ALLOC_MAX_OUTPUT {
        Some(crate::decompress::parallel::rpmalloc_alloc::SystemHugeScope::enter())
    } else {
        None
    };
    let drive_result = if use_thin_t1 {
        chunk_fetcher::drive_thin_t1_oracle(deflate_data, writer, configuration, bytes_written_out)
    } else if let Some(out) = bytes_written_out {
        // Multi-member resume support: capture bytes streamed even on error so
        // a misrouted multi-member stream can resume past the prefix.
        chunk_fetcher::drive_capturing(
            deflate_data,
            writer,
            out_fd,
            parallelization,
            configuration,
            out,
            verbose,
        )
    } else {
        chunk_fetcher::drive(
            deflate_data,
            writer,
            out_fd,
            parallelization,
            configuration,
            verbose,
        )
    };
    let (total_crc, total_size) =
        drive_result.map_err(|e| ReadParallelSmError::DecodeFailed(format!("{e:?}")))?;

    if total_size != expected_size {
        return Err(ReadParallelSmError::SizeMismatch {
            expected: expected_size,
            actual: total_size,
        });
    }
    if total_crc != expected_crc {
        return Err(ReadParallelSmError::CrcMismatch {
            expected: expected_crc,
            actual: total_crc,
        });
    }

    Ok(ReadResult {
        total_crc,
        total_size,
    })
}

// `total_crc` is verified against the trailer inside `read_parallel_sm`; the
// field is kept on the result for completeness even though the current caller
// reads only `total_size`. The allow is unconditional: non-x86 builds never
// construct `ReadResult`, and x86 builds construct it but read only the size.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct ReadResult {
    pub total_crc: u32,
    pub total_size: usize,
}

/// A `Write` that discards the first `skip` bytes handed to it, then forwards
/// the rest to the inner writer. Used by the multi-member RESUME path: after a
/// misrouted multi-member stream fails the single-stream decode, the first
/// member's already-streamed prefix (`skip` bytes) must NOT be re-emitted when
/// the whole stream is re-decoded member-by-member.
#[cfg(parallel_sm)]
struct SkipWriter<'a, W: std::io::Write> {
    inner: &'a mut W,
    skip: usize,
}

#[cfg(parallel_sm)]
impl<W: std::io::Write> std::io::Write for SkipWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.skip == 0 {
            return self.inner.write(buf);
        }
        if buf.len() <= self.skip {
            self.skip -= buf.len();
            return Ok(buf.len());
        }
        let drop = self.skip;
        self.skip = 0;
        self.inner.write_all(&buf[drop..])?;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Counts production `MultiMemberChunked` decodes (a whole-file member walk,
/// each member inflated by the full within-member parallel engine). Deletion-trap
/// discipline (mirror of `single_member::MARKER_PIPELINE_RUNS`): a routing test
/// asserts this advances on a multi-member CLI-shaped decode and stays 0 on the
/// single-member corpus.
#[cfg(parallel_sm)]
pub static MULTI_MEMBER_PIPELINE_RUNS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Production entry for [`crate::decompress::DecodePath::MultiMemberChunked`]:
/// decode a multi-member gzip stream by walking each member and inflating it
/// with the full within-member parallel single-member engine, streaming output
/// in member order. Byte-exact and per-member CRC32 + ISIZE verified (each
/// member is decoded by [`read_parallel_sm`], which parses and checks that
/// member's own trailer). Trailing garbage after a valid member boundary ends
/// the walk cleanly (matching gzip(1) and the sequential path). Shares its
/// implementation with the misroute re-entry ([`read_parallel_sm_resume_multi`])
/// started at offset 0 — the `SkipWriter` drops 0 bytes on this fresh entry.
///
/// Routed only for MIXED "GZ" ++ plain concatenations (the deterministic route
/// that the BGZF fast path would truncate), NOT for plain dominant/few-member
/// distributions — the member-walk was measured to REGRESS those on M1 (see
/// `DecodePath::MultiMemberChunked` docs). It is a member-walk, NOT the
/// rapidgzip-faithful whole-file-block-finder cross-member continuation (the
/// gate-phase core).
#[cfg(parallel_sm)]
pub fn read_parallel_sm_multi<W: std::io::Write>(
    gzip_data: &[u8],
    writer: &mut W,
    parallelization: usize,
    target_compressed_chunk_bytes: usize,
    verbose: bool,
) -> Result<ReadResult, ReadParallelSmError> {
    MULTI_MEMBER_PIPELINE_RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    read_parallel_sm_resume_multi(
        gzip_data,
        writer,
        0,
        parallelization,
        target_compressed_chunk_bytes,
        verbose,
    )
}

/// Decode a possibly-multi-member gzip stream, RESUMING past `already_written`
/// output bytes already streamed by a failed single-member attempt.
///
/// gzip multi-member semantics (RFC 1952 §2.2; `cat a.gz b.gz`, pigz, log
/// rotation): the classifier's `is_likely_multi_member` only scans the first
/// 16 MiB, so a stream whose SECOND member begins past that window is misrouted
/// to the single-member parallel path. That path treats the whole buffer as one
/// deflate stream and errors near the member boundary (the block finder cannot
/// cross member 1's gzip footer into member 2's header). This driver walks each
/// member's deflate body to its boundary, decodes each member with the SAME
/// pure-Rust parallel engine (per-member CRC32 + ISIZE verified), and
/// concatenates output in order — no C-FFI, faithful to gzip(1).
///
/// `already_written` is the contiguous, validated prefix the failed attempt
/// streamed (chunks are written in order only after they validate); a
/// `SkipWriter` drops exactly that many bytes so the prefix is not duplicated.
/// Output streams through the `Write` trait (the resume path does not use the
/// zero-copy `out_fd` writev — correctness over speed on this rare path).
#[cfg(parallel_sm)]
pub fn read_parallel_sm_resume_multi<W: std::io::Write>(
    gzip_data: &[u8],
    writer: &mut W,
    already_written: usize,
    parallelization: usize,
    target_compressed_chunk_bytes: usize,
    verbose: bool,
) -> Result<ReadResult, ReadParallelSmError> {
    use crate::decompress::parallel::gzip_format;
    use crate::decompress::scan_inflate;
    use std::io::Write as _;

    let mut skip = SkipWriter {
        inner: writer,
        skip: already_written,
    };

    let mut offset = 0usize;
    let mut total_size = 0usize;
    let mut members = 0usize;

    while offset < gzip_data.len() {
        let remaining = &gzip_data[offset..];
        // Stop cleanly at the end (no trailing junk) — fewer than the minimum
        // header+trailer bytes, or no gzip magic, means we have consumed every
        // member.
        if remaining.len() < 18 || remaining[0] != 0x1f || remaining[1] != 0x8b {
            break;
        }

        // Header → start of this member's deflate body.
        let (_hdr, header_size) =
            gzip_format::read_header(remaining).map_err(|_| ReadParallelSmError::InvalidHeader)?;
        if remaining.len() < header_size + 8 {
            return Err(ReadParallelSmError::InvalidFormat);
        }

        // Walk this member's deflate stream to its BFINAL end (byte-aligned),
        // pure-Rust, bounded memory. The 8-byte gzip trailer follows.
        let deflate_len = scan_inflate::deflate_stream_byte_len(&remaining[header_size..])
            .map_err(|e| ReadParallelSmError::DecodeFailed(format!("member boundary: {e}")))?;
        let member_end = header_size + deflate_len + 8;
        if member_end > remaining.len() {
            return Err(ReadParallelSmError::InvalidFormat);
        }
        let member = &remaining[..member_end];

        // Decode exactly this one member with the parallel engine; it parses
        // the member's own trailer and verifies CRC32 + ISIZE.
        let r = read_parallel_sm(
            member,
            &mut skip,
            None,
            parallelization,
            target_compressed_chunk_bytes,
            verbose,
        )?;
        total_size += r.total_size;
        members += 1;
        offset += member_end;
    }

    if members == 0 {
        return Err(ReadParallelSmError::InvalidFormat);
    }
    skip.flush()
        .map_err(|_| ReadParallelSmError::InvalidFormat)?;

    Ok(ReadResult {
        total_crc: 0,
        total_size,
    })
}

/// Production entry for [`crate::decompress::DecodePath::MultiMemberGrid`]:
/// decode a multi-member gzip stream as ONE whole-file chunk grid so the
/// dominant member's deflate blocks spread across ALL workers (instead of the
/// member-walk's one-worker-per-member plateau). The whole file (minus only the
/// FIRST member's gzip header) is handed to the SAME parallel chunk pipeline
/// used for single-member; `ChunkConfiguration::multi_member = true` makes every
/// decode arm walk member boundaries (footer → next header → empty-window reset →
/// continue), and the consumer runs the per-member CRC32 + ISIZE verifier
/// (`MemberVerifier`, design §4) over resolved chunks in decode order. Byte-exact
/// and per-member verified; pure-Rust; no C-FFI.
///
/// Faithful to rapidgzip's `GzipChunkFetcher` cross-member continuation: the
/// finder spans the whole file (chunk 0 starts at the first member's first
/// deflate block, i.e. bit 0 of the header-stripped slice), and chunks that
/// straddle a member boundary continue into the next member rather than stopping
/// at its final BFINAL.
#[cfg(parallel_sm)]
pub fn read_parallel_sm_grid<W: std::io::Write>(
    gzip_data: &[u8],
    writer: &mut W,
    out_fd: Option<i32>,
    parallelization: usize,
    target_compressed_chunk_bytes: usize,
    verbose: bool,
) -> Result<ReadResult, ReadParallelSmError> {
    use crate::decompress::parallel::chunk_data::ChunkConfiguration;
    use crate::decompress::parallel::chunk_fetcher;
    use crate::decompress::parallel::compressed_vector::CompressionType;
    use crate::decompress::parallel::gzip_format;

    // Slab auto-gate input (same as the single-member driver).
    crate::decompress::parallel::rpmalloc_alloc::set_decode_threads(parallelization);

    // Strip ONLY the first member's gzip header. Everything after (all member
    // bodies, interior footers+headers, and the FINAL member's footer) stays in
    // the slice: `decode_chunk_*_multi` consumes interior footers/headers and
    // clean-stops at EOF after the last member's footer (§3.1/§3.2). Chunk 0
    // therefore starts at bit 0 of this slice with an empty window — identical
    // to the single-member chunk 0.
    let (_hdr, header_size) =
        gzip_format::read_header(gzip_data).map_err(|_| ReadParallelSmError::InvalidHeader)?;
    if gzip_data.len() < header_size + 8 {
        return Err(ReadParallelSmError::InvalidFormat);
    }
    let input = &gzip_data[header_size..];

    // Expansion-ratio reserve hint (§5a). The old code recomputed this from a
    // FULL-FILE `scan_member_boundaries_fast` pass — a T-invariant serial scan of
    // the whole compressed file, run again here AFTER `classify_gzip` already
    // scanned it, so the grid critical path paid the O(file) scan TWICE. On a
    // large compressible-dominant multi-member stream those two scans were ~50%
    // of the T16 wall (Amdahl). We drop this scan entirely and pass 0 (unknown) →
    // the per-chunk reserve uses the historical 8× fallback in
    // `compute_initial_reserve`, which is grow-safe (a wrong hint only affects
    // initial capacity, never correctness — per-member CRC/ISIZE is the oracle)
    // AND RSS-neutral (a Vec's reserved-but-unwritten pages are not resident;
    // only bytes actually decoded fault in). Per-member boundaries are still
    // discovered DURING the parallel decode by the cross-member chunk walk.
    let expansion_ratio_ceil: u16 = 0;

    let sparsity = window_sparsity_kill_switch();
    let configuration = ChunkConfiguration {
        split_chunk_size: target_compressed_chunk_bytes,
        max_decoded_chunk_size: 20 * target_compressed_chunk_bytes,
        crc32_enabled: true,
        window_sparsity: sparsity,
        window_compression_type: if sparsity {
            None
        } else {
            Some(CompressionType::None)
        },
        expansion_ratio_ceil,
        // THE grid flag: every decode arm walks member boundaries; the consumer
        // runs per-member CRC32 + ISIZE verification.
        multi_member: true,
    };

    // Drive the whole-file grid. Per-member CRC32 + ISIZE are verified INSIDE
    // the consumer (`MemberVerifier`), which returns a terminal error on any
    // member mismatch or a torn final member — so there is NO single-trailer
    // check here (a multi-member stream has one trailer PER member, not one
    // global trailer). `total_crc` is the whole-output rolling CRC and is
    // intentionally NOT compared to any single trailer.
    let (_total_crc, total_size) = chunk_fetcher::drive(
        input,
        writer,
        out_fd,
        parallelization,
        configuration,
        verbose,
    )
    .map_err(|e| ReadParallelSmError::DecodeFailed(format!("{e:?}")))?;

    Ok(ReadResult {
        total_crc: 0,
        total_size,
    })
}

#[cfg_attr(
    not(all(feature = "isal-compression", target_arch = "x86_64")),
    allow(dead_code)
)]
#[derive(Debug)]
pub enum ReadParallelSmError {
    InvalidHeader,
    InvalidFormat,
    DecodeFailed(String),
    SizeMismatch { expected: usize, actual: usize },
    CrcMismatch { expected: u32, actual: u32 },
}

impl std::fmt::Display for ReadParallelSmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadParallelSmError::InvalidHeader => write!(f, "invalid gzip header"),
            ReadParallelSmError::InvalidFormat => write!(f, "input below parallel SM minimum"),
            ReadParallelSmError::DecodeFailed(s) => write!(f, "chunk decode failed: {s}"),
            ReadParallelSmError::SizeMismatch { expected, actual } => {
                write!(f, "output size mismatch: expected {expected}, got {actual}")
            }
            ReadParallelSmError::CrcMismatch { expected, actual } => {
                write!(
                    f,
                    "CRC32 mismatch: expected {expected:08x}, got {actual:08x}"
                )
            }
        }
    }
}

impl std::error::Error for ReadParallelSmError {}

#[cfg(all(test, feature = "isal-compression", target_arch = "x86_64"))]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_gzip(payload: &[u8]) -> Vec<u8> {
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(payload).unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn read_parallel_sm_roundtrip() {
        let payload: Vec<u8> = (0..64 * 1024).map(|i| (i % 251) as u8).collect();
        let gzip = make_gzip(&payload);
        let mut out = Vec::new();
        let result = read_parallel_sm(&gzip, &mut out, None, 4, 512 * 1024, false).unwrap();
        assert_eq!(out, payload);
        assert_eq!(result.total_size, payload.len());
    }

    /// High-thread parallel-SM stress: 16 MiB of incompressible data per
    /// fixture (compresses ~1:1 into many stored blocks) decoded at
    /// T=16, four PRNG seeds varying the block layout. Each decode runs
    /// on a worker thread under a hard 60s bound, so any hang in the
    /// parallel driver fails this test instead of wedging the suite.
    ///
    /// This is a general integration guard, NOT the deterministic
    /// regression for the EOF byte-alignment-padding hang — that hang
    /// only triggers when `consumer_loop` schedules a sub-byte tail
    /// chunk, which depends on the exact confirmed-block layout and is
    /// not reliably forced here. The deterministic reproducer for it
    /// lives in `chunk_decode` — see
    /// `decode_chunk_isal_terminates_on_sub_byte_eof_padding`.
    #[test]
    fn read_parallel_sm_roundtrip_incompressible_high_thread() {
        let seeds: [u64; 4] = [
            0x9e3779b97f4a7c15,
            0xc2b2ae3d27d4eb4f,
            0x165667b19e3779f9,
            0xff51afd7ed558ccd,
        ];
        for (i, seed) in seeds.into_iter().enumerate() {
            let mut original = vec![0u8; 16 * 1024 * 1024];
            let mut state = seed;
            for b in &mut original {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                *b = (state >> 33) as u8;
            }
            let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(6));
            enc.write_all(&original).unwrap();
            let gzip = enc.finish().unwrap();

            let (tx, rx) = std::sync::mpsc::channel();
            let original_len = original.len();
            let handle = std::thread::spawn(move || {
                let mut out = Vec::with_capacity(original_len);
                let res = read_parallel_sm(&gzip, &mut out, None, 16, 1024 * 1024, false);
                let _ = tx.send((res, out));
            });

            match rx.recv_timeout(std::time::Duration::from_secs(60)) {
                Ok((res, out)) => {
                    if let Err(e) = res {
                        panic!("read_parallel_sm errored (seed {i}): {e:?}");
                    }
                    assert_eq!(out, original, "parallel-SM output mismatch (seed {i})");
                    handle.join().unwrap();
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => panic!(
                    "read_parallel_sm did not return within 60s (seed {i}) — \
                     parallel-SM hang on the gzip EOF byte-alignment padding tail"
                ),
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    panic!("decode thread panicked (seed {i})")
                }
            }
        }
    }
}
