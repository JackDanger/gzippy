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
//! Kill-switch: `GZIPPY_WINDOW_SPARSITY=1` restores the old always-on behavior
//! (sparsity=true, compression=None/auto). Effect counter `SPARSITY_DECODE_COUNT`
//! in `chunk_data` tracks actual executions; visible via `GZIPPY_VERBOSE`.

use std::sync::OnceLock;

/// Returns `true` iff the `GZIPPY_WINDOW_SPARSITY=1` kill-switch is active,
/// restoring the pre-port always-on sparsity behavior.
fn window_sparsity_kill_switch() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("GZIPPY_WINDOW_SPARSITY").ok().as_deref() == Some("1"))
}

/// Decompress one single-member gzip buffer with the parallel chunk pipeline.
#[cfg(parallel_sm)]
pub fn read_parallel_sm<W: std::io::Write>(
    gzip_data: &[u8],
    writer: &mut W,
    out_fd: Option<i32>,
    parallelization: usize,
    target_compressed_chunk_bytes: usize,
) -> Result<ReadResult, ReadParallelSmError> {
    read_parallel_sm_inner(
        gzip_data,
        writer,
        out_fd,
        parallelization,
        target_compressed_chunk_bytes,
        None,
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
) -> Result<ReadResult, ReadParallelSmError> {
    read_parallel_sm_inner(
        gzip_data,
        writer,
        out_fd,
        parallelization,
        target_compressed_chunk_bytes,
        Some(bytes_written_out),
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
) -> Result<ReadResult, ReadParallelSmError> {
    use crate::decompress::parallel::chunk_data::ChunkConfiguration;
    use crate::decompress::parallel::chunk_fetcher;
    use crate::decompress::parallel::compressed_vector::CompressionType;
    use crate::decompress::parallel::gzip_format;

    // Slab auto-gate input: stored at EVERY decode entry (atomic, never
    // once-cached) so consecutive decodes with different T re-gate correctly.
    crate::decompress::parallel::rpmalloc_alloc::set_decode_threads(parallelization);

    // REMOVAL-ORACLE NOSTORE produces GARBAGE output bytes by design. Refuse to
    // write them to a REGULAR FILE so a forgotten env var can never leave a
    // plausible-looking corrupt artifact on disk — the sanctioned shape is
    // `-c ... > /dev/null` (char device / pipe). No-op when the knob is unset.
    if crate::decompress::parallel::removal_oracle::nostore_enabled() {
        if let Some(fd) = out_fd {
            let mut st: libc::stat = unsafe { std::mem::zeroed() };
            if unsafe { libc::fstat(fd, &mut st) } == 0
                && (st.st_mode & libc::S_IFMT) == libc::S_IFREG
            {
                return Err(ReadParallelSmError::DecodeFailed(
                    "GZIPPY_ORACLE_NOSTORE refuses regular-file output (bytes are \
                     garbage); redirect to /dev/null"
                        .into(),
                ));
            }
        }
    }

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
            ((numer + denom - 1) / denom).max(2).min(u16::MAX as u64) as u16
        }
    };

    // Faithful port of vendor `applyChunkDataConfiguration` at keepIndex=false
    // (ParallelGzipReader.hpp:1320-1330, tools/rapidgzip.cpp:47):
    //   windowSparsity   = keepIndex && windowSparsity = false
    //   windowCompressionType = keepIndex ? userValue : Some(NONE) = Some(NONE)
    // Kill-switch GZIPPY_WINDOW_SPARSITY=1 restores old always-on behavior.
    let sparsity = window_sparsity_kill_switch();
    let configuration = ChunkConfiguration {
        split_chunk_size: target_compressed_chunk_bytes,
        max_decoded_chunk_size: 20 * target_compressed_chunk_bytes,
        // Gate-2 CRC removal oracle: GZIPPY_ORACLE_CRC_OFF=1 disables the
        // per-chunk CRC32 accumulation (bytes stay correct). Default true.
        crc32_enabled: !crate::decompress::parallel::removal_oracle::crc_off_enabled(),
        // Default (keepIndex=false, vendor benchmark): sparsity OFF — skip the
        // 32 KiB `getUsedWindowSymbols` scan per chunk finalize.
        window_sparsity: sparsity,
        // Default (keepIndex=false): store windows uncompressed (CompressionType::None).
        // Kill-switch (old behavior): None → auto-select (heuristic in window_compression_type()).
        window_compression_type: if sparsity {
            None
        } else {
            Some(CompressionType::None)
        },
        expansion_ratio_ceil,
    };

    // Phase-timing: gzip envelope (header+footer) parsed + chunk config built.
    crate::decompress::parallel::phase_timing::mark("envelope_parsed");

    // Clean-window oracle (GZIPPY_CLEAN_WINDOW_ORACLE=1, default OFF): decode
    // every chunk with its true predecessor window — no speculation, no marker
    // bootstrap, no append_markered/absorb_isal_tail/narrow copies — to size
    // whether the marker pipeline is the rapidgzip gap. CRC/size still verified
    // below, so the known-window path's correctness is checked too.
    // THIN-T1 PRODUCTION PATH (route-A scaffold shed). At T==1 the decode is
    // strictly sequential front-to-back, so EVERY block already has its 32 KiB
    // predecessor window — the parallel block-finder / WindowMap / marker arming
    // / prefetch / threadpool scaffold is pure overhead. The thin serial rolling-
    // window driver over the SAME shared `decode_chunk` kernel sheds it (route-A
    // oracle: thin/libdeflate≈1.08 vs prod/libdeflate≈1.22 on this box). CRC32 +
    // ISIZE are verified below exactly as for the parallel path, and bytes_written
    // is captured so the multi-member-misroute resume net still works.
    // `GZIPPY_NO_THIN_T1=1` forces the legacy parallel path at T1 (AB re-verify).
    let force_thin_oracle = std::env::var_os("GZIPPY_THIN_T1_ORACLE").is_some();
    // T1 serial-eligible: strictly T==1, no clean-window oracle, not force-thin.
    let t1_serial = parallelization <= 1
        && std::env::var_os("GZIPPY_NO_THIN_T1").is_none()
        && std::env::var_os("GZIPPY_CLEAN_WINDOW_ORACLE").is_none();
    // T1-MONOLITH (igzip-shaped single-buffer path): a deliberate, T1-gated
    // divergence from the rapidgzip chunk pipeline toward the igzip monolith
    // (plans/T1-MONOLITH-DIVERGENCE-LEDGER.md). It is OPT-IN (GZIPPY_MONOLITH=1),
    // NOT the default: the pre-registered falsifier measurement (Intel+AMD,
    // plans/T1-MONOLITH-RESULTS.md) found it FALSIFIED — it REGRESSES past the
    // legacy thin-T1 driver because the single ISIZE buffer first-touches the
    // whole output (~4× igzip's page faults), the cost igzip's streaming small
    // reused buffer avoids. Kept opt-in only so the falsifier stays reproducible;
    // the production T1 default remains thin-T1. T>1 NEVER takes this path.
    // Legacy full-ISIZE monolith (FALSIFIED — fault-storm): opt-in only via
    // GZIPPY_MONOLITH=1 so the prior falsifier stays reproducible.
    let use_old_monolith =
        t1_serial && !force_thin_oracle && std::env::var_os("GZIPPY_MONOLITH").is_some();
    // T1-MONOLITH-STREAMING: one continuous serial decode that STREAMS through a
    // small resident buffer (no fault-storm — fixes the prior full-ISIZE
    // monolith). It IS byte-exact, fault-storm-free, and DOES shed the per-chunk
    // scaffold INSTRUCTIONS (fulcrum optgate: instr/byte RESOLVED-improved on
    // silesia/nasa/monorepo/squishy), but fulcrum optgate REFUSED the wall win as
    // INSTRUCTION-ONLY — cyc/byte did NOT improve beyond spread (only 2.8-4.6% of
    // the gz→igzip cyc/byte gap closed; residual is kernel cycle-efficiency, NOT
    // the scaffold). Per the governing policy (FULCRUM is the sole oracle; a
    // default-path change must ride a gated WALL win), it is OPT-IN
    // (GZIPPY_STREAM_MONOLITH=1), NOT the default; production T1 stays thin-T1
    // (byte-identical to before this cycle). See
    // plans/T1-MONOLITH-FINISH-RESULTS.md. Native-only; T>1 never takes it.
    let use_stream_monolith = {
        #[cfg(not(isal_clean_tail))]
        {
            t1_serial
                && !force_thin_oracle
                && !use_old_monolith
                && std::env::var_os("GZIPPY_STREAM_MONOLITH").is_some()
        }
        #[cfg(isal_clean_tail)]
        {
            false
        }
    };
    let use_thin_t1 = (t1_serial && !use_old_monolith && !use_stream_monolith) || force_thin_oracle;
    let drive_result = if use_old_monolith {
        chunk_fetcher::drive_monolith_t1(
            deflate_data,
            writer,
            configuration,
            expected_size,
            bytes_written_out,
        )
    } else if use_stream_monolith {
        #[cfg(not(isal_clean_tail))]
        {
            chunk_fetcher::drive_monolith_streaming_t1(
                deflate_data,
                writer,
                configuration,
                expected_size,
                bytes_written_out,
            )
        }
        #[cfg(isal_clean_tail)]
        {
            unreachable!("stream monolith is native-only")
        }
    } else if use_thin_t1 {
        chunk_fetcher::drive_thin_t1_oracle(deflate_data, writer, configuration, bytes_written_out)
    } else if std::env::var_os("GZIPPY_CLEAN_WINDOW_ORACLE").is_some() {
        chunk_fetcher::drive_clean_window_oracle(
            deflate_data,
            writer,
            parallelization,
            configuration,
        )
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
        )
    } else {
        chunk_fetcher::drive(deflate_data, writer, out_fd, parallelization, configuration)
    };
    let (total_crc, total_size) =
        drive_result.map_err(|e| ReadParallelSmError::DecodeFailed(format!("{e:?}")))?;

    // FIXED-SLEEP coordination-isolation mode produces GARBAGE output (zeros)
    // by design — it is a wall-only measurement of the coordination chain
    // with decode replaced by a fixed sleep. Skip CRC/size verification in
    // that mode only (zero production change when unset).
    let sleep_mode = crate::decompress::parallel::decode_bypass::sleep_decode_enabled();
    // REMOVAL-ORACLE NOSTORE: output bytes are garbage (stores elided) — CRC and
    // ISIZE cannot match. Skip verification in that mode only. NODECODE replay
    // stays VERIFIED: its replay hits are byte-correct by construction and its
    // misses run the real decode, so verification doubles as the honesty check.
    let nostore_mode = crate::decompress::parallel::removal_oracle::nostore_enabled();
    // CRC removal oracle: calculator disabled → total_crc is 0, so skip ONLY the
    // CRC verify (bytes/ISIZE still correct and verified).
    let crc_off_mode = crate::decompress::parallel::removal_oracle::crc_off_enabled();
    if !sleep_mode && !nostore_mode {
        if total_size != expected_size {
            return Err(ReadParallelSmError::SizeMismatch {
                expected: expected_size,
                actual: total_size,
            });
        }
        if !crc_off_mode && total_crc != expected_crc {
            return Err(ReadParallelSmError::CrcMismatch {
                expected: expected_crc,
                actual: total_crc,
            });
        }
    }

    // Phase-timing: trailer CRC32 + ISIZE verified (the finalize/verify phase).
    crate::decompress::parallel::phase_timing::mark("crc_verified");

    // Allocator visibility (GZIPPY_RPMALLOC_STATS=1): show whether the
    // decode's span allocations were warm-reused (cache) or re-mapped (faults).
    crate::decompress::parallel::rpmalloc_alloc::dump_global_stats("post-decode");

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
        let result = read_parallel_sm(&gzip, &mut out, None, 4, 512 * 1024).unwrap();
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
                let res = read_parallel_sm(&gzip, &mut out, None, 16, 1024 * 1024);
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
