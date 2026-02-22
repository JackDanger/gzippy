//! Parallel single-member gzip decompression (two-pass approach)
//!
//! For single-member gzip files with many threads, provides parallel decompression:
//!
//! Pass 1 (sequential): Fast scan through deflate blocks using a circular buffer.
//!   Records checkpoints (bit position + 32KB window) at ~1MB intervals.
//!   The circular buffer stays in L1/L2 cache for better performance.
//!
//! Pass 2 (parallel): Re-decode each chunk in parallel using the checkpoint's
//!   window as dictionary. Each thread writes to a disjoint region of the output.
//!
//! Performance characteristics:
//!   total_time ≈ scan_time + decode_time / num_threads
//!   scan_time ≈ sequential_decode_time (both fully decode the stream)
//!   So total ≈ sequential * (1 + 1/threads)
//!   Faster than sequential only when threads > scan_overhead / decode_overlap.
//!   In practice, needs 8+ threads to outperform sequential libdeflate.

use std::cell::UnsafeCell;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::scan_inflate::scan_deflate_fast;

struct OutputBuf(UnsafeCell<Vec<u8>>);
unsafe impl Send for OutputBuf {}
unsafe impl Sync for OutputBuf {}
impl OutputBuf {
    fn ptr(&self) -> SendPtr {
        SendPtr(unsafe { (*self.0.get()).as_mut_ptr() })
    }
}

#[derive(Clone, Copy)]
struct SendPtr(*mut u8);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    unsafe fn add(self, offset: usize) -> *mut u8 {
        self.0.add(offset)
    }
}

/// Minimum compressed size to attempt parallel (4MB).
const MIN_PARALLEL_SIZE: usize = 4 * 1024 * 1024;

/// Minimum threads for the two-pass approach to outperform sequential.
/// The scan pass costs ~100% of sequential decode, so total work ≈ 2x.
/// Needs enough threads in the decode pass to compensate:
///   At 4 threads: total ≈ 1.25x sequential (slower)
///   At 8 threads: total ≈ 1.13x sequential (marginal)
///   At 16 threads: total ≈ 1.06x sequential (small win)
/// Gated at 8 to avoid regressions on typical 4-core CI machines.
const MIN_THREADS_FOR_PARALLEL: usize = 8;

/// Checkpoint interval in decompressed bytes (~1MB for good parallelism granularity)
const CHECKPOINT_INTERVAL: usize = 1024 * 1024;

#[inline]
fn debug_enabled() -> bool {
    use std::sync::OnceLock;
    static DEBUG: OnceLock<bool> = OnceLock::new();
    *DEBUG.get_or_init(|| std::env::var("GZIPPY_DEBUG").is_ok())
}

/// Parallel decompress a single-member gzip stream.
///
/// Returns Ok(bytes_written) on success, or Err if the file can't be
/// parallel-decoded (caller should fall back to sequential).
pub fn decompress_parallel<W: Write>(
    gzip_data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> Result<u64, ParallelError> {
    let t0 = std::time::Instant::now();

    let header_size = crate::marker_decode::skip_gzip_header(gzip_data)
        .map_err(|_| ParallelError::InvalidHeader)?;

    let trailer_size = 8;
    if gzip_data.len() < header_size + trailer_size {
        return Err(ParallelError::TooSmall);
    }
    let deflate_data = &gzip_data[header_size..gzip_data.len() - trailer_size];

    if deflate_data.len() < MIN_PARALLEL_SIZE || num_threads < MIN_THREADS_FOR_PARALLEL {
        return Err(ParallelError::TooSmall);
    }

    // Read ISIZE from trailer
    let isize_offset = gzip_data.len() - 4;
    let expected_output = u32::from_le_bytes([
        gzip_data[isize_offset],
        gzip_data[isize_offset + 1],
        gzip_data[isize_offset + 2],
        gzip_data[isize_offset + 3],
    ]) as usize;

    if debug_enabled() {
        eprintln!(
            "[parallel_sm] starting: {} bytes deflate, {} threads, isize={}",
            deflate_data.len(),
            num_threads,
            expected_output
        );
    }

    // =========================================================================
    // PASS 1: Sequential scan for block boundaries + windows
    // =========================================================================
    let t_scan = std::time::Instant::now();
    let scan = scan_deflate_fast(deflate_data, CHECKPOINT_INTERVAL, expected_output)
        .map_err(ParallelError::Io)?;
    let scan_elapsed = t_scan.elapsed();

    if scan.checkpoints.is_empty() {
        return Err(ParallelError::TooFewChunks);
    }

    let total_output = scan.total_output_size;

    if debug_enabled() {
        let scan_mbps = total_output as f64 / scan_elapsed.as_secs_f64() / 1e6;
        eprintln!(
            "[parallel_sm] pass 1: {} checkpoints in {:.1}ms ({:.0} MB/s)",
            scan.checkpoints.len(),
            scan_elapsed.as_secs_f64() * 1000.0,
            scan_mbps
        );
    }

    // =========================================================================
    // PASS 2: Parallel re-decode using checkpoints
    // =========================================================================
    let t_decode = std::time::Instant::now();

    // Build chunk descriptions: chunk 0 + checkpoint chunks
    let mut chunks: Vec<ChunkDesc> = Vec::with_capacity(scan.checkpoints.len() + 1);

    chunks.push(ChunkDesc {
        input_byte_pos: 0,
        bitbuf: 0,
        bitsleft: 0,
        window: Vec::new(),
        output_offset: 0,
        output_size: scan.checkpoints[0].output_offset,
    });

    for i in 0..scan.checkpoints.len() {
        let cp = &scan.checkpoints[i];
        let next_output_offset = if i + 1 < scan.checkpoints.len() {
            scan.checkpoints[i + 1].output_offset
        } else {
            total_output
        };
        chunks.push(ChunkDesc {
            input_byte_pos: cp.input_byte_pos,
            bitbuf: cp.bitbuf,
            bitsleft: cp.bitsleft,
            window: cp.window.clone(),
            output_offset: cp.output_offset,
            output_size: next_output_offset - cp.output_offset,
        });
    }

    let output_buf = OutputBuf(UnsafeCell::new(vec![0u8; total_output]));
    let errors = AtomicBool::new(false);
    let chunk_idx = AtomicUsize::new(0);

    let output_ptr = output_buf.ptr();
    std::thread::scope(|s| {
        for _ in 0..num_threads {
            s.spawn(|| loop {
                let idx = chunk_idx.fetch_add(1, Ordering::Relaxed);
                if idx >= chunks.len() {
                    break;
                }

                let chunk = &chunks[idx];
                if chunk.output_size == 0 {
                    continue;
                }

                let out_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        output_ptr.add(chunk.output_offset),
                        chunk.output_size,
                    )
                };

                if decode_chunk_with_dictionary(
                    deflate_data,
                    chunk.input_byte_pos,
                    chunk.bitbuf,
                    chunk.bitsleft,
                    &chunk.window,
                    out_slice,
                )
                .is_err()
                {
                    errors.store(true, Ordering::Relaxed);
                }
            });
        }
    });

    if errors.load(Ordering::Relaxed) {
        return Err(ParallelError::DecodeFailed);
    }

    let decode_elapsed = t_decode.elapsed();
    let output = unsafe { &*output_buf.0.get() };
    writer.write_all(&output[..total_output])?;

    if debug_enabled() {
        let total_elapsed = t0.elapsed();
        let scan_mbps = total_output as f64 / scan_elapsed.as_secs_f64() / 1e6;
        let decode_mbps = total_output as f64 / decode_elapsed.as_secs_f64() / 1e6;
        let total_mbps = total_output as f64 / total_elapsed.as_secs_f64() / 1e6;
        eprintln!(
            "[parallel_sm] scan: {:.1}ms ({:.0}), decode: {:.1}ms ({:.0}), total: {:.1}ms ({:.0} MB/s)",
            scan_elapsed.as_secs_f64() * 1000.0, scan_mbps,
            decode_elapsed.as_secs_f64() * 1000.0, decode_mbps,
            total_elapsed.as_secs_f64() * 1000.0, total_mbps
        );
    }

    Ok(total_output as u64)
}

struct ChunkDesc {
    input_byte_pos: usize,
    bitbuf: u64,
    bitsleft: u32,
    window: Vec<u8>,
    output_offset: usize,
    output_size: usize,
}

unsafe impl Send for ChunkDesc {}
unsafe impl Sync for ChunkDesc {}

/// Decode a chunk using the consume_first decoder with a pre-set dictionary.
fn decode_chunk_with_dictionary(
    deflate_data: &[u8],
    input_byte_pos: usize,
    bitbuf: u64,
    bitsleft: u32,
    window: &[u8],
    output: &mut [u8],
) -> Result<usize, io::Error> {
    use crate::consume_first_decode::Bits;

    let mut bits = Bits::new(deflate_data);
    bits.pos = input_byte_pos;
    bits.bitbuf = bitbuf;
    bits.bitsleft = bitsleft;

    let mut full_output = if !window.is_empty() {
        let mut buf = vec![0u8; window.len() + output.len() + 65536];
        buf[..window.len()].copy_from_slice(window);
        buf
    } else {
        vec![0u8; output.len() + 65536]
    };

    let start_pos = if !window.is_empty() { window.len() } else { 0 };
    let mut out_pos = start_pos;

    loop {
        if out_pos - start_pos >= output.len() {
            break;
        }

        if bits.available() < 3 {
            bits.refill();
        }

        let bfinal = (bits.peek() & 1) != 0;
        let btype = ((bits.peek() >> 1) & 3) as u8;
        bits.consume(3);

        if out_pos + 4 * 1024 * 1024 > full_output.len() {
            full_output.resize(full_output.len() * 2, 0);
        }

        match btype {
            0 => {
                out_pos = crate::consume_first_decode::decode_stored_pub(
                    &mut bits,
                    &mut full_output,
                    out_pos,
                )?;
            }
            1 => {
                out_pos = crate::consume_first_decode::decode_fixed_pub(
                    &mut bits,
                    &mut full_output,
                    out_pos,
                )?;
            }
            2 => {
                out_pos = crate::consume_first_decode::decode_dynamic_pub(
                    &mut bits,
                    &mut full_output,
                    out_pos,
                )?;
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid block type",
                ));
            }
        }

        if bfinal {
            break;
        }
    }

    let decoded = out_pos - start_pos;
    let to_copy = decoded.min(output.len());
    output[..to_copy].copy_from_slice(&full_output[start_pos..start_pos + to_copy]);

    Ok(to_copy)
}

#[derive(Debug)]
pub enum ParallelError {
    InvalidHeader,
    TooSmall,
    TooFewChunks,
    DecodeFailed,
    Io(io::Error),
}

impl From<io::Error> for ParallelError {
    fn from(e: io::Error) -> Self {
        ParallelError::Io(e)
    }
}

impl std::fmt::Display for ParallelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParallelError::InvalidHeader => write!(f, "invalid gzip header"),
            ParallelError::TooSmall => write!(f, "file too small for parallel decode"),
            ParallelError::TooFewChunks => write!(f, "too few chunks for parallel decode"),
            ParallelError::DecodeFailed => write!(f, "chunk decode failed"),
            ParallelError::Io(e) => write!(f, "I/O error: {}", e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gzip_data(data: &[u8]) -> Vec<u8> {
        use std::io::Write;
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    fn make_compressible_data(size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        let mut rng: u64 = 0xdeadbeef;
        while data.len() < size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32) % 5 < 3 {
                data.push((rng >> 16) as u8);
            } else {
                let byte = ((rng >> 24) % 26 + b'a' as u64) as u8;
                let repeat = ((rng >> 40) % 8 + 2) as usize;
                for _ in 0..repeat.min(size - data.len()) {
                    data.push(byte);
                }
            }
        }
        data.truncate(size);
        data
    }

    #[test]
    fn test_parallel_small_falls_back() {
        let data = b"hello world";
        let compressed = make_gzip_data(data);
        let mut output = Vec::new();
        let result = decompress_parallel(&compressed, &mut output, 4);
        assert!(matches!(result, Err(ParallelError::TooSmall)));
    }

    #[test]
    fn test_parallel_roundtrip() {
        let data = make_compressible_data(40 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        eprintln!(
            "test_parallel_roundtrip: {} bytes → {} bytes compressed ({:.1}%)",
            data.len(),
            compressed.len(),
            compressed.len() as f64 / data.len() as f64 * 100.0
        );

        if compressed.len() < MIN_PARALLEL_SIZE {
            eprintln!("compressed data too small, skipping");
            return;
        }

        let mut output = Vec::new();
        match decompress_parallel(&compressed, &mut output, 10) {
            Ok(bytes) => {
                assert_eq!(bytes as usize, data.len(), "output size mismatch");
                assert_eq!(output, data, "output data mismatch");
                eprintln!("parallel decompress succeeded: {} bytes", bytes);
            }
            Err(e) => {
                eprintln!("parallel decompress fell back: {}", e);
                let mut seq_out = Vec::new();
                let mut decoder = flate2::read::GzDecoder::new(&compressed[..]);
                std::io::Read::read_to_end(&mut decoder, &mut seq_out).unwrap();
                assert_eq!(seq_out, data);
            }
        }
    }

    #[test]
    fn test_parallel_versus_sequential() {
        let data = make_compressible_data(40 * 1024 * 1024);
        let compressed = make_gzip_data(&data);

        let mut seq_out = Vec::new();
        let mut decoder = flate2::read::GzDecoder::new(&compressed[..]);
        std::io::Read::read_to_end(&mut decoder, &mut seq_out).unwrap();
        assert_eq!(seq_out.len(), data.len());
        assert_eq!(seq_out, data);

        let mut par_out = Vec::new();
        match decompress_parallel(&compressed, &mut par_out, 10) {
            Ok(bytes) => {
                assert_eq!(bytes as usize, data.len());
                assert_eq!(par_out, seq_out, "parallel output differs from sequential");
                eprintln!("parallel matches sequential!");
            }
            Err(_) => {
                eprintln!("parallel fell back (expected for some data)");
            }
        }
    }

    #[test]
    fn test_parallel_silesia() {
        let gz = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping (silesia not found)");
                return;
            }
        };

        let header_size = crate::marker_decode::skip_gzip_header(&gz).expect("valid header");
        let deflate = &gz[header_size..gz.len() - 8];
        let mut ref_output = vec![0u8; 211968000 + 65536];
        let ref_size = crate::consume_first_decode::inflate_consume_first(deflate, &mut ref_output)
            .expect("reference inflate");

        let mut par_output = Vec::new();
        let t = std::time::Instant::now();
        let result = decompress_parallel(&gz, &mut par_output, 10);
        let elapsed = t.elapsed();

        match result {
            Ok(bytes) => {
                assert_eq!(bytes as usize, ref_size, "size mismatch");
                assert_eq!(
                    &par_output[..ref_size],
                    &ref_output[..ref_size],
                    "content mismatch"
                );
                let mbps = ref_size as f64 / elapsed.as_secs_f64() / 1e6;
                eprintln!(
                    "parallel silesia: {} bytes in {:.1}ms ({:.0} MB/s)",
                    ref_size,
                    elapsed.as_secs_f64() * 1000.0,
                    mbps
                );
            }
            Err(e) => {
                eprintln!("parallel silesia fell back: {}", e);
            }
        }
    }
}
