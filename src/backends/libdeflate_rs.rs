//! Route A: pure-Rust port of libdeflate (the `streaming-libdeflate-rs`
//! crate, vendored as an optional dep gated on `streaming-libdeflate-rs`
//! feature) wrapped for in-memory gzip decompression.
//!
//! `plans/unified-decoder.md` §4.1: this is the cheapest bake-off route
//! — if it lands ≤5pp of ISA-L FFI on neurotic silesia, §4.5 outcome 1
//! triggers (vendor + ship; Routes B/C/D die).
//!
//! Validated correctness: feed through the three-oracle harness
//! (`tests::three_oracle_diff`); divergence from libdeflate-sys or
//! zlib-ng surfaces immediately.

#![cfg(feature = "streaming-libdeflate-rs")]

use streaming_libdeflate_rs::{
    decompress_gzip::libdeflate_gzip_decompress,
    libdeflate_alloc_decode_tables,
    streams::{
        deflate_chunked_buffer_input::DeflateChunkedBufferInput,
        deflate_chunked_buffer_output::DeflateChunkedBufferOutput,
    },
    DeflateInput,
};

/// Decompress a full gzip stream in memory.
///
/// Returns the decoded bytes. Errors propagate as
/// `LibdeflateError` from the underlying crate (BadData / ShortOutput
/// / InsufficientSpace per RFC 1952 violations).
pub fn decompress_gzip(gz: &[u8]) -> Result<Vec<u8>, String> {
    // Input adapter: serve the in-memory bytes via a refill closure.
    // The closure reads at most `buf.len()` bytes per call from the
    // gz cursor and returns how many it filled (0 = EOF).
    let mut cursor = 0usize;
    let mut input_stream = DeflateChunkedBufferInput::new(
        |buf: &mut [u8]| {
            let remaining = gz.len().saturating_sub(cursor);
            let take = remaining.min(buf.len());
            buf[..take].copy_from_slice(&gz[cursor..cursor + take]);
            cursor += take;
            take
        },
        // 64 KiB I/O chunk — matches the crate's typical use shape.
        64 * 1024,
    );

    // Output adapter: collect into a Vec via a callback per chunk.
    let collected: std::sync::Arc<std::sync::Mutex<Vec<u8>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let collected_clone = std::sync::Arc::clone(&collected);
    let mut output_stream = DeflateChunkedBufferOutput::new(
        move |chunk: &[u8]| -> Result<(), ()> {
            collected_clone.lock().unwrap().extend_from_slice(chunk);
            Ok(())
        },
        64 * 1024,
    );

    let mut decoder = libdeflate_alloc_decode_tables();

    // Drain in a loop in case the gzip stream has multiple members.
    while {
        input_stream.ensure_overread_length();
        input_stream.has_valid_bytes_slow()
    } {
        libdeflate_gzip_decompress(&mut decoder, &mut input_stream, &mut output_stream)
            .map_err(|e| format!("libdeflate_gzip_decompress: {e:?}"))?;
    }

    // Extract the collected bytes; drop the Arc so we can move out.
    drop(output_stream);
    let result = std::sync::Arc::try_unwrap(collected)
        .map_err(|_| "Arc<Mutex<Vec<u8>>> still has multiple refs".to_string())?
        .into_inner()
        .map_err(|e| format!("Mutex poisoned: {e}"))?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn compress(data: &[u8], level: u32) -> Vec<u8> {
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(level));
        enc.write_all(data).unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn route_a_decompress_empty() {
        let gz = compress(b"", 6);
        let out = decompress_gzip(&gz).unwrap();
        assert_eq!(out, b"");
    }

    #[test]
    fn route_a_decompress_short() {
        let payload = b"the quick brown fox jumps over the lazy dog";
        let gz = compress(payload, 6);
        let out = decompress_gzip(&gz).unwrap();
        assert_eq!(out, payload);
    }

    #[test]
    fn route_a_decompress_long_repetitive() {
        let payload = b"abcdefghijklmnopqrstuvwxyz".repeat(10_000);
        let gz = compress(&payload, 9);
        let out = decompress_gzip(&gz).unwrap();
        assert_eq!(out, payload);
    }

    #[test]
    fn route_a_decompress_random_l1() {
        let mut rng: u64 = 0xcafef00d_deadbeef;
        let mut data = Vec::with_capacity(64 * 1024);
        for _ in 0..(64 * 1024) {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            data.push((rng >> 24) as u8);
        }
        let gz = compress(&data, 1);
        let out = decompress_gzip(&gz).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn route_a_decompress_silesia_if_available() {
        let candidates = [
            "benchmark_data/silesia-large.gz",
            "benchmark_data/silesia.tar.gz",
            "benchmark_data/silesia-gzip.tar.gz",
        ];
        let Some(gz) = candidates.iter().find_map(|p| std::fs::read(p).ok()) else {
            eprintln!("[route-A silesia] no corpus, skipping");
            return;
        };
        // Ground truth via flate2.
        use std::io::Read;
        let mut gz_dec = flate2::read::GzDecoder::new(&gz[..]);
        let mut expected = Vec::new();
        gz_dec.read_to_end(&mut expected).unwrap();

        let got = decompress_gzip(&gz).unwrap();
        assert_eq!(got.len(), expected.len(), "length mismatch");
        const CHUNK: usize = 4096;
        for i in (0..got.len()).step_by(CHUNK) {
            let end = (i + CHUNK).min(got.len());
            assert_eq!(
                &got[i..end],
                &expected[i..end],
                "byte mismatch [{i}..{end}]"
            );
        }
        eprintln!("[route-A silesia] byte-perfect on {} bytes", got.len());
    }
}
