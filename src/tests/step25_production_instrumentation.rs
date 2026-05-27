//! Step 2.5: instrument production `read_stream` to find the actual
//! chunked-vs-monolithic gap on a real decode.
//!
//! This is an `--ignored` test that runs only when explicitly invoked.
//! It exercises a real ~1 MiB silesia-shaped decode via the parallel-SM
//! pipeline (the production caller pattern), then reads the
//! `READ_STREAM_*` counters to characterize the production calling
//! shape.
//!
//! Hypothesis from the advisor's third-pass verdict: production
//! callers in `gzip_chunk.rs:240-289` write into chunk.data's spare
//! capacity directly, so they MIGHT be calling read_stream with a
//! large (~128 KiB or more) output buffer rather than the tiny
//! chunked-bench shape (64 KiB). If avg bytes/call is large,
//! production is in monolithic-like mode and there's no chunked-mode
//! gap to fix.
//!
//! Run:
//! ```text
//! cargo test --release --features pure-rust-inflate -- step25 --ignored --nocapture
//! ```

#[cfg(test)]
#[cfg(all(target_arch = "x86_64", feature = "pure-rust-inflate"))]
mod tests {
    use crate::decompress::inflate::resumable::{
        READ_STREAM_BYTES_OUT, READ_STREAM_CALLS, READ_STREAM_OUTPUT_BUF_BYTES,
    };
    use std::io::Write;
    use std::sync::atomic::Ordering;

    fn make_payload(size: usize) -> Vec<u8> {
        // Mixed entropy — representative of real workloads.
        let mut data = Vec::with_capacity(size);
        let mut rng: u64 = 0xfeed_face_c0de_d00d;
        while data.len() < size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32) % 5 < 3 {
                data.push(((rng >> 16) % 95 + 32) as u8);
            } else {
                let len = ((rng >> 24) % 8 + 2) as usize;
                let b = ((rng >> 32) % 26 + b'a' as u64) as u8;
                for _ in 0..len.min(size - data.len()) {
                    data.push(b);
                }
            }
        }
        data.truncate(size);
        data
    }

    #[test]
    #[ignore = "diagnostic: run with --ignored --nocapture to see production calling shape"]
    fn step25_production_calling_shape() {
        // Reset counters.
        READ_STREAM_CALLS.store(0, Ordering::Relaxed);
        READ_STREAM_BYTES_OUT.store(0, Ordering::Relaxed);
        READ_STREAM_OUTPUT_BUF_BYTES.store(0, Ordering::Relaxed);

        // 24 MiB raw, ~14 MiB compressed — large enough to clear the
        // 10 MiB parallel-SM routing threshold AND get many chunks
        // through the worker pool.
        let payload = make_payload(24 * 1024 * 1024);
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(1));
        enc.write_all(&payload).unwrap();
        let compressed = enc.finish().unwrap();
        assert!(
            compressed.len() > 10 * 1024 * 1024,
            "compressed must exceed 10 MiB to route parallel-SM (got {})",
            compressed.len()
        );

        let _lock = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let mut out = Vec::with_capacity(payload.len() + 1024);
        crate::decompress::decompress_single_member(&compressed, &mut out, 4)
            .expect("parallel-SM decode");
        assert_eq!(out, payload, "production decode must be byte-perfect");

        let calls = READ_STREAM_CALLS.load(Ordering::Relaxed);
        let bytes_out = READ_STREAM_BYTES_OUT.load(Ordering::Relaxed);
        let buf_bytes = READ_STREAM_OUTPUT_BUF_BYTES.load(Ordering::Relaxed);
        let avg_out = if calls > 0 { bytes_out / calls } else { 0 };
        let avg_buf = if calls > 0 { buf_bytes / calls } else { 0 };
        let fill_pct = if buf_bytes > 0 {
            (bytes_out as f64 / buf_bytes as f64) * 100.0
        } else {
            0.0
        };

        println!("\n=== Step 2.5: Production calling-shape diagnostic ===");
        println!(
            "Input:  {} bytes raw, {} bytes compressed (~{}% ratio)",
            payload.len(),
            compressed.len(),
            (compressed.len() * 100) / payload.len()
        );
        println!("Decode: parallel-SM, T=4");
        println!();
        println!("read_stream call counters:");
        println!("  total calls:           {}", calls);
        println!(
            "  total bytes produced:  {} ({:.1} MiB)",
            bytes_out,
            bytes_out as f64 / (1024.0 * 1024.0)
        );
        println!(
            "  total buffer bytes:    {} ({:.1} MiB)",
            buf_bytes,
            buf_bytes as f64 / (1024.0 * 1024.0)
        );
        println!(
            "  avg bytes/call:        {} ({:.1} KiB)",
            avg_out,
            avg_out as f64 / 1024.0
        );
        println!(
            "  avg buffer/call:       {} ({:.1} KiB)",
            avg_buf,
            avg_buf as f64 / 1024.0
        );
        println!("  buffer fill rate:      {:.1}%", fill_pct);
        println!();
        println!("Interpretation:");
        if avg_out >= 1024 * 1024 {
            println!("  avg_out >= 1 MiB → production is MONOLITHIC-LIKE.");
            println!("  Our bench's 64 KiB chunked harness is UNREALISTIC.");
            println!("  Production does not see the chunked-mode gap.");
        } else if avg_out >= 64 * 1024 {
            println!("  avg_out in [64 KiB, 1 MiB] → production is CHUNKED.");
            println!("  Our 64 KiB chunked harness is REPRESENTATIVE.");
            println!("  Production sees the chunked-mode gap — worth fixing.");
        } else {
            println!("  avg_out < 64 KiB → production is HEAVILY CHUNKED.");
            println!("  Even worse than our bench; per-call overhead dominates.");
            println!("  Per-call setup reduction is the highest-leverage lever.");
        }
        println!();
    }
}
